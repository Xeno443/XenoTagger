"""Batch processing tab - a folder through the started pipeline
(core.batch.run_batch, shared with the CLI)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gradio as gr
from PIL import Image as PILImage

from core import pipeline
from core.batch import find_images, run_batch
from ui import pipeline_status, state
from ui.context import BuildContext
from ui.dialogs import browse_directory_ui
from ui.notify import notify
from ui.operation import (
    COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET, RUN_BUSY, RUN_IDLE, interrupt_button_ui, operation_blocked_by,
    operation_end, operation_should_stop, operation_start,
)
from ui.operation import run_controls_update as run_controls_update_for

log = logging.getLogger("app")  # fixed name, see app.py


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def run_batch_ui(directory_str, overwrite, trigger_word):
    blocked = operation_blocked_by()
    if blocked:
        yield blocked, "", None, "", RUN_BUSY, gr.update(), gr.update(), gr.update()
        return

    log.info("Batch requested: %s (overwrite=%s)", directory_str, overwrite)
    directory = Path(directory_str) if directory_str else None
    if not directory or not directory.is_dir():
        log.warning("Batch: not a directory: %s", directory_str)
        yield f"Not a directory: {directory_str}", "", None, "", gr.update(), gr.update(), gr.update(), gr.update()
        return
    plan = pipeline.running_plan()
    if plan is None:
        yield "Start the pipeline first (Settings -> Pipeline).", "", None, "", *(gr.update(),) * 4
        return
    images = find_images(directory)
    if not images:
        log.warning("Batch: no images found in %s", directory)
        yield "No images found in that directory.", "", None, "", *(gr.update(),) * 4
        return

    operation_start("batch", "Batch captioning")
    try:
        running_state = (RUN_IDLE, COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET)
        idle_state = (RUN_IDLE, COL_SHOWN, COL_HIDDEN, INTERRUPT_RESET)
        yield "Processing...", "", None, "", *running_state
        total = len(images)

        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def progress_cb(i, total_, path, status, text, detail):
            q.put(("progress", i, path, status, text, detail))

        def on_stage_cb(path: Path, stage: str) -> None:
            q.put(("stage", path, stage))

        def should_stop() -> bool:
            # Also stops when a pipeline backend crashed - otherwise every
            # remaining image would just collect the same error.
            return operation_should_stop("batch") or bool(pipeline_status.snapshot().not_ready)

        def worker():
            try:
                result_holder["result"] = run_batch(
                    plan, directory, trigger_word=(trigger_word or "").strip(), overwrite=overwrite,
                    progress_cb=progress_cb, should_stop=should_stop, on_stage=on_stage_cb,
                )
            except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash app
                log.exception("Batch failed")
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        start_time = time.monotonic()
        thread.start()

        # One infotext line that always carries the fullest currently
        # known context (file, step, N/total, avg, ETA) - a separate
        # stage-only line would be replaced by the next progress line
        # before anyone could read it.
        last_file = ""
        last_image = None
        last_caption = ""
        processed = 0
        current_file = ""
        current_stage: Optional[str] = None

        def infotext_text() -> str:
            elapsed = time.monotonic() - start_time
            avg = elapsed / processed if processed else 0.0
            eta = avg * (total - processed)
            text = current_file or f"{total} image(s)"
            text += f" · {processed}/{total} processed · avg {avg:.1f}s/image · ETA {_format_duration(eta)}"
            if current_stage:
                text += f" - {current_stage}"
            return text

        while (item := q.get()) is not None:
            if item[0] == "stage":
                _, path, stage = item
                current_file, current_stage = path.name, stage
                yield infotext_text(), last_file, last_image, last_caption, *(gr.update(),) * 4
                continue

            _, i, path, status, text, detail = item
            processed, current_file, current_stage = i, path.name, None
            # Skipped images were never sent to a model - nothing new to
            # preview, keep showing the last one that was.
            if status not in ("skipped", "conflict"):
                last_file = path.name
                try:
                    # Gradio won't serve a path outside its CWD/temp dir -
                    # batch folders can be anywhere, so hand it image data.
                    with PILImage.open(path) as img:
                        last_image = img.copy()
                except OSError as exc:
                    log.warning("Could not load preview for %s: %s", path, exc)
                    last_image = None
                if status == "done":
                    last_caption = text or ""
                elif status == "failed":
                    last_caption = f"(not captioned - {detail}; see .txt.issue)"
                else:
                    last_caption = "(interrupted)"
            yield infotext_text(), last_file, last_image, last_caption, *(gr.update(),) * 4

        thread.join()
        if "error" in result_holder:
            message = f"Unexpected error: {result_holder['error']}"
            notify(f"Batch captioning: {message}", level="error")
            yield message, last_file, last_image, last_caption, *idle_state
            return
        r = result_holder["result"]
        crashed = pipeline_status.snapshot().not_ready
        verb = f"Stopped ({', '.join(crashed)} stopped unexpectedly)" if crashed else ("Stopped" if r.aborted else "Done")
        summary = (
            f"{verb}: {r.done} captioned, {r.skipped} skipped, {r.failed} failed "
            f"in {_format_duration(time.monotonic() - start_time)}"
        )
        if r.conflicts:
            summary += (f". {len(r.conflicts)} image(s) skipped because another image has the same "
                        f"name: {', '.join(p.name for p in r.conflicts[:6])}{'...' if len(r.conflicts) > 6 else ''}")
        # Fires even if the user's still on the Batch tab - start a long
        # batch, go do something else, still find out when it's done.
        notify(f"Batch captioning {summary}", level="warning" if (r.failed or r.conflicts or crashed) else "info")
        yield summary, last_file, last_image, last_caption, *idle_state
    finally:
        operation_end()


def interrupt_batch_ui():
    return interrupt_button_ui("batch")

def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Batch processing", interactive=False) as batch_tab:
        with gr.Row():
            batch_dir = gr.Textbox(label="Directory of images", scale=4)
            batch_browse_btn = gr.Button("Browse...", scale=1)

        with gr.Row():
            batch_trigger = gr.Textbox(
                label="Trigger word",
                value=cfg.trigger_word,
            )
            batch_overwrite = gr.Checkbox(
                label="Overwrite existing captions", value=cfg.overwrite_existing
            )

        with gr.Row():
            batch_last_image = gr.Image(
                label="Preview", interactive=False, show_label=False,
                buttons=[],
            )
            with gr.Column():
                batch_last_file = gr.Textbox(label="Last file processed", interactive=False)
                batch_last_caption = gr.Textbox(label="Last caption created", interactive=False)

        # Same two-Column swap as the Single-image tab (see that tab
        # and app.py's own module docstring for why: a Row of two
        # plain gr.Column()s is the only thing confirmed to align
        # predictably, no CSS needed) - kept here for consistency even
        # though this row has no third sibling to misalign against.
        with gr.Row():
            with gr.Column() as batch_run_col:
                batch_run_btn = gr.Button("Run batch", variant="primary")
            with gr.Column(visible=False) as batch_interrupt_col:
                batch_interrupt_btn = gr.Button("Interrupt", variant="stop")
            with gr.Column():
                # Gated purely on "is batch_dir a real directory" -
                # there's no cheap way to know whether the last batch
                # run in it actually succeeded (batch.py is stateless,
                # see its own module docstring), so that's
                # deliberately not part of the check. Click wiring is
                # in app.py (it crosses into the Review tab) - see
                # _send_batch_to_review_ui's own docstring.
                batch_send_review_btn = gr.Button("Send to Review", interactive=False)

        batch_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        batch_browse_btn.click(browse_directory_ui, [batch_dir], [batch_dir])
        batch_run_btn.click(
            run_batch_ui,
            [batch_dir, batch_overwrite, batch_trigger],
            [
                batch_infotext, batch_last_file, batch_last_image, batch_last_caption,
                batch_run_btn, batch_run_col, batch_interrupt_col, batch_interrupt_btn,
            ],
        )
        batch_interrupt_btn.click(interrupt_batch_ui, [], [batch_interrupt_btn])
        batch_dir.change(
            lambda d: gr.update(interactive=bool(d and Path(d).is_dir())),
            [batch_dir], [batch_send_review_btn],
        )

    return SimpleNamespace(
        tab=batch_tab,
        run_controls=[batch_run_btn, batch_run_col, batch_interrupt_col, batch_interrupt_btn],
        dir=batch_dir,
        send_review_btn=batch_send_review_btn,
    )


def run_controls_update(op, ready: bool) -> tuple:
    """Updates for build()'s run_controls - see operation.run_controls_update."""
    return run_controls_update_for("batch", op, ready)
