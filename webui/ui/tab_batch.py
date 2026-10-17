"""Batch processing tab."""

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

from core.batch import find_images, run_batch
from backends.llama.server import ServerError, is_healthy
from ui import state
from ui.context import BuildContext
from ui.dialogs import browse_directory_ui
from backends.llama.session import display_base_url, get_client
from ui.notify import notify
from ui.operation import (
    COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET, RUN_BUSY, RUN_IDLE, operation_blocked_by, operation_end,
    operation_interrupt_click, operation_should_stop, operation_start,
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


def run_batch_ui(directory_str, overwrite, trigger_word_override):
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

    operation_start("batch", "Batch captioning")
    try:
        cfg = state.current_cfg
        base_url = display_base_url(cfg)
        already_up = is_healthy(base_url)
        running_state = (RUN_IDLE, COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET)
        if cfg.server_mode == "external":
            yield ("Processing..." if already_up else "Connecting to external server..."), "", None, "", *running_state
        else:
            yield ("Processing..." if already_up else "Starting server (loading model)..."), "", None, "", *running_state

        idle_state = (RUN_IDLE, COL_SHOWN, COL_HIDDEN, INTERRUPT_RESET)
        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Batch: server error: %s", exc)
            yield f"Server error: {exc}", "", None, "", *idle_state
            return

        if not already_up:
            yield "Processing...", "", None, "", gr.update(), gr.update(), gr.update(), gr.update()

        images = find_images(directory)
        if not images:
            log.warning("Batch: no images found in %s", directory)
            yield "No images found in that directory.", "", None, "", *idle_state
            return
        total = len(images)

        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def progress_cb(i, total_, path, status, caption, resize_note):
            q.put(("progress", i, path, status, caption, resize_note))

        def on_stage_cb(path: Path, stage: str) -> None:
            q.put(("stage", path, stage))

        def worker():
            try:
                result_holder["result"] = run_batch(
                    directory, client, state.current_cfg,
                    overwrite=overwrite,
                    trigger_word=trigger_word_override,
                    progress_cb=progress_cb,
                    should_stop=lambda: operation_should_stop("batch"),
                    on_stage=on_stage_cb,
                )
            except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash app
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        start_time = time.monotonic()
        thread.start()

        # processed/current_file/current_stage/resize_suffix all feed one
        # combined infotext line, rebuilt on every queue item (stage or
        # progress) - a stage-only line was tried first and reverted: the
        # progress line that follows it lands a moment later and replaces
        # it outright, so the "N/total processed · avg · ETA" context was
        # only ever visible for a blink between two images. One line that
        # always carries the fullest currently-known context avoids that.
        last_file = ""
        last_image = None
        last_caption = ""
        processed = 0
        current_file = ""
        current_stage: Optional[str] = None
        resize_suffix = ""

        def infotext_text() -> str:
            elapsed = time.monotonic() - start_time
            avg = elapsed / processed if processed else 0.0
            eta = avg * (total - processed)
            text = current_file or f"{total} image(s)"
            if resize_suffix:
                text += f" · resized {resize_suffix}"
            text += f" · {processed}/{total} processed · avg {avg:.1f}s/image · ETA {_format_duration(eta)}"
            if current_stage:
                text += f" - {current_stage}"
            return text

        while True:
            item = q.get()
            if item is None:
                break
            if item[0] == "stage":
                _, path, stage = item
                if path.name != current_file:
                    # A new image started - the previous one's resize
                    # note (if any) no longer applies to what's showing.
                    resize_suffix = ""
                current_file = path.name
                current_stage = stage
                yield infotext_text(), last_file, last_image, last_caption, gr.update(), gr.update(), gr.update(), gr.update()
                continue

            _, i, path, status, caption, resize_note = item
            processed = i
            current_file = path.name
            current_stage = None  # between images - no stage in flight until the next on_stage
            resize_suffix = resize_note or ""

            # Skipped images were never sent to the model at all (already
            # captioned) - nothing to preview, so leave the preview
            # fields showing whatever the last actually-processed image
            # was instead of decoding and flashing up an unrelated,
            # untouched file.
            if status != "skipped":
                last_file = path.name
                try:
                    # Gradio refuses to serve a raw path outside its CWD/
                    # temp dir (InvalidPathError) - batch directories can
                    # be anywhere on disk, so hand it decoded image data
                    # instead of a path at all.
                    with PILImage.open(path) as img:
                        last_image = img.copy()
                except OSError as exc:
                    log.warning("Could not load preview for %s: %s", path, exc)
                    last_image = None
                if status == "ok":
                    last_caption = caption or ""
                elif status == "truncated":
                    last_caption = "(truncated - not saved, see .txt.issue)"
                else:
                    last_caption = "(failed - see .txt.issue)"
            yield infotext_text(), last_file, last_image, last_caption, gr.update(), gr.update(), gr.update(), gr.update()

        thread.join()
        if "error" in result_holder:
            message = f"Unexpected error: {result_holder['error']}"
            notify(f"Batch captioning: {message}", level="error")
            yield message, last_file, last_image, last_caption, *idle_state
        else:
            r = result_holder["result"]
            total_elapsed = time.monotonic() - start_time
            verb = "Aborted" if r.aborted else "Done"
            summary = (
                f"{verb}: {r.processed} captioned, {r.truncated} truncated, "
                f"{r.skipped} skipped, {r.failed} failed "
                f"in {_format_duration(total_elapsed)}"
            )
            # Fires even if the user's still watching the Batch tab - a
            # long batch is exactly the kind of thing this was asked for
            # (start it, go do something else, still find out when it's
            # done or hit trouble without having to stay on this tab).
            notify(f"Batch captioning {summary}", level="warning" if r.failed else "info")
            yield summary, last_file, last_image, last_caption, *idle_state
    finally:
        operation_end()


def interrupt_batch_ui():
    action = operation_interrupt_click("batch")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


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


def run_controls_update(op, reachable: bool) -> tuple:
    """Updates for build()'s run_controls - see operation.run_controls_update."""
    return run_controls_update_for("batch", op, reachable)
