"""Single image tab - one uploaded image through the started pipeline
(core.pipeline.caption_pieces), shown as the rendered caption. Nothing is
written next to the image; Save caption writes the rendered text."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from core import config as config_mod
from core import pipeline
from ui import pipeline_status, state
from ui.context import BuildContext
from ui.operation import (
    COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET, RUN_BUSY, RUN_IDLE, interrupt_button_ui, operation_blocked_by,
    operation_end, operation_should_stop, operation_start,
)
from ui.operation import run_controls_update as run_controls_update_for

log = logging.getLogger("app")  # fixed name, see app.py


def run_single_ui(image_path, trigger_word: str):
    blocked = operation_blocked_by()
    if blocked:
        yield gr.update(), blocked, RUN_BUSY, gr.update(), gr.update(), gr.update()
        return
    if not image_path:
        yield "", "Please choose an image first.", gr.update(), gr.update(), gr.update(), gr.update()
        return
    plan = pipeline.running_plan()
    if plan is None:
        yield "", "Start the pipeline first (Settings -> Pipeline).", gr.update(), gr.update(), gr.update(), gr.update()
        return

    operation_start("single", "Single-image captioning")
    try:
        log.info("Single-image caption requested: %s", image_path)
        running_state = (RUN_IDLE, COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET)
        idle_state = (RUN_IDLE, COL_SHOWN, COL_HIDDEN, INTERRUPT_RESET)
        yield "", "Processing...", *running_state

        # caption_pieces blocks, so it runs in a worker thread; on_stage
        # reaches this generator through the queue, so the infotext can
        # show which step is running.
        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def worker():
            try:
                result_holder["result"] = pipeline.caption_pieces(
                    plan, image_path, {pipeline.TRIGGER_PIECE: (trigger_word or "").strip()},
                    should_stop=lambda: operation_should_stop("single") or bool(pipeline_status.snapshot().not_ready),
                    on_stage=q.put,
                )
            except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash app
                log.exception("Single-image caption failed")
                result_holder["error"] = str(exc)
            q.put(None)

        start = time.monotonic()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while (stage := q.get()) is not None:
            yield gr.update(), f"{stage}...", gr.update(), gr.update(), gr.update(), gr.update()
        thread.join()

        if "error" in result_holder:
            yield "", f"Captioning failed: {result_holder['error']}", *idle_state
            return
        result = result_holder["result"]
        caption = pipeline.render_pieces(plan, result.pieces)
        elapsed = f"{time.monotonic() - start:.1f}s"
        # Interrupt clicked: a second click stops the pipeline, which shows
        # up as a broken connection - still just "interrupted" to the user.
        if result.interrupted or operation_should_stop("single"):
            status = f"Interrupted after {elapsed}."
        elif result.errors:
            status = f"Finished in {elapsed} with problems: " + " ".join(result.errors)
        else:
            notes = "; ".join(f"{piece}: {out.note}" for piece, out in result.outputs.items() if out.note)
            status = f"Finished in {elapsed}" + (f" ({notes})" if notes else "") + "."
        yield caption, status, *idle_state
    finally:
        operation_end()


def interrupt_single_ui():
    return interrupt_button_ui("single")


PROJECT_ROOT = config_mod.ROOT_DIR  # one level above webui/


def _is_gradio_temp_upload(path: Path) -> bool:
    """True if `path` is a copy Gradio made in its own temp cache rather
    than the real file - which is always true for a browser drag-and-drop
    or click-to-upload, since browsers never expose a local file's real
    absolute path to the page. gr.Image gives us that temp copy's path,
    not the original's, so we can't just save next to it in that case.
    """
    import tempfile

    gradio_temp = Path(os.environ.get("GRADIO_TEMP_DIR") or (Path(tempfile.gettempdir()) / "gradio"))
    try:
        path.resolve().relative_to(gradio_temp.resolve())
        return True
    except ValueError:
        return False


def clear_single_result_ui() -> tuple[str, str]:
    """Wired to the image box's own .change() - fires on upload, clear, or
    replace, so a stale caption/status from a previous image never lingers
    next to a new or now-empty image slot.
    """
    return "", ""


def save_single_ui(image_path, caption_text: str) -> str:
    if not image_path:
        return "No image loaded."
    if not caption_text.strip():
        return "Nothing to save."

    src = Path(image_path)
    if not _is_gradio_temp_upload(src):
        txt_path = src.with_suffix(".txt")
        txt_path.write_text(caption_text, encoding="utf-8")
        log.info("Saved caption to %s", txt_path)
        return f"Saved to {txt_path}"

    log.debug("save_single_ui(): %s is a Gradio temp upload - showing save dialog", src)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    save_path = filedialog.asksaveasfilename(
        initialdir=str(PROJECT_ROOT),
        initialfile=src.with_suffix(".txt").name,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    root.destroy()

    if not save_path:
        log.info("Save caption cancelled by user")
        return "Save cancelled."
    Path(save_path).write_text(caption_text, encoding="utf-8")
    log.info("Saved caption to %s", save_path)
    return f"Saved to {save_path}"


def build(ctx: BuildContext):
    cfg = state.current_cfg
    # interactive=False at construction: the very first paint, before
    # demo.load's chain has run even once, would otherwise show the tab
    # clickable regardless of whether the pipeline is started. Corrected
    # by the frame's startup check (app.py's _startup_ui); Batch does the
    # same.
    with gr.Tab("Single image", id="single", interactive=False) as single_tab:
        with gr.Row(equal_height=True):
            with gr.Column():
                single_image = gr.Image(
                    type="filepath", label="Image", sources=["upload"]
                )
            with gr.Column():
                single_trigger = gr.Textbox(
                    label="Trigger word",
                    value=cfg.trigger_word,
                )
                single_caption = gr.Textbox(
                    label="Caption", lines=20, interactive=True
                )
        # Confirmed live: a Row of exactly two plain gr.Column()s (no
        # CSS at all) aligns correctly with Row A above. So instead of
        # stacking Caption/Interrupt inside one shared Column (which
        # needed CSS overrides that threw alignment off), each gets
        # its OWN Column, and it's the whole COLUMN's visibility that
        # toggles - not just the button inside it. At any moment
        # exactly two of these three Columns are visible (Caption+Save
        # idle, Interrupt+Save running) - always that same proven
        # "Row of two plain Columns" shape, whichever two they are.
        with gr.Row():
            with gr.Column() as single_run_col:
                single_run_btn = gr.Button("Caption", variant="primary")
            # Separate component from single_run_btn on purpose - see
            # the Operation tracking section: an Interrupt that shares
            # the same button/event as the long-running call would sit
            # queued behind it and never actually reach the server
            # while captioning is in flight.
            with gr.Column(visible=False) as single_interrupt_col:
                single_interrupt_btn = gr.Button("Interrupt", variant="stop")
            with gr.Column():
                single_save_btn = gr.Button("Save caption", interactive=False)
        single_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        single_run_btn.click(
            run_single_ui,
            [single_image, single_trigger],
            [
                single_caption, single_infotext,
                single_run_btn, single_run_col, single_interrupt_col, single_interrupt_btn,
            ],
        )
        single_interrupt_btn.click(interrupt_single_ui, [], [single_interrupt_btn])
        single_save_btn.click(
            save_single_ui, [single_image, single_caption], [single_infotext]
        )
        single_caption.change(
            lambda text: gr.update(interactive=bool(text.strip())),
            [single_caption], [single_save_btn],
        )
        single_image.change(clear_single_result_ui, [], [single_caption, single_infotext])

    return SimpleNamespace(
        tab=single_tab,
        run_controls=[single_run_btn, single_run_col, single_interrupt_col, single_interrupt_btn],
    )


def run_controls_update(op, ready: bool) -> tuple:
    """Updates for build()'s run_controls - see operation.run_controls_update."""
    return run_controls_update_for("single", op, ready)
