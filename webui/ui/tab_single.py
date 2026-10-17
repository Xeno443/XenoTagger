"""Single image tab."""

from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from core import config as config_mod
from core.captioner import caption_image
from backends.llama.client import ClientError
from backends.llama.server import ServerError, is_healthy
from ui import state
from ui.context import BuildContext
from backends.llama.session import display_base_url, get_client
from ui.operation import (
    COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET, RUN_BUSY, RUN_IDLE, operation_blocked_by, operation_end,
    operation_interrupt_click, operation_start,
)
from ui.operation import run_controls_update as run_controls_update_for

log = logging.getLogger("app")  # fixed name, see app.py


def run_single_ui(image_path, trigger_word_override: str):
    blocked = operation_blocked_by()
    if blocked:
        yield gr.update(), blocked, RUN_BUSY, gr.update(), gr.update(), gr.update()
        return

    if not image_path:
        yield "", "Please choose an image first.", gr.update(), gr.update(), gr.update(), gr.update()
        return

    operation_start("single", "Single-image captioning")
    try:
        log.info("Single-image caption requested: %s", image_path)
        cfg = state.current_cfg
        base_url = display_base_url(cfg)
        already_up = is_healthy(base_url)
        running_state = (RUN_IDLE, COL_HIDDEN, COL_SHOWN, INTERRUPT_RESET)
        if cfg.server_mode == "external":
            yield "", ("Processing..." if already_up else "Connecting to external server..."), *running_state
        else:
            yield "", ("Processing..." if already_up else "Starting server (loading model)..."), *running_state

        idle_state = (RUN_IDLE, COL_SHOWN, COL_HIDDEN, INTERRUPT_RESET)
        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Single-image caption: server error: %s", exc)
            yield "", f"Server error: {exc}", *idle_state
            return

        if not already_up:
            yield "", "Processing...", gr.update(), gr.update(), gr.update(), gr.update()

        # A plain blocking caption_image() call can't yield mid-call, so
        # showing a live stage change ("captioning" -> "tagging (Hydra)")
        # in single_infotext needs the same background-thread + queue
        # shape run_batch_ui already uses - the queue is what lets on_stage
        # (called synchronously inside caption_image(), from this worker
        # thread) reach a yield in this generator without blocking on it.
        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def on_stage_cb(stage: str) -> None:
            q.put(stage)

        def worker():
            try:
                result_holder["caption"], result_holder["result"], _, _ = caption_image(
                    image_path, client, cfg, trigger_word=trigger_word_override,
                    on_stage=on_stage_cb,
                )
            except ClientError as exc:
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            stage = q.get()
            if stage is None:
                break
            yield gr.update(), f"{stage}...", gr.update(), gr.update(), gr.update(), gr.update()
        thread.join()

        if "error" in result_holder:
            log.warning("Single-image caption failed for %s: %s", image_path, result_holder["error"])
            yield "", f"Captioning failed: {result_holder['error']}", *idle_state
            return

        caption = result_holder["caption"]
        result = result_holder["result"]
        speed = f", {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""

        if result.truncated:
            # llama-server clamps actual generation to whatever fits in
            # (context_size - prompt_tokens), even if that's below the
            # requested max_tokens. If it stopped short of max_tokens, raising
            # max_tokens further won't help - context_size is the real ceiling.
            if result.completion_tokens < cfg.max_tokens:
                advice = (
                    f"prompt used {result.prompt_tokens} tokens, leaving no room "
                    f"in the {cfg.context_size}-token context for the requested "
                    f"{cfg.max_tokens} - raise Context size in Settings"
                )
            else:
                advice = "raise Max tokens in Settings for a full caption"
            status = (
                f"Finished in {result.elapsed_s:.1f}s, but CUT OFF at {result.completion_tokens} "
                f"tokens{speed} ({advice})"
            )
        else:
            status = f"Finished in {result.elapsed_s:.1f}s ({result.completion_tokens} tokens{speed})"
        if result.resize_note:
            status = f"Resized {result.resize_note}. {status}"
        yield caption, status, *idle_state
    finally:
        operation_end()


def interrupt_single_ui():
    action = operation_interrupt_click("single")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


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
    # interactive=False at construction (option B from the design
    # discussion - the simpler fallback, not a dedicated splash
    # overlay): the very first paint, before demo.load's chain has
    # had a chance to run even once, would otherwise show every
    # tab clickable regardless of real server state. Corrected
    # within the first real check (_startup_reachability_ui,
    # wired at the bottom of app.py's build_app) - this same gap applied to
    # Models and Batch too (see their own Tab() calls below), not
    # just Single-image.
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


def run_controls_update(op, reachable: bool) -> tuple:
    """Updates for build()'s run_controls - see operation.run_controls_update."""
    return run_controls_update_for("single", op, reachable)
