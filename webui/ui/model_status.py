"""The one-line model status + "Show in Models" button a backend's Settings
page shows (Backend.model_status). The button's click is wired by the
frame (app.py), since it switches to another top-level tab; the backend's
settings_ui returns it as its handle's models_btn."""

from __future__ import annotations

import gradio as gr

from core.backend import Backend
from ui import state
from ui.context import BuildContext


def model_status_row(ctx: BuildContext, backend: Backend) -> gr.Button:
    # Plain Row, not Group - a Button in a Group gets squashed into one
    # slab with its neighbors (see CLAUDE.md's gr.Group() gotcha).
    with gr.Row():
        status_md = gr.Markdown(backend.model_status(state.current_cfg))
        show_btn = gr.Button("Show in Models", scale=0, min_width=160)
    # Cheap (config + a file-exists check), and the active model can
    # change from the Models tab while this page isn't visible.
    ctx.on_tick(lambda: backend.model_status(state.current_cfg), [], [status_md])
    return show_btn
