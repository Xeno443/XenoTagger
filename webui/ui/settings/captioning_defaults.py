"""Settings -> Captioning defaults."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui import state
from ui.context import BuildContext


def collect_settings(trigger_word, overwrite_existing) -> dict:
    return dict(trigger_word=trigger_word, overwrite_existing=bool(overwrite_existing))


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Captioning defaults", id="captioning-defaults-settings") as captioning_defaults_settings_tab:
        with gr.Group():
            trigger_word = gr.Textbox(label="Default trigger word", value=cfg.trigger_word)
            overwrite_existing = gr.Checkbox(
                label="Overwrite existing captions by default", value=cfg.overwrite_existing
            )

    return SimpleNamespace(
        tab=captioning_defaults_settings_tab,
        tab_label="Captioning defaults",
        tab_id="captioning-defaults-settings",
        gated=True,
        save_inputs=[trigger_word, overwrite_existing],
    )
