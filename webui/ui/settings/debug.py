"""Settings -> Debug."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui import state
from ui.context import BuildContext


def collect_settings(debug_tab_enabled) -> dict:
    return dict(debug_tab_enabled=bool(debug_tab_enabled))


def load_values(cfg) -> tuple:
    return (cfg.debug_tab_enabled,)


def build(ctx: BuildContext):
    cfg = state.current_cfg
    # id="debug-settings" - the one never-gated sub-tab, where entering
    # Settings lands during a caption job (ui.tab_settings'
    # _settings_subtab_entry_ui): unlike the other sub-tabs (all disabled
    # while a caption is running - see _settings_gating_ui), this is just a
    # checkbox with no live effect of its own, so there's never a reason
    # to gate it.
    with gr.Tab("Debug", id="debug-settings") as debug_settings_tab:
        with gr.Group():
            debug_tab_enabled = gr.Checkbox(
                label="Enable Debuglog tab (requires app restart)",
                value=cfg.debug_tab_enabled,
            )

    return SimpleNamespace(
        tab=debug_settings_tab,
        tab_label="Debug",
        tab_id="debug-settings",
        gated=False,
        save_inputs=[debug_tab_enabled],
    )
