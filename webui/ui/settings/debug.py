"""Settings -> Debug."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui import state
from ui.context import BuildContext


def build(ctx: BuildContext):
    cfg = state.current_cfg
    # id="debug-settings" - the one fixed, never-gated
    # fallback landing spot for _fallback_tab_safety_ui
    # (app.py, see its own comment): unlike the other four
    # sub-tabs (all disabled while a caption is
    # running - see ui.tab_settings' _settings_gating_ui) and unlike the
    # top-level Debuglog tab (conditionally
    # hidden by debug_tab_enabled), this sub-tab is just a
    # checkbox with no live effect of its own, so there's
    # never a reason to gate it.
    with gr.Tab("Debug", id="debug-settings"):
        with gr.Group():
            debug_tab_enabled = gr.Checkbox(
                label="Enable Debuglog tab (requires app restart)",
                value=cfg.debug_tab_enabled,
            )

    return SimpleNamespace(debug_tab_enabled=debug_tab_enabled)
