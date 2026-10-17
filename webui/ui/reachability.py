"""Server-reachability gating shared by the frame (app.py's startup and
reachability-refresh handlers) and the llama backend's autostart
(backends/llama/settings_ui.py): the Single/Batch tabs' interactive
state, the Llama sub-tab's Start/Stop buttons, and the popup for a
reachability transition.

Today this is all about the llama backend, the only captioner - it
becomes "is the active captioner ready" with the v0.9 step-3 caption
pipeline. The Models tab isn't gated: downloads work in every mode, and
only llama's own "Set as active model" depends on the server
(backends.llama.session.set_active_blocked).
"""

from __future__ import annotations

import gradio as gr

from backends.llama.server import ServerStatus
from backends.llama.session import is_server_managed_by_us, managed_base_url, update_reachability_tracking
from core.config import AppConfig
from ui.notify import notify


def track_reachability_transitions(cfg: AppConfig, status: ServerStatus) -> None:
    """Shows the one-shot "server stopped responding" popup backends.llama.
    session's transition tracker returns. Must run before the status bar's
    next status_label() read (it updates the crash flag that label reads)."""
    message = update_reachability_tracking(status, cfg)
    if message:
        notify(message, level="warning")


def tab_gating_updates(cfg: AppConfig, status: ServerStatus) -> tuple:
    """(single_tab, batch_tab, start_btn, end_btn) updates. Single/Batch
    are disabled unless something is reachable at all."""
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = llama_lifecycle_button_updates(cfg, status)
    return single_batch_update, single_batch_update, start_btn_update, end_btn_update


def llama_lifecycle_button_updates(cfg: AppConfig, status: ServerStatus):
    """(start_btn_update, end_btn_update) for the Llama settings tab's
    two lifecycle buttons - shared by every tab_gating_updates() caller
    so they can't disagree. Start is only
    ever useful in managed mode with something installed but not already
    running (installed-and-not-running, not installed-and-running or
    external, where it'd have nothing to do); End only when this session
    actually owns whatever's currently running - see
    is_server_managed_by_us's docstring on why that's the one thing that
    gates it, not just "something's running".

    Start also tracks Save-settings' primary/orange styling, but only
    while it's actually clickable - variant, not just interactive, so
    "ready to click" reads as visually distinct from "nothing to do right
    now" rather than relying on interactive=False's subtler dimming alone
    to carry that. End stays plain (like Restart app) regardless of
    state - it's the less common action, not the one to draw the eye to."""
    can_start = cfg.server_mode == "managed" and status.installed and not status.reachable
    start_update = gr.update(interactive=can_start, variant="primary" if can_start else "secondary")
    owned = cfg.server_mode == "managed" and status.reachable and is_server_managed_by_us(managed_base_url(cfg))
    end_update = gr.update(interactive=owned)
    return start_update, end_update
