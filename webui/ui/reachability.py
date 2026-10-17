"""Server-reachability gating shared by the frame (app.py's startup and
reachability-refresh handlers) and the llama backend's autostart
(backends/llama/settings_ui.py): the Models/Single/Batch tabs'
interactive state, the Llama sub-tab's Start/Stop buttons, and the
popups for reachability transitions.

Today this is all about the llama backend, the only captioner - it
becomes "is the active captioner ready" with the v0.9 step-3 caption
pipeline.
"""

from __future__ import annotations

import gradio as gr

from backends.llama.server import ServerStatus
from backends.llama.session import (
    cached_controllable, is_server_managed_by_us, managed_base_url, note_controllable_transition,
    update_reachability_tracking,
)
from core.config import AppConfig
from ui.downloads_queue import download_abort_all
from ui.notify import notify


def track_reachability_transitions(cfg: AppConfig, status: ServerStatus, controllable: bool) -> None:
    """Shows the one-shot popups backends.llama.session's two transition
    trackers return - "model management is disabled" (which also aborts
    every queued download - switching away from a controllable state
    likely means they're no longer wanted) and "server stopped
    responding". Must run before the status bar's next status_label() read
    (it updates the crash flag that label reads)."""
    message = note_controllable_transition(controllable, cfg, status)
    if message:
        notify(message)
        download_abort_all()
    message = update_reachability_tracking(status, cfg)
    if message:
        notify(message, level="warning")


def tab_gating_updates(cfg: AppConfig, status: ServerStatus, controllable: bool) -> tuple:
    """(models_tab, single_tab, batch_tab, start_btn, end_btn) updates.
    Models is disabled unless controllable; Single/Batch unless something
    is reachable at all."""
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = llama_lifecycle_button_updates(cfg, status)
    return models_tab_update, single_batch_update, single_batch_update, start_btn_update, end_btn_update


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


def gating_from_cache(cfg: AppConfig, status: ServerStatus) -> tuple:
    """tab_gating_updates() with controllable read from the cache."""
    return tab_gating_updates(cfg, status, cached_controllable(cfg))
