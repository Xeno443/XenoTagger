"""Settings -> Llama: llama.cpp install, managed/external server config,
sampling and image resizing. The only file in backends/llama/ that
imports Gradio or ui/ - loaded by ui.tab_settings via
LlamaBackend.settings_ui_module (see ui.tab_settings for the contract).

No Start/Stop or autostart here: the pipeline starts and stops
llama-server (Settings -> Pipeline), and prompts are its variants there.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Optional

import gradio as gr

from core import config as config_mod
from core.config import AppConfig
from core.downloads import format_size
from ui import state
from ui.context import BuildContext
from ui.notify import notify
from ui.model_status import model_status_row

from . import BACKEND
from .install import (
    BACKENDS as LLAMA_BACKENDS, DEFAULT_BACKEND as LLAMA_DEFAULT_BACKEND,
    install as llama_install_run, installed_info as llama_installed_info, plan_install as llama_plan_install,
)
from .server import get_loaded_model_name, is_healthy
from .session import managed_base_url

log = logging.getLogger(__name__)

def _verify_external_ui(external_url: str) -> str:
    """Tests whatever URL is currently in the External server URL field -
    not necessarily state.current_cfg.external_url, since the user may be
    checking a value they haven't saved yet. A real check beyond a bare
    /health 200: also asks /props (get_loaded_model_name, server.py) so
    the result says what model is actually loaded there, not just that
    *something* answered."""
    base_url = external_url.rstrip("/")
    if not is_healthy(base_url):
        return f"No server responding at {base_url}."
    model = get_loaded_model_name(base_url)
    if model:
        return f"Reachable - model loaded: {model}."
    return f"Reachable at {base_url}, but couldn't read which model is loaded (no /props response)."


# --------------------------------------------------------- llama.cpp install
#
# A single-operation worker, not a queue (only one llama.cpp install ever
# makes sense at a time) - its own lock/thread/abort-flag rather than
# reusing ui.downloads_queue's _download_lock, because an install overwrites the exact
# binary/DLLs a running llama-server.exe might have open, which is a
# fundamentally different hazard than a curated model download (which
# never touches anything currently in use). _llama_install_start_ui
# refuses to start while the managed port is reachable for exactly this
# reason.

_install_lock = threading.RLock()
_install_thread: Optional[threading.Thread] = None
_install_state = {"phase": "idle", "bytes_done": 0, "bytes_total": 0}
_install_abort_requested = False
# One-shot (message, level) popped by the next _llama_install_status_ui
# tick (success or failure) - not left sitting in _install_state, because
# that would mean every 2s tick after a completed install keeps
# re-announcing the same message into llama_lifecycle_infotext, stomping
# on whatever the Verify button wrote there in the meantime.
# Mirrors ui.downloads_queue's _download_needs_refresh single-shot-flag idiom. Also
# used to fire a one-shot notify() popup on completion/failure, so
# finding out an install finished doesn't require still being on the
# Llama sub-tab when it does.
_install_announce: Optional[tuple[str, str]] = None  # (message, level)


def _install_worker(backend_id: str) -> None:
    global _install_thread, _install_announce
    label = LLAMA_BACKENDS[backend_id].label

    def on_progress(written: int) -> None:
        with _install_lock:
            _install_state["bytes_done"] = written

    def on_phase(phase: str) -> None:
        with _install_lock:
            _install_state["phase"] = phase

    try:
        plan = llama_plan_install(backend_id)
        with _install_lock:
            _install_state["bytes_total"] = plan.total_bytes
        completed = llama_install_run(
            plan, should_abort=lambda: _install_abort_requested,
            on_progress=on_progress, on_phase=on_phase,
        )
        message = f"Installed {label} ({plan.build_tag})." if completed else "Install aborted."
        level = "info"
    except Exception as exc:
        log.exception("llama.cpp install failed (backend=%s)", backend_id)
        message = f"Install failed: {exc}"
        level = "error"
    with _install_lock:
        _install_state.update(phase="idle", bytes_done=0, bytes_total=0)
        _install_announce = (message, level)
        _install_thread = None


def _llama_install_start_ui(backend_id: str) -> str:
    """Kicks off a background install and returns immediately - actual
    progress/completion is picked up by the status_timer via
    _llama_install_status_ui below, the same fire-and-forget shape as
    download_enqueue's own click handlers."""
    global _install_thread, _install_abort_requested
    if not backend_id:
        return "No backend selected."
    with _install_lock:
        if _install_thread is not None and _install_thread.is_alive():
            return "An install is already in progress."
    if is_healthy(managed_base_url(state.current_cfg)):
        return "Stop the pipeline first (Settings -> Pipeline) - llama-server is running."
    with _install_lock:
        _install_abort_requested = False
        _install_state.update(phase="planning", bytes_done=0, bytes_total=0)
        _install_thread = threading.Thread(target=_install_worker, args=(backend_id,), daemon=True)
        _install_thread.start()
    return f"Installing {LLAMA_BACKENDS[backend_id].label} - resolving latest release..."


def abort_background() -> None:
    """Aborts a running llama.cpp install. Bare flag-set, no UI return
    value - mirrors ui.downloads_queue's download_abort_all(), called both
    by _llama_install_abort_ui below and, as this sub-tab's
    abort_background hook, by ui.tab_settings' restart_app_ui (which
    doesn't need or want a UI-facing message, the process is about to be
    replaced)."""
    global _install_abort_requested
    with _install_lock:
        _install_abort_requested = True


def _llama_install_abort_ui() -> str:
    with _install_lock:
        if _install_thread is None or not _install_thread.is_alive():
            return "Nothing to abort."
    abort_background()
    return "Aborting install..."


def _llama_install_status_ui():
    """(row_visibility, progress_html, lifecycle_infotext_update,
    backend_dropdown_update, install_btn_update, abort_btn_update,
    installed_caption_update) - wired to the same 2s status_timer as the
    download-status row, plus called directly after Install/Abort clicks
    for immediate feedback.

    lifecycle_infotext_update only ever carries a real value on the tick
    that pops a pending _install_announce (every other tick is a no-op
    gr.update(), so it never overwrites a Verify message that isn't
    actually stale). backend_dropdown_update only pushes a value right
    after an install actually completes (the same announce tick) - NOT on
    every idle poll, which would otherwise fight a user's own in-progress
    dropdown pick before they've clicked Install/Reinstall. The rest are
    read-only derived state (label text and interactive= only), safe to
    recompute every tick."""
    global _install_announce
    with _install_lock:
        phase = _install_state["phase"]
        bytes_done = _install_state["bytes_done"]
        bytes_total = _install_state["bytes_total"]
        announce = _install_announce
        _install_announce = None
    running = phase in ("planning", "downloading", "extracting")

    if announce is not None:
        notify(announce[0], level=announce[1])
    infotext_update = gr.update(value=announce[0]) if announce is not None else gr.update()

    info = llama_installed_info()
    backend_update = (
        gr.update(value=info["backend_id"])
        if (announce is not None and info is not None and info.get("backend_id") in LLAMA_BACKENDS)
        else gr.update()
    )
    install_btn_update = gr.update(
        value="Reinstall llama.cpp" if info is not None else "Install llama.cpp",
        interactive=not running,
    )
    abort_btn_update = gr.update(interactive=running)
    installed_caption_update = gr.update(value=_llama_installed_text())

    if not running:
        status_row_update = gr.update(visible=False)
        status_html_update = gr.update(value="")
    else:
        if phase == "planning":
            html = "<div>Resolving latest llama.cpp release...</div>"
        elif phase == "extracting":
            html = "<div>Extracting...</div>"
        else:
            total = max(bytes_total, 1)
            pct = min(100, int(bytes_done * 100 / total))
            html = (
                f"<progress value=\"{bytes_done}\" max=\"{total}\" style=\"width:100%;\"></progress>"
                f"<div>Downloading: {pct}% ({format_size(bytes_done)} / {format_size(bytes_total)})</div>"
            )
        status_row_update = gr.update(visible=True)
        status_html_update = gr.update(value=html)
    return (
        status_row_update, status_html_update, infotext_update,
        backend_update, install_btn_update, abort_btn_update, installed_caption_update,
    )


def _llama_installed_text() -> str:
    info = llama_installed_info()
    if info is None:
        return "Not installed yet."
    return f"Installed: {info.get('label', info.get('backend_id'))} ({info.get('build_tag')})"


def _llama_backend_dropdown_default() -> str:
    """Pre-selects the backend dropdown to whatever's actually installed,
    rather than always defaulting to LLAMA_DEFAULT_BACKEND - so a user who
    already has, say, CUDA 12.4 installed sees that reflected instead of
    the newest-by-default CUDA 13.3 sitting there looking unrelated to
    their "Reinstall" button."""
    info = llama_installed_info()
    if info and info.get("backend_id") in LLAMA_BACKENDS:
        return info["backend_id"]
    return LLAMA_DEFAULT_BACKEND


def _llama_install_button_label() -> str:
    return "Reinstall llama.cpp" if llama_installed_info() is not None else "Install llama.cpp"


def collect_settings(
    server_mode, server_port, external_url, n_gpu_layers, context_size, extra_server_args,
    temperature, top_p, max_tokens, request_timeout,
    resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
) -> dict:
    """AppConfig updates from build()'s save_inputs, in that order.
    model_name/mmproj_name are deliberately NOT settable from this form -
    they live entirely in the Models tab ("Set as active model"), so Save
    carries forward whatever's already configured."""
    return dict(
        server_mode=server_mode,
        server_port=int(server_port),
        external_url=external_url,
        n_gpu_layers=n_gpu_layers.strip() or "auto",
        context_size=int(context_size),
        extra_server_args=extra_server_args,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        request_timeout=int(request_timeout),
        resize_enabled=bool(resize_enabled),
        resize_target_mp=float(resize_target_mp),
        snap_enabled=bool(snap_enabled),
        snap_multiple=int(snap_multiple),
    )


def load_values(cfg: AppConfig) -> tuple:
    return (
        cfg.server_mode, cfg.server_port, cfg.external_url, cfg.n_gpu_layers, cfg.context_size,
        cfg.extra_server_args, cfg.temperature, cfg.top_p, cfg.max_tokens, cfg.request_timeout,
        cfg.resize_enabled, cfg.resize_target_mp, cfg.snap_enabled, cfg.snap_multiple,
    )


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Llama", id=f"{BACKEND.id}-settings") as llama_settings_tab:
        gr.Markdown(config_mod.LLAMA_TAB_INTRO)
        installed_backend_text = gr.Markdown(_llama_installed_text())
        show_in_models_btn = model_status_row(ctx, BACKEND)
        with gr.Group():
            server_mode = gr.Radio(
                choices=[
                    ("Managed: the pipeline starts and stops a local llama-server", "managed"),
                    ("External: a llama-server you run yourself", "external"),
                ],
                value=cfg.server_mode,
                label="Server mode",
                elem_id="server-mode-radio",
            )

        # gr.Column, not gr.Group - only visible= toggling is
        # needed here, not Group's bordered/connected-siblings
        # styling, which squashes button rows into a merged slab
        # (see the gr.Group() gotcha).
        with gr.Column(visible=(cfg.server_mode == "managed")) as managed_server_group:
            # The dropdown+Install+Abort row stays OUTSIDE the Group
            # below - a Row of buttons inside a Group loses the gap
            # between them and merges into one flat bar.
            with gr.Row():
                llama_backend_dropdown = gr.Dropdown(
                    choices=[(b.label, bid) for bid, b in LLAMA_BACKENDS.items()],
                    value=_llama_backend_dropdown_default(),
                    show_label=False, container=False, scale=3,
                )
                install_llama_btn = gr.Button(_llama_install_button_label(), scale=2)
                abort_install_btn = gr.Button("Abort install", scale=1, interactive=False)
            with gr.Group():
                with gr.Row(visible=False) as llama_install_status_row:
                    llama_install_status_text = gr.HTML(container=False)
                server_port = gr.Number(
                    label="Port (managed mode)", value=cfg.server_port, precision=0
                )
                with gr.Row():
                    n_gpu_layers = gr.Textbox(
                        label="GPU layers ('auto', 'all', or an exact number)",
                        value=cfg.n_gpu_layers,
                    )
                    context_size = gr.Number(label="Context size", value=cfg.context_size, precision=0)
                extra_server_args = gr.Textbox(
                    label="Extra llama-server arguments", value=cfg.extra_server_args
                )

        with gr.Column(visible=(cfg.server_mode == "external")) as external_server_group:
            external_url = gr.Textbox(label="External server URL", value=cfg.external_url)
            verify_external_btn = gr.Button("Verify")

        with gr.Group():
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=cfg.temperature, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=cfg.top_p, label="Top-p")
                max_tokens = gr.Number(label="Max tokens", value=cfg.max_tokens, precision=0)
            request_timeout = gr.Number(
                label="Request timeout (seconds)", value=cfg.request_timeout, precision=0
            )

        # Image resizing only affects what's sent to llama-server - Hydra
        # and Furrence2 do their own image preparation.
        with gr.Group():
            resize_enabled = gr.Checkbox(
                label="Downscale oversized images before sending them to the model",
                value=cfg.resize_enabled,
            )
            resize_target_mp = gr.Slider(
                0.1, 4.0, value=cfg.resize_target_mp, step=0.1,
                label="Target resolution (megapixels)",
            )
            with gr.Row():
                snap_enabled = gr.Checkbox(label="Scale to multiple of", value=cfg.snap_enabled)
                snap_multiple = gr.Number(value=cfg.snap_multiple, precision=0, show_label=False)

        # Bottom of the tab, right above the shared Save settings/Restart
        # app row (outside this Tabs() block - see ui.tab_settings).
        llama_lifecycle_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # Purely cosmetic (which fields are relevant to the selected
        # mode) - server_mode itself stays the one source of truth.
        server_mode.change(
            lambda mode: (gr.update(visible=mode == "managed"), gr.update(visible=mode == "external")),
            [server_mode],
            [managed_server_group, external_server_group],
        )
        verify_external_btn.click(_verify_external_ui, [external_url], [llama_lifecycle_infotext])

        _llama_install_status_outputs = [
            llama_install_status_row, llama_install_status_text, llama_lifecycle_infotext,
            llama_backend_dropdown, install_llama_btn, abort_install_btn, installed_backend_text,
        ]
        install_llama_btn.click(
            _llama_install_start_ui, [llama_backend_dropdown], [llama_lifecycle_infotext]
        ).then(_llama_install_status_ui, [], _llama_install_status_outputs)
        abort_install_btn.click(
            _llama_install_abort_ui, [], [llama_lifecycle_infotext]
        ).then(_llama_install_status_ui, [], _llama_install_status_outputs)

    ctx.on_tick(_llama_install_status_ui, [], _llama_install_status_outputs)
    ctx.on_load(_llama_install_status_ui, [], _llama_install_status_outputs)

    return SimpleNamespace(
        tab=llama_settings_tab,
        tab_label="Llama",
        tab_id=f"{BACKEND.id}-settings",
        gated=True,
        models_btn=show_in_models_btn,
        # collect_settings()'s parameter order.
        save_inputs=[
            server_mode, server_port, external_url, n_gpu_layers, context_size, extra_server_args,
            temperature, top_p, max_tokens, request_timeout,
            resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
        ],
    )