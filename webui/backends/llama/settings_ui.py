"""Settings -> Llama: llama.cpp install, managed/external server config,
Start/Stop, the prompt/sampling fields, and llama-server's launch-time
autostart. The only file in backends/llama/ that imports Gradio or ui/ -
loaded by ui.tab_settings via LlamaBackend.settings_ui_module, which
expects build(ctx), collect_settings(), and optionally after_save() and
abort_background() (see ui.tab_settings)."""

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
from ui.context import REACHABILITY, BuildContext
from ui.notify import notify
from ui.model_status import model_status_row
from ui.reachability import tab_gating_updates

from . import BACKEND
from .install import (
    BACKENDS as LLAMA_BACKENDS, DEFAULT_BACKEND as LLAMA_DEFAULT_BACKEND,
    install as llama_install_run, installed_info as llama_installed_info, plan_install as llama_plan_install,
)
from .server import get_loaded_model_name, is_healthy
from .session import (
    cached_status, is_server_managed_by_us, managed_base_url, refresh_reachability_cache, stop_managed,
    try_start_managed_llama,
)

log = logging.getLogger(__name__)


def _autostart_managed_llama_ui():
    """Registered as this backend's launch-time autoload step (see
    build()) - core.lifecycle.autoload_order runs it after the fixed-VRAM
    backends' (Hydra's), and app.py chains it after the startup check, so
    the initial UI has already rendered with tabs correctly
    enabled/disabled - autostart is a background convenience layered on
    top of that correct initial state, never a substitute for it. Only
    actually does anything for managed mode, installed, not already
    running, and the checkbox on (BACKEND.wants_autoload) - the one case a
    user would otherwise have to click Start for themselves. Silent no-op
    in every other case - no popup for "didn't autostart", only for
    actually doing something. Returns app.py's reachability outputs (see
    ui.context.REACHABILITY), and sets current_tab_label_state to "Single
    image" when it redirects there - a server-pushed switch doesn't fire
    main_tabs.select() on its own (see app.py's _refresh_reachability_ui
    docstring for the bug this class of fix addresses)."""
    cfg = state.current_cfg
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.skip())
    if not BACKEND.wants_autoload(cfg):
        return noop
    notify("Starting managed llama-server automatically (autostart is enabled)...")
    ok, _ = try_start_managed_llama(cfg)
    if not ok:
        return noop
    return (*tab_gating_updates(cfg, cached_status()), gr.update(selected="single"), "Single image")


def _start_managed_llama_ui():
    """A generator, not a plain function - starting can take a while
    (model load), and a single return value would leave the status box
    silent/frozen for the whole duration with no sign anything's
    happening (the chained _refresh_reachability_ui after this click only
    fires once this generator is fully exhausted, so it can't help with
    mid-wait feedback either - same pattern already used for
    run_single_ui's "Starting server (loading model)..." yield, for the
    exact same reason)."""
    yield "Starting llama-server (this can take a while if a model needs to load)..."
    _, message = try_start_managed_llama(state.current_cfg)
    yield message


def _end_managed_llama_ui():
    """Only ever meaningfully clickable when is_server_managed_by_us() is
    true (see its interactive= wiring below) - but stop_managed() is
    itself already safe to call regardless, since it only ever touches
    session.py's _session["managed"], which is only ever non-None when this session
    is the one that actually spawned the process (see server.py's
    docstring). The ownership check here is just to report accurately
    what happened for a stale/racing click, not to gate the action. A
    generator for the same reason _start_managed_llama_ui is - stopping
    can take up to ManagedServer.stop()'s own 10s wait before it falls
    back to a hard kill."""
    yield "Stopping llama-server..."
    owned = is_server_managed_by_us(managed_base_url(state.current_cfg))
    stop_managed()
    refresh_reachability_cache(state.current_cfg)
    yield "llama-server stopped." if owned else "Nothing to stop - this session doesn't own the running server."


def _verify_external_ui(external_url: str) -> str:
    """Tests whatever URL is currently in the External server URL field -
    not necessarily state.current_cfg.external_url, since the user may be
    checking a value they haven't saved yet. A real check beyond a bare
    /health 200: also asks /props (get_loaded_model_name, server.py -
    the same call the status bar's "Model: X" already uses) so the result
    says what model is actually loaded there, not just that *something*
    answered."""
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
# on whatever the Start/End/Verify buttons wrote there in the meantime.
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
        return "Stop the running llama-server first (\"Stop llama server\" above) before installing/reinstalling."
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
    installed_caption_update, autostart_checkbox_update) - wired to the
    same 2s status_timer as the download-status row, plus called
    directly after Install/Abort clicks for immediate feedback.

    lifecycle_infotext_update only ever carries a real value on the tick
    that pops a pending _install_announce (every other tick is a no-op
    gr.update(), so it never overwrites a Start/Stop/Verify message that
    isn't actually stale). backend_dropdown_update only pushes a value
    right after an install actually completes (the same announce tick) -
    NOT on every idle poll, which would otherwise fight a user's own
    in-progress dropdown pick before they've clicked Install/Reinstall.
    install_btn_update/abort_btn_update/installed_caption_update/
    autostart_checkbox_update are read-only derived state (label text
    and interactive= only), safe to recompute every tick with no such
    risk - autostart_managed_llama only makes sense once something is
    actually installed to autostart."""
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
    autostart_update = gr.update(interactive=info is not None)

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
        autostart_update,
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
    autostart_managed_llama, prompt_template, temperature, top_p, max_tokens, request_timeout,
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
        autostart_managed_llama=bool(autostart_managed_llama),
        prompt_template=prompt_template,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        request_timeout=int(request_timeout),
    )


def after_save(old_cfg: AppConfig, new_cfg: AppConfig) -> Optional[str]:
    """Reacts to a saved change in the server-process-affecting fields and
    returns what to append to "Settings saved." (None if nothing to add).
    Prompt/sampling fields are just written through, live immediately, no
    process interaction. Server-process fields (mode, port, GPU layers,
    context size, extra args) each need a specific reaction: switching to
    external stops any managed server we own; switching to managed never
    auto-starts (the user must do that explicitly, even with autostart on
    - autostart is for the *next app launch*, not this save); changing the
    process args while staying managed stops the current one (a no-op if
    we don't own it - see stop_managed) and, only if autostart is enabled,
    immediately starts a fresh one with the new settings."""
    switched_to_external = old_cfg.server_mode != "external" and new_cfg.server_mode == "external"
    switched_to_managed = old_cfg.server_mode == "external" and new_cfg.server_mode == "managed"
    process_args_changed = (
        old_cfg.server_port, old_cfg.n_gpu_layers, old_cfg.context_size, old_cfg.extra_server_args
    ) != (
        new_cfg.server_port, new_cfg.n_gpu_layers, new_cfg.context_size, new_cfg.extra_server_args
    )

    if switched_to_external:
        stop_managed()
        return "Switched to external mode."
    if switched_to_managed:
        return (
            "Switched to managed mode - click \"Start llama server\" "
            "in this tab when you're ready (autostart applies on the next app launch, not now)."
        )
    if new_cfg.server_mode == "managed" and process_args_changed:
        stop_managed()
        if new_cfg.autostart_managed_llama:
            _, start_message = try_start_managed_llama(new_cfg)
            return start_message
        return "Click \"Start llama server\" to apply the new server settings."
    return None


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Llama", id=f"{BACKEND.id}-settings") as llama_settings_tab:
        gr.Markdown(config_mod.LLAMA_TAB_INTRO)
        # Same position as hydra_status_md on the Hydra tab -
        # a status line right under the intro, above any
        # Group/section, not folded into the managed-mode
        # config box below (moved out of there live).
        installed_backend_text = gr.Markdown(_llama_installed_text())
        show_in_models_btn = model_status_row(ctx, BACKEND)
        with gr.Group():
            server_mode = gr.Radio(
                choices=[
                    (
                        "Managed: the llama server is managed by the UI, model management enabled",
                        "managed",
                    ),
                    (
                        "External: the llama server is managed by the user",
                        "external",
                    ),
                ],
                value=cfg.server_mode,
                label="Server mode",
                elem_id="server-mode-radio",
            )

        # gr.Column, not gr.Group - only visible= toggling is
        # needed here, not Group's bordered/connected-siblings
        # styling, which (per the Models tab's own gotcha,
        # confirmed live there) squashes the button rows below
        # into a merged slab instead of leaving them looking
        # like real buttons. Column has no such styling.
        with gr.Column(visible=(cfg.server_mode == "managed")) as managed_server_group:
            # The dropdown+Install+Abort row stays OUTSIDE the
            # Group below, on the tab's plain background - a
            # Row of buttons inside a Group loses the gap
            # between them and merges into one flat bar (see
            # the Models tab's own gotcha above; confirmed
            # live there, then anticipated again here before
            # even trying it). Everything from the install-
            # status row onward is plain config fields (no
            # buttons), so grouping THOSE for the shared grey
            # background doesn't carry the same risk.
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
                autostart_managed_llama = gr.Checkbox(
                    label="Autostart on app launch (if installed but not already running)",
                    value=cfg.autostart_managed_llama,
                    interactive=(llama_installed_info() is not None),
                )
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
            with gr.Row():
                start_llama_btn = gr.Button(
                    "Start llama server", interactive=False, variant="secondary"
                )
                end_llama_btn = gr.Button("Stop llama server", interactive=False)

        with gr.Column(visible=(cfg.server_mode == "external")) as external_server_group:
            external_url = gr.Textbox(label="External server URL", value=cfg.external_url)
            verify_external_btn = gr.Button("Verify")

        with gr.Group():
            prompt_template = gr.Textbox(
                label="Prompt template", value=cfg.prompt_template, lines=4
            )
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=cfg.temperature, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=cfg.top_p, label="Top-p")
                max_tokens = gr.Number(label="Max tokens", value=cfg.max_tokens, precision=0)
            request_timeout = gr.Number(
                label="Request timeout (seconds)", value=cfg.request_timeout, precision=0
            )

        # Shared by both mode-conditional groups above (only one
        # of which is ever visible at a time), rather than one
        # status box per group - Start/Stop/Verify never fire at
        # the same moment anyway. Kept at the very bottom of the
        # tab, so it lands right above the shared Save settings/
        # Restart app row below (outside this Tabs() block
        # entirely - see ui.tab_settings' own layout).
        llama_lifecycle_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # Purely cosmetic (which fields are relevant to the
        # selected mode) - server_mode itself is still the
        # one source of truth for which mode is active;
        # this never needs to be read back, only pushed to.
        server_mode.change(
            lambda mode: (gr.update(visible=mode == "managed"), gr.update(visible=mode == "external")),
            [server_mode],
            [managed_server_group, external_server_group],
        )
        # Both chain the same immediate refresh
        # save_settings_btn already uses (see its own
        # comment) - Start/Stop change server state
        # directly, so waiting up to 8s for the next
        # reachability_timer tick to reflect it would be
        # the exact same staleness that fix already
        # addressed for settings saves. current_tab_label_
        # state is "Settings" here too (these buttons only
        # exist on that tab), so push_to_settings is
        # always a no-op, not a real redirect.
        ctx.after_server_change(start_llama_btn.click(_start_managed_llama_ui, [], [llama_lifecycle_infotext]))
        ctx.after_server_change(end_llama_btn.click(_end_managed_llama_ui, [], [llama_lifecycle_infotext]))
        verify_external_btn.click(_verify_external_ui, [external_url], [llama_lifecycle_infotext])

        _llama_install_status_outputs = [
            llama_install_status_row, llama_install_status_text, llama_lifecycle_infotext,
            llama_backend_dropdown, install_llama_btn, abort_install_btn, installed_backend_text,
            autostart_managed_llama,
        ]
        install_llama_btn.click(
            _llama_install_start_ui, [llama_backend_dropdown], [llama_lifecycle_infotext]
        ).then(_llama_install_status_ui, [], _llama_install_status_outputs)
        abort_install_btn.click(
            _llama_install_abort_ui, [], [llama_lifecycle_infotext]
        ).then(_llama_install_status_ui, [], _llama_install_status_outputs)

    ctx.on_tick(_llama_install_status_ui, [], _llama_install_status_outputs)
    ctx.on_load(_llama_install_status_ui, [], _llama_install_status_outputs)
    ctx.on_autoload(BACKEND.id, _autostart_managed_llama_ui, REACHABILITY)

    return SimpleNamespace(
        tab=llama_settings_tab,
        tab_label="Llama",
        tab_id=f"{BACKEND.id}-settings",
        gated=True,
        start_btn=start_llama_btn,
        end_btn=end_llama_btn,
        models_btn=show_in_models_btn,
        # collect_settings()'s parameter order.
        save_inputs=[
            server_mode, server_port, external_url, n_gpu_layers, context_size, extra_server_args,
            autostart_managed_llama, prompt_template, temperature, top_p, max_tokens, request_timeout,
        ],
    )
