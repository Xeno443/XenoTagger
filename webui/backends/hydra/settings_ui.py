"""Settings -> Hydra: dependency install, Load/Unload, the tagger
settings, and Hydra's launch-time autoload. The only file in
backends/hydra/ that imports Gradio or ui/ - loaded by ui.tab_settings via
HydraBackend.settings_ui_module (see ui.tab_settings for what it must
provide)."""

from __future__ import annotations

import logging
import re
import threading
from types import SimpleNamespace
from typing import Optional

import gradio as gr

from backends import BACKENDS
from core.lifecycle import load_with_coexistence
from ui import state
from ui.context import BuildContext
from ui.dialogs import browse_tag_vocab_file_ui
from ui.model_status import model_status_row
from ui.notify import notify
from ui.operation import operation_blocked_by

from . import BACKEND
from . import classifier as hydra_classifier
from .install import install_deps as hydra_install_deps

log = logging.getLogger(__name__)

# --------------------------------------------------------------- Hydra install/lifecycle
#
# Two separate concerns, kept apart the same way llama's own install
# section (backends/llama/settings_ui.py) is kept apart from ui.downloads_queue:
#   - Installing Hydra's Python deps (torch + friends) is a pip subprocess
#     - its own lock/thread/state below. Its per-line output is NOT
#     streamed into the UI (a dedicated Textbox for this was tried and
#     removed - too noisy for a settings page); install.py's own
#     log.debug("pip: %s", line) call already captures every line to the
#     debug log unconditionally (see its own docstring), regardless of
#     whether a UI callback is listening, so nothing is lost by not
#     wiring one here - only completion/failure surfaces to the UI, via
#     hydra_lifecycle_infotext + a notify() popup.
#   - Downloading hydra-3.5.safetensors is just another file - it's
#     Hydra's row in the combined Models tab (HydraBackend.model_downloads),
#     through the same shared download queue (ui.downloads_queue) as
#     every other backend's models.
# Loading/unloading the model itself is not a background job at all - it
# blocks for a few seconds at most (~1GB), so it's a plain generator
# click handler, the same shape as _start_managed_llama_ui/
# _end_managed_llama_ui in backends/llama/settings_ui.py (no dedicated timer needed - unlike
# llama-server's reachability, "is Hydra loaded" can only change via our
# own explicit actions here, never externally).

_hydra_install_lock = threading.RLock()
_hydra_install_thread: Optional[threading.Thread] = None
_hydra_install_running = False
# One-shot (message, level) popped by the next _hydra_install_status_ui
# tick, same single-shot idiom as llama's own _install_announce (backends/llama/settings_ui.py) -
# so a stale completion message doesn't keep stomping on whatever
# Load/Unload wrote to hydra_lifecycle_infotext in the meantime, and so
# a one-shot notify() popup fires on completion/failure.
_hydra_install_announce: Optional[tuple[str, str]] = None  # (message, level)


_HYDRA_METRIC_RE = re.compile(r"f([0-9]*\.?[0-9]+)@([0-9]*\.?[0-9]+)")


def _hydra_metric_to_sliders(metric: str) -> tuple[float, float]:
    """Confidence/Threshold slider values for an "f<beta>@<precision>"
    metric string - falls back to the app's own default (not
    AppConfig.hydra_metric's literal default, which would go stale) for
    anything else (a hand-edited "csi..." string, "default", or garbage
    left over from before these sliders existed)."""
    if (match := _HYDRA_METRIC_RE.fullmatch(metric.strip())) is not None:
        return float(match.group(1)), float(match.group(2))
    return 0.5, 0.1


def hydra_sliders_to_metric(confidence: float, threshold: float) -> str:
    return f"f{confidence:g}@{threshold:g}"


def collect_settings(
    hydra_enabled, hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
    hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_autoload_model, hydra_tag_vocab_path,
) -> dict:
    """AppConfig updates from build()'s save_inputs, in that order. The
    two metric sliders map to the one hydra_metric string."""
    return dict(
        hydra_enabled=bool(hydra_enabled),
        hydra_device=hydra_device,
        hydra_metric=hydra_sliders_to_metric(float(hydra_confidence), float(hydra_threshold)),
        hydra_implications=hydra_implications,
        hydra_exclude_categories=hydra_exclude_categories,
        hydra_exclude_tags=hydra_exclude_tags,
        hydra_max_tags=int(hydra_max_tags),
        hydra_autoload_model=bool(hydra_autoload_model),
        hydra_tag_vocab_path=hydra_tag_vocab_path,
    )


def _hydra_status_text() -> str:
    """Dependencies + loaded state - whether the model is downloaded is the
    model status line below it (ui.model_status)."""
    st = hydra_classifier.status()
    parts = [
        f"Dependencies: {'installed' if st.deps_installed else 'not installed'}.",
        f"Loaded on {st.device}." if st.loaded else "Not loaded.",
    ]
    return " ".join(parts)


def _hydra_install_worker() -> None:
    global _hydra_install_thread, _hydra_install_running, _hydra_install_announce
    try:
        completed = hydra_install_deps()
        message = "Hydra dependencies installed." if completed else "Hydra dependency install failed - see debug log."
        level = "info" if completed else "error"
    except Exception as exc:
        log.exception("Hydra dependency install failed")
        message = f"Install failed: {exc}"
        level = "error"
    with _hydra_install_lock:
        _hydra_install_running = False
        _hydra_install_announce = (message, level)
        _hydra_install_thread = None


def _hydra_install_start_ui() -> str:
    global _hydra_install_thread, _hydra_install_running
    with _hydra_install_lock:
        if _hydra_install_thread is not None and _hydra_install_thread.is_alive():
            return "An install is already in progress."
        _hydra_install_running = True
        _hydra_install_thread = threading.Thread(target=_hydra_install_worker, daemon=True)
        _hydra_install_thread.start()
    return "Installing Hydra dependencies (torch + friends - this downloads several GB)..."


def _hydra_install_status_ui():
    """(lifecycle_infotext_update, status_md_update, install_btn_update,
    load_btn_update, unload_btn_update, enabled_checkbox_update,
    autoload_checkbox_update) - wired to the same 2s status_timer as
    everything else, plus called directly after Install/Load/Unload
    clicks for immediate feedback. The three buttons and two checkboxes
    are all read-only derived state (interactive= only), safe to
    recompute every tick with no risk of fighting an in-flight user
    edit, unlike lifecycle_infotext (one-shot, see _install_announce's
    own comment for why)."""
    global _hydra_install_announce
    with _hydra_install_lock:
        running = _hydra_install_running
        announce = _hydra_install_announce
        _hydra_install_announce = None

    if announce is not None:
        notify(announce[0], level=announce[1])
    infotext_update = gr.update(value=announce[0]) if announce is not None else gr.update()
    hydra_status_update = gr.update(value=_hydra_status_text())

    st = hydra_classifier.status()
    # not running is load-bearing here, not just for the Install button
    # below - deps_installed() (find_spec("torch")) can flip True while
    # install_deps()'s pip subprocess is still mid-write (e.g. torch's
    # own __init__.py exists before its multi-GB CUDA DLLs are fully
    # written), so without this, Load/Enable/Autoload can all go
    # clickable mid-install and race a load against an incomplete
    # torch install.
    ready = st.deps_installed and st.model_downloaded and not running
    return (
        infotext_update,
        hydra_status_update,
        gr.update(interactive=not st.deps_installed and not running),
        gr.update(interactive=ready and not st.loaded),
        gr.update(interactive=st.loaded),
        gr.update(interactive=ready),
        gr.update(interactive=ready),
    )


def _load_hydra_ui():
    """The shared "load Hydra" UI flow - used by BOTH the manual "Load
    Hydra model" button and the launch-time autoload, so correctness
    doesn't depend on which of the two triggered it, or on the autoload
    chain's own fixed-before-adaptive ordering (core.lifecycle.
    autoload_order) staying right forever. That ordering just avoids the
    pointless stop-immediately-after-start dance during a normal dual-
    autostart launch; core.lifecycle.load_with_coexistence is what
    actually makes it safe regardless: if hydra_device is "cuda", every
    running adaptive backend this app owns (llama-server) is stopped
    first so Hydra gets first pick of free VRAM, then restarted so its
    "auto" resizes itself around whatever Hydra actually claimed. Never
    touches a server this session didn't start.

    Refuses outright (no stop/restart at all) if a caption operation is
    currently active - killing the server out from under an in-flight
    Single/Batch/Review request would break it. In practice the Hydra
    settings sub-tab is already disabled while one is running
    (_settings_gating_ui), so this is the same "second line of defense
    against a stray click" operation_blocked_by() already is elsewhere,
    not the primary gate.

    Fires one notify() popup on the final outcome - this can now take as
    long as a full llama-server restart (reloading a 10-30GB model, not
    just Hydra's own ~1GB), long enough to deserve one regardless of
    which of the two triggers fired it, same as this project's other
    background-completion popups."""
    if hydra_classifier.status().loaded:
        yield "Hydra model is already loaded.", gr.update()
        return

    blocked = operation_blocked_by()
    if blocked:
        message = f"Can't load Hydra model right now - {blocked}"
        yield message, gr.update()
        notify(message, level="warning")
        return

    for step in load_with_coexistence(BACKEND, BACKENDS, state.current_cfg):
        # Status line refreshes once Hydra's own load has happened
        # (restarting/done), not while llama-server is being stopped.
        status_update = gr.update(value=_hydra_status_text()) if step.phase in ("restarting", "done") else gr.update()
        yield step.message, status_update
        if step.phase == "done":
            notify(step.message, level=step.level)


def _load_hydra_model_ui():
    """The manual "Load Hydra model" button's click target - see
    _load_hydra_ui()'s own docstring."""
    yield from _load_hydra_ui()


def _autoload_hydra_model_ui():
    """This backend's launch-time autoload step (registered in build()),
    run before llama-server's autostart by core.lifecycle.autoload_order -
    so that when both are enabled, Hydra gets first pick of a clean,
    fully-free VRAM budget and llama-server's own n_gpu_layers="auto"
    (which gracefully adapts to reduced free VRAM, unlike Hydra's own
    all-or-nothing load) sizes itself around whatever's left. Silent no-op
    unless BACKEND.wants_autoload (hydra_enabled, hydra_autoload_model,
    deps+model both present, nothing already loaded) - same "no popup for
    didn't-autostart" convention as llama's autostart.

    The actual load - including the stop/restart dance, moot here in the
    normal case since nothing's started yet this early in a fresh launch,
    but still a real safety net for an edge case like a page reload while
    an owned llama-server is running - is _load_hydra_ui(), the same
    shared flow the manual Load button uses."""
    if not BACKEND.wants_autoload(state.current_cfg):
        yield gr.update(), gr.update()
        return
    notify("Loading Hydra model automatically (autoload is enabled)...")
    yield from _load_hydra_ui()


def _unload_hydra_model_ui():
    if not hydra_classifier.status().loaded:
        yield "Hydra model isn't loaded.", gr.update()
        return
    yield "Unloading Hydra model...", gr.update()
    hydra_classifier.unload()
    yield "Hydra model unloaded.", gr.update(value=_hydra_status_text())


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Hydra", id=f"{BACKEND.id}-settings") as hydra_settings_tab:
        gr.Markdown(
            "Hydra 3.5 (RedRocket) is an optional second-stage e621 tag "
            "classifier - if loaded, its tags are appended after the VLM's "
            "caption on every Single-image/Batch/Review caption. Loading "
            "competes with llama-server for VRAM, so it's never automatic - "
            "install its dependencies below, download its model on the "
            "Models tab, then Load it explicitly below (or enable autoload "
            "for next launch)."
        )
        hydra_status_md = gr.Markdown(_hydra_status_text())
        show_in_models_btn = model_status_row(ctx, BACKEND)

        # Initial interactive= values only - kept fresh
        # afterward by _hydra_install_status_ui (2s timer +
        # after every Install/Load/Unload click below).
        _hydra_st = hydra_classifier.status()
        _hydra_ready = _hydra_st.deps_installed and _hydra_st.model_downloaded
        with gr.Group():
            hydra_autoload_model = gr.Checkbox(
                label="Autoload Hydra model on startup",
                value=cfg.hydra_autoload_model,
                interactive=_hydra_ready,
            )
            hydra_enabled = gr.Checkbox(
                label="Enable Hydra tagging (appends tags after the VLM caption)",
                value=cfg.hydra_enabled,
                interactive=_hydra_ready,
            )

        # Plain Row, not Group - see managed_server_group's own
        # comment (ui.settings.llama) for why (Group's styling squashes
        # button rows into a merged slab).
        with gr.Row():
            hydra_install_btn = gr.Button(
                "Install Hydra dependencies", interactive=not _hydra_st.deps_installed,
            )
            hydra_load_btn = gr.Button(
                "Load Hydra model",
                interactive=(
                    _hydra_st.deps_installed and _hydra_st.model_downloaded
                    and not _hydra_st.loaded
                ),
            )
            hydra_unload_btn = gr.Button("Unload Hydra model", interactive=_hydra_st.loaded)

        with gr.Group():
            hydra_device = gr.Radio(
                choices=[("CUDA", "cuda"), ("CPU", "cpu")],
                value=cfg.hydra_device,
                label="Device (only takes effect after an Unload + Load)",
            )
            _hydra_confidence_default, _hydra_threshold_default = _hydra_metric_to_sliders(
                cfg.hydra_metric
            )
            with gr.Row():
                hydra_confidence = gr.Slider(
                    minimum=0.1, maximum=1.5, step=0.05,
                    value=_hydra_confidence_default,
                    label="Confidence",
                    info=(
                        "Per-tag strictness. Low = looser, more tags "
                        "(favors catching everything). High = stricter, "
                        "fewer but more certain tags (favors precision)."
                    ),
                )
                hydra_threshold = gr.Slider(
                    minimum=0.0, maximum=0.5, step=0.05,
                    value=_hydra_threshold_default,
                    label="Threshold",
                    info=(
                        "Safety floor, independent of Confidence: never let "
                        "a tag through with a measured track record worse "
                        "than this (e.g. 0.1 = wrong no more than 9 times "
                        "out of 10 on Hydra's own validation data)."
                    ),
                )
            hydra_implications = gr.Dropdown(
                choices=[
                    "preserve", "inherit", "constrain", "enforce", "remove",
                    "constrain-remove", "enforce-inherit", "enforce-constrain",
                    "enforce-remove", "off",
                ],
                value=cfg.hydra_implications,
                label="Implications mode",
            )
            with gr.Row():
                hydra_exclude_categories = gr.Textbox(
                    label="Exclude categories (space-separated)", value=cfg.hydra_exclude_categories,
                )
                hydra_exclude_tags = gr.Textbox(
                    label="Exclude tags (space-separated)", value=cfg.hydra_exclude_tags,
                )
            hydra_max_tags = gr.Number(
                label="Max tags appended (0 = no cap)", value=cfg.hydra_max_tags, precision=0,
            )

        # Plain Row, kept OUTSIDE the Group above - same
        # reasoning as hydra_install_btn's Row: a Row
        # containing a Button loses its inter-element gaps
        # and gets squashed into a merged slab when nested
        # inside a Group (see the gr.Group() gotcha).
        # Takes effect on next restart, not on Save - see
        # AppConfig.hydra_tag_vocab_path's own comment.
        with gr.Row():
            hydra_tag_vocab_path = gr.Textbox(
                label="Tag vocabulary CSV (Review tag autocomplete, restart to apply)",
                value=cfg.hydra_tag_vocab_path, scale=4, interactive=False,
            )
            hydra_tag_vocab_browse_btn = gr.Button("Browse...", scale=1)
        hydra_tag_vocab_browse_btn.click(
            browse_tag_vocab_file_ui, [hydra_tag_vocab_path], [hydra_tag_vocab_path]
        )

        # Same placement rationale as llama_lifecycle_infotext
        # (ui.settings.llama) - bottom of the tab, right above the shared Save
        # settings/Restart app row.
        hydra_lifecycle_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        _hydra_install_status_outputs = [
            hydra_lifecycle_infotext, hydra_status_md,
            hydra_install_btn, hydra_load_btn, hydra_unload_btn,
            hydra_enabled, hydra_autoload_model,
        ]
        hydra_install_btn.click(_hydra_install_start_ui, [], [hydra_lifecycle_infotext]).then(
            _hydra_install_status_ui, [], _hydra_install_status_outputs
        )
        # Same immediate-refresh reasoning as Start/Stop
        # llama's own chained calls (backends/llama/settings_ui.py) - a manual Load
        # click can now stop+restart the managed server
        # (see _load_hydra_ui), and
        # waiting up to 8s for reachability_timer's next
        # tick to reflect that would be the same
        # staleness already fixed for Start/Stop/Save.
        ctx.after_server_change(
            hydra_load_btn.click(
                _load_hydra_model_ui, [], [hydra_lifecycle_infotext, hydra_status_md]
            ).then(_hydra_install_status_ui, [], _hydra_install_status_outputs)
        )
        hydra_unload_btn.click(
            _unload_hydra_model_ui, [], [hydra_lifecycle_infotext, hydra_status_md]
        ).then(_hydra_install_status_ui, [], _hydra_install_status_outputs)

    ctx.on_tick(_hydra_install_status_ui, [], _hydra_install_status_outputs)
    ctx.on_load(_hydra_install_status_ui, [], _hydra_install_status_outputs)
    ctx.on_autoload(BACKEND.id, _autoload_hydra_model_ui, [hydra_lifecycle_infotext, hydra_status_md])

    return SimpleNamespace(
        tab=hydra_settings_tab,
        tab_label="Hydra",
        tab_id=f"{BACKEND.id}-settings",
        gated=True,
        models_btn=show_in_models_btn,
        # collect_settings()'s parameter order.
        save_inputs=[
            hydra_enabled, hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
            hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_autoload_model,
            hydra_tag_vocab_path,
        ],
    )
