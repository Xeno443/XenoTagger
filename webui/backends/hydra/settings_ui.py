"""Settings -> Hydra: dependency install and the tagger settings. The only
file in backends/hydra/ that imports Gradio or ui/ - loaded by
ui.tab_settings via HydraBackend.settings_ui_module (see ui.tab_settings
for the contract).

No Load/Unload or autoload here: whether Hydra runs is its Run checkbox
on Settings -> Pipeline, and the pipeline loads it.
"""

from __future__ import annotations

import logging
import re
import threading
from types import SimpleNamespace
from typing import Optional

import gradio as gr

from core.config import AppConfig
from ui import state
from ui.context import BuildContext
from ui.dialogs import browse_tag_vocab_file_ui
from ui.model_status import model_status_row
from ui.notify import notify

from . import BACKEND
from . import classifier as hydra_classifier
from .install import install_deps as hydra_install_deps

log = logging.getLogger(__name__)

# Installing Hydra's Python deps (torch + friends) is a pip subprocess in
# a background thread. Its per-line output only goes to the debug log
# (install.py logs every pip line); only completion/failure surfaces to
# the UI, via hydra_lifecycle_infotext + a notify() popup. Downloading the
# model is Hydra's row on the Models tab (the shared download queue).

_hydra_install_lock = threading.RLock()
_hydra_install_thread: Optional[threading.Thread] = None
_hydra_install_running = False
# One-shot (message, level) popped by the next _hydra_install_status_ui
# tick, so a stale completion message doesn't keep re-announcing itself.
_hydra_install_announce: Optional[tuple[str, str]] = None


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
    hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
    hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_tag_vocab_path,
) -> dict:
    """AppConfig updates from build()'s save_inputs, in that order. The
    two metric sliders map to the one hydra_metric string."""
    return dict(
        hydra_device=hydra_device,
        hydra_metric=hydra_sliders_to_metric(float(hydra_confidence), float(hydra_threshold)),
        hydra_implications=hydra_implications,
        hydra_exclude_categories=hydra_exclude_categories,
        hydra_exclude_tags=hydra_exclude_tags,
        hydra_max_tags=int(hydra_max_tags),
        hydra_tag_vocab_path=hydra_tag_vocab_path,
    )


def load_values(cfg: AppConfig) -> tuple:
    return (
        cfg.hydra_device, *_hydra_metric_to_sliders(cfg.hydra_metric), cfg.hydra_implications,
        cfg.hydra_exclude_categories, cfg.hydra_exclude_tags, cfg.hydra_max_tags, cfg.hydra_tag_vocab_path,
    )


def _hydra_status_text() -> str:
    """Dependencies + loaded state - whether the model is downloaded is the
    model status line below it (ui.model_status)."""
    st = hydra_classifier.status()
    loaded = f"Loaded on {st.device} (by the pipeline)." if st.loaded else "Not loaded."
    return f"Dependencies: {'installed' if st.deps_installed else 'not installed'}. {loaded}"


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
    """(lifecycle_infotext, status_md, install_btn) - the 2s status_timer,
    page load, and right after an Install click."""
    global _hydra_install_announce
    with _hydra_install_lock:
        running = _hydra_install_running
        announce = _hydra_install_announce
        _hydra_install_announce = None

    if announce is not None:
        notify(announce[0], level=announce[1])
    st = hydra_classifier.status()
    return (
        gr.update(value=announce[0]) if announce is not None else gr.update(),
        gr.update(value=_hydra_status_text()),
        gr.update(interactive=not st.deps_installed and not running),
    )


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Hydra", id=f"{BACKEND.id}-settings") as hydra_settings_tab:
        gr.Markdown(
            "Hydra 3.5 (RedRocket) is an e621 tagger. Install its dependencies "
            "below and download its model on the Models tab; whether it runs is "
            "its Run checkbox in Settings -> Pipeline, which also loads it."
        )
        hydra_status_md = gr.Markdown(_hydra_status_text())
        show_in_models_btn = model_status_row(ctx, BACKEND)

        # Plain Row, not Group - a Button in a Group gets squashed into
        # one slab with its neighbors (see the gr.Group() gotcha).
        with gr.Row():
            hydra_install_btn = gr.Button(
                "Install Hydra dependencies", interactive=not hydra_classifier.status().deps_installed,
            )

        with gr.Group():
            hydra_device = gr.Radio(
                choices=[("CUDA", "cuda"), ("CPU", "cpu")],
                value=cfg.hydra_device,
                label="Device",
            )
            _hydra_confidence_default, _hydra_threshold_default = _hydra_metric_to_sliders(cfg.hydra_metric)
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
                label="Max tags (0 = no cap)", value=cfg.hydra_max_tags, precision=0,
            )

        # Plain Row outside the Group, same reason as the Install button.
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

        hydra_lifecycle_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        _hydra_install_status_outputs = [hydra_lifecycle_infotext, hydra_status_md, hydra_install_btn]
        hydra_install_btn.click(_hydra_install_start_ui, [], [hydra_lifecycle_infotext]).then(
            _hydra_install_status_ui, [], _hydra_install_status_outputs
        )

    ctx.on_tick(_hydra_install_status_ui, [], _hydra_install_status_outputs)
    ctx.on_load(_hydra_install_status_ui, [], _hydra_install_status_outputs)

    return SimpleNamespace(
        tab=hydra_settings_tab,
        tab_label="Hydra",
        tab_id=f"{BACKEND.id}-settings",
        gated=True,
        models_btn=show_in_models_btn,
        # collect_settings()'s parameter order.
        save_inputs=[
            hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
            hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_tag_vocab_path,
        ],
    )
