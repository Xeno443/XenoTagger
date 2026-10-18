"""Settings -> Furrence2: dependency install, worker status, caption
length. The only file in backends/furrence2/ that imports Gradio or ui/ -
loaded by ui.tab_settings via Furrence2Backend.settings_ui_module.

No Load/Unload buttons: starting and stopping the worker is the
pipeline's job (start_pipeline/stop_pipeline, v0.9 step 3). The install
runs in a background thread with the same one-shot announce idiom as
Hydra's (backends/hydra/settings_ui.py); its pip output only goes to the
debug log.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Optional

import gradio as gr

from ui import state
from ui.context import BuildContext
from ui.model_status import model_status_row
from ui.notify import notify

from . import BACKEND, worker_client
from .install import install_deps

log = logging.getLogger(__name__)

_install_lock = threading.Lock()
_install_thread: Optional[threading.Thread] = None
_install_announce: Optional[tuple[str, str]] = None  # (message, level), popped by the next tick


def collect_settings(furrence2_length_min, furrence2_length_max) -> dict:
    lo, hi = int(furrence2_length_min), int(furrence2_length_max)
    return dict(furrence2_length_min=min(lo, hi), furrence2_length_max=max(lo, hi))


def _status_text() -> str:
    st = worker_client.status()
    deps = "installed" if st.deps_installed else "not installed"
    if st.running:
        worker = f"Running on {st.device}."
    elif st.crashed:
        worker = f"Stopped unexpectedly - see {worker_client.LOG_PATH.name} in webui/logs."
    else:
        worker = "Not running."
    return f"Dependencies: {deps}. {worker}"


def _install_worker() -> None:
    global _install_thread, _install_announce
    try:
        ok = install_deps()
        announce = ("Furrence2 dependencies installed.", "info") if ok else (
            "Furrence2 dependency install failed - see debug log.", "error")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        log.exception("Furrence2 dependency install failed")
        announce = (f"Install failed: {exc}", "error")
    with _install_lock:
        _install_announce = announce
        _install_thread = None


def _install_start_ui() -> str:
    global _install_thread
    with _install_lock:
        if _install_thread is not None:
            return "An install is already in progress."
        _install_thread = threading.Thread(target=_install_worker, daemon=True)
        _install_thread.start()
    return "Installing Furrence2 dependencies (torch if missing, then transformers 4.46 into furrence2/lib)..."


def _status_ui():
    """(infotext, status line, Install button) - 2s tick, page load, and
    right after an Install click."""
    global _install_announce
    with _install_lock:
        running = _install_thread is not None
        announce, _install_announce = _install_announce, None
    if announce is not None:
        notify(announce[0], level=announce[1])
    st = worker_client.status()
    return (
        gr.update(value=announce[0]) if announce is not None else gr.update(),
        gr.update(value=_status_text()),
        # Reinstalling is allowed (e.g. a broken lib folder), just not twice at once.
        gr.update(interactive=not running, value="Reinstall dependencies" if st.deps_installed else "Install dependencies"),
    )


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Furrence2", id=f"{BACKEND.id}-settings") as furrence2_settings_tab:
        gr.Markdown(
            "Furrence2 (lodestone-horizon/furrence2-large) writes a caption "
            "from the image plus its e621 tags, so it needs a tagger (Hydra) "
            "running before it. It runs in its own background process with "
            "its own copy of transformers 4.46 (the version its model code "
            "needs), installed into the `furrence2/lib` folder. Weights are "
            "non-commercial (cc-by-nc-4.0)."
        )
        status_md = gr.Markdown(_status_text())
        show_in_models_btn = model_status_row(ctx, BACKEND)
        with gr.Row():
            install_btn = gr.Button("Install dependencies")

        with gr.Row():
            length_min = gr.Number(
                label="Caption length, minimum", value=cfg.furrence2_length_min, precision=0,
                info="The prompt's length value is 60 + 2 per tag, kept between these two. Roughly words.",
            )
            length_max = gr.Number(
                label="Caption length, maximum", value=cfg.furrence2_length_max, precision=0,
                info="Above ~200 it starts padding with filler. Set both equal for a fixed length.",
            )

        infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        status_outputs = [infotext, status_md, install_btn]
        install_btn.click(_install_start_ui, [], [infotext]).then(_status_ui, [], status_outputs)

    ctx.on_tick(_status_ui, [], status_outputs)
    ctx.on_load(_status_ui, [], status_outputs)

    return SimpleNamespace(
        tab=furrence2_settings_tab,
        tab_label="Furrence2",
        tab_id=f"{BACKEND.id}-settings",
        gated=True,
        models_btn=show_in_models_btn,
        save_inputs=[length_min, length_max],
    )
