"""Hydra 3.5 (RedRocket) e621 tagger - in-process torch, all-or-nothing
VRAM ("fixed"), so it loads before llama-server at launch and stops/
restarts an owned llama-server when loaded later (core.lifecycle).

  - classifier.py  the loaded-model singleton: load/unload/classify/status
  - install.py     pip deps + weights download
  - vendor/        vendored inference-only RedRocket hydra source
  - settings_ui.py Settings -> Hydra (GUI only)
"""

from __future__ import annotations

from core.backend import Backend
from core.config import AppConfig

from . import classifier


class HydraBackend(Backend):
    id = "hydra"
    label = "Hydra"
    role = "tagger"
    vram = "fixed"
    settings_ui_module = "backends.hydra.settings_ui"

    def is_loaded(self, cfg: AppConfig) -> bool:
        return classifier.status().loaded

    def load(self, cfg: AppConfig) -> tuple[bool, str]:
        try:
            classifier.load(cfg)
        except classifier.HydraError as exc:
            return False, f"Couldn't load Hydra model: {exc}"
        return True, f"Hydra model loaded on {classifier.status().device}."

    def unload(self, cfg: AppConfig) -> None:
        classifier.unload()

    def competes_for_vram(self, cfg: AppConfig) -> bool:
        return cfg.hydra_device == "cuda"

    def wants_autoload(self, cfg: AppConfig) -> bool:
        # Deliberately independent of hydra_enabled being what *uses* a
        # loaded model - autoload only ever fires when both are on.
        if not (cfg.hydra_enabled and cfg.hydra_autoload_model):
            return False
        st = classifier.status()
        return st.deps_installed and st.model_downloaded and not st.loaded

    def status_bar_segments(self, cfg: AppConfig) -> list[tuple[str, str]]:
        """Compact one-word(ish) Hydra state - see settings_ui's
        _hydra_status_text() for the fuller sentence-form version shown on
        the Hydra sub-tab itself; this is deliberately terser, matching the
        llama-server segment's own "running"/"idle"/etc. brevity."""
        if not cfg.hydra_enabled:
            return [("Hydra", "disabled")]
        st = classifier.status()
        if st.loaded:
            return [("Hydra", f"loaded ({st.device})")]
        if not st.deps_installed or not st.model_downloaded:
            return [("Hydra", "not set up")]
        return [("Hydra", "enabled, not loaded")]

    def shutdown(self, cfg: AppConfig) -> None:
        # Unlike llama-server (a separate OS process), Hydra lives in this
        # process's own CUDA context, which os.execv doesn't clean up -
        # unload explicitly so VRAM isn't leaked into the replaced process.
        classifier.unload()


BACKEND = HydraBackend()
