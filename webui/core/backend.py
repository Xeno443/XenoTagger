"""The interface every model backend (webui/backends/<id>/) implements -
everything the rest of the app may know about a backend without reaching
into its internals. Backends subclass Backend and override what applies;
the registry of instances is backends.BACKENDS.

Gradio-free and backend-agnostic: core/ never imports backends/, so code
here (core.lifecycle) only ever works on Backend objects it's handed.

Captioning itself (caption()/tag()) isn't part of this interface yet -
core.captioner still calls the llama client and Hydra classifier
directly. That moves behind this interface with the caption-pipeline
rewrite (v0.9 step 3).
"""

from __future__ import annotations

from .config import AppConfig


class Backend:
    # Stable identifier. Also the prefix of its Settings sub-tab id
    # ("<id>-settings") and of its AppConfig keys ("<id>_...", by
    # convention - older keys like server_mode predate that).
    id: str = ""
    # Human name used in lifecycle messages, e.g. "Stopping llama-server
    # to free VRAM for Hydra...".
    label: str = ""
    # "captioner" (writes the caption line) or "tagger" (writes the tag line).
    role: str = ""
    # "fixed": load() is all-or-nothing onto one device (Hydra).
    # "adaptive": sizes itself to whatever VRAM is free when it starts
    # (llama-server's n_gpu_layers="auto"). Drives core.lifecycle: fixed
    # backends load first at launch, and loading one stops/restarts any
    # running adaptive one so it can resize around it.
    vram: str = "fixed"
    # Dotted path of the backend's Settings sub-tab module, imported only
    # by the GUI (ui.tab_settings) - the only file in a backend allowed to
    # import Gradio. See ui.tab_settings for what it must provide.
    settings_ui_module: str = ""

    def is_loaded(self, cfg: AppConfig) -> bool:
        """Loaded AND ours to unload/reload - an external or foreign
        llama-server is running but never "loaded" in this sense."""
        return False

    def load(self, cfg: AppConfig) -> tuple[bool, str]:
        """Blocking. (ok, user-facing message) - never raises for an
        expected failure (missing deps, OOM, bad model selection)."""
        raise NotImplementedError

    def unload(self, cfg: AppConfig) -> None:
        raise NotImplementedError

    def competes_for_vram(self, cfg: AppConfig) -> bool:
        """Whether loading this right now needs GPU memory at all (e.g.
        False for Hydra on CPU) - no coexistence dance if not."""
        return True

    def wants_autoload(self, cfg: AppConfig) -> bool:
        """True if this backend's autoload/autostart setting is on AND
        loading it right now would actually do something (installed, not
        already loaded/running). The Settings sub-tab's autoload handler
        asks this before acting; core.lifecycle.autoload_order decides
        when that handler runs relative to the other backends'."""
        return False

    def status_bar_segments(self, cfg: AppConfig) -> list[tuple[str, str]]:
        """(label, value) pairs for the global status bar, in order."""
        return []

    def shutdown(self, cfg: AppConfig) -> None:
        """Before an in-place app restart (os.execv skips atexit and
        normal teardown): release anything that would otherwise leak into
        the replaced process - a child process, a CUDA context."""

    def on_exit(self) -> None:
        """Registered with atexit by the GUI's main() only (never at import
        time - scratch scripts import these modules too)."""
