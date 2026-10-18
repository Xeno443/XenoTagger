"""The interface every model backend (webui/backends/<id>/) implements -
everything the rest of the app may know about a backend without reaching
into its internals. Backends subclass Backend and override what applies;
the registry of instances is backends.BACKENDS.

Gradio-free and backend-agnostic: core/ never imports backends/, so code
here (core.lifecycle) only ever works on Backend objects it's handed.

tag()/caption() are the per-image calls the caption pipeline (v0.9 step
3) makes on whatever backends it started: a tagger returns structured
Tags, a captioner returns one CaptionOutput per variant. Expected
failures raise BackendError (each backend's own error class derives from
it), so the pipeline can handle every backend alike. core.captioner
still calls llama and Hydra directly until the pipeline replaces it.

The model methods (list_models() and friends) feed the GUI's combined
Models tab (ui/tab_models.py), which shows every backend's rows in one
table. They're plain data in and out, no Gradio: a backend describes its
models and the choices for one of them (ModelRow, ModelOption), the tab
renders them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from .config import AppConfig
from .downloads import DownloadItem

# What tag()/caption() accept: a file path, or the encoded file's bytes
# (an upload on the Single tab).
ImageInput = Union[str, Path, bytes]


class BackendError(RuntimeError):
    """An expected failure of one tag()/caption() call (not loaded, model
    error, bad image) - the message is user-facing."""


class Interrupted(BackendError):
    """caption() stopped early because should_stop() returned True."""


@dataclass
class Tag:
    name: str  # e621 form, underscores: "looking_at_viewer"
    # e621 category ("general", "species", "meta", ...), except rating
    # tags (safe/questionable/explicit), which are "rating" rather than
    # "meta" so a captioner can drop them separately.
    category: str
    confidence: float


@dataclass
class CaptionOutput:
    text: str
    truncated: bool = False  # hit a length limit, text is cut off
    seconds: float = 0.0
    note: str = ""  # one line of details for logs/infotext, e.g. token counts


@dataclass
class ModelRow:
    """One row of the Models tab table."""
    # Backend-internal id, handed back to model_options()/model_downloads()/
    # set_active() - never shown.
    key: str
    name: str
    source: str  # "Curated" (in a list we ship) or "Manual" (found on disk)
    local: str  # what's on disk, e.g. "2/5" (of 5 curated files) or "1"
    active: bool = False  # the one this backend uses right now (star column)
    note: str = ""  # appended to "Viewing <name>" when the row is selected


@dataclass
class ModelOption:
    """One dropdown under the table for the selected row (llama: Quant,
    mmproj). An empty choices list shows a disabled "N/A"."""
    label: str
    choices: list[tuple[str, str]] = field(default_factory=list)  # (shown label, value)
    value: Optional[str] = None  # preselected; falls back to the first choice


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
    # Captioners only: whether caption() uses the tagger's tags - "no",
    # "optional" (llama: only a prompt containing {tags}) or "required"
    # (Furrence2 can't caption without them).
    takes_tags: str = "no"
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

    def tag(self, cfg: AppConfig, image: ImageInput) -> list[Tag]:
        """Taggers only. Best first, already filtered by this tagger's own
        settings (threshold, exclusions). Raises BackendError."""
        raise NotImplementedError

    def caption(
        self,
        cfg: AppConfig,
        image: ImageInput,
        variant: str = "",
        tags: Optional[list[Tag]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> CaptionOutput:
        """Captioners only. variant is the pipeline variant's backend-
        specific setting (llama: the prompt). tags are the tagger's, if
        one ran. should_stop is polled while generating where the backend
        can stop mid-caption (llama); a backend that's fast per image just
        finishes it. Raises BackendError (Interrupted when stopped)."""
        raise NotImplementedError

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

    # ---- Models tab. `values` below is the selected row's current dropdown
    # values, one per ModelOption in model_options()' order, None for an
    # "N/A" one; trailing extras are ignored.

    # Whether this backend has rows in the Models tab at all (and a "Show
    # in Models" link on its Settings page).
    has_models: bool = False
    # Whether a downloaded row can be made the active one (llama: pick one
    # of several GGUFs). False means there's nothing to choose - Hydra's
    # single model is used as soon as it's on disk.
    selectable_models: bool = False
    # Markdown shown in the Models tab's intro, explaining this backend's rows.
    models_help: str = ""

    def list_models(self, cfg: AppConfig) -> list[ModelRow]:
        """Curated and on-disk models merged into one row each - matching
        the two up is backend-specific. Rescans the disk; called on every
        Models tab refresh, so later calls may reuse this result."""
        return []

    def model_options(self, cfg: AppConfig, key: str) -> list[ModelOption]:
        return []

    def model_downloads(self, cfg: AppConfig, key: str, values: list) -> list[DownloadItem]:
        """What still has to be downloaded for this selection - empty if
        it's all on disk (or nothing downloadable is selected)."""
        return []

    def set_active_problem(self, cfg: AppConfig, key: str, values: list) -> Optional[str]:
        """Why this (fully downloaded) selection can't be made active right
        now, or None if it can. Only asked when selectable_models."""
        return None

    def set_active(self, cfg: AppConfig, key: str, values: list) -> tuple[Optional[dict], str]:
        """(AppConfig updates, message) - updates is None if refused. The
        caller applies and saves the updates; a backend never saves config."""
        return None, "Nothing to set as active."

    def model_status(self, cfg: AppConfig) -> str:
        """One line for this backend's Settings page, next to its "Show in
        Models" button."""
        return ""
