"""Persistent app settings: the AppConfig dataclass, and load()/save() for
its on-disk JSON form.

This is the single source of truth for every tunable knob in the app -
server connection, model selection, image preprocessing, generation
parameters, captioning defaults, and debug logging - and it's shared
verbatim by both entry points: the Gradio GUI (app.py, which keeps one
AppConfig in its module-level `current_cfg` and reassigns it wholesale on
Save rather than mutating fields in place) and the headless CLI (cli.py,
which loads it once per invocation and applies its own argparse overrides
on top). Neither entry point subclasses or extends this - a setting that
should apply to both always belongs here, not duplicated in either caller.

load() merges whatever's on disk over the dataclass's own defaults field
by field (unknown keys in an old/newer settings.json are ignored rather
than erroring, and any field missing from the file just keeps its
built-in default) - this is what lets a new field get added here with a
sensible default and have every existing user's settings.json keep
working unmodified, without a migration step. This module does no
locking of its own; concurrent access to the loaded AppConfig instance
(e.g. a running batch vs. a Settings save from another tab) is the
caller's responsibility - see app.py's `_session_lock`/`_operation_lock`.

A few defaults are deliberate, not arbitrary - see the inline comments
above each field, e.g. `temperature` (0.3, not the more "obvious" 0.0:
fully greedy decoding was found to reliably lock in both repetition loops
and borderline-abliterated-model refusals, since there's then no chance
to sample around a bad top token) and `resize_enabled`/`snap_enabled`
(oversized source images were found to burn most of the context window on
the image alone before any completion budget was left).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

WEBUI_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = WEBUI_DIR / "config" / "settings.json"

DEFAULT_PROMPT = (
    "Describe this image in a single detailed paragraph, written in natural "
    "language, suitable as a caption for LoRA training. Mention subject, "
    "pose, clothing, setting, lighting and style. Do not use markdown."
)

# Shown as the Markdown note at the top of the Models tab (app.py). Not an
# AppConfig field - it's not per-user/persisted state, just UI copy kept
# here so it can be edited without hunting through app.py's Gradio layout
# code. {ignored_substrings} is filled in at render time from
# core.models.IGNORED_SUBSTRINGS, so that part always reflects the real,
# current ignore-list rather than going stale if it's ever changed.
MODELS_TAB_INTRO = (
    "Every `.gguf` under a `webui/models/<folder>/` is picked up, "
    "grouped by folder: files with \"mmproj\" in the name are "
    "projectors, everything else is a selectable quant. Files "
    "matching `{ignored_substrings}` are ignored "
    "(e.g. speculative-decoding draft models). Dropdown choices "
    "marked \"(download, ...)\" come from the curated list and "
    "aren't on disk yet - picking one changes the button below to "
    "\"Download\", which queues it in the background; \"Abort all "
    "downloads\" stops whatever's in flight and clears the rest of "
    "the queue."
)


@dataclass
class AppConfig:
    # Server management: "managed" (start our own if none is running,
    # connect to an existing one otherwise - always on 127.0.0.1, see
    # core.server.MANAGED_HOST) or "external" (never manage a process,
    # always talk to external_url, which may point anywhere).
    server_mode: str = "managed"
    server_port: int = 8901
    external_url: str = "http://127.0.0.1:8080"
    # Whether to start the managed server automatically on app launch, if
    # it's installed but not already running - see app.py's autostart
    # wiring. Ignored right after switching mode to managed (the user must
    # start it explicitly at least once, so a stale/wrong model doesn't
    # silently start loading the moment they flip the radio).
    autostart_managed_llama: bool = False

    # Model selection: "<folder>/<file stem>" of a ModelVariant/MmprojVariant
    # from core.models. Empty mmproj_name means auto-pick the largest mmproj
    # in the selected model's own folder.
    model_name: str = ""
    mmproj_name: str = ""
    # llama-server accepts an exact number, "auto" (fit to free VRAM at
    # startup - its own default), or "all". Kept as a free-form string here
    # so all three forms pass straight through to -ngl.
    n_gpu_layers: str = "auto"
    context_size: int = 4096
    extra_server_args: str = ""

    # Downscale images (in memory, before base64-encoding - never touches
    # the file on disk) to at most this many megapixels before sending them
    # to the vision model. Vision encoders tokenize proportionally to input
    # resolution, so an oversized source image can burn most of the context
    # window on the image alone, starving the actual completion.
    resize_enabled: bool = True
    resize_target_mp: float = 1.0
    # Snap both output dimensions to a multiple of this value - the shorter
    # side by resizing, the longer side by center-cropping the excess
    # rather than stretching it. What SDXL and similar trainers expect
    # (typically 64). Only applies when resize_enabled and a resize
    # actually happens.
    snap_enabled: bool = True
    snap_multiple: int = 64

    # Generation settings.
    prompt_template: str = DEFAULT_PROMPT
    # Low but not 0.0 - greedy (temp=0.0) decoding deterministically locks
    # in a bad top token every time (a repetition loop, a borderline
    # refusal on an imperfectly abliterated model), with no chance of
    # sampling around it. Captioning is still a grounded/factual task
    # though, so this stays well below creative-writing temperatures.
    temperature: float = 0.3
    top_p: float = 0.9
    # Some models "think" before answering (a separate reasoning_content
    # field); their reasoning alone can consume several hundred tokens
    # before any real caption text is produced, so this needs real headroom.
    max_tokens: int = 1024
    # How long to wait for a single captioning request before giving up.
    # Slow hardware, a big prompt/context, or a reasoning model can all
    # push real generation time well past a short default.
    request_timeout: int = 300

    # Captioning output.
    trigger_word: str = ""
    overwrite_existing: bool = False
    recursive_batch: bool = False

    # Off by default: when disabled, no log handler is even attached (not
    # just hidden), so there's no formatting/buffering overhead anywhere in
    # the app from log calls nobody's going to look at. Takes effect on
    # next app restart, same as the other settings that affect UI structure
    # or process startup.
    debug_tab_enabled: bool = False


def load() -> AppConfig:
    defaults = asdict(AppConfig())
    if SETTINGS_PATH.exists():
        try:
            on_disk = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            log.debug("Loaded settings from %s", SETTINGS_PATH)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read %s (%s) - using defaults", SETTINGS_PATH, exc)
            on_disk = {}
        # server_mode was renamed "auto" -> "managed" - an older
        # settings.json predating that rename still says "auto".
        if on_disk.get("server_mode") == "auto":
            on_disk["server_mode"] = "managed"
        # server_host was removed (managed mode now always binds
        # 127.0.0.1 - see core.server.MANAGED_HOST) - drop it explicitly
        # rather than relying on the generic unknown-key filter below, so
        # it's clear this one's gone on purpose, not just unrecognized.
        on_disk.pop("server_host", None)
        defaults.update({k: v for k, v in on_disk.items() if k in defaults})
    else:
        log.debug("No settings.json yet at %s - using built-in defaults", SETTINGS_PATH)
    return AppConfig(**defaults)


def save(cfg: AppConfig) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    log.debug("Saved settings to %s", SETTINGS_PATH)
