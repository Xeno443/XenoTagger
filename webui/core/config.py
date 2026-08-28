"""Persistent app settings, shared by the GUI and the headless CLI."""

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


@dataclass
class AppConfig:
    # Server management: "auto" (start our own if none is running, connect
    # to an existing one otherwise) or "external" (never manage a process,
    # always talk to external_url).
    server_mode: str = "auto"
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    external_url: str = "http://127.0.0.1:8080"

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

    # Generation settings.
    prompt_template: str = DEFAULT_PROMPT
    temperature: float = 0.7
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
