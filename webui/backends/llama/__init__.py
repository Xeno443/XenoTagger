"""llama.cpp VLM captioner - a llama-server that this app either spawns
and owns ("managed", always 127.0.0.1) or just talks to ("external", any
URL). Managed vs. external is internal to this package; to the rest of
the app it's one adaptive-VRAM captioner.

  - server.py      llama-server process lifecycle, health/props checks
  - client.py      chat API calls, image preprocessing
  - session.py     which server this app process owns + the cached
                   reachability state the GUI gates on
  - models.py      local GGUF discovery + curated model list
  - catalog.py     the Models tab rows (folder = row, quant/mmproj dropdowns)
  - install.py     llama-server.exe installer (ggml-org releases)
  - settings_ui.py Settings -> Llama (GUI only)
"""

from __future__ import annotations

from typing import Optional

from core.backend import Backend, ModelOption, ModelRow
from core.config import AppConfig
from core.downloads import DownloadItem

from . import catalog, session
from .server import LLAMA_SERVER_EXE, get_loaded_model_name, is_healthy


class LlamaBackend(Backend):
    id = "llama"
    label = "llama-server"
    role = "captioner"
    vram = "adaptive"
    settings_ui_module = "backends.llama.settings_ui"
    has_models = True
    selectable_models = True
    models_help = catalog.MODELS_HELP

    def is_loaded(self, cfg: AppConfig) -> bool:
        # Ownership only, not a health check - same test "Stop llama
        # server" is gated on. A server we didn't start is never ours to
        # stop/restart.
        return session.is_server_managed_by_us(session.managed_base_url(cfg))

    def load(self, cfg: AppConfig) -> tuple[bool, str]:
        return session.try_start_managed_llama(cfg)

    def unload(self, cfg: AppConfig) -> None:
        session.stop_managed()
        session.refresh_reachability_cache(cfg)

    def wants_autoload(self, cfg: AppConfig) -> bool:
        # Reads the reachability cache the GUI's startup check has just
        # refreshed - no extra network call.
        if cfg.server_mode != "managed" or not cfg.autostart_managed_llama:
            return False
        status = session.cached_status()
        return status.installed and not status.reachable

    def status_bar_segments(self, cfg: AppConfig) -> list[tuple[str, str]]:
        base_url = session.display_base_url(cfg)
        healthy = is_healthy(base_url)
        installed = LLAMA_SERVER_EXE.exists()  # cheap, local - no extra network call
        # Always the server's own answer, never our config - our selection is
        # just what we'd ask it to load next, not necessarily what's actually
        # loaded right now (e.g. external mode, or a server someone else picked).
        model = get_loaded_model_name(base_url) if healthy else None
        return [("llama", session.status_label(cfg, healthy, installed)), ("Model", model or "n/a")]

    def shutdown(self, cfg: AppConfig) -> None:
        session.stop_managed()

    def on_exit(self) -> None:
        session.stop_managed()

    def list_models(self, cfg: AppConfig) -> list[ModelRow]:
        return catalog.list_models(cfg)

    def model_options(self, cfg: AppConfig, key: str) -> list[ModelOption]:
        return catalog.model_options(cfg, key)

    def model_downloads(self, cfg: AppConfig, key: str, values: list) -> list[DownloadItem]:
        return catalog.model_downloads(cfg, key, values)

    def set_active_problem(self, cfg: AppConfig, key: str, values: list) -> Optional[str]:
        return catalog.set_active_problem(cfg, key, values)

    def set_active(self, cfg: AppConfig, key: str, values: list) -> tuple[Optional[dict], str]:
        return catalog.set_active(cfg, key, values)

    def model_status(self, cfg: AppConfig) -> str:
        return catalog.model_status(cfg)


BACKEND = LlamaBackend()
