"""Hydra 3.5 (RedRocket) e621 tagger - in-process torch, all-or-nothing
VRAM ("fixed"), so the pipeline loads it before llama-server
(core.lifecycle).

  - classifier.py  the loaded-model singleton: load/unload/classify/status
  - install.py     pip deps + weights download
  - vendor/        vendored inference-only RedRocket hydra source
  - settings_ui.py Settings -> Hydra (GUI only)
"""

from __future__ import annotations

from typing import Optional

from core.backend import Backend, ImageInput, ModelRow, Tag
from core.config import AppConfig
from core.downloads import DownloadItem

from . import classifier, install

# Models tab row key - Hydra has exactly one model.
_MODEL_KEY = "hydra-3.5"


class HydraBackend(Backend):
    id = "hydra"
    label = "Hydra"
    role = "tagger"
    vram = "fixed"
    settings_ui_module = "backends.hydra.settings_ui"
    has_models = True
    selectable_models = False  # one model, used as soon as it's downloaded
    models_help = (
        "one model (~1GB), used as soon as it's downloaded. Install its "
        "dependencies in Settings -> Hydra; the pipeline loads it."
    )

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

    def setup_problem(self, cfg: AppConfig) -> Optional[str]:
        if not install.deps_installed():
            return "Hydra's dependencies aren't installed (Settings -> Hydra)."
        if not install.model_downloaded():
            return "Hydra's model isn't downloaded (Models tab)."
        return None

    def tag(self, cfg: AppConfig, image: ImageInput) -> list[Tag]:
        result = classifier.classify(image, cfg)
        categories = classifier.label_categories()
        return [Tag(name, categories.get(name, "general"), float(conf)) for name, conf in result.tags.items()]

    def tags_from_names(self, cfg: AppConfig, names: list[str]) -> list[Tag]:
        # Hand-added names Hydra doesn't know count as general.
        categories = classifier.label_categories()
        return [Tag(name, categories.get(name, "general"), 1.0) for name in names]

    def shutdown(self, cfg: AppConfig) -> None:
        # Unlike llama-server (a separate OS process), Hydra lives in this
        # process's own CUDA context, which os.execv doesn't clean up -
        # unload explicitly so VRAM isn't leaked into the replaced process.
        classifier.unload()

    def list_models(self, cfg: AppConfig) -> list[ModelRow]:
        downloaded = install.model_downloaded()
        return [ModelRow(
            key=_MODEL_KEY, name="RedRocket-Hydra 3.5", source="Curated",
            local="1/1" if downloaded else "0/1", active=downloaded,
            note="" if downloaded else "not downloaded yet",
        )]

    def model_downloads(self, cfg: AppConfig, key: str, values: list) -> list[DownloadItem]:
        if key != _MODEL_KEY or install.model_downloaded():
            return []
        return [DownloadItem(
            url=install.MODEL_URL, dest_path=install.MODEL_PATH,
            label=install.MODEL_FILENAME, size_bytes=install.MODEL_SIZE_BYTES,
        )]

    def model_status(self, cfg: AppConfig) -> str:
        return f"Model: {install.MODEL_FILENAME}" if install.model_downloaded() else "Model: missing"


BACKEND = HydraBackend()
