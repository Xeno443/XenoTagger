"""Furrence2 (lodestone-horizon/furrence2-large) - a Florence-2 fine-tune
that writes a caption from the image plus its e621 tags (Hydra's, or a
tags sidecar). Not llama.cpp-compatible and stuck on transformers 4.46,
so it runs in its own worker process with its own lib folder (see
install.py). All-or-nothing VRAM ("fixed", ~1.5GB), like Hydra.

  - install.py        deps (system torch + the 4.46 lib folder), model
                      download + fp16 conversion
  - worker.py         the worker process (original Florence-2 code)
  - worker_client.py  start/stop/caption from the app side
  - vendor/           the original Florence-2 code + timm stub
  - settings_ui.py    Settings -> Furrence2 (GUI only)

Loading is the pipeline's job (start_pipeline, v0.9 step 3) - there's no
Load button or autoload, and nothing calls caption() yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from core.backend import Backend, ModelRow
from core.config import AppConfig
from core.downloads import DownloadItem

from . import install, worker_client

_MODEL_KEY = "furrence2-large"


class Furrence2Backend(Backend):
    id = "furrence2"
    label = "Furrence2"
    role = "captioner"
    vram = "fixed"
    settings_ui_module = "backends.furrence2.settings_ui"
    has_models = True
    selectable_models = False  # one model, used as soon as it's downloaded
    models_help = (
        "one model (3.3GB download, converted to 1.6GB fp16 right after). "
        "Install its dependencies in Settings -> Furrence2."
    )

    def is_loaded(self, cfg: AppConfig) -> bool:
        return worker_client.status().running

    def load(self, cfg: AppConfig) -> tuple[bool, str]:
        try:
            device = worker_client.start()
        except worker_client.Furrence2Error as exc:
            return False, f"Couldn't start Furrence2: {exc}"
        return True, f"Furrence2 running on {device}."

    def unload(self, cfg: AppConfig) -> None:
        worker_client.stop()

    def status_bar_segments(self, cfg: AppConfig) -> list[tuple[str, str]]:
        st = worker_client.status()
        if st.running:
            return [("Furrence2", f"running ({st.device})")]
        if st.crashed:
            return [("Furrence2", "error")]
        if not st.deps_installed or not st.model_downloaded:
            return [("Furrence2", "not set up")]
        return [("Furrence2", "not running")]

    def shutdown(self, cfg: AppConfig) -> None:
        worker_client.stop()

    def on_exit(self) -> None:
        worker_client.stop()

    def caption(
        self, cfg: AppConfig, image: Union[str, Path, bytes], tags: list[tuple[str, str]], seed: str = "",
    ) -> worker_client.Furrence2Result:
        """tags: (e621 name with underscores, category) pairs, best first -
        Furrence2 can't caption without them. Becomes the Backend
        interface's caption() in step 3. Raises Furrence2Error."""
        return worker_client.caption(
            image, tags, seed=seed,
            length_min=cfg.furrence2_length_min, length_max=cfg.furrence2_length_max,
        )

    def list_models(self, cfg: AppConfig) -> list[ModelRow]:
        missing = len(install.download_items())
        downloaded = missing == 0
        return [ModelRow(
            key=_MODEL_KEY, name="lodestone-horizon/furrence2-large", source="Curated",
            local=f"{2 - missing}/2", active=downloaded,
            note="" if downloaded else "not downloaded yet",
        )]

    def model_downloads(self, cfg: AppConfig, key: str, values: list) -> list[DownloadItem]:
        return install.download_items() if key == _MODEL_KEY else []

    def model_status(self, cfg: AppConfig) -> str:
        if install.model_downloaded():
            return "Model: furrence2-large (fp16), downloaded."
        return "Model: furrence2-large, not downloaded yet."


BACKEND = Furrence2Backend()
