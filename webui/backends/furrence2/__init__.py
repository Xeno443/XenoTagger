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
Load button or autoload, and nothing calls caption() until the pipeline
exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.backend import Backend, CaptionOutput, ImageInput, ModelRow, Tag
from core.config import AppConfig
from core.downloads import DownloadItem

from . import install, worker_client

_MODEL_KEY = "furrence2-large"


class Furrence2Backend(Backend):
    id = "furrence2"
    label = "Furrence2"
    role = "captioner"
    takes_tags = "required"
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

    def setup_problem(self, cfg: AppConfig) -> Optional[str]:
        if not install.deps_installed():
            return "Furrence2's dependencies aren't installed (Settings -> Furrence2)."
        if not install.model_downloaded():
            return "Furrence2's model isn't downloaded (Models tab)."
        return None

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
        self,
        cfg: AppConfig,
        image: ImageInput,
        variant: str = "",
        tags: Optional[list[Tag]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> CaptionOutput:
        """variant is unused (length comes from the settings), and so is
        should_stop - one caption takes about a second. The prompt's tag
        order is shuffled with the file name as seed, so captioning the
        same file again builds the same prompt."""
        if tags is None:
            raise worker_client.Furrence2Error("Furrence2 needs a tagger's tags, and none ran.")
        result = worker_client.caption(
            image, [(t.name, t.category) for t in tags],
            seed="" if isinstance(image, bytes) else Path(image).name,
            length_min=cfg.furrence2_length_min, length_max=cfg.furrence2_length_max,
        )
        return CaptionOutput(
            text=result.caption, seconds=result.seconds,
            note=f"{result.tags_used} tags, length {result.length}, {result.prompt_tokens} prompt tokens",
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
