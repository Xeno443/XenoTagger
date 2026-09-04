"""In-process Hydra 3.5 (RedRocket) e621 tag classifier - core.captioner's
second-stage hook (see captioner.py's own docstring for why that single
hook point exists). Pure Python + torch, imported and run in the same
interpreter as the GUI/CLI (this repo's "no venv, everything in
system\\python" house rule applies to Hydra's deps too - see
core.hydra_install), so unlike llama-server there's no separate
process/port/health-check to manage here - just a lazily-populated
module-level singleton.

load()/unload() are the only things that touch the (heavy, ~1GB, VRAM-
competing-with-llama-server) model itself, and are always explicit calls
driven by Settings -> Hydra's Load/Unload buttons (or autoload at
launch) - see AppConfig.hydra_enabled's own docstring for why loading is
never implicit here, same reasoning as managed llama-server. classify()
never loads anything itself; it raises HydraError immediately if nothing
is loaded, and captioner.py treats that (like any other HydraError) as a
soft failure - skip the tag-append, keep the VLM-only caption - not a
hard error that fails the whole caption.

Calibration (model.calibrate(cfg.hydra_metric)) is recomputed on every
classify() call rather than cached - cheap, a single reduction per label
over an already-in-memory validation tensor - so hydra_metric/
hydra_implications/hydra_exclude_categories/hydra_exclude_tags/
hydra_max_tags all take effect on the very next caption with no reload
needed. Only hydra_device is baked into the loaded model's tensors and
needs an explicit Unload + Load to actually change.

torch and the vendored rr_hydra package are imported lazily inside
functions, not at module top level, so importing this module never
requires torch to be installed - mirrors core.models' own `import gguf`
pattern for the same reason (this module needs to be importable, and
status() needs to work, before the optional deps are ever installed).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .config import AppConfig
from .hydra_install import MODEL_PATH, deps_installed, model_downloaded

log = logging.getLogger(__name__)


class HydraError(RuntimeError):
    pass


@dataclass
class HydraStatus:
    deps_installed: bool
    model_downloaded: bool
    loaded: bool
    device: Optional[str] = None  # the device it's actually loaded on, None if not loaded


@dataclass
class HydraResult:
    tags: dict[str, float]  # already implication/threshold-filtered, sorted desc by probability
    tag_text: str  # comma-joined, underscores -> spaces, ready to append to a caption


_lock = threading.Lock()
_model = None  # a vendor.rr_hydra.model.Hydra instance, once loaded
_device: Optional[str] = None


def status() -> HydraStatus:
    return HydraStatus(
        deps_installed=deps_installed(),
        model_downloaded=model_downloaded(),
        loaded=_model is not None,
        device=_device,
    )


def load(cfg: AppConfig) -> None:
    """Explicit, heavy: loads hydra-3.5.safetensors and moves it onto
    cfg.hydra_device. Raises HydraError with the real underlying reason
    (e.g. a CUDA OOM against an already-running llama-server) rather than
    swallowing failures - surfacing that clearly is the whole point of
    this being an explicit action instead of an implicit lazy-load.
    No-op if a model is already loaded."""
    global _model, _device
    with _lock:
        if _model is not None:
            log.debug("hydra_classifier.load(): already loaded, ignoring")
            return

        if not deps_installed():
            raise HydraError(
                "Hydra's Python dependencies aren't installed yet - "
                "use \"Install Hydra dependencies\" in Settings -> Hydra."
            )
        if not model_downloaded():
            raise HydraError(
                f"{MODEL_PATH} not found - use \"Download Hydra model\" in Settings -> Hydra."
            )

        try:
            import torch

            # Absolute, not relative (`from ..vendor...`) - core/ is
            # imported as a top-level package (app.py/cli.py run with
            # webui/ as sys.path[0] and do `from core import ...`
            # directly, never through a "webui" package umbrella), so
            # core has no parent package for a ".." to climb to. vendor/
            # is importable the same top-level way core itself is.
            from vendor.rr_hydra.model import load_model
        except ImportError as exc:
            raise HydraError(f"Could not import Hydra's dependencies: {exc}") from exc

        device = cfg.hydra_device
        try:
            if device == "cuda" and not torch.cuda.is_available():
                log.warning("hydra_classifier.load(): CUDA requested but not available - falling back to CPU")
                device = "cpu"
        except Exception as exc:  # noqa: BLE001 - e.g. a CUDA runtime left in a bad
            # state by a torch import that raced an in-progress dependency
            # install (see _hydra_install_status_ui's own comment) - must
            # not escape as a raw, uncaught exception, since torch.cuda
            # can stay broken for the rest of this process even after the
            # install finishes, and every retry would otherwise crash the
            # same way instead of surfacing a clean, dismissable message.
            raise HydraError(f"Could not query CUDA availability: {exc}") from exc

        log.info("Loading Hydra model from %s onto %s ...", MODEL_PATH, device)
        try:
            model = load_model(str(MODEL_PATH))
            if device != "cpu":
                model = model.to(device=device)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is (e.g. a CUDA OOM)
            log.error("hydra_classifier.load(): failed to load model: %s", exc)
            raise HydraError(f"Failed to load Hydra model: {exc}") from exc

        _model = model
        _device = device
        log.info("Hydra model loaded on %s", device)


def unload() -> None:
    """Frees the loaded model and empties the CUDA cache. No-op if
    nothing is loaded."""
    global _model, _device
    with _lock:
        if _model is None:
            return
        log.info("Unloading Hydra model")
        _model = None
        _device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def classify(image: Union[str, Path, bytes], cfg: AppConfig) -> HydraResult:
    """Raises HydraError if nothing is currently loaded (see load()) or
    if classification itself fails for any reason - callers such as
    core.captioner.caption_image() are expected to catch this and skip
    the tag-append, not fail the whole caption."""
    model = _model
    if model is None:
        raise HydraError("Hydra model is not loaded - use \"Load Hydra model\" in Settings -> Hydra.")

    try:
        import torch

        from vendor.rr_hydra import image as hydra_image

        source = str(image) if isinstance(image, Path) else image
        tensor = model.load_image(source)
        patches, sizes = hydra_image.stack(
            [tensor], 16, model.image_config.max_seqlen, device=_device,
        )
        with torch.inference_mode():
            output = model.forward(model.from_srgb(patches), sizes).cpu()[0]
        calibration = model.calibrate(cfg.hydra_metric)
        tags = calibration.classify_output(
            output,
            implications=cfg.hydra_implications,
            exclude_categories=cfg.hydra_exclude_categories,
            exclude_labels=cfg.hydra_exclude_tags,
            sort=True,
        )
    except Exception as exc:  # noqa: BLE001 - classification failures are soft, see module docstring
        log.warning("hydra_classifier.classify(): failed: %s", exc)
        raise HydraError(f"Hydra classification failed: {exc}") from exc

    if cfg.hydra_max_tags > 0 and len(tags) > cfg.hydra_max_tags:
        tags = dict(list(tags.items())[: cfg.hydra_max_tags])

    tag_text = ", ".join(label.replace("_", " ") for label in tags)
    return HydraResult(tags=tags, tag_text=tag_text)
