"""Discover and sanity-check GGUF vision models under webui/models/.

Real-world GGUF repos put everything in one flat folder: several quant
variants of the main model, sometimes more than one mmproj precision, and
occasionally unrelated sidecar files (e.g. speculative-decoding "draft"
models). So instead of requiring exactly one model file and one mmproj
file per folder, each main-model quant becomes its own selectable
ModelVariant, each mmproj file becomes its own selectable MmprojVariant,
and callers explicitly pair one of each (see resolve_selection below).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import AppConfig

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CACHE_PATH = Path(__file__).resolve().parent.parent / "config" / "models_cache.json"

GGUF_MAGIC = b"GGUF"

# Filenames containing any of these are neither the main model nor an
# mmproj - e.g. speculative-decoding draft-model sidecars. Extend as new
# kinds of auxiliary GGUF files show up in the wild.
IGNORED_SUBSTRINGS = ("draft", "dflash", "speculative")


@dataclass
class ModelVariant:
    name: str  # "<folder>/<file stem>", shown in the UI
    folder: Path
    model_path: Path
    architecture: Optional[str] = None
    valid: bool = False
    error: Optional[str] = None


@dataclass
class MmprojVariant:
    name: str  # "<folder>/<file stem>"
    folder: Path
    mmproj_path: Path
    valid: bool = False
    error: Optional[str] = None


def _read_gguf_metadata(path: Path) -> tuple[bool, Optional[str], Optional[str]]:
    """Returns (is_valid_gguf, architecture, error). Slow - reads/parses the
    GGUF header, which for a multi-GB file is the expensive part of a scan.
    Callers should go through _cached_metadata instead of calling directly.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError as exc:
        return False, None, f"could not read file: {exc}"
    if magic != GGUF_MAGIC:
        return False, None, "not a valid GGUF file (bad magic bytes)"

    try:
        import gguf  # optional dependency, gives us architecture/quant info

        reader = gguf.GGUFReader(str(path))
        arch_field = reader.fields.get("general.architecture")
        arch = None
        if arch_field is not None and arch_field.parts:
            arch = bytes(arch_field.parts[-1]).decode("utf-8", "ignore")
        return True, arch, None
    except ImportError:
        return True, None, None
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        return True, None, f"gguf header parse warning: {exc}"


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cached_metadata(path: Path, cache: dict) -> tuple[bool, Optional[str], Optional[str]]:
    """Like _read_gguf_metadata, but skips the parse if `path` is unchanged
    (same size + mtime) since it was last cached.
    """
    key = str(path)
    stat = path.stat()

    entry = cache.get(key)
    if entry is not None and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
        # Cache hits are the routine/fast path on every scan - not logged,
        # or this would flood the buffer since scans happen often.
        return entry["valid"], entry.get("architecture"), entry.get("error")

    log.debug("New or changed GGUF, parsing header: %s", path)
    valid, arch, err = _read_gguf_metadata(path)
    if not valid:
        log.debug("%s is not a valid GGUF: %s", path, err)
    cache[key] = {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "valid": valid,
        "architecture": arch,
        "error": err,
    }
    return valid, arch, err


def _scan_all() -> tuple[list[ModelVariant], list[MmprojVariant]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    seen_keys: set[str] = set()
    models: list[ModelVariant] = []
    mmprojs: list[MmprojVariant] = []

    for folder in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        for path in sorted(folder.glob("*.gguf")):
            lower = path.name.lower()
            if any(s in lower for s in IGNORED_SUBSTRINGS):
                continue

            seen_keys.add(str(path))
            display_name = f"{folder.name}/{path.stem}"

            if "mmproj" in lower:
                ok, _, err = _cached_metadata(path, cache)
                mmprojs.append(
                    MmprojVariant(
                        name=display_name, folder=folder, mmproj_path=path,
                        valid=ok, error=err,
                    )
                )
            else:
                ok, arch, err = _cached_metadata(path, cache)
                models.append(
                    ModelVariant(
                        name=display_name, folder=folder, model_path=path,
                        architecture=arch, valid=ok, error=err,
                    )
                )

    # Drop cache entries for files that no longer exist, so it doesn't grow
    # forever as models get swapped out.
    stale_keys = set(cache) - seen_keys
    if stale_keys:
        log.debug("Pruning %d stale cache entries (file no longer present)", len(stale_keys))
        for key in stale_keys:
            del cache[key]
    _save_cache(cache)

    log.debug(
        "Scanned %s: %d model(s), %d mmproj(s) (%d invalid)",
        MODELS_DIR, len(models), len(mmprojs),
        sum(1 for m in models if not m.valid) + sum(1 for m in mmprojs if not m.valid),
    )
    return models, mmprojs


def scan_all() -> tuple[list[ModelVariant], list[MmprojVariant]]:
    """One scan pass returning both lists - use this over the two functions
    below when you need both, to avoid walking webui/models/ twice."""
    return _scan_all()


def scan_model_variants() -> list[ModelVariant]:
    models, _ = _scan_all()
    return models


def scan_mmproj_variants() -> list[MmprojVariant]:
    _, mmprojs = _scan_all()
    return mmprojs


def mmprojs_for_folder(folder: Path) -> list[MmprojVariant]:
    return [m for m in scan_mmproj_variants() if m.folder == folder]


def resolve_selection(cfg: AppConfig) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Returns (model_path, mmproj_path, error) for the configured selection.

    An empty cfg.mmproj_name means "auto-pick the largest mmproj in the
    model's own folder" (largest = usually the highest-precision projector).
    """
    if not cfg.model_name:
        log.debug("resolve_selection: no model configured")
        return None, None, "No model selected."

    model = next((m for m in scan_model_variants() if m.name == cfg.model_name), None)
    if model is None:
        log.warning("resolve_selection: configured model '%s' not found by scan", cfg.model_name)
        return None, None, f"Selected model '{cfg.model_name}' no longer exists."
    if not model.valid:
        log.warning("resolve_selection: model '%s' is invalid: %s", cfg.model_name, model.error)
        return None, None, f"Selected model '{cfg.model_name}' is invalid: {model.error}"

    candidates = mmprojs_for_folder(model.folder)
    if not candidates:
        log.warning("resolve_selection: no mmproj found in %s", model.folder)
        return None, None, f"No mmproj GGUF found in {model.folder}."

    if cfg.mmproj_name:
        mmproj = next((m for m in candidates if m.name == cfg.mmproj_name), None)
        if mmproj is None:
            log.warning("resolve_selection: configured mmproj '%s' not found by scan", cfg.mmproj_name)
            return None, None, f"Selected mmproj '{cfg.mmproj_name}' no longer exists."
    else:
        mmproj = max(candidates, key=lambda m: m.mmproj_path.stat().st_size)
        log.debug("resolve_selection: auto-picked largest mmproj '%s'", mmproj.name)

    if not mmproj.valid:
        log.warning("resolve_selection: mmproj '%s' is invalid: %s", mmproj.name, mmproj.error)
        return None, None, f"Selected mmproj '{mmproj.name}' is invalid: {mmproj.error}"

    log.debug("resolve_selection: model=%s mmproj=%s", model.model_path, mmproj.mmproj_path)
    return model.model_path, mmproj.mmproj_path, None
