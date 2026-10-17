"""What the combined Models tab shows for llama (LlamaBackend's model
methods): one row per model folder, with its quant and mmproj as the two
dropdowns. Gradio-free - it returns core.backend's ModelRow/ModelOption,
and the tab renders them.

One row per model DIRECTORY (models.group_models), not one per quant file
- a folder's several quants and mmproj precisions become the two
dropdowns' choices. Clicking a row only shows its choices; switching
models means a full llama-server restart + multi-GB reload, too expensive
for a stray click, so only "Set as active model" commits one (writes
model_name/mmproj_name). Not-yet-downloaded curated quants/mmprojs are
choices too, labeled "(download, 6.8 GB)" - picking one turns that button
into "Download" - next to the "(on disk, 8.0 GB)" ones.

Values are "<folder>/<file stem>" for both local and curated choices, so
the folder - the row key - can always be read back off a value, and a
local file and its curated entry share one value.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from core.backend import ModelOption, ModelRow
from core.config import AppConfig
from core.downloads import DownloadItem, format_size

from . import session
from .models import (
    IGNORED_SUBSTRINGS, ModelGroup, cache_file_metadata, group_models, load_curated_models, merge_curated, scan_all,
)

log = logging.getLogger(__name__)

MODELS_HELP = (
    "every `.gguf` under a `webui/models/<folder>/` is picked up, "
    "grouped by folder: files with \"mmproj\" in the name are "
    "projectors, everything else is a selectable quant. Files "
    f"matching `{', '.join(IGNORED_SUBSTRINGS)}` are ignored "
    "(e.g. speculative-decoding draft models). In the dropdowns, "
    "\"(on disk, ...)\" is already downloaded, \"(download, ...)\" comes "
    "from the curated list and isn't yet, and ★ is the active one. "
    "\"Set as active model\" takes effect the next time this app "
    "starts llama-server."
)

# The last list_models() scan, reused by the per-row calls in between
# (picking a row, changing a dropdown) - same staleness as the table on
# screen, which list_models() refreshes on every Models tab refresh.
_groups_lock = threading.Lock()
_groups: Optional[list[ModelGroup]] = None


def _scan() -> list[ModelGroup]:
    global _groups
    models, mmprojs = scan_all()
    groups = merge_curated(group_models(models, mmprojs), load_curated_models())
    with _groups_lock:
        _groups = groups
    return groups


def _group(key: str) -> Optional[ModelGroup]:
    with _groups_lock:
        groups = _groups
    if groups is None:
        groups = _scan()
    return next((g for g in groups if g.name == key), None)


def _is_active(cfg: AppConfig, group: ModelGroup) -> bool:
    return any(q.name == cfg.model_name for q in group.quants)


def _local_count(group: ModelGroup) -> str:
    # "n/total" for a curated family (how many of its curated quants are
    # on disk), a plain count for a manual one.
    if group.curated:
        return f"{len(group.quants)}/{len(group.curated.quants)}"
    return str(len(group.quants))


def list_models(cfg: AppConfig) -> list[ModelRow]:
    rows = []
    for g in _scan():
        note = "not downloaded yet - pick a quant below to see download info" if g.curated and not g.quants else ""
        rows.append(ModelRow(
            key=g.name, name=g.name, source="Curated" if g.curated else "Manual",
            local=_local_count(g), active=_is_active(cfg, g), note=note,
        ))
    return rows


def _local_choice_label(name: str, path: Path, active: bool) -> str:
    """"Q8_0 (on disk, 8.0 GB)" - the bare file stem like a curated choice's
    "Q4_K (download, 4.6 GB)", so both read as one format and what's
    already downloaded stands out. The value keeps the "<folder>/" prefix.
    ★ marks the configured one, like the table's star column."""
    stem = name.split("/", 1)[-1]
    star = "★ " if active else ""
    try:
        return f"{star}{stem} (on disk, {format_size(path.stat().st_size)})"
    except OSError:
        return f"{star}{stem} (on disk)"


def model_options(cfg: AppConfig, key: str) -> list[ModelOption]:
    """Quant and mmproj for one folder, preselecting the configured pair if
    this is the active folder. A folder with no quant at all (e.g. only its
    mmproj finished downloading) shows both as N/A."""
    group = _group(key)
    if group is None:
        return [ModelOption("Quant"), ModelOption("mmproj")]

    quant_pairs = [
        (_local_choice_label(q.name, q.model_path, q.name == cfg.model_name), q.name) for q in group.quants
    ]
    mmproj_pairs = [
        (_local_choice_label(m.name, m.mmproj_path, m.name == cfg.mmproj_name), m.name) for m in group.mmprojs
    ]
    if group.curated:
        local_quants = {q.name for q in group.quants}
        for cq in group.curated.quants:
            value = f"{group.name}/{cq.name}"
            if value not in local_quants:
                quant_pairs.append((f"{cq.name} (download, {format_size(cq.size_bytes)})", value))
        local_mmprojs = {m.name for m in group.mmprojs}
        for cm in group.curated.mmprojs:
            value = f"{group.name}/{cm.name}"
            if value not in local_mmprojs:
                mmproj_pairs.append((f"{cm.name} (download, {format_size(cm.size_bytes)})", value))

    if not quant_pairs:
        return [ModelOption("Quant"), ModelOption("mmproj")]
    active = _is_active(cfg, group)
    return [
        ModelOption("Quant", quant_pairs, cfg.model_name if active else None),
        ModelOption("mmproj", mmproj_pairs, cfg.mmproj_name if active else None),
    ]


def _selection(key: str, values: list) -> tuple[Optional[str], Optional[str]]:
    """(quant, mmproj) - a value from another folder counts as nothing
    selected: it's a stale dropdown value from the row the user just
    switched away from (see the Models tab's allow_custom_value)."""
    padded = list(values) + [None, None]
    return tuple(v if v and v.startswith(f"{key}/") else None for v in padded[:2])


def model_downloads(cfg: AppConfig, key: str, values: list) -> list[DownloadItem]:
    """Whichever of the selected quant/mmproj isn't on disk yet, if it's a
    curated one."""
    group = _group(key)
    if group is None or group.curated is None:
        return []
    quant, mmproj = _selection(key, values)
    items = []
    if quant and not any(q.name == quant for q in group.quants):
        stem = quant.split("/", 1)[-1]
        cq = next((c for c in group.curated.quants if c.name == stem), None)
        if cq:
            items.append(DownloadItem(
                url=cq.url, dest_path=group.folder / Path(cq.url).name, label=cq.name, size_bytes=cq.size_bytes,
                after_download=cache_file_metadata,
            ))
    if mmproj and not any(m.name == mmproj for m in group.mmprojs):
        stem = mmproj.split("/", 1)[-1]
        cm = next((c for c in group.curated.mmprojs if c.name == stem), None)
        if cm:
            items.append(DownloadItem(
                url=cm.url, dest_path=group.folder / Path(cm.url).name, label=cm.name, size_bytes=cm.size_bytes,
                after_download=cache_file_metadata,
            ))
    return items


def set_active_problem(cfg: AppConfig, key: str, values: list) -> Optional[str]:
    """What's wrong with the selection itself first (more useful to know),
    then whether the server would pick it up at all."""
    group = _group(key)
    quant, mmproj = _selection(key, values)
    if group is None:
        return "No model selected - click a row in the table first."
    if not group.quants:
        return "No model file in this folder (download incomplete?) - can't be set as active."
    if not quant:
        return "No quant selected."
    if not any(q.name == quant for q in group.quants):
        return f"\"{quant.rsplit('/', 1)[-1]}\" isn't downloaded yet."
    if not mmproj:
        return "This model has no mmproj - can't be set as active."
    if not any(m.name == mmproj for m in group.mmprojs):
        return f"\"{mmproj.rsplit('/', 1)[-1]}\" isn't downloaded yet."
    return session.set_active_blocked(cfg)


def set_active(cfg: AppConfig, key: str, values: list) -> tuple[Optional[dict], str]:
    problem = set_active_problem(cfg, key, values)
    if problem:
        return None, problem
    quant, mmproj = _selection(key, values)
    log.info("Set active model to %s (mmproj=%s)", quant, mmproj)
    if session.is_server_managed_by_us(session.managed_base_url(cfg)):
        how = "Stop and start llama-server in Settings -> Llama to load it."
    else:
        how = "llama-server loads it the next time this app starts it."
    return dict(model_name=quant, mmproj_name=mmproj), f"Active model set to {quant} (mmproj: {mmproj}). {how}"


def model_status(cfg: AppConfig) -> str:
    if cfg.server_mode == "external":
        return "Model: whatever the external server has loaded (see the status bar)."
    if not cfg.model_name:
        return "Model: none selected yet."
    return f"Model: {cfg.model_name} (mmproj: {cfg.mmproj_name or 'largest in its folder'})."
