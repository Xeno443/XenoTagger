"""e621 tag vocabulary for Review's tag editor - a plain CSV parse, no torch,
no relation to Hydra's own (much smaller, classifier-specific) label set in
vendor/rr_hydra. The point of using the full e621 vocabulary rather than
Hydra's own ~8.9k labels is that a user may want to hand-add a tag neither
the VLM nor Hydra caught (an artist credit, a specific object, a species
Hydra wasn't trained on) - restricting to Hydra's own output vocabulary
would block exactly that case.

CSV rows are "tag,category,post_count,\"alias1,alias2,...\"" (the same shape
sd-webui-tagcomplete-style tag CSVs use; category/post_count are parsed but
unused for now). Everything is normalized to space form ("_" -> " ") at
parse time - this must match the space-form tags Hydra already writes into
.tags/the combined caption (hydra_classifier.py's own
`tag_text = ", ".join(label.replace("_", " ") for label in tags)`), or
alias resolution and dropdown-choice matching against existing file content
would silently break.

Parse results are cached by (path, mtime) so repeated calls (e.g. once per
Review page load, once per Settings save) after the first are cheap and a
changed/replaced CSV is picked up automatically without an app restart.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig

log = logging.getLogger(__name__)


@dataclass
class TagVocab:
    tags: list[str] = field(default_factory=list)  # canonical, space-form, CSV order (already sorted by post_count desc)
    alias_to_canonical: dict[str, str] = field(default_factory=dict)  # space-form alias -> space-form canonical


_EMPTY_VOCAB = TagVocab()
_cache: dict[str, tuple[float, TagVocab]] = {}  # path -> (mtime, vocab)
_warned_paths: set[str] = set()  # paths we've already logged a load failure for, so it isn't repeated every call

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
AUTOCOMPLETE_JS_PATH = STATIC_DIR / "tag_autocomplete.js"


def _parse(path: str) -> TagVocab:
    tags: list[str] = []
    alias_to_canonical: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            canonical = row[0].strip().replace("_", " ")
            tags.append(canonical)
            if len(row) > 3 and row[3].strip():
                for alias in row[3].split(","):
                    alias = alias.strip().replace("_", " ")
                    if alias and alias != canonical:
                        alias_to_canonical[alias] = canonical
    return TagVocab(tags=tags, alias_to_canonical=alias_to_canonical)


def get_vocab(path: str) -> TagVocab:
    """Returns TagVocab([], {}) - not an error - if path is empty, missing,
    or unreadable, logging the failure once per path rather than on every
    call (this is polled far more often than a real change happens)."""
    if not path:
        return _EMPTY_VOCAB

    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        if path not in _warned_paths:
            log.warning("Tag vocabulary CSV not found at %s (%s)", path, exc)
            _warned_paths.add(path)
        return _EMPTY_VOCAB

    cached = _cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        vocab = _parse(path)
    except (OSError, csv.Error) as exc:
        if path not in _warned_paths:
            log.warning("Could not parse tag vocabulary CSV %s: %s", path, exc)
            _warned_paths.add(path)
        return _EMPTY_VOCAB

    _warned_paths.discard(path)
    _cache[path] = (mtime, vocab)
    log.debug("Loaded tag vocabulary from %s: %d tags, %d aliases", path, len(vocab.tags), len(vocab.alias_to_canonical))
    return vocab


def resolve(cfg: AppConfig, tag: str) -> str:
    """Maps a typed/selected alias to its canonical tag; returns the input
    unchanged (just stripped) if it's already canonical or unknown."""
    tag = tag.strip()
    vocab = get_vocab(cfg.hydra_tag_vocab_path)
    return vocab.alias_to_canonical.get(tag, tag)


def to_js_payload(cfg: AppConfig) -> dict:
    """The full vocabulary, ready for json.dumps() into the browser - see
    build_autocomplete_head(). No size cap: unlike a gr.Dropdown's
    `choices` (a framework-reactive prop Gradio renders as real DOM
    elements per entry - confirmed sluggish with even a few hundred), this
    ends up as a plain, inert JS array/object that the autocomplete script
    filters by hand, so its size doesn't matter."""
    vocab = get_vocab(cfg.hydra_tag_vocab_path)
    return {"tags": vocab.tags, "aliases": vocab.alias_to_canonical}


def build_autocomplete_head(cfg: AppConfig) -> str:
    """Builds the HTML injected into the page <head> via
    demo.launch(head=...) - the tag vocabulary as a global JS object,
    followed by static/tag_autocomplete.js's contents inline (no separate
    fetch - see this module's docstring on why a large choices list is
    fine here but wasn't as a gr.Dropdown prop). Baked in once at process
    launch, like other settings already documented as "takes effect on
    next restart" (e.g. AppConfig.debug_tab_enabled) - changing
    hydra_tag_vocab_path in Settings won't hot-swap this."""
    payload = json.dumps(to_js_payload(cfg))
    try:
        script = AUTOCOMPLETE_JS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Could not read %s - Review's tag autocomplete will be unavailable: %s", AUTOCOMPLETE_JS_PATH, exc)
        script = ""
    return f"<script>window.XT_TAG_DATA = {payload};</script>\n<script>{script}</script>"
