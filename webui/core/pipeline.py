"""The caption pipeline: which backends run (Settings -> Pipeline, the
pipeline_* config keys), starting/stopping them, turning one image into
pieces, and rendering the caption file from those pieces.

Backend-neutral like core.lifecycle: callers pass the backend list
(backends.BACKENDS), core/ never imports backends/.

Layers, bottom up:
  - make_plan(): config -> ordered Steps (the tagger first, then every
    captioner's checked variants in registry order), or the problems that
    stop it from running. Read-only, loads nothing.
  - start_pipeline()/stop_pipeline(): load/unload the plan's backends.
    Start is all-or-nothing (a failed load unloads what this start already
    loaded) in fixed-before-adaptive VRAM order; stop unloads in reverse,
    only what the pipeline loaded - a llama-server this app didn't start
    is never stopped (Backend.unload already guarantees that).
  - caption_pieces(): one image -> pieces, no file I/O (Single works on
    uploads). Pieces passed in as existing are reused, not regenerated.
  - render(): pieces -> caption text, no models. Also the Pipeline tab's
    preview, re-rendering, and Review.
  - process_file(): the file layer for Batch/CLI - read existing piece
    files, caption what's missing, write pieces, render <stem>.txt.

Files for wolf.png (each piece its own file, so it opens in any editor and
trainers - which only read <stem>.txt - ignore it):
    wolf.hydra.txt           the tagger's tags, ", "-separated, spaces form
    wolf.llama-short.txt     one per captioner piece (backend-variant)
    wolf.furrence2.txt       (a captioner without variants: just its id)
    wolf.trigger.txt         the trigger word used (empty = none, on purpose)
    wolf.txt                 rendered from the above - never edited directly
    wolf.txt.issue           why the last run couldn't finish this image
Pieces are raw: no trigger word, tags in plain spaces form. Only render()
applies the Output settings, so changing them never needs a model call.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .backend import Backend, BackendError, CaptionOutput, ImageInput, Interrupted, Tag
from .config import AppConfig
from .lifecycle import load_order, load_with_coexistence

log = logging.getLogger(__name__)

TAGS_MODES = ("own", "append", "none")
TAG_STYLES = ("spaces", "underscores")
TRIGGER_PIECE = "trigger"
ISSUE_SUFFIX = ".issue"  # same convention as core.batch: <stem>.txt.issue
# Variant names end up in file names (wolf.llama-<name>.txt).
_VARIANT_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")


# ------------------------------------------------------------------ plan

@dataclass(frozen=True)
class Step:
    backend: Backend
    piece: str  # piece id: file <stem>.<piece>.txt
    label: str  # for progress/errors, e.g. "llama-server short"
    variant: str = ""  # the variant's setting, passed to caption()
    uses_tags: bool = False


@dataclass
class Plan:
    cfg: AppConfig  # the config the plan was made from - every call uses this one
    tagger: Optional[Step]
    captioners: list[Step]
    backends: list[Backend]  # distinct, in load order
    problems: list[str] = field(default_factory=list)

    @property
    def piece_ids(self) -> list[str]:
        """Every piece this plan writes per image, trigger included."""
        steps = ([self.tagger] if self.tagger else []) + self.captioners
        return [s.piece for s in steps] + [TRIGGER_PIECE]


def variants_for(cfg: AppConfig, backend: Backend) -> list[dict]:
    """The backend's variants from the config, normalized - a hand-edited
    settings.json can hold anything. Missing = one "default" variant."""
    raw = cfg.pipeline_variants.get(backend.id) if isinstance(cfg.pipeline_variants, dict) else None
    if raw is None:
        return [{"name": "default", "on": True, "value": backend.default_variant_value(cfg)}]
    if not isinstance(raw, list):
        return []
    return [
        {"name": str(v.get("name", "")).strip(), "on": bool(v.get("on", True)), "value": str(v.get("value", ""))}
        for v in raw if isinstance(v, dict)
    ]


def make_plan(cfg: AppConfig, backends: Iterable[Backend]) -> Plan:
    backends = list(backends)
    by_id = {b.id: b for b in backends}
    problems: list[str] = []

    run = cfg.pipeline_run if isinstance(cfg.pipeline_run, list) else []
    for backend_id in run:
        if backend_id not in by_id:
            problems.append(f"Unknown backend \"{backend_id}\" in the pipeline settings.")
    selected = [b for b in backends if b.id in run]  # registry order, no duplicates

    taggers = [b for b in selected if b.role == "tagger"]
    if len(taggers) > 1:
        problems.append(f"Only one tagger can run at a time ({', '.join(b.label for b in taggers)}).")
    tagger = Step(taggers[0], taggers[0].id, taggers[0].label) if taggers else None

    captioners: list[Step] = []
    for b in selected:
        if b.role != "captioner":
            continue
        if not b.variant_label:
            captioners.append(Step(b, b.id, b.label, uses_tags=b.variant_uses_tags(cfg, "")))
            continue
        checked = [v for v in variants_for(cfg, b) if v["on"]]
        if not checked:
            problems.append(f"{b.label}: no variant is checked.")
        seen: set[str] = set()
        for v in checked:
            name = v["name"]
            if not _VARIANT_NAME_RE.fullmatch(name):
                problems.append(
                    f"{b.label} variant \"{name}\": names can only use letters, digits, - and _ "
                    f"(up to 40) - they become file names."
                )
                continue
            if name.lower() in seen:  # Windows file names ignore case
                problems.append(f"{b.label}: two variants are named \"{name}\".")
                continue
            seen.add(name.lower())
            captioners.append(Step(
                b, f"{b.id}-{name}", f"{b.label} {name}", variant=v["value"],
                uses_tags=b.variant_uses_tags(cfg, v["value"]),
            ))

    if not tagger and not captioners and not problems:
        problems.append("Nothing is set to run.")
    if not tagger:
        for step in captioners:
            if step.uses_tags:
                problems.append(f"{step.label} needs tags, but no tagger is set to run.")
    for b in selected:
        if problem := b.setup_problem(cfg):
            problems.append(problem)
    if cfg.pipeline_tags_mode not in TAGS_MODES:
        problems.append(f"Unknown tag handling \"{cfg.pipeline_tags_mode}\".")
    if cfg.pipeline_tag_style not in TAG_STYLES:
        problems.append(f"Unknown tag style \"{cfg.pipeline_tag_style}\".")

    return Plan(cfg, tagger, captioners, load_order(selected), problems)


# ------------------------------------------------------------------ start/stop

@dataclass
class Progress:
    message: str
    done: bool = False  # the last one of a start/stop
    ok: bool = True  # only meaningful when done


@dataclass
class PipelineStatus:
    state: str  # "stopped", "starting", "started", "stopping"
    plan: Optional[Plan]  # what's running (also while starting/stopping)
    message: str  # the latest start/stop progress line, then its outcome


_lock = threading.Lock()  # guards the four below; never held while loading
_state = "stopped"
_plan: Optional[Plan] = None
_loaded: list[Backend] = []  # what this pipeline loaded, in load order
_message = ""


def status() -> PipelineStatus:
    with _lock:
        return PipelineStatus(_state, _plan, _message)


def running_plan() -> Optional[Plan]:
    """The plan to caption with - None unless the pipeline is started."""
    with _lock:
        return _plan if _state == "started" else None


def not_ready() -> list[str]:
    """Labels of the started pipeline's backends that aren't usable any
    more (crashed worker, server gone). Empty when stopped."""
    plan = running_plan()
    if plan is None:
        return []
    return [b.label for b in plan.backends if not b.is_ready(plan.cfg)]


def _set(state: str, plan: Optional[Plan], message: str) -> None:
    global _state, _plan, _message
    with _lock:
        _state, _plan, _message = state, plan, message


def _note(message: str) -> Progress:
    """A progress line, also kept as status().message - so a UI polling
    status() (another browser tab, the 2s timer) shows it live too."""
    global _message
    with _lock:
        _message = message
    return Progress(message)


def _unload_all(plan: Plan) -> Iterator[Progress]:
    while _loaded:
        backend = _loaded.pop()
        yield _note(f"Stopping {backend.label}...")
        try:
            backend.unload(plan.cfg)
        except Exception:  # noqa: BLE001 - keep stopping the others
            log.exception("Unloading %s failed", backend.label)


def start_pipeline(cfg: AppConfig, backends: Iterable[Backend]) -> Iterator[Progress]:
    """A generator of progress messages for a UI to show; the last one has
    done=True. Loads nothing if the plan has problems or the pipeline isn't
    stopped."""
    global _state, _plan, _message
    backends = list(backends)
    plan = make_plan(cfg, backends)
    # Decide under the lock, yield outside it: the caller may well read
    # status() between two progress lines, which takes the same lock.
    refusal: Optional[Progress] = None
    with _lock:
        if _state != "stopped":
            refusal = Progress(f"The pipeline is already {_state}.", done=True, ok=_state == "started")
        elif plan.problems:
            refusal = Progress("Can't start the pipeline: " + " ".join(plan.problems), done=True, ok=False)
            _message = refusal.message
        else:
            _state, _plan, _message = "starting", plan, ""
    if refusal is not None:
        yield refusal
        return
    log.info("Starting pipeline: %s", ", ".join(b.label for b in plan.backends))

    try:
        for backend in plan.backends:
            # Listed before loading, so an abort mid-load still unloads it
            # (unload() of something not loaded is a no-op).
            _loaded.append(backend)
            message = ""
            for step in load_with_coexistence(backend, backends, cfg):
                yield _note(step.message)
                message = step.message
            if not backend.is_ready(cfg):
                log.warning("Pipeline start failed at %s: %s", backend.label, message)
                yield from _unload_all(plan)
                final = f"Couldn't start the pipeline - {message}"
                _set("stopped", None, final)
                yield Progress(final, done=True, ok=False)
                return
    except BaseException:
        # The caller stopped iterating (closed generator) or a load raised -
        # never leave the state stuck at "starting" with models loaded.
        log.warning("Pipeline start aborted - unloading what was loaded")
        for _ in _unload_all(plan):
            pass
        _set("stopped", None, "Pipeline start was aborted.")
        raise

    final = "Pipeline started: " + ", ".join(b.label for b in plan.backends) + "."
    _set("started", plan, final)
    log.info(final)
    yield Progress(final, done=True)


def stop_pipeline() -> Iterator[Progress]:
    """Unloads everything start_pipeline loaded, in reverse order."""
    global _state
    refusal: Optional[Progress] = None
    with _lock:  # decide under the lock, yield outside it - see start_pipeline
        if _state != "started":
            refusal = Progress(
                "The pipeline isn't running." if _state == "stopped" else f"The pipeline is {_state}, wait for it.",
                done=True, ok=_state == "stopped",
            )
        else:
            _state = "stopping"
        plan = _plan
    if refusal is not None:
        yield refusal
        return
    assert plan is not None
    log.info("Stopping pipeline")
    try:
        yield from _unload_all(plan)
    finally:
        # Same reasoning as start: a closed generator still finishes the
        # job instead of leaving "stopping" behind.
        for _ in _unload_all(plan):
            pass
        _set("stopped", None, "Pipeline stopped.")
    yield Progress("Pipeline stopped.", done=True)


# ------------------------------------------------------------------ one image

def tags_to_text(names: list[str]) -> str:
    """Tag piece content: underscored names -> "looking at viewer, solo"."""
    return ", ".join(name.replace("_", " ") for name in names)


def text_to_tags(text: str) -> list[str]:
    """Inverse of tags_to_text, tolerant of hand edits (newlines, extra
    spaces). e621 names never contain spaces, so this is lossless."""
    return ["_".join(t.split()) for t in re.split(r"[,\n]", text) if t.strip()]


@dataclass
class ImageResult:
    pieces: dict[str, str]  # every piece now available: reused + new
    new: list[str] = field(default_factory=list)  # piece ids generated this time
    errors: list[str] = field(default_factory=list)  # one line per failed step
    interrupted: bool = False
    outputs: dict[str, CaptionOutput] = field(default_factory=dict)  # new caption pieces' details

    @property
    def complete(self) -> bool:
        return not self.errors and not self.interrupted


def caption_pieces(
    plan: Plan,
    image: ImageInput,
    existing: Optional[dict[str, str]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> ImageResult:
    """Runs every step whose piece isn't in `existing`. A failed step is
    recorded in errors and the rest still run, except steps that need tags
    when there are none (tagger failed). should_stop is checked between
    steps and passed into caption() (llama stops mid-caption); stopping
    sets interrupted and skips the remaining steps. Never raises
    BackendError."""
    cfg = plan.cfg
    result = ImageResult(pieces=dict(existing or {}))

    def stopping() -> bool:
        if should_stop is not None and should_stop():
            result.interrupted = True
        return result.interrupted

    tags: Optional[list[Tag]] = None
    if (step := plan.tagger) is not None:
        if step.piece in result.pieces:
            tags = step.backend.tags_from_names(cfg, text_to_tags(result.pieces[step.piece]))
        elif not stopping():
            if on_stage:
                on_stage(step.label)
            try:
                tags = step.backend.tag(cfg, image)
                result.pieces[step.piece] = tags_to_text([t.name for t in tags])
                result.new.append(step.piece)
            except BackendError as exc:
                log.warning("%s failed: %s", step.label, exc)
                result.errors.append(f"{step.label}: {exc}")

    for step in plan.captioners:
        if step.piece in result.pieces:
            continue
        if stopping():
            break
        if step.uses_tags and tags is None:
            result.errors.append(f"{step.label}: skipped - it needs tags and the tagger failed.")
            continue
        if on_stage:
            on_stage(step.label)
        try:
            out = step.backend.caption(cfg, image, variant=step.variant, tags=tags, should_stop=should_stop)
        except Interrupted:
            result.interrupted = True
            break
        except BackendError as exc:
            log.warning("%s failed: %s", step.label, exc)
            result.errors.append(f"{step.label}: {exc}")
            continue
        # A cut-off or empty caption is never kept as a piece - the next
        # run would reuse it instead of trying again.
        if out.truncated:
            result.errors.append(f"{step.label}: cut off ({out.note}) - raise the token limit and run again.")
            continue
        if not out.text.strip():
            result.errors.append(f"{step.label}: returned an empty caption.")
            continue
        result.pieces[step.piece] = out.text.strip()
        result.new.append(step.piece)
        result.outputs[step.piece] = out
    return result


# ------------------------------------------------------------------ render

def render(cfg: AppConfig, captions: list[str], tag_names: Optional[list[str]], trigger: str) -> str:
    """The caption file's text: one line per caption (each folded onto one
    line - a trainer reading lines as separate captions must not split
    one), tags per pipeline_tags_mode, the trigger word in front of every
    line unless pipeline_trigger_on is off or the line already starts
    with it. tag_names are underscored, None = no tagger ran."""
    lines = [" ".join(c.split()) for c in captions if c.strip()]

    tag_text = ""
    if tag_names:
        names = tag_names if cfg.pipeline_tag_style == "underscores" else [n.replace("_", " ") for n in tag_names]
        tag_text = cfg.pipeline_tag_separator.join(names)
    if tag_text and cfg.pipeline_tags_mode == "append" and lines:
        lines = [f"{line} {tag_text}" for line in lines]
    elif tag_text and cfg.pipeline_tags_mode != "none":
        lines.append(tag_text)

    trigger = trigger.strip()
    if cfg.pipeline_trigger_on and trigger:
        lines = [
            line if line.lower() == trigger.lower() or line.lower().startswith(f"{trigger.lower()},")
            else f"{trigger}, {line}"
            for line in lines
        ] or [trigger]
    return "\n".join(lines)


def render_pieces(plan: Plan, pieces: dict[str, str]) -> str:
    """render() for a plan's pieces, in the plan's caption order. Pieces
    the plan doesn't have are left out."""
    tag_names = None
    if plan.tagger is not None and plan.tagger.piece in pieces:
        tag_names = text_to_tags(pieces[plan.tagger.piece])
    captions = [pieces[s.piece] for s in plan.captioners if s.piece in pieces]
    return render(plan.cfg, captions, tag_names, pieces.get(TRIGGER_PIECE, ""))


# ------------------------------------------------------------------ files

def caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def piece_path(image_path: Path, piece: str) -> Path:
    return image_path.with_name(f"{image_path.stem}.{piece}.txt")


def issue_path(image_path: Path) -> Path:
    return Path(f"{caption_path(image_path)}{ISSUE_SUFFIX}")


def read_pieces(image_path: Path, piece_ids: Iterable[str]) -> dict[str, str]:
    pieces = {}
    for piece in piece_ids:
        path = piece_path(image_path, piece)
        if path.is_file():
            pieces[piece] = path.read_text(encoding="utf-8").strip()
    return pieces


@dataclass
class FileResult:
    status: str  # "done", "skipped", "failed", "interrupted"
    image: ImageResult
    text: Optional[str] = None  # the rendered caption, when done
    reason: str = ""  # for "skipped"


def process_file(
    plan: Plan,
    image_path: Path,
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> FileResult:
    """One image on disk. Without overwrite:
      - <stem>.txt exists and no piece file at all: skipped - captioned by
        something else (v0.8, another tool, by hand), never overwritten.
      - <stem>.txt exists and every piece the plan needs exists: skipped.
      - otherwise only the missing pieces are generated (e.g. a variant
        added since), then <stem>.txt is rendered fresh.
    overwrite regenerates every piece. trigger_word: None = the config's
    default, "" = none; only written for an image that has no trigger
    file yet (or with overwrite) - an image keeps the trigger word it was
    captioned with.

    A step failing keeps the pieces that worked, writes no <stem>.txt but
    <stem>.txt.issue (the next run only redoes what's missing).
    Interrupted: pieces so far are kept, nothing else written."""
    cfg = plan.cfg
    txt_path = caption_path(image_path)
    existing = {} if overwrite else read_pieces(image_path, plan.piece_ids)
    if not overwrite and txt_path.exists():
        if not existing:
            return FileResult("skipped", ImageResult({}), reason="has a caption made outside the pipeline")
        if all(p in existing for p in plan.piece_ids):
            return FileResult("skipped", ImageResult(existing), reason="already captioned")

    new_trigger = TRIGGER_PIECE not in existing
    if new_trigger:
        existing[TRIGGER_PIECE] = (cfg.trigger_word if trigger_word is None else trigger_word).strip()

    result = caption_pieces(plan, image_path, existing, should_stop, on_stage)
    if new_trigger:
        result.new.insert(0, TRIGGER_PIECE)
    for piece in result.new:
        piece_path(image_path, piece).write_text(result.pieces[piece], encoding="utf-8")

    if result.errors:
        issue_path(image_path).write_text("\n".join(result.errors), encoding="utf-8")
        log.warning("Not writing %s: %s", txt_path.name, "; ".join(result.errors))
        return FileResult("failed", result)
    if result.interrupted:
        return FileResult("interrupted", result)

    text = render_pieces(plan, result.pieces)
    txt_path.write_text(text, encoding="utf-8")
    issue_path(image_path).unlink(missing_ok=True)
    log.info("Captioned %s (%s)", image_path.name, ", ".join(result.new) or "re-rendered")
    return FileResult("done", result, text=text)
