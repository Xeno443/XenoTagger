"""Directory batch-captioning: scans a directory for images and captions
whichever ones need it, shared verbatim by the GUI's Batch tab and the
CLI (cli.py imports run_batch() directly - there is no separate CLI
implementation of this loop).

Deliberately as stateless as the rest of this ecosystem's captioning
tools (TagGUI, JoyCaption's batch node, etc. - none of them track batch
run history either, see the ISSUE_SUFFIX note below and this project's
own design discussion for why that was a conscious choice, not an
oversight): the only "state" that matters is derived fresh from the
directory's contents on every run. A "<image>.txt" sidecar existing next
to an image IS the record of "this one's done" - skip it unless
overwrite is set; there's no separate journal/database that could drift
out of sync with reality if images get added, removed, or a caption file
gets hand-edited between runs.

should_stop, if given, is polled between images (never mid-request - a
graceful "finish the current one, then stop" signal) - see app.py's
Operation-tracking section for what actually drives it from the UI side;
this module has no opinion on where the flag comes from, only that it's
a zero-argument callable returning a bool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from .captioner import caption_image
from .client import ClientError, LlamaClient
from .config import AppConfig

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# (index, total, image_path, status, caption, resize_note). status is one
# of "ok", "skipped", "truncated", "failed". caption is the generated text
# on status == "ok" only, otherwise None. resize_note is set whenever the
# image was actually resized, regardless of status (resizing happens
# before generation, so it's known even if generation then truncated).
ProgressCallback = Callable[[int, int, Path, str, Optional[str], Optional[str]], None]

# A truncated or failed image gets this file instead of a .txt caption -
# so a LoRA trainer (which only looks for "<image>.txt") never sees it,
# and the next batch run picks the image right back up automatically
# (no .txt means it isn't "already captioned"). Content is a one-line
# human-readable reason, meant for a future review UI to surface.
ISSUE_SUFFIX = ".issue"
# Chained onto txt_path exactly like ISSUE_SUFFIX above (so
# "<image>.txt" + NLP_SUFFIX = "<image>.txt.nlp", never "<image>.nlp") -
# matches ISSUE_SUFFIX's own convention, which was already this shape.
# See core.captioner.caption_image/run_batch below for what each actually
# holds (VLM-only caption / Hydra's own tag text).
NLP_SUFFIX = ".nlp"
TAGS_SUFFIX = ".tags"


@dataclass
class BatchResult:
    processed: int = 0
    skipped: int = 0
    truncated: int = 0
    failed: int = 0
    aborted: bool = False
    errors: list[tuple[Path, str]] = field(default_factory=list)


def find_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


@dataclass
class ReviewItem:
    path: Path
    status: str  # see review_item_status() below for exactly what this can be


def review_item_status(image_path: Path) -> str:
    """Classifies a single image by which of .txt/.txt.nlp/.txt.tags/
    .txt.issue exist next to it - shared by scan_review_status() below and
    app.py's _review_maybe_save (so the table's status stays consistent
    with whatever a save just did, not just re-derived from a fresh
    directory scan):

    - "Captioned" - .txt exists, no .txt.nlp/.txt.tags sidecars (Hydra was
      off, or this predates the sidecar feature).
    - "Captioned (nlp)" / "Captioned (tags)" / "Captioned (nlp tags)" -
      .txt exists alongside whichever of the two sidecars are also there
      (see core.batch.run_batch's own sidecar-writing logic).
    - "Sidecar only (nlp)" / "Sidecar only (tags)" / "Sidecar only (nlp tags)"
      - one or both of .txt.nlp/.txt.tags exist but .txt doesn't - an
      inconsistent state (shouldn't normally happen since all three are
      written/kept in sync together, but worth surfacing plainly, with
      which sidecar(s) are the orphans, rather than silently falling back
      to "New" if it ever does, e.g. from a hand-deleted .txt).
    - "Error" - no .txt, no sidecars, but a .txt.issue file exists (a
      batch run failed/truncated on this image).
    - "New" - none of the above exist yet.
    """
    txt_path = image_path.with_suffix(".txt")
    issue_path = Path(f"{txt_path}{ISSUE_SUFFIX}")
    sidecars = [
        name for name, suffix in (("nlp", NLP_SUFFIX), ("tags", TAGS_SUFFIX))
        if Path(f"{txt_path}{suffix}").exists()
    ]

    if txt_path.exists():
        return f"Captioned ({' '.join(sidecars)})" if sidecars else "Captioned"
    if sidecars:
        return f"Sidecar only ({' '.join(sidecars)})"
    if issue_path.exists():
        return "Error"
    return "New"


def scan_review_status(directory: Path) -> list[ReviewItem]:
    """For the Review tab: classifies every image in `directory` by its
    current on-disk state (see review_item_status() above) - a pure
    snapshot, same statelessness as the rest of this module - there's
    nothing to go stale, just call again after edits or another batch run
    to get a fresh read."""
    directory = Path(directory)
    return [ReviewItem(path=p, status=review_item_status(p)) for p in find_images(directory)]


def run_batch(
    directory: Path,
    client: LlamaClient,
    cfg: AppConfig,
    overwrite: bool = False,
    trigger_word: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stage: Optional[Callable[[Path, str], None]] = None,
) -> BatchResult:
    """should_stop, if given, is checked before each image - returning True
    stops the batch after whatever image is currently in flight finishes
    (never mid-request), leaving everything captioned so far in place.

    on_stage, if given, is called with (image_path, stage) - the path is
    added here rather than coming from caption_image() itself (whose own
    on_stage is just (stage) - see its docstring), since only run_batch()
    knows which image is currently in flight; caption_image() processes
    one image at a time and has no reason to know its own path. Called
    from whatever thread run_batch() itself runs on (the GUI runs it in a
    background worker thread, see app.py's run_batch_ui), so it must be a
    plain, Gradio-free callback, never something that touches gr.*
    directly.

    When cfg.hydra_enabled and Hydra actually produced tags for an image,
    two extra sidecars are written alongside the usual .txt:
    "<image>.txt.nlp" (the VLM-only caption, trigger word applied, no
    Hydra tags) and "<image>.txt.tags" (just Hydra's tag_text) - for
    consumers that want the two pieces separately instead of re-splitting
    the combined .txt. Batch-only for now, not written by Single-image or
    Review recaption.

    Sidecar adoption: an image with no .txt yet but an existing .txt.nlp
    and/or .txt.tags (e.g. hand-renamed in from another tool, or left over
    from an interrupted run) has that content reused rather than
    regenerated - only whichever half is actually missing gets a fresh
    model call, and an adopted .txt.tags is never touched by Hydra even if
    cfg.hydra_enabled, so it can't be silently overwritten by a second tag
    source. overwrite=True bypasses all of this and treats every image as
    fresh, exactly like it already does for a pre-existing .txt. See
    readme/faq-caption.md for the full case table.
    """
    directory = Path(directory)
    images = find_images(directory)
    result = BatchResult()
    total = len(images)
    log.info(
        "Starting batch: %d image(s) in %s (overwrite=%s)",
        total, directory, overwrite,
    )

    for i, image_path in enumerate(images, start=1):
        if should_stop and should_stop():
            result.aborted = True
            log.info("Batch stopped by user after %d/%d images", i - 1, total)
            break

        txt_path = image_path.with_suffix(".txt")
        issue_path = Path(f"{txt_path}{ISSUE_SUFFIX}")
        nlp_path = Path(f"{txt_path}{NLP_SUFFIX}")
        tags_path = Path(f"{txt_path}{TAGS_SUFFIX}")
        if txt_path.exists() and not overwrite:
            result.skipped += 1
            log.info("Skipping (caption exists): %s", image_path)
            if progress_cb:
                progress_cb(i, total, image_path, "skipped", None, None)
            continue

        # Adopt whatever sidecar(s) already exist unless overwrite says to
        # ignore everything on disk and start fresh (see run_batch's own
        # docstring above and readme/faq-caption.md).
        adopted_nlp = nlp_path.read_text(encoding="utf-8") if not overwrite and nlp_path.exists() else None
        adopted_tags = tags_path.read_text(encoding="utf-8") if not overwrite and tags_path.exists() else None

        stage_cb = (lambda s, _path=image_path: on_stage(_path, s)) if on_stage else None
        try:
            # An adopted .txt.tags is trusted as-is - Hydra never runs for
            # this image, so it can't clobber or duplicate it.
            call_cfg = replace(cfg, hydra_enabled=False) if adopted_tags is not None else cfg
            caption, outcome, vlm_caption, hydra_tags = caption_image(
                image_path, client, call_cfg, trigger_word=trigger_word, on_stage=stage_cb,
                existing_caption=adopted_nlp,
            )
            if outcome.truncated:
                result.truncated += 1
                reason = f"truncated: cut off at {outcome.completion_tokens}/{cfg.max_tokens} tokens"
                issue_path.write_text(reason, encoding="utf-8")
                log.warning("Truncated, not writing caption for %s: %s", image_path, reason)
                if progress_cb:
                    progress_cb(i, total, image_path, "truncated", None, outcome.resize_note)
                continue

            issue_path.unlink(missing_ok=True)
            if adopted_tags is not None:
                nlp_path.write_text(vlm_caption, encoding="utf-8")
                caption = f"{vlm_caption}\n{adopted_tags}" if adopted_tags.strip() else vlm_caption
            elif cfg.hydra_enabled and hydra_tags is not None:
                nlp_path.write_text(vlm_caption, encoding="utf-8")
                tags_path.write_text(hydra_tags, encoding="utf-8")
            txt_path.write_text(caption, encoding="utf-8")
            result.processed += 1
            if outcome.resize_note:
                log.info("Captioned: %s -> %s (resized %s)", image_path, txt_path, outcome.resize_note)
            else:
                log.info("Captioned: %s -> %s", image_path, txt_path)
            if progress_cb:
                progress_cb(i, total, image_path, "ok", caption, outcome.resize_note)
        except ClientError as exc:
            result.failed += 1
            result.errors.append((image_path, str(exc)))
            log.error("Failed to caption %s: %s", image_path, exc)
            issue_path.write_text(f"failed: {exc}", encoding="utf-8")
            if progress_cb:
                progress_cb(i, total, image_path, "failed", None, None)

    log.info(
        "Batch %s: %d captioned, %d truncated, %d skipped, %d failed",
        "aborted" if result.aborted else "complete",
        result.processed, result.truncated, result.skipped, result.failed,
    )
    return result
