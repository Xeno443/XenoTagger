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
from dataclasses import dataclass, field
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


@dataclass
class BatchResult:
    processed: int = 0
    skipped: int = 0
    truncated: int = 0
    failed: int = 0
    aborted: bool = False
    errors: list[tuple[Path, str]] = field(default_factory=list)


def find_images(directory: Path, recursive: bool = False) -> list[Path]:
    pattern_iter = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        p for p in pattern_iter if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_batch(
    directory: Path,
    client: LlamaClient,
    cfg: AppConfig,
    recursive: bool = False,
    overwrite: bool = False,
    trigger_word: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> BatchResult:
    """should_stop, if given, is checked before each image - returning True
    stops the batch after whatever image is currently in flight finishes
    (never mid-request), leaving everything captioned so far in place.
    """
    directory = Path(directory)
    images = find_images(directory, recursive=recursive)
    result = BatchResult()
    total = len(images)
    log.info(
        "Starting batch: %d image(s) in %s (recursive=%s, overwrite=%s)",
        total, directory, recursive, overwrite,
    )

    for i, image_path in enumerate(images, start=1):
        if should_stop and should_stop():
            result.aborted = True
            log.info("Batch stopped by user after %d/%d images", i - 1, total)
            break

        txt_path = image_path.with_suffix(".txt")
        issue_path = Path(f"{txt_path}{ISSUE_SUFFIX}")
        if txt_path.exists() and not overwrite:
            result.skipped += 1
            log.info("Skipping (caption exists): %s", image_path)
            if progress_cb:
                progress_cb(i, total, image_path, "skipped", None, None)
            continue

        try:
            caption, outcome = caption_image(image_path, client, cfg, trigger_word=trigger_word)
            if outcome.truncated:
                result.truncated += 1
                reason = f"truncated: cut off at {outcome.completion_tokens}/{cfg.max_tokens} tokens"
                issue_path.write_text(reason, encoding="utf-8")
                log.warning("Truncated, not writing caption for %s: %s", image_path, reason)
                if progress_cb:
                    progress_cb(i, total, image_path, "truncated", None, outcome.resize_note)
                continue

            issue_path.unlink(missing_ok=True)
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
