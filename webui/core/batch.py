"""Directory batch-captioning, shared by the GUI batch tab and the CLI."""

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

# (index, total, image_path, status, caption). caption is the generated text
# on status == "ok", otherwise None.
ProgressCallback = Callable[[int, int, Path, str, Optional[str]], None]


@dataclass
class BatchResult:
    processed: int = 0
    skipped: int = 0
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
        if txt_path.exists() and not overwrite:
            result.skipped += 1
            log.info("Skipping (caption exists): %s", image_path)
            if progress_cb:
                progress_cb(i, total, image_path, "skipped", None)
            continue

        try:
            caption = caption_image(image_path, client, cfg, trigger_word=trigger_word)
            txt_path.write_text(caption, encoding="utf-8")
            result.processed += 1
            log.info("Captioned: %s -> %s", image_path, txt_path)
            if progress_cb:
                progress_cb(i, total, image_path, "ok", caption)
        except ClientError as exc:
            result.failed += 1
            result.errors.append((image_path, str(exc)))
            log.error("Failed to caption %s: %s", image_path, exc)
            if progress_cb:
                progress_cb(i, total, image_path, "failed", None)

    log.info(
        "Batch %s: %d captioned, %d skipped, %d failed",
        "aborted" if result.aborted else "complete",
        result.processed, result.skipped, result.failed,
    )
    return result
