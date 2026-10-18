"""Directory batch-captioning on the caption pipeline (core.pipeline),
shared verbatim by the GUI's Batch tab and the CLI - there is no separate
CLI implementation of this loop.

Stateless like the rest of the pipeline: the only "state" is what's on
disk next to each image (its piece files, <stem>.txt, <stem>.txt.issue),
derived fresh on every run - see core.pipeline.process_file for which
images get skipped, filled in or captioned.

Also still holds the Review tab's scan helpers (review_item_status,
scan_review_status), which read the v0.8 sidecar names until Review
moves onto piece files (v0.9 step 3, part 6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .pipeline import Plan, process_file

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# (index, total, image_path, status, caption_text, detail). status is
# core.pipeline.FileResult's ("done", "skipped", "failed", "interrupted")
# or "conflict" (shares its caption files with another image, see
# run_batch). caption_text is the rendered caption on "done", else None.
# detail is a one-line reason/summary (errors, skip reason, piece notes).
ProgressCallback = Callable[[int, int, Path, str, Optional[str], str], None]

# Review tab only (v0.8 sidecars) - see the module docstring.
ISSUE_SUFFIX = ".issue"
NLP_SUFFIX = ".nlp"
TAGS_SUFFIX = ".tags"


@dataclass
class BatchResult:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    aborted: bool = False
    # Images skipped because another image has the same name with a
    # different extension (wolf.png + wolf.jpg would share wolf.txt and
    # every piece file).
    conflicts: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)


def find_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _name_conflicts(images: list[Path]) -> set[Path]:
    """Images whose stem (ignoring case - Windows file names do) isn't
    unique in the folder."""
    by_stem: dict[str, list[Path]] = {}
    for path in images:
        by_stem.setdefault(path.stem.lower(), []).append(path)
    return {p for group in by_stem.values() if len(group) > 1 for p in group}


def run_batch(
    plan: Plan,
    directory: Path,
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stage: Optional[Callable[[Path, str], None]] = None,
) -> BatchResult:
    """Runs process_file over every image in `directory`. should_stop is
    checked before each image and handed down (llama stops mid-caption);
    a stop ends the batch, keeping everything finished so far. on_stage
    gets (image_path, step label) before each model call."""
    images = find_images(Path(directory))
    conflicts = _name_conflicts(images)
    result = BatchResult(conflicts=sorted(conflicts))
    total = len(images)
    log.info("Starting batch: %d image(s) in %s (overwrite=%s)", total, directory, overwrite)
    if conflicts:
        log.warning("Skipping %d image(s) that share a name with another: %s",
                    len(conflicts), ", ".join(p.name for p in sorted(conflicts)))

    for i, image_path in enumerate(images, start=1):
        if should_stop is not None and should_stop():
            result.aborted = True
            log.info("Batch stopped after %d/%d images", i - 1, total)
            break
        if image_path in conflicts:
            result.skipped += 1
            if progress_cb:
                progress_cb(i, total, image_path, "conflict", None,
                            "another image has the same name - rename one of them")
            continue

        stage_cb = (lambda s, _path=image_path: on_stage(_path, s)) if on_stage else None
        r = process_file(plan, image_path, trigger_word, overwrite, should_stop, stage_cb)
        if r.status == "done":
            result.done += 1
            detail = "; ".join(f"{piece}: {out.note}" for piece, out in r.image.outputs.items() if out.note)
        elif r.status == "skipped":
            result.skipped += 1
            detail = r.reason
        elif r.status == "failed":
            result.failed += 1
            detail = "; ".join(r.image.errors)
            result.errors.append((image_path, detail))
        else:  # interrupted
            result.aborted = True
            detail = "interrupted"
        if progress_cb:
            progress_cb(i, total, image_path, r.status, r.text, detail)
        if r.status == "interrupted":
            log.info("Batch stopped during %s", image_path.name)
            break

    log.info(
        "Batch %s: %d captioned, %d skipped, %d failed",
        "stopped" if result.aborted else "complete", result.done, result.skipped, result.failed,
    )
    return result


# ------------------------------------------------------------------ Review (v0.8 sidecars)

@dataclass
class ReviewItem:
    path: Path
    status: str  # see review_item_status() below for exactly what this can be


def review_item_status(image_path: Path) -> str:
    """Classifies a single image by which of .txt/.txt.nlp/.txt.tags/
    .txt.issue exist next to it - shared by scan_review_status() below and
    ui/tab_review.py's _review_maybe_save:

    - "Captioned" - .txt exists, no .txt.nlp/.txt.tags sidecars.
    - "Captioned (nlp)" / "Captioned (tags)" / "Captioned (nlp tags)" -
      .txt exists alongside whichever of the two sidecars are also there.
    - "Sidecar only (...)" - one or both sidecars exist but .txt doesn't.
    - "Error" - no .txt, no sidecars, but a .txt.issue file exists.
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
    """For the Review tab: every image in `directory` with its current
    on-disk state (see review_item_status() above)."""
    directory = Path(directory)
    return [ReviewItem(path=p, status=review_item_status(p)) for p in find_images(directory)]
