"""Background model-download worker - fetches a curated quant/mmproj GGUF
from its HF URL into webui/models/<folder>/, driven by app.py's Download-
queue section (this module itself stays Gradio-free, like the rest of
core/ - it just does the actual transfer).

Streams to a "<name>.gguf.part" sibling file, renaming to the real name
only once the transfer completes fully - so a scan_all() pass during an
in-flight download never sees (and tries to GGUF-parse) a half-written
file; scan_all() only ever looks at *.gguf. The .part file is removed on
any non-success exit (abort or error) - no resumability is implemented,
so there is no reason to leave partial multi-GB debris lying around; the
next attempt just starts over.

should_abort() is polled once per downloaded chunk, mirroring how
batch.py's run_batch() polls its own should_stop between images - a
best-effort, "notices within one unit of work" cancellation, not an
instant kill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

log = logging.getLogger(__name__)

# Small enough that should_abort() is checked often even on a fast link,
# large enough not to add real per-chunk overhead on a multi-GB transfer.
CHUNK_SIZE = 1024 * 1024

REQUEST_TIMEOUT = 30  # seconds of silence (connect or between chunks), not a total-transfer cap


@dataclass
class DownloadItem:
    url: str
    dest_path: Path
    label: str  # shown in the queue status text, e.g. "gemma-4-12b-it-Q4_K_M"
    size_bytes: int


def download_one(
    item: DownloadItem,
    should_abort: Callable[[], bool],
    on_progress: Optional[Callable[[int], None]] = None,
) -> bool:
    """Streams item.url to item.dest_path. Leaves either a complete file
    at dest_path or nothing at all - never a partial one - so a later
    scan can't mistake an interrupted download for a real, usable file.

    Returns True if the file was actually completed and renamed into
    place, False if should_abort() cut it short - lets the caller (app.py's
    worker) know whether to trigger a Models-tab refresh, without it
    having to duplicate the abort-vs-done distinction made here.

    on_progress, if given, is called after every chunk write with the
    cumulative byte count written so far - the UI side (app.py) turns
    that into a percentage/speed display; this module has no opinion on
    how it's shown, only that it's a running total, not a delta."""
    item.dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = item.dest_path.with_name(item.dest_path.name + ".part")
    log.info("Downloading %s -> %s", item.label, item.dest_path)

    try:
        written = 0
        with requests.get(item.url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if should_abort():
                        log.info("Download aborted: %s", item.label)
                        return False
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
                        if on_progress:
                            on_progress(written)
        part_path.replace(item.dest_path)
        log.info("Download complete: %s", item.dest_path)
        return True
    finally:
        # No-op if the rename above already moved it away (the success
        # path) - only actually deletes anything on an early return
        # (abort) or an exception propagating out of the try block.
        part_path.unlink(missing_ok=True)
