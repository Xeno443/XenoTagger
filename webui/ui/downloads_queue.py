"""Background download queue shared by every section of the Models tab
(one queue, one worker thread, one progress row)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import gradio as gr

from core.downloads import DownloadItem, download_one, format_size
from ui.notify import notify

log = logging.getLogger("app")  # fixed name, see app.py

# Separate from ui.operation's _active_operation/operation_lock on purpose -
# downloading a curated model shouldn't block captioning with whatever's
# already loaded, or vice versa, so this gets its own lock and its own
# single background worker thread rather than reusing that mutex. Not
# persisted across a restart (in-memory only, by design - nothing here
# needs to survive the app process ending); restart_app_ui() (ui.tab_settings) calls
# download_abort_all() before its os.execv for the same reason
# operation_force_abort() gets called there - don't leave a background
# thread/state hanging into (or racing) the fresh process.
#
# The actual transfer (core/downloads.py's download_one) is a plain
# streaming HTTP GET with no resumability - "Abort all" (or an app
# restart) simply discards whatever was in flight, and a later re-queue
# starts over from byte zero.

_download_lock = threading.RLock()
_download_queue: list[DownloadItem] = []
_download_current: Optional[DownloadItem] = None
_download_current_bytes = 0  # bytes written so far for _download_current - see download_one's on_progress
_download_started_at: Optional[float] = None  # time.monotonic() when _download_current began, for average speed
_download_abort_requested = False
_download_worker_thread: Optional[threading.Thread] = None
_download_needs_refresh = False  # set True whenever a download actually completes - see _download_triggered_refresh_ui
# Completion/failure (message, level) pairs popped by the next
# download_status_ui tick - a list, not a single slot like
# _install_announce/_hydra_install_announce, since several queued
# downloads can each finish inside one 2s tick window. A plain abort
# (download_one returning False without raising only ever means
# should_abort() fired) is user-initiated and doesn't get announced.
_download_announces: list[tuple[str, str]] = []


def _download_worker() -> None:
    global _download_current, _download_current_bytes, _download_started_at, _download_needs_refresh
    while True:
        with _download_lock:
            if _download_abort_requested or not _download_queue:
                _download_current = None
                return
            _download_current = _download_queue.pop(0)
            _download_current_bytes = 0
            _download_started_at = time.monotonic()

        def on_progress(written: int) -> None:
            global _download_current_bytes
            with _download_lock:
                _download_current_bytes = written

        completed = False
        announce: Optional[tuple[str, str]] = None
        try:
            completed = download_one(_download_current, lambda: _download_abort_requested, on_progress)
            if completed:
                announce = (f"Downloaded {_download_current.label}.", "info")
        except Exception as exc:
            log.exception("Download failed: %s", _download_current.label)
            announce = (f"Download failed: {_download_current.label}: {exc}", "error")
        with _download_lock:
            if announce is not None:
                _download_announces.append(announce)
            _download_current = None
            if completed:
                # Picked up by _download_triggered_refresh_ui on the next
                # status_timer tick (within ~2s) even if nobody's actively
                # watching the Models tab or clicking anything right now -
                # a completed download shouldn't need a manual Refresh
                # click to actually show up as usable.
                _download_needs_refresh = True
            if _download_abort_requested:
                return


def download_enqueue(items: list[DownloadItem]) -> list[DownloadItem]:
    """Adds `items` to the queue, skipping any whose dest_path is already
    queued or currently downloading (e.g. a stray double-click), and
    (re)starts the worker thread if it isn't already running. Returns
    only the items actually added, so the caller can report exactly what
    changed."""
    global _download_worker_thread, _download_abort_requested
    added: list[DownloadItem] = []
    with _download_lock:
        existing_paths = {i.dest_path for i in _download_queue}
        if _download_current is not None:
            existing_paths.add(_download_current.dest_path)
        for item in items:
            if item.dest_path not in existing_paths:
                _download_queue.append(item)
                existing_paths.add(item.dest_path)
                added.append(item)
        if added:
            _download_abort_requested = False
            if _download_worker_thread is None or not _download_worker_thread.is_alive():
                _download_worker_thread = threading.Thread(target=_download_worker, daemon=True)
                _download_worker_thread.start()
    return added


def download_abort_all() -> None:
    global _download_abort_requested
    with _download_lock:
        _download_abort_requested = True
        _download_queue.clear()


def _download_status_html() -> Optional[str]:
    """None means idle - nothing downloading, nothing queued - which the
    UI uses to hide the download-status row entirely rather than show an
    empty one. A plain HTML5 <progress> element rather than any hand-
    rolled div/CSS bar - it needs zero styling of our own to look right
    and already follows the browser's own light/dark handling, which
    matters given this project's habit of avoiding custom CSS without a
    real reason (and the Interrupt-button saga's lesson that CSS here
    can't actually be visually verified before the user tries it)."""
    with _download_lock:
        if _download_current is None and not _download_queue:
            return None

        html = ""
        if _download_current is not None:
            item = _download_current
            done = _download_current_bytes
            total = max(item.size_bytes, 1)  # avoid a div-by-zero for a malformed curated size_bytes
            pct = min(100, int(done * 100 / total))
            elapsed = max(time.monotonic() - (_download_started_at or time.monotonic()), 0.001)
            speed = done / elapsed
            html += (
                f"<progress value=\"{done}\" max=\"{total}\" style=\"width:100%;\"></progress>"
                f"<div>{item.label}: {pct}% "
                f"({format_size(done)} / {format_size(item.size_bytes)}) - {format_size(speed)}/s</div>"
            )
        more = len(_download_queue)
        if more:
            html += f"<div>{more} more queued</div>"
        return html


def download_status_ui():
    """(row_visibility_update, html_update) - wired to the same 2s
    status_timer as everything else, plus called directly after any
    action that changes the queue (from either the VLM or Hydra section -
    both feed the one shared _download_queue) so the row responds
    immediately instead of waiting for the next tick. Also fires a
    one-shot notify() popup per completed/failed download popped from
    _download_announces."""
    with _download_lock:
        pending = list(_download_announces)
        _download_announces.clear()
    for message, level in pending:
        notify(message, level=level)

    html = _download_status_html()
    if html is None:
        return gr.update(visible=False), gr.update(value="")
    return gr.update(visible=True), gr.update(value=html)


def download_consume_needs_refresh() -> bool:
    """True once per completed download (then resets) - read by the Models
    tab's _download_triggered_refresh_ui to decide whether a rescan is
    worth doing on this tick."""
    global _download_needs_refresh
    with _download_lock:
        needs_refresh = _download_needs_refresh
        _download_needs_refresh = False
    return needs_refresh
