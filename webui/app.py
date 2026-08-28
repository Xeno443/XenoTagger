"""Gradio UI: Single-image captioning, Batch processing, Models, Settings,
and an opt-in Debug tab. The CLI (cli.py) is a separate, independent
entry point that shares core/ with this file but not this file itself -
nothing UI-specific belongs in core/.

Two coordination mechanisms guard shared, process-wide state that both
tabs (and Settings' restart actions) can touch concurrently - both are
threading.RLock, not a plain Lock, because each has a function that calls
another lock-holding function of its own while already holding the lock
(get_client() calls _stop_managed(); a plain Lock would deadlock there):

  - `_session`/`_session_lock`: which llama-server (if any) we're
    currently managing. Guards against two requests racing in before any
    server is up yet (e.g. a batch run and a single-image caption both
    arriving at once) each deciding independently that no server exists
    and each starting their own.

  - `_active_operation`/`_operation_lock` ("Operation tracking" section):
    is a long-running job (single-image or batch) currently active, and
    what stage of being interrupted is it in. This is also the mutual-
    exclusion gate that stops single-image and batch from running at
    once, since both ultimately share the one llama-server connection.

The Run/Interrupt buttons are two SEPARATE gr.Button components per tab,
not one button whose label changes - this was tried first and doesn't
work: Gradio disables the triggering component of a still-pending event
client-side, and defaults every event listener's server-side
concurrency_limit to 1, so a second click on the button that's still
running the long captioning/batch call either never reaches the server
or queues silently behind it - never actually interrupts anything. The
fix (and the reason it's structured as two buttons, mirroring how Forge/
A1111 do this in modules/ui_toprow.py: Generate, Interrupt, and Skip are
independent gr.Button()s there too) is that the Interrupt button was
never the one that started the long call, so it's never "busy" and is
always immediately clickable - its own click handler is fast and
non-blocking (flip a flag; on the second click, kill the managed
llama-server process - see core/server.py's ManagedServer.stop for why a
hard kill is the only reliable way to actually interrupt an in-flight
request). First click requests a graceful stop (batch: finish the
current image, then stop before the next one; single-image has no next
item to gracefully stop before, so this click is intentionally a no-op
buffer against a stray double-click); second click force-aborts.

On the Single-image tab specifically, Caption and Interrupt each get
their OWN gr.Column (single_run_col/single_interrupt_col) rather than
sharing one - swapping which Column is visible, not the buttons inside
them. This is a layout fix, not just the concurrency one above: Gradio
gives every gr.Row() its own independent flex computation with no shared
alignment across separate Rows (it's not a table), so the row above
(image | caption box) and this button row only ever lined up because both
happened to be simple two-Column shapes computing the same coincidental
50/50 split. A CSS-based single-Column stack (Interrupt absolutely/grid
positioned over Caption) was tried first and worked functionally, but the
Column's own default padding/gap - which a bare Button in the original
layout never had - broke that coincidental alignment, through several
rounds of trying to CSS-patch it back. Two separate Columns, each either
visible or not, sidesteps the whole problem: at any moment exactly two of
the three Columns (Caption, Interrupt, Save caption) are visible, always
that same plain "Row of two Columns" shape confirmed to align correctly -
whichever two they happen to be. See ui_css.py's own note on the
abandoned CSS approach for the full story.

Because button state is normally only pushed by whichever specific
browser connection's generator is actively running (run_single_ui /
run_batch_ui's own yields), a reloaded page or a second browser tab would
otherwise show stale "idle" buttons even while something is genuinely
running elsewhere. _operation_button_states() closes that gap: it
recomputes both tabs' Run/Interrupt appearance fresh from
_active_operation on every call, and is wired into the same 2-second
status_timer that already refreshes the status bar, plus demo.load - so
any view self-corrects within one tick instead of staying wrong
indefinitely.

Restart (both restart_server_ui and restart_app_ui) doesn't block just
because something's running, nor does it silently kill it out from under
whatever's in flight - it force-aborts (skipping the two-click grace
period; see _operation_force_abort) and actually waits for that
operation to finish noticing and clean up (_wait_for_operation_to_end)
before touching the server. This matters most for restart_app_ui, whose
os.execv replaces the entire process with zero chance for any cleanup to
run afterward - waiting first means an interrupted operation gets an
observable, handled failure instead of just vanishing without a trace.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# Must be set before `import gradio` - some of its telemetry checks read
# this at import time, not just when building a Blocks instance.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from core import config as config_mod
from core.batch import ISSUE_SUFFIX, ReviewItem, find_images, run_batch, scan_review_status
from core.captioner import caption_image
from core.client import ClientError, LlamaClient
from core.config import AppConfig
from core.models import (
    IGNORED_SUBSTRINGS, MODELS_DIR, ModelGroup, format_size, group_models,
    load_curated_models, merge_curated, resolve_selection, scan_all,
)
from core.server import LOG_PATH, ManagedServer, ServerError, get_loaded_model_name, is_healthy, resolve_server
from ui_css import ALL_CSS

log = logging.getLogger("app")  # fixed name, not __name__ - which is "__main__"
                                  # when run directly (the real usage) but
                                  # "app" when imported (e.g. in tests) -
                                  # this way logging.getLogger("app") always
                                  # refers to the right logger either way.

current_cfg: AppConfig = config_mod.load()

_session = {"managed": None, "client": None, "base_url": None}
# Guards every read/decide/write of _session - without it, two requests
# racing in at once (e.g. a batch run and a single-image caption, both
# arriving before any server is up yet) could both decide no server exists
# and each start their own, stomping on each other's _session writes and
# possibly killing one process mid-launch via the other's _stop_managed().
# Reentrant (not a plain Lock) because get_client() calls _stop_managed()
# internally while already holding it - a plain Lock would deadlock there.
_session_lock = threading.RLock()

# Debug tab: captures log output from our own Python code (core.* modules
# all log via the standard `logging` module, but nothing was ever attached
# to see it before this) into a bounded in-memory buffer.
_PY_LOG_BUFFER: deque[str] = deque(maxlen=2000)


class _BufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _PY_LOG_BUFFER.append(self.format(record))


if current_cfg.debug_tab_enabled:
    _log_handler = _BufferLogHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_log_handler)
    # Deliberately NOT touching the root logger's level. Elevating only our
    # own namespaces to DEBUG (rather than the root, which every library's
    # logger inherits from when it has no level of its own) means Gradio's
    # httpx/httpcore/asyncio internals - and anything else - stay at their
    # own quiet default instead of flooding this with unrelated chatter.
    logging.getLogger("core").setLevel(logging.DEBUG)
    logging.getLogger("app").setLevel(logging.DEBUG)


def _stop_managed() -> None:
    with _session_lock:
        if _session["managed"] is not None:
            _session["managed"].stop()
        _session["managed"] = None
        _session["client"] = None
        _session["base_url"] = None


atexit.register(_stop_managed)


# ------------------------------------------------------ Operation tracking
#
# Single source of truth for "is something long-running active, and what".
# The Single-image and Batch tabs' Run buttons double as Interrupt buttons
# once running (same Generate/Interrupt pattern most Stable Diffusion
# WebUIs use): first click while running asks for a graceful stop (batch:
# finish the current image, then stop before the next one; single-image
# has no queue to gracefully stop before, so this click is just a
# deliberate one-click buffer against an accidental double-click), second
# click hard-aborts by killing the managed llama-server outright - the
# only actually reliable way to stop an in-flight request. (llama-server
# itself has open upstream bugs where it doesn't reliably notice a
# disconnected client and keeps generating regardless - closing our end of
# the connection alone can't be trusted to stop it; see ManagedServer.stop
# in core/server.py for the real kill.)
#
# This also acts as the mutual-exclusion gate for the race discussed
# above _session_lock: only one kind of operation ("single" or "batch")
# may be active at a time, since both ultimately share the one
# llama-server connection.

@dataclass
class _Operation:
    kind: str  # "single" or "batch"
    label: str  # human-readable, shown in the status bar
    stop_requested: bool = False
    abort_requested: bool = False


_active_operation: Optional[_Operation] = None
_operation_lock = threading.RLock()


def _operation_blocked_by() -> Optional[str]:
    """None if nothing is currently running. Otherwise a message naming
    what's active - covers both a different kind of operation and an
    accidental duplicate start of the same kind (the Run button disables
    itself while busy, but this is a second line of defense, e.g. against
    a stray click landing just before that update reaches the browser)."""
    with _operation_lock:
        op = _active_operation
        if op is not None:
            return f"{op.label} is currently running - interrupt it from its own tab first."
        return None


def _operation_start(kind: str, label: str) -> None:
    global _active_operation
    with _operation_lock:
        _active_operation = _Operation(kind=kind, label=label)


def _operation_end() -> None:
    global _active_operation
    with _operation_lock:
        _active_operation = None


def _operation_interrupt_click(kind: str) -> str:
    """Call when the Run/Interrupt button is clicked while `kind` is
    already the active operation. Returns "stopping" on the first click
    (graceful - just sets a flag the running loop checks), "aborting" on
    the second (hard - kills the server, which is done outside the lock
    since _stop_managed() acquires its own _session_lock)."""
    with _operation_lock:
        op = _active_operation
        if op is None or op.kind != kind:
            return "idle"
        if not op.stop_requested:
            op.stop_requested = True
            log.info("%s: stop requested (graceful)", op.label)
            return "stopping"
        op.abort_requested = True
        log.info("%s: abort requested (killing server)", op.label)
    _stop_managed()
    return "aborting"


def _operation_should_stop(kind: str) -> bool:
    with _operation_lock:
        op = _active_operation
        return op is not None and op.kind == kind and op.stop_requested


def _operation_status_text() -> str:
    with _operation_lock:
        op = _active_operation
        if op is None:
            return "idle"
        if op.abort_requested:
            return f"{op.label} (aborting)"
        if op.stop_requested:
            return f"{op.label} (stopping)"
        return op.label


def _interrupt_label_state(op: Optional["_Operation"], kind: str) -> tuple[str, bool]:
    """(label, interactive) for `kind`'s Interrupt button, given the
    active operation (or None). Used by _operation_button_states, which
    has to derive this fresh from scratch since it isn't the one that
    just clicked Interrupt - interrupt_single_ui/interrupt_batch_ui don't
    need this themselves, they already know which of the two states just
    happened from _operation_interrupt_click's own return value."""
    if op is None or op.kind != kind:
        return "Interrupt", True
    if op.abort_requested:
        return "Interrupting...", False
    if op.stop_requested:
        return "Interrupt (click again to abort now)", True
    return "Interrupt", True


def _operation_button_states():
    """Ground-truth Run/Interrupt appearance for all three tabs, recomputed
    fresh from _active_operation every call - not just whatever a single
    running generator last pushed. Wired into the periodic status timer
    and page load below, so a reloaded page, a second browser tab, or a
    missed/delayed update all self-correct within one tick instead of
    showing stale state indefinitely.

    All three use column-visibility swapping (single_run_col <->
    single_interrupt_col, etc. - see the Single-image tab and app.py's own
    module docstring for why: confirmed live that a Row of two plain
    gr.Column()s aligns correctly with a neighboring row with zero CSS, so
    toggling which two of N Columns are visible reuses that proven shape
    instead of hand-tuned CSS. Batch/Review's own rows have no neighboring
    row to misalign against, but use the same pattern anyway for
    consistency).

    Review recaptioning is its own "review" operation kind, distinct from
    "single" even though it calls the exact same caption_image() - if it
    shared the "single" kind, the Single-image tab's own Interrupt would
    light up whenever Review (not it) was the one running, which would be
    backwards. A third kind gives it its own correct run/interrupt state
    while still mutually excluding against the other two for free (the
    existing "blocked if op.kind != kind" check needs no changes to cover
    a third kind). Review's other nav (prev/next/table/dir/browse/scan)
    goes a step further than a plain disabled Run button, though: it's
    disabled whenever ANYTHING is active anywhere, not just when Review
    itself is - simplest consistent rule, matching what the user asked
    for (nothing on the tab should be interactable mid-recaption) without
    special-casing "safe" vs "unsafe" interactions to allow selectively.

    Returns (single_run_btn, single_run_col, single_interrupt_col,
    single_interrupt_btn, batch_run_btn, batch_run_col,
    batch_interrupt_col, batch_interrupt_btn, review_recaption_col,
    review_interrupt_col, review_interrupt_btn, review_prev_btn,
    review_next_btn, review_table, review_dir, review_browse_btn,
    review_scan_btn).
    """
    with _operation_lock:
        op = _active_operation
        single_blocked = op is not None and op.kind != "single"
        single_running = op is not None and op.kind == "single"
        single_label, single_ok = _interrupt_label_state(op, "single")
        batch_blocked = op is not None and op.kind != "batch"
        batch_running = op is not None and op.kind == "batch"
        batch_label, batch_ok = _interrupt_label_state(op, "batch")
        review_running = op is not None and op.kind == "review"
        review_label, review_ok = _interrupt_label_state(op, "review")
        review_nav = _REVIEW_NAV_IDLE if op is None else _REVIEW_NAV_BUSY

    return (
        gr.update(interactive=not single_blocked),
        gr.update(visible=not single_running),
        gr.update(visible=single_running),
        gr.update(value=single_label, variant="stop", interactive=single_ok),
        gr.update(interactive=not batch_blocked),
        gr.update(visible=not batch_running),
        gr.update(visible=batch_running),
        gr.update(value=batch_label, variant="stop", interactive=batch_ok),
        gr.update(visible=not review_running),
        gr.update(visible=review_running),
        gr.update(value=review_label, variant="stop", interactive=review_ok),
        *review_nav,
    )


def _operation_force_abort() -> None:
    """Immediately hard-aborts whatever's running, if anything - skips the
    two-click grace period (for Restart, which needs the operation gone
    now, not eventually). No-op if nothing is running."""
    with _operation_lock:
        op = _active_operation
        if op is None:
            return
        op.stop_requested = True
        op.abort_requested = True
        log.info("%s: force-aborted (restart requested)", op.label)
    _stop_managed()


def _wait_for_operation_to_end(timeout: float = 30.0) -> None:
    """Blocks until the active operation's own thread notices the abort
    and calls _operation_end() - killing the server (above) should make
    its in-flight request fail almost immediately, so this is normally
    quick; the timeout is just a safety net against hanging forever."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _operation_lock:
            if _active_operation is None:
                return
        time.sleep(0.2)
    log.warning("Timed out waiting for the running operation to stop before restart")


def get_client(cfg: AppConfig) -> LlamaClient:
    with _session_lock:
        if cfg.server_mode == "external":
            desired_base = cfg.external_url.rstrip("/")
        else:
            desired_base = f"http://{cfg.server_host}:{cfg.server_port}"

        if _session["client"] is not None and _session["base_url"] == desired_base:
            log.debug("get_client(): reusing existing client for %s", desired_base)
            return _session["client"]

        model_path = mmproj_path = None
        if cfg.server_mode != "external":
            model_path, mmproj_path, error = resolve_selection(cfg)
            if error:
                log.warning("get_client(): %s", error)
                raise ServerError(error)

        _stop_managed()
        log.debug("get_client(): resolving server (mode=%s)", cfg.server_mode)
        base_url, managed = resolve_server(cfg, model_path, mmproj_path)
        _session["managed"] = managed
        _session["client"] = LlamaClient(base_url, timeout=cfg.request_timeout)
        _session["base_url"] = base_url
        return _session["client"]


def restart_server_ui() -> str:
    log.info("User cleared the server connection from Settings")
    # If a single-image or batch operation is currently running, force-abort
    # it now (rather than block this action) and wait for that to actually
    # finish before touching the server out from under it - see the
    # Operation tracking section above for why a hard kill is the only
    # reliable way to interrupt an in-flight request.
    _operation_force_abort()
    _wait_for_operation_to_end()
    _stop_managed()
    return "Server connection cleared. It will (re)connect/(re)start on the next request."


def restart_app_ui() -> None:
    """Replaces this process with a fresh one via os.execv - the standard
    Python self-restart trick, so newly-saved settings.json values (or a
    just-toggled Debug tab) take effect without anyone needing a terminal.
    Never returns: the process image is gone the moment execv succeeds, so
    stop our own managed llama-server first (execv skips atexit entirely).
    """
    log.info("User triggered app restart from Settings")
    _operation_force_abort()
    _wait_for_operation_to_end()
    _stop_managed()
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------- Single tab

# Interrupt is a SEPARATE component from single_run_btn on purpose - see
# the Operation tracking section: an Interrupt sharing the same
# button/event as the long-running call would sit queued behind it and
# never actually reach the server while captioning is in flight. It gets
# its own Column (single_interrupt_col) rather than sharing Caption's, so
# swapping which one is visible reuses the plain-two-Column-Row shape
# confirmed to align correctly with the row above - see the Single-image
# tab and _operation_button_states' docstring for the full story.
_RUN_IDLE = gr.update(interactive=True)
_RUN_BUSY = gr.update(interactive=False)
_COL_SHOWN = gr.update(visible=True)
_COL_HIDDEN = gr.update(visible=False)
_INTERRUPT_RESET = gr.update(value="Interrupt", variant="stop", interactive=True)


def run_single_ui(image_path, trigger_word_override: str):
    blocked = _operation_blocked_by()
    if blocked:
        yield gr.update(), blocked, _RUN_BUSY, gr.update(), gr.update(), gr.update()
        return

    if not image_path:
        yield "", "Please choose an image first.", gr.update(), gr.update(), gr.update(), gr.update()
        return

    _operation_start("single", "Single-image captioning")
    try:
        log.info("Single-image caption requested: %s", image_path)
        cfg = current_cfg
        base_url = _display_base_url()
        already_up = is_healthy(base_url)
        running_state = (_RUN_IDLE, _COL_HIDDEN, _COL_SHOWN, _INTERRUPT_RESET)
        if cfg.server_mode == "external":
            yield "", ("Processing..." if already_up else "Connecting to external server..."), *running_state
        else:
            yield "", ("Processing..." if already_up else "Starting server (loading model)..."), *running_state

        idle_state = (_RUN_IDLE, _COL_SHOWN, _COL_HIDDEN, _INTERRUPT_RESET)
        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Single-image caption: server error: %s", exc)
            yield "", f"Server error: {exc}", *idle_state
            return

        if not already_up:
            yield "", "Processing...", gr.update(), gr.update(), gr.update(), gr.update()

        try:
            caption, result = caption_image(
                image_path, client, cfg, trigger_word=trigger_word_override
            )
        except ClientError as exc:
            log.warning("Single-image caption failed for %s: %s", image_path, exc)
            yield "", f"Captioning failed: {exc}", *idle_state
            return

        speed = f", {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""

        if result.truncated:
            # llama-server clamps actual generation to whatever fits in
            # (context_size - prompt_tokens), even if that's below the
            # requested max_tokens. If it stopped short of max_tokens, raising
            # max_tokens further won't help - context_size is the real ceiling.
            if result.completion_tokens < cfg.max_tokens:
                advice = (
                    f"prompt used {result.prompt_tokens} tokens, leaving no room "
                    f"in the {cfg.context_size}-token context for the requested "
                    f"{cfg.max_tokens} — raise Context size in Settings"
                )
            else:
                advice = "raise Max tokens in Settings for a full caption"
            status = (
                f"Finished in {result.elapsed_s:.1f}s, but CUT OFF at {result.completion_tokens} "
                f"tokens{speed} ({advice})"
            )
        else:
            status = f"Finished in {result.elapsed_s:.1f}s ({result.completion_tokens} tokens{speed})"
        if result.resize_note:
            status = f"Resized {result.resize_note}. {status}"
        yield caption, status, *idle_state
    finally:
        _operation_end()


def interrupt_single_ui():
    action = _operation_interrupt_click("single")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level above webui/


def _is_gradio_temp_upload(path: Path) -> bool:
    """True if `path` is a copy Gradio made in its own temp cache rather
    than the real file - which is always true for a browser drag-and-drop
    or click-to-upload, since browsers never expose a local file's real
    absolute path to the page. gr.Image gives us that temp copy's path,
    not the original's, so we can't just save next to it in that case.
    """
    import tempfile

    gradio_temp = Path(os.environ.get("GRADIO_TEMP_DIR") or (Path(tempfile.gettempdir()) / "gradio"))
    try:
        path.resolve().relative_to(gradio_temp.resolve())
        return True
    except ValueError:
        return False


def clear_single_result_ui() -> tuple[str, str]:
    """Wired to the image box's own .change() - fires on upload, clear, or
    replace, so a stale caption/status from a previous image never lingers
    next to a new or now-empty image slot.
    """
    return "", ""


def save_single_ui(image_path, caption_text: str) -> str:
    if not image_path:
        return "No image loaded."
    if not caption_text.strip():
        return "Nothing to save."

    src = Path(image_path)
    if not _is_gradio_temp_upload(src):
        txt_path = src.with_suffix(".txt")
        txt_path.write_text(caption_text, encoding="utf-8")
        log.info("Saved caption to %s", txt_path)
        return f"Saved to {txt_path}"

    log.debug("save_single_ui(): %s is a Gradio temp upload - showing save dialog", src)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    save_path = filedialog.asksaveasfilename(
        initialdir=str(PROJECT_ROOT),
        initialfile=src.with_suffix(".txt").name,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    root.destroy()

    if not save_path:
        log.info("Save caption cancelled by user")
        return "Save cancelled."
    Path(save_path).write_text(caption_text, encoding="utf-8")
    log.info("Saved caption to %s", save_path)
    return f"Saved to {save_path}"


# ----------------------------------------------------------------- Batch tab

def browse_directory_ui(current_value: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = current_value if current_value and Path(current_value).is_dir() else None
    selected = filedialog.askdirectory(initialdir=initial) if initial else filedialog.askdirectory()
    root.destroy()
    if selected:
        log.debug("browse_directory_ui(): user picked %s", selected)
    return selected or current_value


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def run_batch_ui(directory_str, recursive, overwrite, trigger_word_override):
    blocked = _operation_blocked_by()
    if blocked:
        yield blocked, "", None, "", _RUN_BUSY, gr.update(), gr.update(), gr.update()
        return

    log.info("Batch requested: %s (recursive=%s, overwrite=%s)", directory_str, recursive, overwrite)
    directory = Path(directory_str) if directory_str else None
    if not directory or not directory.is_dir():
        log.warning("Batch: not a directory: %s", directory_str)
        yield f"Not a directory: {directory_str}", "", None, "", gr.update(), gr.update(), gr.update(), gr.update()
        return

    _operation_start("batch", "Batch captioning")
    try:
        cfg = current_cfg
        base_url = _display_base_url()
        already_up = is_healthy(base_url)
        running_state = (_RUN_IDLE, _COL_HIDDEN, _COL_SHOWN, _INTERRUPT_RESET)
        if cfg.server_mode == "external":
            yield ("Processing..." if already_up else "Connecting to external server..."), "", None, "", *running_state
        else:
            yield ("Processing..." if already_up else "Starting server (loading model)..."), "", None, "", *running_state

        idle_state = (_RUN_IDLE, _COL_SHOWN, _COL_HIDDEN, _INTERRUPT_RESET)
        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Batch: server error: %s", exc)
            yield f"Server error: {exc}", "", None, "", *idle_state
            return

        if not already_up:
            yield "Processing...", "", None, "", gr.update(), gr.update(), gr.update(), gr.update()

        images = find_images(directory, recursive=recursive)
        if not images:
            log.warning("Batch: no images found in %s", directory)
            yield "No images found in that directory.", "", None, "", *idle_state
            return

        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def progress_cb(i, total, path, status, caption, resize_note):
            q.put((i, total, path, status, caption, resize_note))

        def worker():
            try:
                result_holder["result"] = run_batch(
                    directory, client, current_cfg,
                    recursive=recursive,
                    overwrite=overwrite,
                    trigger_word=trigger_word_override,
                    progress_cb=progress_cb,
                    should_stop=lambda: _operation_should_stop("batch"),
                )
            except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash app
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        start_time = time.monotonic()
        thread.start()

        last_file = ""
        last_image = None
        last_caption = ""
        while True:
            item = q.get()
            if item is None:
                break
            i, total, path, status, caption, resize_note = item

            elapsed = time.monotonic() - start_time
            avg = elapsed / i if i else 0.0
            eta = avg * (total - i)
            status_text = f"{i}/{total} processed · avg {avg:.1f}s/image · ETA {_format_duration(eta)}"
            if resize_note:
                status_text += f" · resized {resize_note}"

            # Skipped images were never sent to the model at all (already
            # captioned) - nothing to preview, so leave the preview
            # fields showing whatever the last actually-processed image
            # was instead of decoding and flashing up an unrelated,
            # untouched file.
            if status != "skipped":
                last_file = path.name
                try:
                    # Gradio refuses to serve a raw path outside its CWD/
                    # temp dir (InvalidPathError) - batch directories can
                    # be anywhere on disk, so hand it decoded image data
                    # instead of a path at all.
                    with PILImage.open(path) as img:
                        last_image = img.copy()
                except OSError as exc:
                    log.warning("Could not load preview for %s: %s", path, exc)
                    last_image = None
                if status == "ok":
                    last_caption = caption or ""
                elif status == "truncated":
                    last_caption = "(truncated - not saved, see .txt.issue)"
                else:
                    last_caption = "(failed - see .txt.issue)"
            yield status_text, last_file, last_image, last_caption, gr.update(), gr.update(), gr.update(), gr.update()

        thread.join()
        if "error" in result_holder:
            yield f"Unexpected error: {result_holder['error']}", last_file, last_image, last_caption, *idle_state
        else:
            r = result_holder["result"]
            total_elapsed = time.monotonic() - start_time
            verb = "Aborted" if r.aborted else "Done"
            summary = (
                f"{verb}: {r.processed} captioned, {r.truncated} truncated, "
                f"{r.skipped} skipped, {r.failed} failed "
                f"in {_format_duration(total_elapsed)}"
            )
            yield summary, last_file, last_image, last_caption, *idle_state
    finally:
        _operation_end()


def interrupt_batch_ui():
    action = _operation_interrupt_click("batch")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


# ---------------------------------------------------------------- Review tab
#
# Browse a directory's images and captions side by side, edit inline, and
# re-run captioning on just the one currently shown - no automated bad-
# caption detection (see status column instead), matching how TagGUI and
# similar tools handle this: a human looks, a human decides.
#
# State lives in three gr.State components (Gradio has no server-side
# per-session storage otherwise): review_items_state (list[ReviewItem],
# the current directory's scan_review_status() snapshot),
# review_index_state (int, which item is currently shown, -1 if none),
# review_loaded_caption_state (str, exactly what was on disk when the
# CURRENT item was loaded - compared against the live Textbox value to
# detect a real edit before auto-saving on navigate-away).
#
# Recaption is its own "review" operation kind (not reusing "single",
# even though it's literally the same caption_image() call the
# Single-image tab makes) - see _operation_button_states' docstring for
# why a third kind, rather than sharing "single", is actually necessary
# here. It still mutually-excludes against Single-image/Batch the same
# way they already exclude each other, for free (all three ultimately
# share the one llama-server connection). It gets its own Run/Interrupt
# Column pair (review_recaption_col/review_interrupt_col) built with the
# exact same proven two-Column-swap pattern used elsewhere.


def _review_status_table(items: list[ReviewItem]) -> list[list[str]]:
    return [[item.path.name, item.status] for item in items]


def _review_load(items: list[ReviewItem], index: int):
    """Returns (image, caption_text, loaded_caption_text) for items[index],
    or (None, "", "") if index is out of range. loaded_caption_text is
    exactly what's on disk right now - the baseline later compared against
    the live Textbox to detect an edit worth auto-saving."""
    if not items or not (0 <= index < len(items)):
        return None, "", ""
    item = items[index]
    try:
        with PILImage.open(item.path) as img:
            image = img.copy()
    except OSError as exc:
        log.warning("Review: could not load image %s: %s", item.path, exc)
        image = None
    txt_path = item.path.with_suffix(".txt")
    try:
        caption = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    except OSError as exc:
        log.warning("Review: could not read %s: %s", txt_path, exc)
        caption = ""
    return image, caption, caption


def _review_maybe_save(items: list[ReviewItem], index: int, loaded_caption: str, current_caption: str) -> None:
    """Auto-save on navigate-away. Only writes if the caption actually
    changed AND isn't empty - clearing the box to blank never deletes an
    existing caption, it just leaves the file untouched (agreed: deleting
    a caption should be a deliberate act, not an accident from clearing
    text to retype it - may want an explicit clear/delete action later)."""
    if not items or not (0 <= index < len(items)):
        return
    current = current_caption or ""
    if current.strip() == "" or current == loaded_caption:
        return
    item = items[index]
    txt_path = item.path.with_suffix(".txt")
    txt_path.write_text(current, encoding="utf-8")
    Path(f"{txt_path}{ISSUE_SUFFIX}").unlink(missing_ok=True)
    item.status = "captioned"
    log.info("Review: saved caption for %s", item.path)


def _review_position_text(items: list[ReviewItem], index: int, prefix: str = "") -> str:
    if not items:
        return "No images found."
    if not (0 <= index < len(items)):
        return f"{len(items)} image(s) found."
    item = items[index]
    return f"{prefix}{index + 1}/{len(items)} — {item.path.name} ({item.status})"


def review_scan_ui(directory_str: str):
    directory = Path(directory_str) if directory_str else None
    if not directory or not directory.is_dir():
        log.warning("Review: not a directory: %s", directory_str)
        return f"Not a directory: {directory_str}", [], -1, "", None, "", _review_status_table([])

    items = scan_review_status(directory)
    log.info("Review: scanned %s - %d image(s)", directory, len(items))
    if not items:
        return f"No images found in {directory}", items, -1, "", None, "", _review_status_table(items)

    image, caption, loaded = _review_load(items, 0)
    return (
        _review_position_text(items, 0),
        items, 0, loaded,
        image, caption,
        _review_status_table(items),
    )


def review_prev_ui(items: list[ReviewItem], index: int, loaded_caption: str, current_caption: str):
    _review_maybe_save(items, index, loaded_caption, current_caption)
    new_index = max(0, index - 1) if items else -1
    image, caption, loaded = _review_load(items, new_index)
    return _review_position_text(items, new_index), items, new_index, loaded, image, caption, _review_status_table(items)


def review_next_ui(items: list[ReviewItem], index: int, loaded_caption: str, current_caption: str):
    _review_maybe_save(items, index, loaded_caption, current_caption)
    new_index = min(len(items) - 1, index + 1) if items else -1
    image, caption, loaded = _review_load(items, new_index)
    return _review_position_text(items, new_index), items, new_index, loaded, image, caption, _review_status_table(items)


def review_table_select_ui(
    items: list[ReviewItem], index: int, loaded_caption: str, current_caption: str, evt: gr.SelectData
):
    _review_maybe_save(items, index, loaded_caption, current_caption)
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    image, caption, loaded = _review_load(items, row)
    return _review_position_text(items, row), items, row, loaded, image, caption, _review_status_table(items)


# Mirrors _RUN_IDLE/_RUN_BUSY from the Single-image tab - reused here
# since Review's nav controls (prev/next/table/dir/browse/scan) need the
# exact same "disabled while a 'single'-kind operation is running
# anywhere" treatment as part of the requirement that recaptioning
# disables the rest of this tab's navigation, not just morph its own
# button. See _review_nav_state below and _operation_button_states.
_REVIEW_NAV_BUSY = tuple(gr.update(interactive=False) for _ in range(6))
_REVIEW_NAV_IDLE = tuple(gr.update(interactive=True) for _ in range(6))
# 3 (recaption_col, interrupt_col, interrupt_btn) + 6 (_REVIEW_NAV_BUSY/IDLE
# width) = 9 - must match running_state/idle_state's length below exactly,
# so it's a named constant rather than a repeated magic number.
_REVIEW_STATE_NOOP = tuple(gr.update() for _ in range(9))


def review_recaption_ui(items: list[ReviewItem], index: int, current_caption: str):
    if not items or not (0 <= index < len(items)):
        yield gr.update(), "No image loaded.", *_REVIEW_STATE_NOOP
        return

    blocked = _operation_blocked_by()
    if blocked:
        yield gr.update(), blocked, *_REVIEW_STATE_NOOP
        return

    item = items[index]
    _operation_start("review", "Review recaptioning")
    try:
        log.info("Review recaption requested: %s", item.path)
        cfg = current_cfg
        base_url = _display_base_url()
        already_up = is_healthy(base_url)
        running_state = (_COL_HIDDEN, _COL_SHOWN, _INTERRUPT_RESET, *_REVIEW_NAV_BUSY)
        msg = "Processing..." if already_up else "Starting server (loading model)..."
        yield current_caption, msg, *running_state

        idle_state = (_COL_SHOWN, _COL_HIDDEN, _INTERRUPT_RESET, *_REVIEW_NAV_IDLE)
        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Review recaption: server error: %s", exc)
            yield current_caption, f"Server error: {exc}", *idle_state
            return

        if not already_up:
            yield current_caption, "Processing...", *_REVIEW_STATE_NOOP

        try:
            caption, result = caption_image(item.path, client, cfg, trigger_word=None)
        except ClientError as exc:
            log.warning("Review recaption failed for %s: %s", item.path, exc)
            yield current_caption, f"Recaptioning failed: {exc}", *idle_state
            return

        speed = f", {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""
        note = f"CUT OFF at {result.completion_tokens} tokens{speed}" if result.truncated else f"{result.completion_tokens} tokens{speed}"
        status = f"Recaptioned in {result.elapsed_s:.1f}s ({note}) — not saved yet, navigate away or edit to keep it"
        if result.resize_note:
            status = f"Resized {result.resize_note}. {status}"
        # Deliberately NOT auto-saved here - populates the box like any
        # manual edit would, so the usual navigate-away auto-save (and
        # the "never save an emptied box" rule) applies uniformly whether
        # the text came from typing or from a fresh model result.
        yield caption, status, *idle_state
    finally:
        _operation_end()


def interrupt_review_ui():
    action = _operation_interrupt_click("review")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


# -------------------------------------------------------------- Settings tab

def save_settings_ui(
    server_mode, server_host, server_port, external_url,
    n_gpu_layers, context_size, extra_server_args,
    resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
    prompt_template, temperature, top_p, max_tokens, request_timeout,
    trigger_word, overwrite_existing, recursive_batch, debug_tab_enabled,
) -> str:
    global current_cfg
    # model_name/mmproj_name are deliberately NOT settable from this form -
    # they live entirely in the Models tab now (models_set_active_ui), so
    # carry forward whatever's already configured rather than defaulting
    # to empty just because this form has no field for them.
    current_cfg = AppConfig(
        server_mode=server_mode,
        server_host=server_host,
        server_port=int(server_port),
        external_url=external_url,
        model_name=current_cfg.model_name,
        mmproj_name=current_cfg.mmproj_name,
        n_gpu_layers=n_gpu_layers.strip() or "auto",
        context_size=int(context_size),
        extra_server_args=extra_server_args,
        resize_enabled=bool(resize_enabled),
        resize_target_mp=float(resize_target_mp),
        snap_enabled=bool(snap_enabled),
        snap_multiple=int(snap_multiple),
        prompt_template=prompt_template,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        request_timeout=int(request_timeout),
        trigger_word=trigger_word,
        overwrite_existing=bool(overwrite_existing),
        recursive_batch=bool(recursive_batch),
        debug_tab_enabled=bool(debug_tab_enabled),
    )
    log.info(
        "Settings saved: server_mode=%s model=%s ngl=%s",
        current_cfg.server_mode, current_cfg.model_name or "(none)", current_cfg.n_gpu_layers,
    )
    config_mod.save(current_cfg)
    return "Settings saved. Use \"Restart server connection\" if you changed the model or server settings, or restart the app if you changed the Debug tab setting."


# ---------------------------------------------------------------- Models tab
#
# One row per model DIRECTORY (core.models.group_models), not one row per
# quant file - a folder's several quants and mmproj precisions become the
# two dropdowns' choices for whichever row is currently selected, not
# separate top-level rows each. Clicking a table row only loads that
# group's options into the dropdowns for viewing/editing - it does NOT
# immediately become the active model (switching models means a full
# llama-server restart + multi-GB reload on the next request, too
# expensive to trigger from a stray click); only the explicit "Set as
# active model" button commits a choice, writing straight to
# settings.json the same way Settings' Save button already does for
# everything else.

def _models_status_marker(cfg: AppConfig, group: ModelGroup) -> str:
    if any(q.name == cfg.model_name for q in group.quants):
        return "✓ active"
    if group.quants:
        return ""
    if group.curated:
        return "downloadable"
    return ""


def _models_table_rows(groups: list[ModelGroup]) -> list[list[str]]:
    cfg = current_cfg
    return [[g.name, _models_status_marker(cfg, g)] for g in groups]


def _models_dropdown_updates(
    group: Optional[ModelGroup], preferred_quant: Optional[str] = None, preferred_mmproj: Optional[str] = None
):
    """(quant_update, mmproj_update) for `group` - both show "N/A" and
    stay disabled if group is None or has no quants at all (e.g. a folder
    whose main model download hasn't finished, only its mmproj has);
    mmproj alone shows "N/A" if the folder has quants but no usable
    mmproj. Deliberately never uses an empty choices=[] with value=None
    to mean "nothing selected" - a Dropdown whose value isn't reliably
    cleared by value=None can be left showing a stale previous value
    (e.g. "N/A" from a different row) against newly-empty choices, which
    Gradio then rejects outright ("Value: X is not in the list of
    choices: []"). Always giving both a real, matching single choice
    instead sidesteps that regardless of the exact clearing behavior.

    preferred_quant/preferred_mmproj select which of the group's choices
    to actually show, if given and valid - falls back to the first choice
    otherwise. Passing these matters: without it, every caller silently
    means "show the alphabetically-first quant/mmproj", which is wrong
    for both models_refresh_ui (should show whatever's ACTUALLY
    configured, cfg.model_name/mmproj_name) and models_set_active_ui
    (should show whatever was JUST committed) - either defaulting to
    "first in the list" regardless would visually contradict its own
    just-reported status text.

    Choices are (label, value) pairs so a not-yet-downloaded curated
    quant/mmproj can carry a "(download, 6.8 GB)" label while its VALUE
    stays the same "<folder>/<stem>" shape a real local one would have -
    that's what lets models_set_active_ui tell the two apart later just by
    checking whether the value matches a real ModelVariant/MmprojVariant."""
    if group is None:
        na = gr.update(choices=["N/A"], value="N/A", interactive=False)
        return na, na

    quant_pairs = [(q.name, q.name) for q in group.quants]
    local_quant_names = {q.name for q in group.quants}
    if group.curated:
        for cq in group.curated.quants:
            value = f"{group.folder.name}/{cq.name}"
            if value not in local_quant_names:
                quant_pairs.append((f"{cq.name} (download, {format_size(cq.size_bytes)})", value))

    mmproj_pairs = [(m.name, m.name) for m in group.mmprojs]
    local_mmproj_names = {m.name for m in group.mmprojs}
    if group.curated:
        for cm in group.curated.mmprojs:
            value = f"{group.folder.name}/{cm.name}"
            if value not in local_mmproj_names:
                mmproj_pairs.append((f"{cm.name} (download, {format_size(cm.size_bytes)})", value))

    if not quant_pairs:
        na = gr.update(choices=["N/A"], value="N/A", interactive=False)
        return na, na
    quant_values = [v for _, v in quant_pairs]
    quant_value = preferred_quant if preferred_quant in quant_values else quant_values[0]
    quant_update = gr.update(choices=quant_pairs, value=quant_value, interactive=True)

    if not mmproj_pairs:
        mmproj_update = gr.update(choices=["N/A"], value="N/A", interactive=False)
    else:
        mmproj_values = [v for _, v in mmproj_pairs]
        mmproj_value = preferred_mmproj if preferred_mmproj in mmproj_values else mmproj_values[0]
        mmproj_update = gr.update(choices=mmproj_pairs, value=mmproj_value, interactive=True)
    return quant_update, mmproj_update


def models_refresh_ui():
    """Rescans webui/models/, rebuilds the grouped table, and re-selects
    whichever group is currently the active model (if any) so a refresh
    shows real state instead of resetting the dropdowns to nothing."""
    models, mmprojs = scan_all()
    groups = merge_curated(group_models(models, mmprojs), load_curated_models())
    table = _models_table_rows(groups)

    cfg = current_cfg
    active_group = next((g for g in groups if any(q.name == cfg.model_name for q in g.quants)), None)
    quant_update, mmproj_update = _models_dropdown_updates(
        active_group, preferred_quant=cfg.model_name, preferred_mmproj=cfg.mmproj_name
    )
    folder_key = str(active_group.folder) if active_group else None
    status = f"{len(groups)} model folder(s) found." if groups else f"No models found under {MODELS_DIR}."

    return table, groups, folder_key, quant_update, mmproj_update, status


def models_table_select_ui(groups: list[ModelGroup], evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not groups or not (0 <= row < len(groups)):
        return None, gr.update(), gr.update(), gr.update()
    group = groups[row]
    quant_update, mmproj_update = _models_dropdown_updates(group)
    if not group.quants:
        if group.curated:
            note = " — not downloaded yet; pick a quant below to see download info"
        else:
            note = " — no model file found (download incomplete?), can't be selected"
    elif not group.mmprojs:
        note = " — no mmproj found, can't be selected"
    else:
        note = ""
    return str(group.folder), quant_update, mmproj_update, f"Viewing {group.name}{note}."


def models_set_active_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    global current_cfg
    noop_dropdowns = (gr.update(), gr.update())
    if not quant_name or quant_name == "N/A":
        return _models_table_rows(groups), *noop_dropdowns, "No model selected — click a row in the table first."

    group = next((g for g in groups if any(q.name == quant_name for q in g.quants)), None)
    if group is None:
        # quant_name didn't match any LOCAL ModelVariant - it's a curated
        # "(download, ...)" choice that isn't on disk yet (see
        # _models_dropdown_updates: value shape is identical either way,
        # so this is the only place that actually tells them apart).
        shown = quant_name.rsplit("/", 1)[-1]
        return _models_table_rows(groups), *noop_dropdowns, (
            f"\"{shown}\" isn't downloaded yet - download support is coming soon, "
            "pick an already-downloaded quant for now."
        )

    if mmproj_name == "N/A":
        # "N/A" is the sentinel _models_dropdown_updates uses specifically
        # for "this folder has no usable mmproj" - not a real empty/auto
        # value, so it must never be written as if it meant that (would
        # silently commit an unselectable model, only failing later at
        # server-start time instead of refusing here where the reason is
        # actually known).
        return _models_table_rows(groups), *noop_dropdowns, "This model has no mmproj — can't be set as active."
    if mmproj_name and not any(m.name == mmproj_name for m in group.mmprojs):
        shown = mmproj_name.rsplit("/", 1)[-1]
        return _models_table_rows(groups), *noop_dropdowns, (
            f"\"{shown}\" isn't downloaded yet - download support is coming soon, "
            "pick an already-downloaded mmproj for now."
        )
    mmproj_value = "" if mmproj_name is None else mmproj_name
    current_cfg = replace(current_cfg, model_name=quant_name, mmproj_name=mmproj_value)
    config_mod.save(current_cfg)
    log.info("Models tab: set active model to %s (mmproj=%s)", quant_name, mmproj_value or "(auto)")
    mmproj_note = f"mmproj: {mmproj_value}" if mmproj_value else "mmproj: auto-pick largest"

    # Rebuild the dropdowns fresh from the group that was just committed,
    # rather than leaving whatever was on screen before untouched - that
    # stale-state gap (a disabled "N/A" mmproj dropdown surviving into a
    # later action that assumes real choices) is what actually caused the
    # "value N/A not in choices []" error. `group` is already the right
    # one, found above.
    quant_update, mmproj_update = _models_dropdown_updates(
        group, preferred_quant=quant_name, preferred_mmproj=mmproj_name
    )
    return (
        _models_table_rows(groups),
        quant_update, mmproj_update,
        f"Active model set to {quant_name} ({mmproj_note}). "
        "Click \"Restart server connection\" below to actually load it.",
    )


# ------------------------------------------------------------------ Status bar

def _display_base_url() -> str:
    cfg = current_cfg
    if cfg.server_mode == "external":
        return cfg.external_url.rstrip("/")
    return f"http://{cfg.server_host}:{cfg.server_port}"


def _server_status_text(base_url: str, healthy: bool) -> str:
    # Just "managed" (we started it) vs "remote" (already running somewhere
    # - local-but-not-ours, external mode, doesn't matter which) - the
    # adjacent "Model: n/a" already covers unhealthy/crashed, so a third
    # state for that here would just be saying the same thing twice.
    cfg = current_cfg
    if cfg.server_mode == "external":
        return "remote"

    managed = _session.get("managed")
    connected = _session.get("client") is not None and _session.get("base_url") == base_url
    if connected and managed is not None:
        return "managed"
    return "remote" if healthy else "stopped"


def get_status_text() -> str:
    base_url = _display_base_url()
    healthy = is_healthy(base_url)
    # Always the server's own answer, never our config - our selection is
    # just what we'd ask it to load next, not necessarily what's actually
    # loaded right now (e.g. external mode, or a server someone else picked).
    model = get_loaded_model_name(base_url) if healthy else None
    model = model or "n/a"
    return (
        f"Python {platform.python_version()} &nbsp;·&nbsp; "
        f"Gradio {gr.__version__} &nbsp;·&nbsp; "
        f"llama-server: {_server_status_text(base_url, healthy)} &nbsp;·&nbsp; "
        f"Model: {model} &nbsp;·&nbsp; "
        f"{_operation_status_text()}"
    )


# ---------------------------------------------------------------- Debug tab

def get_python_debug_text() -> str:
    # Newest first - the boxes don't reliably autoscroll on periodic
    # Timer.tick updates (Gradio's autoscroll is gated on streaming/
    # generating state, not plain polled value replacement), so the latest
    # line needs to be visible without scrolling.
    return "\n".join(reversed(_PY_LOG_BUFFER)) or "(no log output yet)"


def get_llama_debug_text() -> str:
    if _session.get("managed") is None:
        return "n/a"
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        return "\n".join(reversed(text.splitlines())) if text else "(empty)"
    except OSError as exc:
        return f"(could not read {LOG_PATH}: {exc})"


def clear_debug_ui() -> tuple[str, str]:
    _PY_LOG_BUFFER.clear()
    if _session.get("managed") is not None:
        try:
            LOG_PATH.write_text("", encoding="utf-8")
        except OSError:
            pass
    return get_python_debug_text(), get_llama_debug_text()


def build_app() -> gr.Blocks:
    cfg = current_cfg

    with gr.Blocks(title="XenoTagger", analytics_enabled=False) as demo:
        gr.Markdown("# XenoTagger — LoRA dataset captioning")

        with gr.Tab("Single image"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    single_image = gr.Image(
                        type="filepath", label="Image", sources=["upload"]
                    )
                with gr.Column():
                    single_trigger = gr.Textbox(
                        label="Trigger word",
                        value=cfg.trigger_word,
                    )
                    single_caption = gr.Textbox(
                        label="Caption", lines=20, interactive=True
                    )
            # Confirmed live: a Row of exactly two plain gr.Column()s (no
            # CSS at all) aligns correctly with Row A above. So instead of
            # stacking Caption/Interrupt inside one shared Column (which
            # needed CSS overrides that threw alignment off), each gets
            # its OWN Column, and it's the whole COLUMN's visibility that
            # toggles - not just the button inside it. At any moment
            # exactly two of these three Columns are visible (Caption+Save
            # idle, Interrupt+Save running) - always that same proven
            # "Row of two plain Columns" shape, whichever two they are.
            with gr.Row():
                with gr.Column() as single_run_col:
                    single_run_btn = gr.Button("Caption", variant="primary")
                # Separate component from single_run_btn on purpose - see
                # the Operation tracking section: an Interrupt that shares
                # the same button/event as the long-running call would sit
                # queued behind it and never actually reach the server
                # while captioning is in flight.
                with gr.Column(visible=False) as single_interrupt_col:
                    single_interrupt_btn = gr.Button("Interrupt", variant="stop")
                with gr.Column():
                    single_save_btn = gr.Button("Save caption", interactive=False)
            single_status = gr.Textbox(show_label=False, container=False, interactive=False)

            single_run_btn.click(
                run_single_ui,
                [single_image, single_trigger],
                [
                    single_caption, single_status,
                    single_run_btn, single_run_col, single_interrupt_col, single_interrupt_btn,
                ],
            )
            single_interrupt_btn.click(interrupt_single_ui, [], [single_interrupt_btn])
            single_save_btn.click(
                save_single_ui, [single_image, single_caption], [single_status]
            )
            single_caption.change(
                lambda text: gr.update(interactive=bool(text.strip())),
                [single_caption], [single_save_btn],
            )
            single_image.change(clear_single_result_ui, [], [single_caption, single_status])

        with gr.Tab("Batch processing"):
            with gr.Row():
                batch_dir = gr.Textbox(label="Directory of images", scale=4)
                batch_browse_btn = gr.Button("Browse...", scale=1)

            with gr.Row():
                batch_trigger = gr.Textbox(
                    label="Trigger word",
                    value=cfg.trigger_word,
                )
                with gr.Column():
                    batch_recursive = gr.Checkbox(label="Recursive", value=cfg.recursive_batch)
                    batch_overwrite = gr.Checkbox(
                        label="Overwrite existing captions", value=cfg.overwrite_existing
                    )

            # Same two-Column swap as the Single-image tab (see that tab
            # and app.py's own module docstring for why: a Row of two
            # plain gr.Column()s is the only thing confirmed to align
            # predictably, no CSS needed) - kept here for consistency even
            # though this row has no third sibling to misalign against.
            with gr.Row():
                with gr.Column() as batch_run_col:
                    batch_run_btn = gr.Button("Run batch", variant="primary")
                with gr.Column(visible=False) as batch_interrupt_col:
                    batch_interrupt_btn = gr.Button("Interrupt", variant="stop")

            with gr.Row():
                batch_last_image = gr.Image(
                    label="Preview", interactive=False, show_label=False,
                    buttons=[],
                )
                with gr.Column():
                    batch_last_file = gr.Textbox(label="Last file processed", interactive=False)
                    batch_last_caption = gr.Textbox(label="Last caption created", interactive=False)

            batch_status = gr.Textbox(show_label=False, container=False, interactive=False)

            batch_browse_btn.click(browse_directory_ui, [batch_dir], [batch_dir])
            batch_run_btn.click(
                run_batch_ui,
                [batch_dir, batch_recursive, batch_overwrite, batch_trigger],
                [
                    batch_status, batch_last_file, batch_last_image, batch_last_caption,
                    batch_run_btn, batch_run_col, batch_interrupt_col, batch_interrupt_btn,
                ],
            )
            batch_interrupt_btn.click(interrupt_batch_ui, [], [batch_interrupt_btn])

        with gr.Tab("Review"):
            with gr.Row():
                review_dir = gr.Textbox(label="Directory of images", scale=4)
                with gr.Column(scale=1, min_width=120):
                    review_browse_btn = gr.Button("Browse...")
                    review_scan_btn = gr.Button("Scan")

            with gr.Row(equal_height=True):
                review_prev_btn = gr.Button("←", scale=1, min_width=60)
                review_image = gr.Image(
                    label="Image", interactive=False, show_label=False, buttons=[], scale=8, height=480,
                )
                review_next_btn = gr.Button("→", scale=1, min_width=60)

            with gr.Row():
                with gr.Column() as review_recaption_col:
                    review_recaption_btn = gr.Button("Recaption")
                # Separate component from review_recaption_btn on purpose,
                # same reasoning as every other Interrupt in this app - see
                # the Operation tracking section.
                with gr.Column(visible=False) as review_interrupt_col:
                    review_interrupt_btn = gr.Button("Interrupt", variant="stop")

            review_caption = gr.Textbox(label="Caption", lines=4, interactive=True)

            review_table = gr.Dataframe(
                headers=["File", "Status"], datatype=["str", "str"],
                interactive=False, row_count=(0, "dynamic"),
            )

            review_status = gr.Textbox(show_label=False, container=False, interactive=False)

            # See the Review tab's handler-function docstrings above for
            # what each of these three actually holds.
            review_items_state = gr.State([])
            review_index_state = gr.State(-1)
            review_loaded_caption_state = gr.State("")

            _review_nav_outputs = [
                review_status, review_items_state, review_index_state, review_loaded_caption_state,
                review_image, review_caption, review_table,
            ]
            review_browse_btn.click(browse_directory_ui, [review_dir], [review_dir])
            review_scan_btn.click(review_scan_ui, [review_dir], _review_nav_outputs)
            review_prev_btn.click(
                review_prev_ui,
                [review_items_state, review_index_state, review_loaded_caption_state, review_caption],
                _review_nav_outputs,
            )
            review_next_btn.click(
                review_next_ui,
                [review_items_state, review_index_state, review_loaded_caption_state, review_caption],
                _review_nav_outputs,
            )
            review_table.select(
                review_table_select_ui,
                [review_items_state, review_index_state, review_loaded_caption_state, review_caption],
                _review_nav_outputs,
            )
            review_recaption_btn.click(
                review_recaption_ui,
                [review_items_state, review_index_state, review_caption],
                [
                    review_caption, review_status,
                    review_recaption_col, review_interrupt_col, review_interrupt_btn,
                    review_prev_btn, review_next_btn, review_table,
                    review_dir, review_browse_btn, review_scan_btn,
                ],
            )
            review_interrupt_btn.click(interrupt_review_ui, [], [review_interrupt_btn])

        with gr.Tab("Models"):
            gr.Markdown(
                "Every `.gguf` under a `webui/models/<folder>/` is picked up, "
                "grouped by folder: files with \"mmproj\" in the name are "
                "projectors, everything else is a selectable quant. Files "
                f"matching `{', '.join(IGNORED_SUBSTRINGS)}` are ignored "
                "(e.g. speculative-decoding draft models). Dropdown choices "
                "marked \"(download, ...)\" come from the curated list and "
                "aren't on disk yet - download support is coming soon."
            )
            models_table = gr.Dataframe(
                headers=["Model", "Status"], datatype=["str", "str"],
                interactive=False, row_count=(0, "dynamic"),
            )
            with gr.Row():
                models_quant_dropdown = gr.Dropdown(label="Quant", interactive=False)
                models_mmproj_dropdown = gr.Dropdown(label="mmproj", interactive=False)
            with gr.Row():
                models_set_active_btn = gr.Button("Set as active model", variant="primary")
                restart_server_btn = gr.Button("Restart server connection")
                refresh_models_btn = gr.Button("Refresh")
            models_status = gr.Textbox(show_label=False, container=False, interactive=False)

            # See this tab's own handler-function docstrings above for what
            # each of these two actually holds.
            models_groups_state = gr.State([])
            models_selected_folder_state = gr.State(None)

            _models_scan_outputs = [
                models_table, models_groups_state, models_selected_folder_state,
                models_quant_dropdown, models_mmproj_dropdown, models_status,
            ]
            refresh_models_btn.click(models_refresh_ui, [], _models_scan_outputs)
            models_table.select(
                models_table_select_ui,
                [models_groups_state],
                [models_selected_folder_state, models_quant_dropdown, models_mmproj_dropdown, models_status],
            )
            models_set_active_btn.click(
                models_set_active_ui,
                [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
                [models_table, models_quant_dropdown, models_mmproj_dropdown, models_status],
            )
            restart_server_btn.click(restart_server_ui, [], [models_status])

        with gr.Tab("Settings"):
            with gr.Group():
                gr.Markdown("### llama-server")
                server_mode = gr.Radio(
                    choices=[
                        (
                            "Auto - use an already-running server if there is one; "
                            "otherwise start llama-server.exe and stop it again when done",
                            "auto",
                        ),
                        (
                            "External - always connect to a server you start yourself, "
                            "possibly on a different machine",
                            "external",
                        ),
                    ],
                    value=cfg.server_mode,
                    label="Server mode",
                )
                with gr.Row():
                    server_host = gr.Textbox(label="Host (auto mode)", value=cfg.server_host)
                    server_port = gr.Number(label="Port (auto mode)", value=cfg.server_port, precision=0)
                external_url = gr.Textbox(
                    label="External server URL (external mode)", value=cfg.external_url
                )
                gr.Markdown("Model/mmproj selection now lives on the **Models** tab.")
                with gr.Row():
                    n_gpu_layers = gr.Textbox(
                        label="GPU layers ('auto', 'all', or an exact number)",
                        value=cfg.n_gpu_layers,
                    )
                    context_size = gr.Number(label="Context size", value=cfg.context_size, precision=0)
                extra_server_args = gr.Textbox(
                    label="Extra llama-server arguments", value=cfg.extra_server_args
                )
                gr.Markdown("\"Restart server connection\" now lives on the **Models** tab.")

            with gr.Group():
                gr.Markdown("### Generation")
                prompt_template = gr.Textbox(
                    label="Prompt template", value=cfg.prompt_template, lines=4
                )
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=cfg.temperature, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=cfg.top_p, label="Top-p")
                    max_tokens = gr.Number(label="Max tokens", value=cfg.max_tokens, precision=0)
                request_timeout = gr.Number(
                    label="Request timeout (seconds)", value=cfg.request_timeout, precision=0
                )

            with gr.Group():
                gr.Markdown("### Image resizing")
                resize_enabled = gr.Checkbox(
                    label="Downscale oversized images before sending to the model",
                    value=cfg.resize_enabled,
                )
                resize_target_mp = gr.Slider(
                    0.1, 4.0, value=cfg.resize_target_mp, step=0.1,
                    label="Target resolution (megapixels)",
                )
                with gr.Row():
                    snap_enabled = gr.Checkbox(
                        label="Scale to multiple of", value=cfg.snap_enabled,
                    )
                    snap_multiple = gr.Number(
                        value=cfg.snap_multiple, precision=0, show_label=False,
                    )

            with gr.Group():
                gr.Markdown("### Captioning defaults")
                trigger_word = gr.Textbox(label="Default trigger word", value=cfg.trigger_word)
                with gr.Row():
                    overwrite_existing = gr.Checkbox(
                        label="Overwrite existing captions by default", value=cfg.overwrite_existing
                    )
                    recursive_batch = gr.Checkbox(
                        label="Recursive batch by default", value=cfg.recursive_batch
                    )

            with gr.Group():
                gr.Markdown("### Debug")
                debug_tab_enabled = gr.Checkbox(
                    label="Enable Debug tab (requires app restart)",
                    value=cfg.debug_tab_enabled,
                )

            save_settings_btn = gr.Button("Save settings", variant="primary")
            restart_app_btn = gr.Button("Restart app")
            settings_status = gr.Textbox(label="Status", interactive=False)

            settings_inputs = [
                server_mode, server_host, server_port, external_url,
                n_gpu_layers, context_size, extra_server_args,
                resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
                prompt_template, temperature, top_p, max_tokens, request_timeout,
                trigger_word, overwrite_existing, recursive_batch, debug_tab_enabled,
            ]
            save_settings_btn.click(save_settings_ui, settings_inputs, [settings_status])
            restart_app_btn.click(restart_app_ui, [], [])

        with gr.Tab("Debug", visible=cfg.debug_tab_enabled):
            gr.Markdown("**Python debug log**")
            debug_python_box = gr.Textbox(
                lines=18, max_lines=18, interactive=False, show_label=False,
                value=get_python_debug_text(),
            )
            gr.Markdown("**llama-server output** (n/a unless started by this app)")
            debug_llama_box = gr.Textbox(
                lines=18, max_lines=18, interactive=False, show_label=False,
                value=get_llama_debug_text(),
            )
            debug_clear_btn = gr.Button("Clear")
            debug_clear_btn.click(clear_debug_ui, [], [debug_python_box, debug_llama_box])

        status_bar = gr.Markdown(get_status_text(), elem_id="status-bar")
        _run_interrupt_btns = [
            single_run_btn, single_run_col, single_interrupt_col, single_interrupt_btn,
            batch_run_btn, batch_run_col, batch_interrupt_col, batch_interrupt_btn,
            review_recaption_col, review_interrupt_col, review_interrupt_btn,
            review_prev_btn, review_next_btn, review_table,
            review_dir, review_browse_btn, review_scan_btn,
        ]
        status_timer = gr.Timer(2.0)
        status_timer.tick(get_status_text, [], [status_bar])
        status_timer.tick(_operation_button_states, [], _run_interrupt_btns)
        if cfg.debug_tab_enabled:
            status_timer.tick(get_python_debug_text, [], [debug_python_box])
            status_timer.tick(get_llama_debug_text, [], [debug_llama_box])
        demo.load(get_status_text, [], [status_bar])
        demo.load(_operation_button_states, [], _run_interrupt_btns)
        demo.load(models_refresh_ui, [], _models_scan_outputs)

    return demo


UI_PORT = 7901


def main() -> None:
    demo = build_app()
    demo.queue()
    demo.launch(server_port=UI_PORT, footer_links=[], css=ALL_CSS)


if __name__ == "__main__":
    main()
