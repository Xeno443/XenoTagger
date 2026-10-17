"""Operation tracking: is a long-running caption job (single/batch/review)
active, and what stage of being interrupted is it in. Also the shared
Run/Interrupt button states every caption tab uses."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import gradio as gr

from ui.llama_session import stop_managed

log = logging.getLogger("app")  # fixed name, see app.py

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
# above ui.llama_session's _session_lock: only one kind of operation ("single" or "batch")
# may be active at a time, since both ultimately share the one
# llama-server connection.

@dataclass
class _Operation:
    kind: str  # "single" or "batch"
    label: str  # human-readable, shown in the status bar
    stop_requested: bool = False
    abort_requested: bool = False


_active_operation: Optional[_Operation] = None
operation_lock = threading.RLock()


def operation_blocked_by() -> Optional[str]:
    """None if nothing is currently running. Otherwise a message naming
    what's active - covers both a different kind of operation and an
    accidental duplicate start of the same kind (the Run button disables
    itself while busy, but this is a second line of defense, e.g. against
    a stray click landing just before that update reaches the browser)."""
    with operation_lock:
        op = _active_operation
        if op is not None:
            return f"{op.label} is currently running - interrupt it from its own tab first."
        return None


def current_operation() -> Optional[_Operation]:
    """The active operation, or None. Hold operation_lock around the call
    and every read of the result if the reads need to be consistent with
    each other (op's flags can change from another thread)."""
    return _active_operation


def operation_start(kind: str, label: str) -> None:
    global _active_operation
    with operation_lock:
        _active_operation = _Operation(kind=kind, label=label)


def operation_end() -> None:
    global _active_operation
    with operation_lock:
        _active_operation = None


def operation_interrupt_click(kind: str) -> str:
    """Call when the Run/Interrupt button is clicked while `kind` is
    already the active operation. Returns "stopping" on the first click
    (graceful - just sets a flag the running loop checks), "aborting" on
    the second (hard - kills the server, which is done outside the lock
    since stop_managed() acquires its own _session_lock)."""
    with operation_lock:
        op = _active_operation
        if op is None or op.kind != kind:
            return "idle"
        if not op.stop_requested:
            op.stop_requested = True
            log.info("%s: stop requested (graceful)", op.label)
            return "stopping"
        op.abort_requested = True
        log.info("%s: abort requested (killing server)", op.label)
    stop_managed()
    return "aborting"


def operation_should_stop(kind: str) -> bool:
    with operation_lock:
        op = _active_operation
        return op is not None and op.kind == kind and op.stop_requested


def operation_status_text() -> str:
    with operation_lock:
        op = _active_operation
        if op is None:
            return "idle"
        if op.abort_requested:
            return f"{op.label} (aborting)"
        if op.stop_requested:
            return f"{op.label} (stopping)"
        return op.label


def interrupt_label_state(op: Optional["_Operation"], kind: str) -> tuple[str, bool]:
    """(label, interactive) for `kind`'s Interrupt button, given the
    active operation (or None). Used by run_controls_update, which
    has to derive this fresh from scratch since it isn't the one that
    just clicked Interrupt - interrupt_single_ui/interrupt_batch_ui don't
    need this themselves, they already know which of the two states just
    happened from operation_interrupt_click's own return value."""
    if op is None or op.kind != kind:
        return "Interrupt", True
    if op.abort_requested:
        return "Interrupting...", False
    if op.stop_requested:
        return "Interrupt (click again to abort now)", True
    return "Interrupt", True


def run_controls_update(kind: str, op: Optional[_Operation], reachable: bool) -> tuple:
    """(run_btn, run_col, interrupt_col, interrupt_btn) updates for the tab
    whose operation kind is `kind` - one tab's share of app.py's
    _update_ui_status (see its docstring). Caller holds operation_lock."""
    blocked = (op is not None and op.kind != kind) or not reachable
    running = op is not None and op.kind == kind
    label, ok = interrupt_label_state(op, kind)
    return (
        gr.update(interactive=not blocked),
        gr.update(visible=not running),
        gr.update(visible=running),
        gr.update(value=label, variant="stop", interactive=ok),
    )


def settings_subtabs_disabled() -> bool:
    """The one, sole question of whether the four operation-sensitive
    Settings sub-tabs (Llama, Hydra, Image resizing, Captioning defaults)
    are currently off-limits - today that's just "is a job running", but
    kept as its own function (not inlined into _settings_gating_ui or
    _settings_subtab_entry_ui, both of which call this rather than
    checking _active_operation themselves) so both of those callers can
    never drift apart on what "disabled" means, even if a future reason
    to disable these sub-tabs gets added later - it only has to change
    here."""
    with operation_lock:
        return _active_operation is not None


def operation_force_abort() -> None:
    """Immediately hard-aborts whatever's running, if anything - skips the
    two-click grace period (for Restart, which needs the operation gone
    now, not eventually). No-op if nothing is running."""
    with operation_lock:
        op = _active_operation
        if op is None:
            return
        op.stop_requested = True
        op.abort_requested = True
        log.info("%s: force-aborted (restart requested)", op.label)
    stop_managed()


def wait_for_operation_to_end(timeout: float = 30.0) -> None:
    """Blocks until the active operation's own thread notices the abort
    and calls operation_end() - killing the server (above) should make
    its in-flight request fail almost immediately, so this is normally
    quick; the timeout is just a safety net against hanging forever."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with operation_lock:
            if _active_operation is None:
                return
        time.sleep(0.2)
    log.warning("Timed out waiting for the running operation to stop before restart")


# Interrupt is a SEPARATE component from single_run_btn on purpose - see
# the Operation tracking section: an Interrupt sharing the same
# button/event as the long-running call would sit queued behind it and
# never actually reach the server while captioning is in flight. It gets
# its own Column (single_interrupt_col) rather than sharing Caption's, so
# swapping which one is visible reuses the plain-two-Column-Row shape
# confirmed to align correctly with the row above - see the Single-image
# tab and app.py's _update_ui_status' docstring for the full story.
RUN_IDLE = gr.update(interactive=True)
RUN_BUSY = gr.update(interactive=False)
COL_SHOWN = gr.update(visible=True)
COL_HIDDEN = gr.update(visible=False)
INTERRUPT_RESET = gr.update(value="Interrupt", variant="stop", interactive=True)
