"""Gradio UI: Single-image captioning, Batch processing, Models, Settings,
and an opt-in Debug tab. The CLI (cli.py) is a separate, independent
entry point that shares core/ with this file but not this file itself -
nothing UI-specific belongs in core/.

Three coordination mechanisms guard shared, process-wide state that both
tabs (and Settings' restart actions) can touch concurrently - the first
two are threading.RLock, not a plain Lock, because each has a function
that calls another lock-holding function of its own while already
holding the lock (get_client() calls _stop_managed(); a plain Lock would
deadlock there):

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

  - `current_cfg`/`_config_lock` (plain Lock - neither writer calls back
    into itself or the other while holding it): guards the read-modify-
    write of current_cfg + config_mod.save() in save_settings_ui() and
    models_set_active_ui(), the only two places that mutate current_cfg
    wholesale. Without it, two overlapping saves could each read a stale
    current_cfg and the second write would silently clobber fields the
    first one just changed.

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
running elsewhere. _update_ui_status() closes that gap: it
recomputes both tabs' Run/Interrupt appearance fresh from
_active_operation on every call, and is wired into the same 2-second
status_timer that already refreshes the status bar, plus demo.load - so
any view self-corrects within one tick instead of staying wrong
indefinitely.

restart_app_ui doesn't block just because something's running, nor does
it silently kill it out from under whatever's in flight - it force-
aborts (skipping the two-click grace period; see _operation_force_abort)
and actually waits for that operation to finish noticing and clean up
(_wait_for_operation_to_end) before touching the server. This matters
because os.execv replaces the entire process with zero chance for any
cleanup to run afterward - waiting first means an interrupted operation
gets an observable, handled failure instead of just vanishing without a
trace. The Models tab's own "Manage llama server" button used to have an
in-place equivalent (restart_server_ui, force-abort + stop + refresh
right there) but is now pure navigation to Settings -> Llama - actually
stopping/starting only ever happens from that sub-tab's Start/End
buttons, which are already correctly gated (disabled mid-job, see
_settings_gating_ui) so there's no force-abort race left for a plain
redirect to need to guard against.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import queue
import re
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
from core import hydra_classifier
from core.batch import ISSUE_SUFFIX, ReviewItem, find_images, run_batch, scan_review_status
from core.captioner import caption_image
from core.client import ClientError, LlamaClient
from core.config import AppConfig
from core.downloads import DownloadItem, download_one
from core.hydra_install import (
    MODEL_PATH as HYDRA_MODEL_PATH, MODEL_SIZE_BYTES as HYDRA_MODEL_SIZE_BYTES, MODEL_URL as HYDRA_MODEL_URL,
    install_deps as hydra_install_deps,
)
from core.llama_install import (
    BACKENDS as LLAMA_BACKENDS, DEFAULT_BACKEND as LLAMA_DEFAULT_BACKEND,
    install as llama_install_run, installed_info as llama_installed_info, plan_install as llama_plan_install,
)
from core.models import (
    IGNORED_SUBSTRINGS, MODELS_DIR, ModelGroup, format_size, group_models,
    load_curated_models, merge_curated, resolve_selection, scan_all,
)
from core.server import (
    LLAMA_SERVER_EXE, LOG_PATH, MANAGED_HOST, ManagedServer, ServerError, ServerStatus, check_status,
    get_loaded_model_name, is_healthy, resolve_server,
)
from ui_css import ALL_CSS

log = logging.getLogger("app")  # fixed name, not __name__ - which is "__main__"
                                  # when run directly (the real usage) but
                                  # "app" when imported (e.g. in tests) -
                                  # this way logging.getLogger("app") always
                                  # refers to the right logger either way.

# gr.Error is deliberately not in here, even though it's the "error"-
# flavored popup - unlike gr.Info/gr.Warning (plain calls that post a
# toast and let execution continue), gr.Error only displays anything
# when *raised* and allowed to propagate all the way out to Gradio's own
# dispatcher, which aborts whatever function raised it. That's the wrong
# shape for _notify's callers below - none of them want firing a popup
# to also cut their own function off before it returns its normal
# outputs. So "error" still logs as a real log.error() line, it just
# surfaces as a (non-fatal) warning-styled toast rather than a red one.
_POPUP_FNS = {"info": gr.Info, "warning": gr.Warning, "error": gr.Warning}


def _notify(text: str, level: str = "info") -> None:
    """The one place a Gradio popup toast gets fired - replaces what used
    to be scattered direct gr.Info/gr.Warning calls. level picks both the
    log.<level>() call (auto-captured by the debug log/file when enabled -
    see _setup_debug_logging) and the matching popup type - see
    _POPUP_FNS above for why "error" still pops a warning-styled toast,
    not a red one. This always pops a toast - for a message that should
    only be logged, call log.<level>() directly instead.

    Must be called from inside a real Gradio callback (click/tick/load
    handler) - like gr.Info/Warning themselves, it needs Gradio's request
    context, so it can't be called from a raw background thread (e.g. the
    batch/download/install worker threads) - those still have to route
    through a queue/state that a polling handler relays from, the same
    pattern already used for lifecycle infotext boxes."""
    getattr(log, level)(text)
    _POPUP_FNS[level](text)


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

# Guards the read-modify-write of current_cfg + config_mod.save() in
# save_settings_ui() and models_set_active_ui() - the only two places that
# mutate current_cfg wholesale. Without it, two overlapping saves (e.g.
# two browser tabs) could each read a stale current_cfg and the second
# write would silently clobber fields the first one just changed. Plain
# Lock, not RLock like the others here - neither function calls back into
# itself or the other while already holding it, so there's no reentrancy
# need.
_config_lock = threading.Lock()

# Debug tab: captures log output from our own Python code (core.* modules
# all log via the standard `logging` module, but nothing was ever attached
# to see it before this) into a bounded in-memory buffer, AND a plain text
# file - the buffer is what the Debug tab's textbox polls, the file is for
# anything that can't/didn't have the tab open when something happened
# (including, deliberately, a coding assistant working on this repo, which
# has no way to peek into this process's own live memory - unlike the
# textbox, a file is something that can just be read after the fact).
_PY_LOG_BUFFER: deque[str] = deque(maxlen=2000)
PY_LOG_PATH = Path(__file__).resolve().parent / "logs" / "app.log"


class _BufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _PY_LOG_BUFFER.append(self.format(record))


def _setup_debug_logging() -> None:
    """Attaches the buffer + file handlers below, if debug_tab_enabled -
    called once from main(), deliberately NOT at module import time. This
    module gets imported directly by ad-hoc test scripts too (this
    project's normal testing approach this session, e.g. `import app`
    from a scratch script to exercise a handler function without a live
    server) - attaching a mode="w" FileHandler at import time meant every
    such import truncated PY_LOG_PATH out from under whatever the
    actually-running app process had already written to it. Scoping
    this to the real entrypoint keeps a plain `import app` side-effect-
    free on disk."""
    if not current_cfg.debug_tab_enabled:
        return
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    buffer_handler = _BufferLogHandler()
    buffer_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(buffer_handler)

    # Truncated fresh on every app start, matching core.server's own
    # LOG_PATH (llama-server.log) convention - this is meant as a live
    # window into the CURRENT run, not an accumulating history across
    # restarts (which, this session, happen often during iteration).
    PY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(PY_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)

    # Deliberately NOT touching the root logger's level. Elevating only our
    # own namespaces to DEBUG (rather than the root, which every library's
    # logger inherits from when it has no level of its own) means Gradio's
    # httpx/httpcore/asyncio internals - and anything else - stay at their
    # own quiet default instead of flooding this with unrelated chatter.
    logging.getLogger("core").setLevel(logging.DEBUG)
    logging.getLogger("app").setLevel(logging.DEBUG)


def _stop_managed() -> None:
    """The reset below always runs, even if .stop() itself raises (e.g.
    ManagedServer.stop()'s second process.wait(10) - after the kill() -
    has no exception handling of its own, so a process that's unusually
    slow to actually exit could propagate a TimeoutExpired here). Without
    the try/except, that would skip the reset entirely and leave _session
    pointing at a stale ManagedServer/base_url - and since
    _is_server_managed_by_us() only checks whether _session matches a
    base_url, not whether the object it's holding is still meaningfully
    "ours," anything ELSE that later answers on that same port (a
    manually-started server on the same default port, most plausibly)
    would then be misreported as owned by this session. Reset first,
    worry about the process separately - our own bookkeeping about what
    we're responsible for shouldn't stay wrong just because killing it
    was slow."""
    global _expected_stop
    with _session_lock:
        if _session["managed"] is not None:
            _expected_stop = True
            try:
                _session["managed"].stop()
            except Exception:
                log.exception("Error stopping managed llama-server - clearing session anyway")
        _session["managed"] = None
        _session["client"] = None
        _session["base_url"] = None


# Registered from main() (not here, at module level) - this module gets
# imported directly by scratch/test scripts too (this project's normal
# testing approach), and an unconditional atexit.register(_stop_managed)
# at import time meant any such throwaway script that happened to populate
# _session (e.g. by calling get_client() directly while testing) would kill
# a real, wanted llama-server the moment the script's own process exited
# normally - _stop_managed() has no way to tell "a real app instance is
# shutting down" apart from "some process that imported this module
# exited", so the fix is to only ever register the hook in the former
# case. See _setup_debug_logging()'s docstring for the same lesson
# learned earlier this session, for a lower-stakes case (log truncation
# rather than killing a real process).


# ------------------------------------------------------- Server reachability
#
# "Can Single/Batch/Review's Recaption actually work right now" - a single,
# always-live condition (not a one-time first-run check), reused by
# _update_ui_status() below to gate those three actions the same way
# "something else is running" already gates them, and by the startup check
# (demo.load, near the bottom of build_app) to decide whether to land on
# the Single-image tab or on Settings with an explanation.
#
# "managed" mode has no implicit lazy-start anymore - starting the server
# is always an explicit action (the "Start llama server" button, or
# autostart at launch - see the server-lifecycle section further down),
# never a side effect of clicking Caption. So "reachable" here means
# exactly one thing for both modes: is something actually answering a
# health check right now. "Installed but not started yet" is just as
# unusable as "not installed at all" from this section's point of view -
# see core.server.ServerStatus.installed for how the two get told apart
# for status-text/button purposes elsewhere. "external" mode has no local
# fallback we control at all, so it always needs a real live health check.
#
# The mode-level categorization (including the base "controllable" -
# see core.server.ServerStatus) lives in core/server.py's check_status(),
# which never starts or stops anything, unlike resolve_server(). It's a
# real network call though (is_healthy), so it's cached here and
# refreshed on its own slower timer (see reachability_timer, further
# down) rather than the fast 2s status_timer that drives everything else -
# no reason to hit the network that often just to keep a button's
# interactive state current.
#
# Whether the Models tab is actually usable is a stricter, session-aware
# question check_status() can't answer on its own (it has no idea which
# server, if any, THIS session started) - see _cached_controllable()
# below for that narrowing, and _refresh_reachability_ui for how it
# drives the Models tab itself (disabling it, and evicting the user to
# Settings if they're on it when that happens).

_reachability_lock = threading.RLock()
_reachability_cache = ServerStatus(reachable=True, controllable=True, installed=True)  # optimistic default until the first real check


def _refresh_reachability_cache() -> ServerStatus:
    global _reachability_cache
    result = check_status(current_cfg)
    with _reachability_lock:
        _reachability_cache = result
    return result


def _cached_reachable() -> bool:
    with _reachability_lock:
        return _reachability_cache.reachable


def _is_server_managed_by_us(base_url: str) -> bool:
    """True if THIS session started (or is otherwise responsible for
    stopping) whatever's currently answering at base_url - what the status
    bar calls "running" rather than a healthy-but-unowned server. Session-
    local only; doesn't know about anything from a previous app run (see
    core/server.py's docstring on why that's on purpose)."""
    managed = _session.get("managed")
    connected = _session.get("client") is not None and _session.get("base_url") == base_url
    return connected and managed is not None


def _managed_base_url(cfg: AppConfig) -> str:
    return f"http://{MANAGED_HOST}:{cfg.server_port}"


def _cached_controllable() -> bool:
    """Whether the Models tab should be usable right now - stricter than
    ServerStatus.controllable's mode-level check (see its docstring):
    external mode is never controllable, same as before, but a running
    managed-mode server also has to be one THIS session started. A
    running server we merely reused (leftover from a previous run, or
    started by hand) might be running any model at all, and there's no
    reliable way to tell - see the module docstring's note on the removed
    PID-tracking approach - so it's treated the same as external mode:
    not ours to manage. A not-yet-running managed-mode server (that's
    otherwise installed) stays controllable, since starting it fresh
    would use whatever we pick."""
    with _reachability_lock:
        status = _reachability_cache
    if not status.controllable:
        return False
    cfg = current_cfg
    if cfg.server_mode == "managed" and status.reachable:
        return _is_server_managed_by_us(_managed_base_url(cfg))
    return True


_last_known_controllable = True  # matches the optimistic cache default above
_last_known_reachable: Optional[bool] = None  # None = no real check has completed yet
_down_is_crash = False  # persists for a whole down-streak, not just its first tick - see _update_reachability_tracking
_expected_stop = False  # set by _stop_managed() itself, consumed by _update_reachability_tracking - see both docstrings


def _note_controllable_transition(controllable: bool, cfg: AppConfig, status: ServerStatus) -> None:
    """Fires exactly one gr.Info the moment the Models tab becomes
    unusable (not on every reachability_timer tick for as long as it stays
    that way), whether that's discovered at startup or later. Also aborts
    any in-flight/queued downloads at that same moment (a deliberate
    reversal of the earlier "let them finish" behavior - decided this
    session that switching away from a controllable state likely means
    those downloads are no longer wanted at all) - done here, not gated
    on which tab the user happens to be viewing, since the download queue
    itself doesn't care whether Models is the active tab right now.

    The message is picked from cfg/status rather than one fixed string -
    confirmed live that the "isn't one this app started" wording is
    flatly wrong for managed-mode-but-nothing-installed: there's no
    active llama-server to have started or not started in that case, the
    reachability warning already covers "nothing's set up yet" on its
    own, so this only needs to add the ownership-specific explanation
    when there's actually something running to have an opinion about."""
    global _last_known_controllable
    if not controllable and _last_known_controllable:
        if cfg.server_mode == "managed" and not status.installed:
            _notify("Model management is disabled: llama.cpp isn't installed yet, so there's nothing to manage.")
        else:
            _notify(
                "Model management is disabled: the active llama-server isn't one this app "
                "started, so what it currently has loaded isn't something the Models tab can "
                "show you or change."
            )
        _download_abort_all()
    _last_known_controllable = controllable


def _update_reachability_tracking(status: ServerStatus, cfg: AppConfig) -> None:
    """Updates _last_known_reachable/_down_is_crash and fires exactly one
    popup for a genuinely NEW, UNEXPECTED transition into unreachable -
    never a repeat every poll while it stays that way, and never at all
    for a transition _stop_managed() itself caused (see _expected_stop -
    stopping a server we manage is never a crash by definition, whether
    that's the explicit End button, a mode switch, or a settings-save
    that needs to restart it; _stop_managed() is the only thing that ever
    calls ManagedServer.stop(), so any call to it is inherently
    deliberate). Must run before _status_label() on the same poll, since
    the label reads _down_is_crash.

    Three cases on a transition into "not reachable":
      - was_reachable is True and _expected_stop is False (a real crash:
        it was answering a moment ago, managed+installed, now isn't, and
        nothing here asked for that) -> the specific "it may have
        crashed" warning, and _down_is_crash stays True for the rest of
        this down-streak (not just this one tick).
      - was_reachable is True and _expected_stop is True (we stopped it
        ourselves) -> no warning, _down_is_crash stays False - same
        "idle" label as a server that was simply never started, not
        "error".
      - was_reachable is None (the very first check this process has ever
        made) -> a freshly-installed-but-never-started managed server is
        just the normal starting state, not an alarm - no popup at all in
        that specific case; everything else (not installed, external down
        from the start) still gets the plain informational warning.
      - was_reachable is False (already known down, still down) -> no
        popup, nothing changed.
    Recovering (reachable again) always clears _down_is_crash and
    _expected_stop."""
    global _last_known_reachable, _down_is_crash, _expected_stop
    was_reachable = _last_known_reachable
    _last_known_reachable = status.reachable
    if status.reachable:
        _down_is_crash = False
        _expected_stop = False
        return
    managed_installed = cfg.server_mode == "managed" and status.installed
    if was_reachable:
        expected = _expected_stop
        _expected_stop = False
        _down_is_crash = not expected
        if _down_is_crash:
            if managed_installed:
                _notify("The managed llama-server stopped responding - it may have crashed. Restart it from Settings.", level="warning")
            else:
                _notify(f"No captioning server is reachable - {status.reason} Set it up on the Settings tab.", level="warning")
    elif was_reachable is None and not managed_installed:
        _notify(f"No captioning server is reachable - {status.reason} Set it up on the Settings tab.", level="warning")
    # was_reachable is False (steady-state down), or None+managed_installed
    # (fresh idle) - no popup either way.


def _status_label(cfg: AppConfig, reachable: bool, installed: bool) -> str:
    """The status-bar's server-state label. Decoupled from ServerStatus
    (just the two fields it actually needs) since its one caller,
    get_status_text(), does its own fresh/uncached health check rather
    than reading the reachability cache - see that function's own
    comment on why the status bar's "Model: X" needs to always be fresh,
    not cached. Reads _down_is_crash (module state, maintained by
    _update_reachability_tracking() on the separate 8s reachability-cache
    cycle) rather than tracking its own transition history - sharing that
    one flag keeps this fast/frequent 2s-driven label consistent with the
    slower cycle's crash detection instead of re-deciding it twice.
    external: connected/unreachable. managed: running, N/A (not
    installed), error (installed, was running, now isn't - likely
    crashed), or idle (installed, never got started, or cleanly stopped -
    not alarming)."""
    if cfg.server_mode == "external":
        return "connected" if reachable else "unreachable"
    if reachable:
        return "running"
    if not installed:
        return "N/A"
    return "error" if _down_is_crash else "idle"


def _refresh_reachability_ui(current_tab_label: str):
    """Wired to reachability_timer.tick(). Refreshes the cache (picked up
    by the next status_timer tick for Single/Batch/Recaption's own Run/
    Recaption BUTTON gating, see the cache's own comment above) and
    separately updates three TABS themselves: Models (disabled unless
    controllable - see _cached_controllable) and Single-image/Batch
    processing (disabled unless reachable at all - captioning there is
    pointless, not just "not ours to redirect", the moment nothing can
    answer). If the user is on one of these when it becomes disabled,
    pushes them off to Settings rather than leaving them stranded on a tab
    they can no longer switch back into. Never touches main_tabs'
    selection otherwise, so it can't yank anyone off Review mid-task -
    Review's own filesystem-only nav doesn't need a server at all (see
    _update_ui_status' docstring), so its tab stays reachable regardless;
    only its Recaption button is gated, same as Single/Batch's Run
    buttons.

    Also updates current_tab_label_state to "Settings" whenever it
    redirects there - main_tabs.select() (which is what normally keeps
    that state current, see _on_main_tab_select) only fires on an actual
    user click, never on a server-pushed gr.update(selected=...) like the
    one this function itself issues. Without this, the tracked label
    would silently go stale the moment THIS function redirects someone,
    and every later poll would keep reading the pre-redirect tab -
    confirmed live: this exact gap was yanking users from Settings->Llama
    to Settings->Debug a few seconds later, via _fallback_tab_safety_ui
    still seeing the stale "Single image"/"Batch processing" label and
    concluding they were still stuck on a disabled tab. gr.skip() leaves
    it untouched on ticks that don't redirect."""
    cfg = current_cfg
    status = _refresh_reachability_cache()
    controllable = _cached_controllable()
    _note_controllable_transition(controllable, cfg, status)
    _update_reachability_tracking(status, cfg)
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = _llama_lifecycle_button_updates(cfg, status)
    push_to_settings = (not controllable and current_tab_label == "Models") or (
        not status.reachable and current_tab_label in ("Single image", "Batch processing")
    )
    tabs_update = gr.update(selected="settings") if push_to_settings else gr.update()
    label_update = "Settings" if push_to_settings else gr.skip()
    return (
        models_tab_update, single_batch_update, single_batch_update,
        start_btn_update, end_btn_update, tabs_update, label_update,
    )


def _on_main_tab_select(evt: gr.SelectData) -> str:
    """Wired to main_tabs.select() - see current_tab_label_state's own
    comment for why this is tracked at all."""
    return evt.value


def _goto_llama_settings_ui():
    """The Models tab's "Manage llama server" button's click target -
    pure navigation to Settings -> Llama, no server action of its own
    (see this module's own docstring for why that's safe: the Start/End
    buttons there are already gated off while a job's running, so this
    can't be used to bypass anything the way an in-place restart-from-
    Models-tab could).

    Explicitly asserts every destination (main_tabs, settings_tabs,
    current_settings_subtab_state, current_tab_label_state) rather than a
    bare gr.update() no-op on any of them - _refresh_reachability_ui's own
    docstring covers why a no-op can't be trusted to self-correct a
    still-pending prior push to a Tabs component's selected=, and
    current_tab_label_state/current_settings_subtab_state both need to
    reflect this redirect immediately since neither main_tabs.select() nor
    settings_tabs.select() reliably fire on a server-pushed switch like
    this one (see their own comments)."""
    return gr.update(selected="settings"), gr.update(selected="llama-settings"), "llama-settings", "Settings"


def _llama_lifecycle_button_updates(cfg: AppConfig, status: ServerStatus):
    """(start_btn_update, end_btn_update) for the Llama settings tab's
    two lifecycle buttons - shared by _refresh_reachability_ui and
    _startup_reachability_ui so the two can't disagree. Start is only
    ever useful in managed mode with something installed but not already
    running (installed-and-not-running, not installed-and-running or
    external, where it'd have nothing to do); End only when this session
    actually owns whatever's currently running - see
    _is_server_managed_by_us's docstring on why that's the one thing that
    gates it, not just "something's running".

    Start also tracks Save-settings' primary/orange styling, but only
    while it's actually clickable - variant, not just interactive, so
    "ready to click" reads as visually distinct from "nothing to do right
    now" rather than relying on interactive=False's subtler dimming alone
    to carry that. End stays plain (like Restart app) regardless of
    state - it's the less common action, not the one to draw the eye to."""
    can_start = cfg.server_mode == "managed" and status.installed and not status.reachable
    start_update = gr.update(interactive=can_start, variant="primary" if can_start else "secondary")
    owned = cfg.server_mode == "managed" and status.reachable and _is_server_managed_by_us(_managed_base_url(cfg))
    end_update = gr.update(interactive=owned)
    return start_update, end_update


def _startup_reachability_ui():
    """Wired to demo.load() - decides which tab a freshly-loaded page
    lands on, and the Models/Single-image/Batch-processing tabs' initial
    interactive state (see _refresh_reachability_ui, which this shares its
    per-tab logic with). A real (not cached) check, since the optimistic
    default cache value would otherwise land everyone on Single-image once
    with everything enabled, even on a machine where nothing's been set up
    yet. Also sets current_tab_label_state to match wherever it actually
    lands - the very first value driving every later poll, before any
    real main_tabs.select() has ever fired (see _refresh_reachability_ui's
    docstring for why this matters and what silently goes wrong without
    it)."""
    cfg = current_cfg
    status = _refresh_reachability_cache()
    controllable = _cached_controllable()
    _note_controllable_transition(controllable, cfg, status)
    _update_reachability_tracking(status, cfg)
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = _llama_lifecycle_button_updates(cfg, status)
    tab_target, tab_label = ("single", "Single image") if status.reachable else ("settings", "Settings")
    return (
        gr.update(selected=tab_target), models_tab_update, single_batch_update, single_batch_update,
        start_btn_update, end_btn_update, tab_label,
    )


def _fallback_tab_safety_ui(current_tab_label: str):
    """Cheap, cache-only 2s-timer safety net: if the user is currently
    sitting on a tab the cached state says is disabled, push them to
    Settings -> Debug (the one sub-tab that's never gated on anything -
    see its own comment) regardless of why. A catch-all in case the
    primary per-transition push logic in _refresh_reachability_ui/
    _startup_reachability_ui missed a case - reads the same caches those
    already refresh, no network call of its own, safe to run this often.
    Also updates current_tab_label_state to "Settings" when it redirects,
    same reasoning as _refresh_reachability_ui - a server-pushed tab
    switch doesn't fire main_tabs.select() on its own."""
    controllable = _cached_controllable()
    reachable = _cached_reachable()
    on_disabled_models = current_tab_label == "Models" and not controllable
    on_disabled_single_batch = current_tab_label in ("Single image", "Batch processing") and not reachable
    if on_disabled_models or on_disabled_single_batch:
        return gr.update(selected="settings"), gr.update(selected="debug-settings"), "Settings"
    return gr.update(), gr.update(), gr.skip()


def _autostart_managed_llama_ui():
    """Chained onto demo.load AFTER the initial UI has already rendered
    with tabs correctly enabled/disabled (see the demo.load chain below) -
    autostart is a background convenience layered on top of that correct
    initial state, never a substitute for it. Only actually does anything
    for managed mode, installed, not already running, and the checkbox
    on - exactly "state 2" (see the state table this was designed
    against), the one case a user would otherwise have to click Start for
    themselves. Silent no-op in every other case (external mode, nothing
    installed, already running, or autostart off) - no popup for "didn't
    autostart", only for actually doing something. Reads the cache
    _startup_reachability_ui (immediately before this in the same chain)
    already refreshed - no extra network call just to decide whether to
    bother. Also sets current_tab_label_state to "Single image" when it
    redirects there - same reasoning as the other tab-pushing functions
    above (a server-pushed switch doesn't fire main_tabs.select() on its
    own; see _refresh_reachability_ui's docstring for the bug this class
    of fix addresses)."""
    cfg = current_cfg
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.skip())
    if cfg.server_mode != "managed" or not cfg.autostart_managed_llama:
        return noop
    with _reachability_lock:
        status = _reachability_cache
    if not status.installed or status.reachable:
        return noop
    _notify("Starting managed llama-server automatically (autostart is enabled)...")
    ok, _ = _try_start_managed_llama()
    if not ok:
        return noop
    with _reachability_lock:
        status = _reachability_cache
    controllable = _cached_controllable()
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = _llama_lifecycle_button_updates(cfg, status)
    return (
        models_tab_update, single_batch_update, single_batch_update,
        start_btn_update, end_btn_update, gr.update(selected="single"), "Single image",
    )


def _autoload_hydra_model_ui():
    """Chained onto the END of the same demo.load chain
    _autostart_managed_llama_ui is already on (after it, never before or
    in parallel) - deliberately, so that when both autostart-llama and
    autoload-Hydra are enabled, Hydra's load attempt happens against
    whatever VRAM llama-server's own autostart already claimed. That
    ordering is the actual real-world scenario worth exercising (see
    AppConfig.hydra_autoload_model's own docstring) - loading Hydra
    first, then llama-server, would never surface the coexistence
    problem this is meant to test for. Silent no-op unless hydra_enabled,
    hydra_autoload_model, deps+model are both present, and nothing is
    already loaded - same "no popup for didn't-autostart" convention as
    _autostart_managed_llama_ui itself. A real load failure (e.g. a CUDA
    OOM) is still surfaced via gr.Info + the status text, exactly like a
    manual Load click would."""
    cfg = current_cfg
    noop = (gr.update(), gr.update())
    if not (cfg.hydra_enabled and cfg.hydra_autoload_model):
        return noop
    st = hydra_classifier.status()
    if not (st.deps_installed and st.model_downloaded) or st.loaded:
        return noop
    _notify("Loading Hydra model automatically (autoload is enabled)...")
    try:
        hydra_classifier.load(cfg)
        message = f"Hydra model loaded on {hydra_classifier.status().device}."
    except hydra_classifier.HydraError as exc:
        message = f"Couldn't autoload Hydra model: {exc}"
    return gr.update(value=message), gr.update(value=_hydra_status_text())


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
    active operation (or None). Used by _update_ui_status, which
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


def _update_ui_status():
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
    a third kind).

    review_recaption_btn's own interactive state follows the same
    "blocked by something else" pattern single_run_btn/batch_run_btn
    already use - recaptioning genuinely can't run concurrently with
    Single/Batch (all three share the one llama-server connection), so
    it needs to be visibly disabled the same way, not just refuse with a
    message after being clicked.

    All three Run/Recaption buttons are ALSO blocked when no server is
    reachable right now (_cached_reachable(), see the Server reachability
    section above) - starting a caption request when there's nothing to
    answer it is exactly as pointless as starting a second operation
    while one's already running, so it's gated the same way. This is why
    the check is cached rather than computed fresh here: it involves a
    real network call for external mode, and this function runs on every
    2s status_timer tick.

    Review's OTHER nav (prev/next/table/dir/browse/scan), though, is
    pure filesystem/UI - scanning or browsing a folder doesn't touch the
    llama-server at all, so there's no real conflict with a Single/Batch
    job running elsewhere. It's disabled only while REVIEW ITSELF is
    recaptioning (op.kind == "review"), not whenever anything anywhere
    is active - browsing a different, already-tagged folder while a
    Batch job elsewhere is still running is a normal, useful thing to
    want to do.

    Returns (single_run_btn, single_run_col, single_interrupt_col,
    single_interrupt_btn, batch_run_btn, batch_run_col,
    batch_interrupt_col, batch_interrupt_btn, review_recaption_btn,
    review_recaption_col, review_interrupt_col, review_interrupt_btn,
    review_prev_btn, review_next_btn, review_table, review_dir,
    review_browse_btn, review_scan_btn).
    """
    reachable = _cached_reachable()
    with _operation_lock:
        op = _active_operation
        single_blocked = (op is not None and op.kind != "single") or not reachable
        single_running = op is not None and op.kind == "single"
        single_label, single_ok = _interrupt_label_state(op, "single")
        batch_blocked = (op is not None and op.kind != "batch") or not reachable
        batch_running = op is not None and op.kind == "batch"
        batch_label, batch_ok = _interrupt_label_state(op, "batch")
        review_blocked = (op is not None and op.kind != "review") or not reachable
        review_running = op is not None and op.kind == "review"
        review_label, review_ok = _interrupt_label_state(op, "review")
        review_nav = _REVIEW_NAV_BUSY if review_running else _REVIEW_NAV_IDLE

    return (
        gr.update(interactive=not single_blocked),
        gr.update(visible=not single_running),
        gr.update(visible=single_running),
        gr.update(value=single_label, variant="stop", interactive=single_ok),
        gr.update(interactive=not batch_blocked),
        gr.update(visible=not batch_running),
        gr.update(visible=batch_running),
        gr.update(value=batch_label, variant="stop", interactive=batch_ok),
        gr.update(interactive=not review_blocked),
        gr.update(visible=not review_running),
        gr.update(visible=review_running),
        gr.update(value=review_label, variant="stop", interactive=review_ok),
        *review_nav,
    )


def _settings_subtabs_disabled() -> bool:
    """The one, sole question of whether the four operation-sensitive
    Settings sub-tabs (Llama, Hydra, Image resizing, Captioning defaults)
    are currently off-limits - today that's just "is a job running", but
    kept as its own function (not inlined into _settings_gating_ui or
    _settings_subtab_entry_ui, both of which call this rather than
    checking _active_operation themselves) so both of those callers can
    never drift apart on what "disabled" means, even if a future reason
    to disable these sub-tabs gets added later - it only has to change
    here."""
    with _operation_lock:
        return _active_operation is not None


def _settings_gating_ui():
    """Disables the operation-sensitive Settings sub-tabs' own
    clickability (see _settings_subtabs_disabled) while a Single/Batch/
    Recaption job is actively running, so a job's settings can't change
    out from under it mid-flight. The Debug sub-tab is deliberately
    excluded - its one checkbox already needs a full app restart to take
    effect regardless (see its own comment), so there's nothing it could
    disrupt by staying editable. Deliberately does NOT touch the fields
    inside those sub-tabs, only the tabs' own interactive= - see
    _settings_subtab_entry_ui below for what actually protects someone
    already sitting on one of these when a job starts (a redirect on
    next entry, not disabling the fields themselves)."""
    update = gr.update(interactive=not _settings_subtabs_disabled())
    return update, update, update, update


# gr.SelectData.value for a Tab is ALWAYS its label, never its id -
# confirmed via gradio/layouts/tabs.py's own EVENTS docstring ("value
# referring to the label of the Tab"), and via a live debug-log capture
# that showed "last_chosen=Hydra" (the label) instead of the expected
# "hydra-settings" (the id). id is otherwise only used for the selected=
# kwarg when PUSHING a tab switch (gr.Tab's own docstring: "required if
# you wish to control the selected tab from a predict function") - so
# without this table, a value read off a real click could never be fed
# back into gr.update(selected=...) and have it actually match anything.
# Three of the five sub-tabs (Hydra, Image resizing, Captioning
# defaults) never had an id at all until this fix, which independently
# meant they could never have been a valid push target either way - see
# their own gr.Tab(...) declarations.
_SETTINGS_SUBTAB_ID_BY_LABEL = {
    "Llama": "llama-settings",
    "Hydra": "hydra-settings",
    "Image resizing": "image-resizing-settings",
    "Captioning defaults": "captioning-defaults-settings",
    "Debug": "debug-settings",
}


def _on_settings_subtab_select(evt: gr.SelectData):
    """Wired to settings_tabs.select() - tracks the sub-tab the user
    actually, genuinely clicked into (current_settings_subtab_state) as
    a stable id (see _SETTINGS_SUBTAB_ID_BY_LABEL above - evt.value
    itself is a label, not usable directly), separately from whatever
    settings_tabs is currently DISPLAYING. Read by
    _settings_subtab_entry_ui below.

    Also guards against recording "debug-settings" while the sub-tabs
    are disabled: it's the only place a real click could land during
    that window anyway (everything else is unclickable), so there's
    nothing informative to record, and gr.skip() leaves whatever was
    genuinely there before (e.g. "hydra-settings") untouched. This
    guard's real necessity - i.e. whether a select event ever actually
    fires here that ISN'T a genuine click, such as a side effect of
    _settings_subtab_entry_ui's own redirect - was never independently
    confirmed; it's kept as cheap, harmless insurance regardless, now
    that it's at least comparing against the right kind of value (an id,
    not a label) to ever have a chance of matching."""
    tab_id = _SETTINGS_SUBTAB_ID_BY_LABEL[evt.value]
    disabled = _settings_subtabs_disabled()
    if tab_id == "debug-settings" and disabled:
        log.debug("settings subtab select: label=%s id=%s disabled=%s -> IGNORED", evt.value, tab_id, disabled)
        return gr.skip()
    log.debug("settings subtab select: label=%s id=%s disabled=%s -> RECORDED", evt.value, tab_id, disabled)
    return tab_id


def _settings_subtab_entry_ui(last_chosen: str):
    """Wired to settings_tab.select() - the moment the user enters the
    Settings tab (from anywhere else), not on a timer: confirmed live
    that someone already sitting on a sub-tab when a job starts is left
    alone (matches _settings_gating_ui only disabling the tabs' own
    clickability, not evicting an existing occupant), so there's nothing
    for this to correct except at the moment of a fresh entry.

    Always explicitly asserts a destination - either "debug-settings" or
    last_chosen - never a bare gr.update() no-op: a no-op was confirmed
    live NOT to reliably undo a still-pending prior push to a Tabs
    component's selected= (see git history/session notes for the
    original bug this replaced), so "leave it alone and hope it already
    reverted" isn't trustworthy here regardless of which branch applies.

    Because this recomputes _settings_subtabs_disabled() fresh on every
    single entry rather than remembering "did I force this last time",
    there's no flag to clear and nothing that can go stale: if the job
    ended 5 seconds ago or 5 minutes ago, disabled is simply False now,
    and the else branch below re-shows last_chosen - whatever that
    genuinely was, Debug included if that's really where the user left
    off - with no separate revert-to-Llama special case needed."""
    disabled = _settings_subtabs_disabled()
    if disabled and last_chosen != "debug-settings":
        log.debug("settings subtab entry: last_chosen=%s disabled=%s -> pushing debug-settings", last_chosen, disabled)
        return gr.update(selected="debug-settings")
    log.debug("settings subtab entry: last_chosen=%s disabled=%s -> pushing %s", last_chosen, disabled, last_chosen)
    return gr.update(selected=last_chosen)


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
            desired_base = _managed_base_url(cfg)

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


def _try_start_managed_llama() -> tuple[bool, str]:
    """Attempts to start (or confirm-reuse) the managed llama-server for
    the current config, blocking until ready or failed. Shared by the
    "Start llama server" button, autostart-at-launch, and a settings-save
    that needs to restart with new server-process args - one
    implementation of "make sure a working managed connection exists,"
    not three that could drift apart. Reuses get_client() - the exact
    same connection-resolution Single/Batch/Recaption already use,
    including its existing resolve_selection() model validation, so a
    missing/invalid model selection is reported the same way here as it
    would be from any other entry point."""
    try:
        get_client(current_cfg)
    except ServerError as exc:
        return False, f"Couldn't start llama-server: {exc}"
    _refresh_reachability_cache()
    return True, "llama-server is running."


def _start_managed_llama_ui():
    """A generator, not a plain function - starting can take a while
    (model load), and a single return value would leave the status box
    silent/frozen for the whole duration with no sign anything's
    happening (the chained _refresh_reachability_ui after this click only
    fires once this generator is fully exhausted, so it can't help with
    mid-wait feedback either - same pattern already used for
    run_single_ui's "Starting server (loading model)..." yield, for the
    exact same reason)."""
    yield "Starting llama-server (this can take a while if a model needs to load)..."
    _, message = _try_start_managed_llama()
    yield message


def _end_managed_llama_ui():
    """Only ever meaningfully clickable when _is_server_managed_by_us() is
    true (see its interactive= wiring below) - but _stop_managed() is
    itself already safe to call regardless, since it only ever touches
    _session["managed"], which is only ever non-None when this session
    is the one that actually spawned the process (see core/server.py's
    docstring). The ownership check here is just to report accurately
    what happened for a stale/racing click, not to gate the action. A
    generator for the same reason _start_managed_llama_ui is - stopping
    can take up to ManagedServer.stop()'s own 10s wait before it falls
    back to a hard kill."""
    yield "Stopping llama-server..."
    owned = _is_server_managed_by_us(_managed_base_url(current_cfg))
    _stop_managed()
    _refresh_reachability_cache()
    yield "llama-server stopped." if owned else "Nothing to stop - this session doesn't own the running server."


def _verify_external_ui(external_url: str) -> str:
    """Tests whatever URL is currently in the External server URL field -
    not necessarily current_cfg.external_url, since the user may be
    checking a value they haven't saved yet. A real check beyond a bare
    /health 200: also asks /props (get_loaded_model_name, core/server.py -
    the same call the status bar's "Model: X" already uses) so the result
    says what model is actually loaded there, not just that *something*
    answered."""
    base_url = external_url.rstrip("/")
    if not is_healthy(base_url):
        return f"No server responding at {base_url}."
    model = get_loaded_model_name(base_url)
    if model:
        return f"Reachable - model loaded: {model}."
    return f"Reachable at {base_url}, but couldn't read which model is loaded (no /props response)."


def restart_app_ui() -> None:
    """Replaces this process with a fresh one via os.execv - the standard
    Python self-restart trick, so newly-saved settings.json values (or a
    just-toggled Debug tab) take effect without anyone needing a terminal.
    Never returns: the process image is gone the moment execv succeeds, so
    stop our own managed llama-server first (execv skips atexit entirely),
    and abort any in-flight/queued downloads or llama.cpp install too -
    all in-memory only and won't survive this regardless, better to say
    so cleanly. Also unload Hydra if loaded - unlike llama-server (a
    separate OS process _stop_managed kills outright), Hydra's model
    lives in THIS process's own CUDA context, which execv does not
    clean up on its own (it replaces the process image but skips normal
    interpreter/CUDA teardown) - explicit unload() first avoids leaking
    VRAM into the replaced process.
    """
    log.info("User triggered app restart from Settings")
    _operation_force_abort()
    _wait_for_operation_to_end()
    _download_abort_all()
    _install_abort_all()
    _stop_managed()
    hydra_classifier.unload()
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------- Single tab

# Interrupt is a SEPARATE component from single_run_btn on purpose - see
# the Operation tracking section: an Interrupt sharing the same
# button/event as the long-running call would sit queued behind it and
# never actually reach the server while captioning is in flight. It gets
# its own Column (single_interrupt_col) rather than sharing Caption's, so
# swapping which one is visible reuses the plain-two-Column-Row shape
# confirmed to align correctly with the row above - see the Single-image
# tab and _update_ui_status' docstring for the full story.
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

        # A plain blocking caption_image() call can't yield mid-call, so
        # showing a live stage change ("captioning" -> "tagging (Hydra)")
        # in single_infotext needs the same background-thread + queue
        # shape run_batch_ui already uses - the queue is what lets on_stage
        # (called synchronously inside caption_image(), from this worker
        # thread) reach a yield in this generator without blocking on it.
        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def on_stage_cb(stage: str) -> None:
            q.put(stage)

        def worker():
            try:
                result_holder["caption"], result_holder["result"] = caption_image(
                    image_path, client, cfg, trigger_word=trigger_word_override,
                    on_stage=on_stage_cb,
                )
            except ClientError as exc:
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            stage = q.get()
            if stage is None:
                break
            yield gr.update(), f"{stage}...", gr.update(), gr.update(), gr.update(), gr.update()
        thread.join()

        if "error" in result_holder:
            log.warning("Single-image caption failed for %s: %s", image_path, result_holder["error"])
            yield "", f"Captioning failed: {result_holder['error']}", *idle_state
            return

        caption = result_holder["caption"]
        result = result_holder["result"]
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
        total = len(images)

        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def progress_cb(i, total_, path, status, caption, resize_note):
            q.put(("progress", i, path, status, caption, resize_note))

        def on_stage_cb(path: Path, stage: str) -> None:
            q.put(("stage", path, stage))

        def worker():
            try:
                result_holder["result"] = run_batch(
                    directory, client, current_cfg,
                    recursive=recursive,
                    overwrite=overwrite,
                    trigger_word=trigger_word_override,
                    progress_cb=progress_cb,
                    should_stop=lambda: _operation_should_stop("batch"),
                    on_stage=on_stage_cb,
                )
            except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash app
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        start_time = time.monotonic()
        thread.start()

        # processed/current_file/current_stage/resize_suffix all feed one
        # combined infotext line, rebuilt on every queue item (stage or
        # progress) - a stage-only line was tried first and reverted: the
        # progress line that follows it lands a moment later and replaces
        # it outright, so the "N/total processed · avg · ETA" context was
        # only ever visible for a blink between two images. One line that
        # always carries the fullest currently-known context avoids that.
        last_file = ""
        last_image = None
        last_caption = ""
        processed = 0
        current_file = ""
        current_stage: Optional[str] = None
        resize_suffix = ""

        def infotext_text() -> str:
            elapsed = time.monotonic() - start_time
            avg = elapsed / processed if processed else 0.0
            eta = avg * (total - processed)
            text = current_file or f"{total} image(s)"
            if resize_suffix:
                text += f" · resized {resize_suffix}"
            text += f" · {processed}/{total} processed · avg {avg:.1f}s/image · ETA {_format_duration(eta)}"
            if current_stage:
                text += f" - {current_stage}"
            return text

        while True:
            item = q.get()
            if item is None:
                break
            if item[0] == "stage":
                _, path, stage = item
                if path.name != current_file:
                    # A new image started - the previous one's resize
                    # note (if any) no longer applies to what's showing.
                    resize_suffix = ""
                current_file = path.name
                current_stage = stage
                yield infotext_text(), last_file, last_image, last_caption, gr.update(), gr.update(), gr.update(), gr.update()
                continue

            _, i, path, status, caption, resize_note = item
            processed = i
            current_file = path.name
            current_stage = None  # between images - no stage in flight until the next on_stage
            resize_suffix = resize_note or ""

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
            yield infotext_text(), last_file, last_image, last_caption, gr.update(), gr.update(), gr.update(), gr.update()

        thread.join()
        if "error" in result_holder:
            message = f"Unexpected error: {result_holder['error']}"
            _notify(f"Batch captioning: {message}", level="error")
            yield message, last_file, last_image, last_caption, *idle_state
        else:
            r = result_holder["result"]
            total_elapsed = time.monotonic() - start_time
            verb = "Aborted" if r.aborted else "Done"
            summary = (
                f"{verb}: {r.processed} captioned, {r.truncated} truncated, "
                f"{r.skipped} skipped, {r.failed} failed "
                f"in {_format_duration(total_elapsed)}"
            )
            # Fires even if the user's still watching the Batch tab - a
            # long batch is exactly the kind of thing this was asked for
            # (start it, go do something else, still find out when it's
            # done or hit trouble without having to stay on this tab).
            _notify(f"Batch captioning {summary}", level="warning" if r.failed else "info")
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
# Single-image tab makes) - see _update_ui_status' docstring for
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
# button. See _review_nav_state below and _update_ui_status.
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

        # Same background-thread + queue shape as run_single_ui/
        # run_batch_ui - see run_single_ui's comment for why a plain
        # blocking caption_image() call can't yield a live stage change
        # on its own.
        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def on_stage_cb(stage: str) -> None:
            q.put(stage)

        def worker():
            try:
                result_holder["caption"], result_holder["result"] = caption_image(
                    item.path, client, cfg, trigger_word=None,
                    on_stage=on_stage_cb,
                )
            except ClientError as exc:
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            stage = q.get()
            if stage is None:
                break
            yield current_caption, f"{stage}...", *_REVIEW_STATE_NOOP
        thread.join()

        if "error" in result_holder:
            log.warning("Review recaption failed for %s: %s", item.path, result_holder["error"])
            yield current_caption, f"Recaptioning failed: {result_holder['error']}", *idle_state
            return

        caption = result_holder["caption"]
        result = result_holder["result"]
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
    server_mode, server_port, external_url,
    n_gpu_layers, context_size, extra_server_args, autostart_managed_llama,
    resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
    prompt_template, temperature, top_p, max_tokens, request_timeout,
    trigger_word, overwrite_existing, recursive_batch, debug_tab_enabled,
    hydra_enabled, hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
    hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_autoload_model,
) -> str:
    """Per-category behavior on save (see the design discussion this was
    built from): image resizing/captioning defaults/prompt-template-bundle
    fields are just written through, live immediately, no process
    interaction - already true by construction, nothing special needed
    for them below. Server-process-affecting fields (mode, port, GPU
    layers, context size, extra args) each need a specific reaction:
    switching to external stops any managed server we own; switching to
    managed never auto-starts (the user must do that explicitly, even
    with autostart on - autostart is for the *next app launch*, not this
    save); changing the process args while staying managed stops the
    current one (a no-op if we don't own it - see _stop_managed) and,
    only if autostart is enabled, immediately starts a fresh one with the
    new settings."""
    global current_cfg
    old_cfg = current_cfg
    # model_name/mmproj_name are deliberately NOT settable from this form -
    # they live entirely in the Models tab now (models_set_active_ui), so
    # carry forward whatever's already configured rather than defaulting
    # to empty just because this form has no field for them.
    with _config_lock:
        current_cfg = AppConfig(
            server_mode=server_mode,
            server_port=int(server_port),
            external_url=external_url,
            model_name=current_cfg.model_name,
            mmproj_name=current_cfg.mmproj_name,
            n_gpu_layers=n_gpu_layers.strip() or "auto",
            context_size=int(context_size),
            extra_server_args=extra_server_args,
            autostart_managed_llama=bool(autostart_managed_llama),
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
            hydra_enabled=bool(hydra_enabled),
            hydra_device=hydra_device,
            hydra_metric=_hydra_sliders_to_metric(float(hydra_confidence), float(hydra_threshold)),
            hydra_implications=hydra_implications,
            hydra_exclude_categories=hydra_exclude_categories,
            hydra_exclude_tags=hydra_exclude_tags,
            hydra_max_tags=int(hydra_max_tags),
            hydra_autoload_model=bool(hydra_autoload_model),
        )
        new_cfg = current_cfg
        config_mod.save(new_cfg)
    log.info(
        "Settings saved: server_mode=%s model=%s ngl=%s",
        new_cfg.server_mode, new_cfg.model_name or "(none)", new_cfg.n_gpu_layers,
    )

    switched_to_external = old_cfg.server_mode != "external" and new_cfg.server_mode == "external"
    switched_to_managed = old_cfg.server_mode == "external" and new_cfg.server_mode == "managed"
    process_args_changed = (
        old_cfg.server_port, old_cfg.n_gpu_layers, old_cfg.context_size, old_cfg.extra_server_args
    ) != (
        new_cfg.server_port, new_cfg.n_gpu_layers, new_cfg.context_size, new_cfg.extra_server_args
    )

    if switched_to_external:
        _stop_managed()
        message = "Settings saved. Switched to external mode."
    elif switched_to_managed:
        message = (
            "Settings saved. Switched to managed mode - click \"Start llama server\" "
            "in this tab when you're ready (autostart applies on the next app launch, not now)."
        )
    elif new_cfg.server_mode == "managed" and process_args_changed:
        _stop_managed()
        if new_cfg.autostart_managed_llama:
            _, start_message = _try_start_managed_llama()
            message = f"Settings saved. {start_message}"
        else:
            message = "Settings saved. Click \"Start llama server\" to apply the new server settings."
    else:
        message = "Settings saved."

    return message


# ----------------------------------------------------------- Download queue
#
# Separate from _active_operation/_operation_lock above on purpose -
# downloading a curated model shouldn't block captioning with whatever's
# already loaded, or vice versa, so this gets its own lock and its own
# single background worker thread rather than reusing that mutex. Not
# persisted across a restart (in-memory only, by design - nothing here
# needs to survive the app process ending); restart_app_ui() below calls
# _download_abort_all() before its os.execv for the same reason
# _operation_force_abort() gets called there - don't leave a background
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
# _download_status_ui tick - a list, not a single slot like
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


def _download_enqueue(items: list[DownloadItem]) -> list[DownloadItem]:
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


def _download_abort_all() -> None:
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


def _download_status_ui():
    """(row_visibility_update, html_update) - wired to the same 2s
    status_timer as everything else, plus called directly after any
    action that changes the queue so the row responds immediately
    instead of waiting for the next tick. Also fires a one-shot _notify()
    popup per completed/failed download popped from _download_announces -
    this function is wired to status_timer.tick twice (once for the
    Models tab's row, once for Hydra's), but draining the list under
    _download_lock means only whichever of the two calls runs first in a
    given tick actually finds anything to announce, so a finished
    download still only pops one popup, not two."""
    with _download_lock:
        pending = list(_download_announces)
        _download_announces.clear()
    for message, level in pending:
        _notify(message, level=level)

    html = _download_status_html()
    if html is None:
        return gr.update(visible=False), gr.update(value="")
    return gr.update(visible=True), gr.update(value=html)


# --------------------------------------------------------- llama.cpp install
#
# A single-operation worker, not a queue (only one llama.cpp install ever
# makes sense at a time) - its own lock/thread/abort-flag rather than
# reusing _download_lock above, because an install overwrites the exact
# binary/DLLs a running llama-server.exe might have open, which is a
# fundamentally different hazard than a curated model download (which
# never touches anything currently in use). _llama_install_start_ui
# refuses to start while the managed port is reachable for exactly this
# reason.

_install_lock = threading.RLock()
_install_thread: Optional[threading.Thread] = None
_install_state = {"phase": "idle", "bytes_done": 0, "bytes_total": 0}
_install_abort_requested = False
# One-shot (message, level) popped by the next _llama_install_status_ui
# tick (success or failure) - not left sitting in _install_state, because
# that would mean every 2s tick after a completed install keeps
# re-announcing the same message into llama_lifecycle_infotext, stomping
# on whatever the Start/End/Verify buttons wrote there in the meantime.
# Mirrors _download_needs_refresh's single-shot-flag idiom above. Also
# used to fire a one-shot _notify() popup on completion/failure, so
# finding out an install finished doesn't require still being on the
# Llama sub-tab when it does.
_install_announce: Optional[tuple[str, str]] = None  # (message, level)


def _install_worker(backend_id: str) -> None:
    global _install_thread, _install_announce
    label = LLAMA_BACKENDS[backend_id].label

    def on_progress(written: int) -> None:
        with _install_lock:
            _install_state["bytes_done"] = written

    def on_phase(phase: str) -> None:
        with _install_lock:
            _install_state["phase"] = phase

    try:
        plan = llama_plan_install(backend_id)
        with _install_lock:
            _install_state["bytes_total"] = plan.total_bytes
        completed = llama_install_run(
            plan, should_abort=lambda: _install_abort_requested,
            on_progress=on_progress, on_phase=on_phase,
        )
        message = f"Installed {label} ({plan.build_tag})." if completed else "Install aborted."
        level = "info"
    except Exception as exc:
        log.exception("llama.cpp install failed (backend=%s)", backend_id)
        message = f"Install failed: {exc}"
        level = "error"
    with _install_lock:
        _install_state.update(phase="idle", bytes_done=0, bytes_total=0)
        _install_announce = (message, level)
        _install_thread = None


def _llama_install_start_ui(backend_id: str) -> str:
    """Kicks off a background install and returns immediately - actual
    progress/completion is picked up by the status_timer via
    _llama_install_status_ui below, the same fire-and-forget shape as
    _download_enqueue's own click handlers."""
    global _install_thread, _install_abort_requested
    if not backend_id:
        return "No backend selected."
    with _install_lock:
        if _install_thread is not None and _install_thread.is_alive():
            return "An install is already in progress."
    if is_healthy(_managed_base_url(current_cfg)):
        return "Stop the running llama-server first (\"End llama server\" above) before installing/reinstalling."
    with _install_lock:
        _install_abort_requested = False
        _install_state.update(phase="planning", bytes_done=0, bytes_total=0)
        _install_thread = threading.Thread(target=_install_worker, args=(backend_id,), daemon=True)
        _install_thread.start()
    return f"Installing {LLAMA_BACKENDS[backend_id].label} - resolving latest release..."


def _install_abort_all() -> None:
    """Bare flag-set, no UI return value - mirrors _download_abort_all()
    above, called both by _llama_install_abort_ui below and by
    restart_app_ui (which doesn't need or want a UI-facing message, the
    process is about to be replaced)."""
    global _install_abort_requested
    with _install_lock:
        _install_abort_requested = True


def _llama_install_abort_ui() -> str:
    with _install_lock:
        if _install_thread is None or not _install_thread.is_alive():
            return "Nothing to abort."
    _install_abort_all()
    return "Aborting install..."


def _llama_install_status_ui():
    """(row_visibility, progress_html, lifecycle_infotext_update,
    installed_backend_text_update) - wired to the same 2s status_timer as
    the download-status row, plus called directly after Install/Abort
    clicks for immediate feedback. lifecycle_infotext_update only ever
    carries a real value on the tick that pops a pending _install_announce
    (every other tick is a no-op gr.update(), so it never overwrites a
    Start/End/Verify message that isn't actually stale); installed_backend_
    text_update is always recomputed fresh from disk - just a marker-file
    read, cheap enough not to need the same one-shot treatment."""
    global _install_announce
    with _install_lock:
        phase = _install_state["phase"]
        bytes_done = _install_state["bytes_done"]
        bytes_total = _install_state["bytes_total"]
        announce = _install_announce
        _install_announce = None

    if announce is not None:
        _notify(announce[0], level=announce[1])
    infotext_update = gr.update(value=announce[0]) if announce is not None else gr.update()
    installed_update = gr.update(value=_llama_installed_text())

    if phase not in ("planning", "downloading", "extracting"):
        return gr.update(visible=False), gr.update(value=""), infotext_update, installed_update

    if phase == "planning":
        html = "<div>Resolving latest llama.cpp release...</div>"
    elif phase == "extracting":
        html = "<div>Extracting...</div>"
    else:
        total = max(bytes_total, 1)
        pct = min(100, int(bytes_done * 100 / total))
        html = (
            f"<progress value=\"{bytes_done}\" max=\"{total}\" style=\"width:100%;\"></progress>"
            f"<div>Downloading: {pct}% ({format_size(bytes_done)} / {format_size(bytes_total)})</div>"
        )
    return gr.update(visible=True), gr.update(value=html), infotext_update, installed_update


def _llama_installed_text() -> str:
    info = llama_installed_info()
    if info is None:
        return "Not installed yet."
    return f"Installed: {info.get('label', info.get('backend_id'))} ({info.get('build_tag')})"


# --------------------------------------------------------------- Hydra install/lifecycle
#
# Two separate concerns, kept apart the same way llama's own install
# section is kept apart from _download_lock above:
#   - Installing Hydra's Python deps (torch + friends) is a pip subprocess
#     streaming text output over time - its own lock/thread/state below,
#     since that's a fundamentally different shape of progress than a
#     byte-counted file transfer.
#   - Downloading hydra-3.5.safetensors is just another file - it reuses
#     the EXISTING curated-model download queue (_download_enqueue et al.
#     above) rather than inventing a second download mechanism; only the
#     on-screen progress row is Hydra-tab-local (a Gradio component can
#     only be laid out once, so the Models tab's own download_status_row/
#     download_status_text can't also be reused here - but the underlying
#     queue/worker/_download_status_html() state they poll is global and
#     happily shared).
# Loading/unloading the model itself is not a background job at all - it
# blocks for a few seconds at most (~1GB), so it's a plain generator
# click handler, the same shape as _start_managed_llama_ui/
# _end_managed_llama_ui above (no dedicated timer needed - unlike
# llama-server's reachability, "is Hydra loaded" can only change via our
# own explicit actions here, never externally).

_hydra_install_lock = threading.RLock()
_hydra_install_thread: Optional[threading.Thread] = None
_hydra_install_running = False
_hydra_install_output: list[str] = []  # last 200 lines of pip's own output, newest last
# One-shot (message, level) popped by the next _hydra_install_status_ui
# tick, same single-shot idiom as llama's own _install_announce above -
# so a stale completion message doesn't keep stomping on whatever
# Load/Unload wrote to hydra_lifecycle_infotext in the meantime, and so
# a one-shot _notify() popup fires on completion/failure.
_hydra_install_announce: Optional[tuple[str, str]] = None  # (message, level)


_HYDRA_METRIC_RE = re.compile(r"f([0-9]*\.?[0-9]+)@([0-9]*\.?[0-9]+)")


def _hydra_metric_to_sliders(metric: str) -> tuple[float, float]:
    """Confidence/Threshold slider values for an "f<beta>@<precision>"
    metric string - falls back to the app's own default (not
    AppConfig.hydra_metric's literal default, which would go stale) for
    anything else (a hand-edited "csi..." string, "default", or garbage
    left over from before these sliders existed)."""
    if (match := _HYDRA_METRIC_RE.fullmatch(metric.strip())) is not None:
        return float(match.group(1)), float(match.group(2))
    return 0.5, 0.1


def _hydra_sliders_to_metric(confidence: float, threshold: float) -> str:
    return f"f{confidence:g}@{threshold:g}"


def _hydra_status_text() -> str:
    st = hydra_classifier.status()
    parts = [
        f"Dependencies: {'installed' if st.deps_installed else 'not installed'}.",
        f"Model: {'downloaded' if st.model_downloaded else 'not downloaded'} ({HYDRA_MODEL_PATH}).",
        f"Loaded on {st.device}." if st.loaded else "Not loaded.",
    ]
    return " ".join(parts)


def _hydra_install_worker() -> None:
    global _hydra_install_thread, _hydra_install_running, _hydra_install_announce
    with _hydra_install_lock:
        _hydra_install_output.clear()

    def on_output(line: str) -> None:
        with _hydra_install_lock:
            _hydra_install_output.append(line)
            del _hydra_install_output[:-200]

    try:
        completed = hydra_install_deps(on_output)
        message = "Hydra dependencies installed." if completed else "Hydra dependency install failed - see output above."
        level = "info" if completed else "error"
    except Exception as exc:
        log.exception("Hydra dependency install failed")
        message = f"Install failed: {exc}"
        level = "error"
    with _hydra_install_lock:
        _hydra_install_running = False
        _hydra_install_announce = (message, level)
        _hydra_install_thread = None


def _hydra_install_start_ui() -> str:
    global _hydra_install_thread, _hydra_install_running
    with _hydra_install_lock:
        if _hydra_install_thread is not None and _hydra_install_thread.is_alive():
            return "An install is already in progress."
        _hydra_install_running = True
        _hydra_install_thread = threading.Thread(target=_hydra_install_worker, daemon=True)
        _hydra_install_thread.start()
    return "Installing Hydra dependencies (torch + friends - this downloads several GB)..."


def _hydra_install_status_ui():
    """(output_box_visible, output_box_value, lifecycle_infotext_update,
    hydra_status_md_update) - wired to the same 2s status_timer as
    everything else, plus called directly after the Install click for
    immediate feedback."""
    global _hydra_install_announce
    with _hydra_install_lock:
        running = _hydra_install_running
        output_text = "\n".join(_hydra_install_output[-40:])
        announce = _hydra_install_announce
        _hydra_install_announce = None

    if announce is not None:
        _notify(announce[0], level=announce[1])
    infotext_update = gr.update(value=announce[0]) if announce is not None else gr.update()
    hydra_status_update = gr.update(value=_hydra_status_text())
    return gr.update(visible=running, value=output_text), infotext_update, hydra_status_update


def _hydra_download_model_ui() -> str:
    added = _download_enqueue([
        DownloadItem(
            url=HYDRA_MODEL_URL, dest_path=HYDRA_MODEL_PATH,
            label="hydra-3.5.safetensors", size_bytes=HYDRA_MODEL_SIZE_BYTES,
        )
    ])
    if not added:
        return "Already downloaded, or already queued."
    return "Queued hydra-3.5.safetensors for download (~1GB)."


def _load_hydra_model_ui():
    if hydra_classifier.status().loaded:
        yield "Hydra model is already loaded.", gr.update()
        return
    yield "Loading Hydra model...", gr.update()
    try:
        hydra_classifier.load(current_cfg)
        message = f"Hydra model loaded on {hydra_classifier.status().device}."
    except hydra_classifier.HydraError as exc:
        message = f"Couldn't load Hydra model: {exc}"
    yield message, gr.update(value=_hydra_status_text())


def _unload_hydra_model_ui():
    if not hydra_classifier.status().loaded:
        yield "Hydra model isn't loaded.", gr.update()
        return
    yield "Unloading Hydra model...", gr.update()
    hydra_classifier.unload()
    yield "Hydra model unloaded.", gr.update(value=_hydra_status_text())


# Mirrors models_refresh_ui()'s 7-value return shape (Models tab section
# below) - the "nothing changed, don't touch anything" return for
# _download_triggered_refresh_ui when no download has completed since the
# last tick, so a full models_refresh_ui() re-scan only actually runs
# when something on disk might really have changed.
_MODELS_REFRESH_NOOP = tuple(gr.update() for _ in range(7))


def _download_triggered_refresh_ui(selected_folder: Optional[str] = None):
    """Wired to the same 2s status_timer as the download-status row -
    auto-refreshes the Models tab the moment a queued download actually
    finishes, so a newly-downloaded quant/mmproj becomes selectable
    without a manual Refresh click. Matters when several items are
    queued: the user can set the first one that finishes active while
    the rest keep downloading in the background, instead of waiting for
    the whole queue. Cheap when nothing changed - just a lock + flag
    check, not a re-scan.

    selected_folder (models_selected_folder_state) is passed straight
    through to models_refresh_ui so this auto-triggered refresh doesn't
    yank the view back to whatever's currently ACTIVE (or to nothing, if
    nothing is) - it's specifically an auto-refresh firing on its own
    schedule, possibly while the user is mid-download-queueing on a
    model that isn't active yet, so preserving whatever row they're
    actually looking at matters more here than for a manual Refresh
    click."""
    global _download_needs_refresh
    with _download_lock:
        needs_refresh = _download_needs_refresh
        _download_needs_refresh = False
    if not needs_refresh:
        return _MODELS_REFRESH_NOOP
    return models_refresh_ui(selected_folder)


def download_abort_all_ui(selected_folder: Optional[str] = None):
    """Also force-refreshes the Models tab, unlike a plain completed-
    download tick: if several items were queued and some already
    finished before this abort, those are real, already-complete
    downloads that need to show up right away, not whenever the next
    tick happens to notice. Not used by restart_app_ui(), which calls
    _download_abort_all() directly - the process is about to be replaced
    there, so refreshing a UI that's about to vanish would be pointless.

    selected_folder is passed through to models_refresh_ui for the same
    reason as _download_triggered_refresh_ui above - don't discard
    whatever row the user's currently viewing just because this refresh
    was forced rather than the flag-driven kind."""
    _download_abort_all()
    row_update, text_update = _download_status_ui()
    return (*models_refresh_ui(selected_folder), row_update, text_update)


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
# everything else. That button's own label/behavior is itself decided by
# _action_mode_for_selection(): "Set as active model" when both the
# selected quant and mmproj are already local, "Download" when either
# isn't - see the Download-queue section above for what a "Download"
# click actually schedules. Its click handler (models_action_ui) always
# recomputes this fresh from the dropdowns' actual current values, so
# clicking is correct regardless of what the button currently displays.
#
# The label also updates live as you change either dropdown within a row
# (models_selection_change_ui, wired to both dropdowns' .change() below) -
# this previously crashed the whole request ("Value: X is not in the list
# of choices: [...]"): switching table rows reprograms a dropdown's
# choices AND value in the same event, and a .change() listener on that
# same dropdown could fire mid-flight with the value it held a moment
# ago, which Gradio's Dropdown.preprocess() then validated against the
# already-replaced choices list and rejected - a framework-level check
# that runs before any of our own code, so no amount of defensive Python
# here could have caught it. Fixed at the actual source: both dropdowns
# now have allow_custom_value=True (see their definitions below), which
# turns "value not in choices" from a hard error into a harmless pass-
# through - our own handlers already treat an unrecognized value as
# "nothing usable selected."

_ACTION_BTN_LABELS = {"set_active": "Set as active model", "download": "Download", "disabled": "Set as active model"}


def _action_mode_for_selection(
    group: Optional[ModelGroup], quant_value: Optional[str], mmproj_value: Optional[str], controllable: bool = True
) -> str:
    """"set_active" (both selections already local), "download" (either
    isn't), or "disabled" (nothing valid selected, or controllable is
    False) - single source of truth for both the action button's label
    (_models_dropdown_updates) and models_action_ui's click dispatch, so
    the two can never silently disagree about what a given selection
    means.

    controllable (see app.py's _cached_controllable) gates the whole
    Models tab, download included: if the active connection isn't one
    this app manages, there's no server the user could ever point at
    something downloaded here either - see _cached_controllable's own
    docstring for why "not ours" and "might already be running the wrong
    model" are the same condition."""
    if not controllable:
        return "disabled"
    if group is None or not quant_value or quant_value == "N/A" or mmproj_value == "N/A":
        return "disabled"
    quant_local = any(q.name == quant_value for q in group.quants)
    mmproj_local = mmproj_value is None or any(m.name == mmproj_value for m in group.mmprojs)
    return "set_active" if (quant_local and mmproj_local) else "download"


def _action_button_update(mode: str):
    return gr.update(value=_ACTION_BTN_LABELS[mode], interactive=(mode != "disabled"))


def _group_for_quant_value(groups: list[ModelGroup], quant_name: Optional[str]) -> Optional[ModelGroup]:
    """quant_name is always "<folder_name>/<stem>" (see _models_dropdown_
    updates) whether it's a local or a not-yet-downloaded curated choice -
    so the group it belongs to can always be found by folder name alone,
    without needing to know which case it is first."""
    if not quant_name or quant_name == "N/A":
        return None
    folder_name = quant_name.split("/", 1)[0]
    return next((g for g in groups if g.name == folder_name), None)


def models_selection_change_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """Wired to both dropdowns' .change() - recomputes the action button
    from _action_mode_for_selection, the same single decision function
    every other writer (_models_dropdown_updates, models_action_ui) goes
    through, so this can never disagree with them about what a given
    selection means. quant_name/mmproj_name may occasionally be a stale
    value from a split second ago (see allow_custom_value on the
    dropdowns' definitions) - _group_for_quant_value simply won't find a
    matching group for a value that no longer belongs to what's on
    screen, which correctly resolves to "disabled" here rather than
    crashing or showing something wrong."""
    group = _group_for_quant_value(groups, quant_name)
    mode = _action_mode_for_selection(group, quant_name, mmproj_name, _cached_controllable())
    return _action_button_update(mode)


def _models_star(cfg: AppConfig, group: ModelGroup) -> str:
    return "★" if any(q.name == cfg.model_name for q in group.quants) else ""


def _models_source(group: ModelGroup) -> str:
    return "Curated" if group.curated else "Manual"


def _models_quant_count(group: ModelGroup) -> str:
    # "n/total" for a curated family (how many of the curated quants are
    # actually downloaded) - just a plain count for a manual one, since
    # there's no curated total to compare against.
    if group.curated:
        return f"{len(group.quants)}/{len(group.curated.quants)}"
    return str(len(group.quants))


def _models_table_rows(groups: list[ModelGroup]) -> list[list[str]]:
    cfg = current_cfg
    return [
        [_models_star(cfg, g), g.name, _models_source(g), _models_quant_count(g)]
        for g in groups
    ]


def _local_choice_label(name: str, path: Path) -> str:
    """Dropdown label for an already-local quant/mmproj, sized to match
    a not-yet-downloaded curated choice's "(download, 6.8 GB)" label -
    so the two look like variations on one format, not two different
    conventions, when they sit in the same dropdown."""
    try:
        return f"{name} ({format_size(path.stat().st_size)})"
    except OSError:
        return name


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
    checking whether the value matches a real ModelVariant/MmprojVariant.

    Also returns a third update, for the action button - computed here
    via _action_mode_for_selection rather than a Dropdown .change()
    listener, since this is the one place that already knows exactly
    which quant/mmproj value ends up selected after the preferred_*
    fallback logic runs, so there's no second copy of that resolution
    logic that could silently drift out of sync with this one."""
    if group is None:
        na = gr.update(choices=["N/A"], value="N/A", interactive=False)
        return na, na, _action_button_update("disabled")

    quant_pairs = [(_local_choice_label(q.name, q.model_path), q.name) for q in group.quants]
    local_quant_names = {q.name for q in group.quants}
    if group.curated:
        for cq in group.curated.quants:
            value = f"{group.folder.name}/{cq.name}"
            if value not in local_quant_names:
                quant_pairs.append((f"{cq.name} (download, {format_size(cq.size_bytes)})", value))

    mmproj_pairs = [(_local_choice_label(m.name, m.mmproj_path), m.name) for m in group.mmprojs]
    local_mmproj_names = {m.name for m in group.mmprojs}
    if group.curated:
        for cm in group.curated.mmprojs:
            value = f"{group.folder.name}/{cm.name}"
            if value not in local_mmproj_names:
                mmproj_pairs.append((f"{cm.name} (download, {format_size(cm.size_bytes)})", value))

    if not quant_pairs:
        na = gr.update(choices=["N/A"], value="N/A", interactive=False)
        return na, na, _action_button_update("disabled")
    quant_values = [v for _, v in quant_pairs]
    quant_value = preferred_quant if preferred_quant in quant_values else quant_values[0]
    quant_update = gr.update(choices=quant_pairs, value=quant_value, interactive=True)

    if not mmproj_pairs:
        mmproj_update = gr.update(choices=["N/A"], value="N/A", interactive=False)
        mmproj_value = "N/A"
    else:
        mmproj_values = [v for _, v in mmproj_pairs]
        mmproj_value = preferred_mmproj if preferred_mmproj in mmproj_values else mmproj_values[0]
        mmproj_update = gr.update(choices=mmproj_pairs, value=mmproj_value, interactive=True)

    mode = _action_mode_for_selection(group, quant_value, mmproj_value, _cached_controllable())
    return quant_update, mmproj_update, _action_button_update(mode)


def models_refresh_ui(selected_folder: Optional[str] = None):
    """Rescans webui/models/, rebuilds the grouped table, and re-selects
    a group for the dropdowns - preferring `selected_folder` (the folder
    the caller was already viewing, typically models_selected_folder_
    state) if it's given and still a real group, falling back to
    whichever group is the currently active model, else nothing.

    Preserving the caller's current view like this matters most for the
    auto-triggered refresh that fires when a background download
    completes (_download_triggered_refresh_ui) - without it, finishing a
    download for a model that isn't active yet would silently reset the
    dropdowns back to whatever WAS active (or to nothing) right out from
    under the user, mid-download-queueing, regardless of who actually
    triggered this particular refresh."""
    models, mmprojs = scan_all()
    groups = merge_curated(group_models(models, mmprojs), load_curated_models())
    table = _models_table_rows(groups)

    cfg = current_cfg
    view_group = next((g for g in groups if str(g.folder) == selected_folder), None) if selected_folder else None
    if view_group is not None:
        is_active_group = any(q.name == cfg.model_name for q in view_group.quants)
    else:
        view_group = next((g for g in groups if any(q.name == cfg.model_name for q in g.quants)), None)
        is_active_group = view_group is not None

    quant_update, mmproj_update, action_update = _models_dropdown_updates(
        view_group,
        preferred_quant=cfg.model_name if is_active_group else None,
        preferred_mmproj=cfg.mmproj_name if is_active_group else None,
    )
    folder_key = str(view_group.folder) if view_group else None
    status = f"{len(groups)} model folder(s) found." if groups else f"No models found under {MODELS_DIR}."

    return table, groups, folder_key, quant_update, mmproj_update, action_update, status


def models_table_select_ui(groups: list[ModelGroup], evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not groups or not (0 <= row < len(groups)):
        return None, gr.update(), gr.update(), gr.update(), gr.update()
    group = groups[row]
    quant_update, mmproj_update, action_update = _models_dropdown_updates(group)
    if not group.quants:
        if group.curated:
            note = " — not downloaded yet; pick a quant below to see download info"
        else:
            note = " — no model file found (download incomplete?), can't be selected"
    elif not group.mmprojs:
        note = " — no mmproj found, can't be selected"
    else:
        note = ""
    return str(group.folder), quant_update, mmproj_update, action_update, f"Viewing {group.name}{note}."


def models_set_active_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """Only reached via models_action_ui when the current selection is
    fully local ("set_active" mode) - the refusal branches below are a
    defensive fallback for a stale/racing click, not the normal path for
    a not-yet-downloaded pick (that's models_download_ui's job now)."""
    global current_cfg
    noop = (gr.update(), gr.update(), gr.update())
    if not _cached_controllable():
        return _models_table_rows(groups), *noop, (
            "Model selection has no effect right now - connected to a server this app doesn't control "
            "(external mode, or an unrecognized server already using the managed port)."
        )
    if not quant_name or quant_name == "N/A":
        return _models_table_rows(groups), *noop, "No model selected — click a row in the table first."

    group = next((g for g in groups if any(q.name == quant_name for q in g.quants)), None)
    if group is None:
        shown = quant_name.rsplit("/", 1)[-1]
        return _models_table_rows(groups), *noop, (
            f"\"{shown}\" isn't downloaded yet - use the Download button, not Set as active."
        )

    if mmproj_name == "N/A":
        # "N/A" is the sentinel _models_dropdown_updates uses specifically
        # for "this folder has no usable mmproj" - not a real empty/auto
        # value, so it must never be written as if it meant that (would
        # silently commit an unselectable model, only failing later at
        # server-start time instead of refusing here where the reason is
        # actually known).
        return _models_table_rows(groups), *noop, "This model has no mmproj — can't be set as active."
    if mmproj_name and not any(m.name == mmproj_name for m in group.mmprojs):
        shown = mmproj_name.rsplit("/", 1)[-1]
        return _models_table_rows(groups), *noop, (
            f"\"{shown}\" isn't downloaded yet - use the Download button, not Set as active."
        )
    mmproj_value = "" if mmproj_name is None else mmproj_name
    with _config_lock:
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
    quant_update, mmproj_update, action_update = _models_dropdown_updates(
        group, preferred_quant=quant_name, preferred_mmproj=mmproj_name
    )
    return (
        _models_table_rows(groups),
        quant_update, mmproj_update, action_update,
        f"Active model set to {quant_name} ({mmproj_note}). "
        "Click \"Restart server connection\" below to actually load it.",
    )


def models_download_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """Only reached via models_action_ui when the current selection has
    something not yet local ("download" mode). Queues whichever of
    quant/mmproj isn't on disk yet - possibly both - and leaves the
    dropdowns/table alone, since nothing about what's locally available
    has actually changed yet."""
    noop = (gr.update(), gr.update(), gr.update())
    if not quant_name or quant_name == "N/A":
        return _models_table_rows(groups), *noop, "No model selected — click a row in the table first."

    group = _group_for_quant_value(groups, quant_name)
    if group is None or group.curated is None:
        return _models_table_rows(groups), *noop, "Nothing curated to download for this selection."

    to_queue: list[DownloadItem] = []
    if not any(q.name == quant_name for q in group.quants):
        stem = quant_name.split("/", 1)[1]
        cq = next((c for c in group.curated.quants if c.name == stem), None)
        if cq:
            to_queue.append(DownloadItem(
                url=cq.url, dest_path=group.folder / Path(cq.url).name,
                label=cq.name, size_bytes=cq.size_bytes,
            ))
    if mmproj_name not in (None, "N/A") and not any(m.name == mmproj_name for m in group.mmprojs):
        stem = mmproj_name.split("/", 1)[1]
        cm = next((c for c in group.curated.mmprojs if c.name == stem), None)
        if cm:
            to_queue.append(DownloadItem(
                url=cm.url, dest_path=group.folder / Path(cm.url).name,
                label=cm.name, size_bytes=cm.size_bytes,
            ))

    if not to_queue:
        return _models_table_rows(groups), *noop, "Already downloaded — nothing to queue."

    added = _download_enqueue(to_queue)
    if not added:
        return _models_table_rows(groups), *noop, "Already queued or downloading."
    labels = ", ".join(i.label for i in added)
    log.info("Models tab: queued for download: %s", labels)
    return _models_table_rows(groups), *noop, f"Queued for download: {labels}."


def models_action_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """The single "Set as active model"/"Download" button's click target -
    decides which of the two the current selection actually means (same
    logic _models_dropdown_updates used to pick the button's label, via
    _action_mode_for_selection) and dispatches to it, then refreshes the
    download-status row too in case this action changed the queue."""
    group = _group_for_quant_value(groups, quant_name)
    mode = _action_mode_for_selection(group, quant_name, mmproj_name, _cached_controllable())
    if mode == "download":
        table_u, quant_u, mmproj_u, action_u, status = models_download_ui(groups, quant_name, mmproj_name)
    else:
        table_u, quant_u, mmproj_u, action_u, status = models_set_active_ui(groups, quant_name, mmproj_name)
    row_u, text_u = _download_status_ui()
    return table_u, quant_u, mmproj_u, action_u, status, row_u, text_u


# ------------------------------------------------------------------ Status bar

def _display_base_url() -> str:
    cfg = current_cfg
    if cfg.server_mode == "external":
        return cfg.external_url.rstrip("/")
    return _managed_base_url(cfg)


def _hydra_footer_label(cfg: AppConfig) -> str:
    """Compact one-word(ish) Hydra state for the status bar - see
    _hydra_status_text() for the fuller sentence-form version shown on
    the Hydra sub-tab itself; this is deliberately terser, matching the
    llama-server segment's own "running"/"idle"/etc. brevity."""
    if not cfg.hydra_enabled:
        return "disabled"
    st = hydra_classifier.status()
    if st.loaded:
        return f"loaded ({st.device})"
    if not st.deps_installed or not st.model_downloaded:
        return "not set up"
    return "enabled, not loaded"


def get_status_text() -> str:
    cfg = current_cfg
    base_url = _display_base_url()
    healthy = is_healthy(base_url)
    installed = LLAMA_SERVER_EXE.exists()  # cheap, local - no extra network call
    # Always the server's own answer, never our config - our selection is
    # just what we'd ask it to load next, not necessarily what's actually
    # loaded right now (e.g. external mode, or a server someone else picked).
    model = get_loaded_model_name(base_url) if healthy else None
    model = model or "n/a"
    return (
        f"Python {platform.python_version()} &nbsp;·&nbsp; "
        f"Gradio {gr.__version__} &nbsp;·&nbsp; "
        f"llama-server: {_status_label(cfg, healthy, installed)} &nbsp;·&nbsp; "
        f"Model: {model} &nbsp;·&nbsp; "
        f"Hydra: {_hydra_footer_label(cfg)} &nbsp;·&nbsp; "
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
    try:
        PY_LOG_PATH.write_text("", encoding="utf-8")
    except OSError:
        pass
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

        # Declared up front, outside main_tabs (rather than down by the
        # rest of the reachability wiring, where it's used more locally) -
        # save_settings_btn's click handler, inside the Settings tab body
        # below, needs it as an input too, see that .then() call. Must NOT
        # be declared as a direct child of `with gr.Tabs()` even though
        # gr.State has no DOM footprint of its own and Gradio's own
        # validation explicitly allows it there (see Tabs.__exit__) -
        # confirmed live that doing so still corrupts the rendered tab
        # strip (labels duplicating/shifting on click), so it's declared
        # here instead, before main_tabs even opens.
        current_tab_label_state = gr.State("Single image")

        with gr.Tabs() as main_tabs:
            # interactive=False at construction (option B from the design
            # discussion - the simpler fallback, not a dedicated splash
            # overlay): the very first paint, before demo.load's chain has
            # had a chance to run even once, would otherwise show every
            # tab clickable regardless of real server state. Corrected
            # within the first real check (_startup_reachability_ui,
            # wired at the bottom of build_app) - this same gap applied to
            # Models and Batch too (see their own Tab() calls below), not
            # just Single-image.
            with gr.Tab("Single image", id="single", interactive=False) as single_tab:
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
                single_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                single_run_btn.click(
                    run_single_ui,
                    [single_image, single_trigger],
                    [
                        single_caption, single_infotext,
                        single_run_btn, single_run_col, single_interrupt_col, single_interrupt_btn,
                    ],
                )
                single_interrupt_btn.click(interrupt_single_ui, [], [single_interrupt_btn])
                single_save_btn.click(
                    save_single_ui, [single_image, single_caption], [single_infotext]
                )
                single_caption.change(
                    lambda text: gr.update(interactive=bool(text.strip())),
                    [single_caption], [single_save_btn],
                )
                single_image.change(clear_single_result_ui, [], [single_caption, single_infotext])

            with gr.Tab("Batch processing", interactive=False) as batch_tab:
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

                batch_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                batch_browse_btn.click(browse_directory_ui, [batch_dir], [batch_dir])
                batch_run_btn.click(
                    run_batch_ui,
                    [batch_dir, batch_recursive, batch_overwrite, batch_trigger],
                    [
                        batch_infotext, batch_last_file, batch_last_image, batch_last_caption,
                        batch_run_btn, batch_run_col, batch_interrupt_col, batch_interrupt_btn,
                    ],
                )
                batch_interrupt_btn.click(interrupt_batch_ui, [], [batch_interrupt_btn])

            with gr.Tab("Review") as review_tab:
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
                    interactive=False, row_count=(0, "dynamic"), buttons=[],
                )

                review_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                # See the Review tab's handler-function docstrings above for
                # what each of these three actually holds.
                review_items_state = gr.State([])
                review_index_state = gr.State(-1)
                review_loaded_caption_state = gr.State("")

                _review_nav_outputs = [
                    review_infotext, review_items_state, review_index_state, review_loaded_caption_state,
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
                        review_caption, review_infotext,
                        review_recaption_col, review_interrupt_col, review_interrupt_btn,
                        review_prev_btn, review_next_btn, review_table,
                        review_dir, review_browse_btn, review_scan_btn,
                    ],
                )
                review_interrupt_btn.click(interrupt_review_ui, [], [review_interrupt_btn])

            with gr.Tab("Models", interactive=False) as models_tab:
                gr.Markdown(
                    config_mod.MODELS_TAB_INTRO.format(ignored_substrings=", ".join(IGNORED_SUBSTRINGS))
                )
                models_table = gr.Dataframe(
                    headers=["A", "Model", "Source", "Quants"], datatype=["str", "str", "str", "str"],
                    interactive=False, row_count=(0, "dynamic"),
                )
                with gr.Row():
                    # allow_custom_value=True is load-bearing, not cosmetic: it
                    # skips Gradio's strict "submitted value must be in the
                    # current choices" check, which otherwise crashes the whole
                    # request when a table-row switch reprograms these dropdowns'
                    # choices at the same moment a .change() event from the old
                    # row is still in flight (see the Models tab handlers' own
                    # comment block below for the full story). Our own handlers
                    # already treat an unrecognized value as "not selected" -
                    # this only removes Gradio's redundant, crash-prone copy of
                    # that same check.
                    models_quant_dropdown = gr.Dropdown(label="Quant", interactive=False, allow_custom_value=True)
                    models_mmproj_dropdown = gr.Dropdown(label="mmproj", interactive=False, allow_custom_value=True)
                with gr.Row():
                    models_action_btn = gr.Button("Set as active model", variant="primary", interactive=False)
                    restart_server_btn = gr.Button("Manage llama server")
                    refresh_models_btn = gr.Button("Refresh")
                with gr.Row(visible=False) as download_status_row:
                    download_status_text = gr.HTML(container=False, scale=4)
                    download_abort_btn = gr.Button("Abort all downloads", scale=1)
                models_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                # See this tab's own handler-function docstrings above for what
                # each of these two actually holds.
                models_groups_state = gr.State([])
                models_selected_folder_state = gr.State(None)

                _models_scan_outputs = [
                    models_table, models_groups_state, models_selected_folder_state,
                    models_quant_dropdown, models_mmproj_dropdown, models_action_btn, models_infotext,
                ]
                refresh_models_btn.click(models_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
                models_table.select(
                    models_table_select_ui,
                    [models_groups_state],
                    [
                        models_selected_folder_state, models_quant_dropdown, models_mmproj_dropdown,
                        models_action_btn, models_infotext,
                    ],
                )
                models_action_btn.click(
                    models_action_ui,
                    [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
                    [
                        models_table, models_quant_dropdown, models_mmproj_dropdown, models_action_btn, models_infotext,
                        download_status_row, download_status_text,
                    ],
                )
                download_abort_btn.click(
                    download_abort_all_ui,
                    [models_selected_folder_state],
                    [*_models_scan_outputs, download_status_row, download_status_text],
                )
                models_quant_dropdown.change(
                    models_selection_change_ui,
                    [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
                    [models_action_btn],
                )
                models_mmproj_dropdown.change(
                    models_selection_change_ui,
                    [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
                    [models_action_btn],
                )

            with gr.Tab("Settings", id="settings") as settings_tab:
                # Declared here, before settings_tabs opens, not as a
                # direct child of `with gr.Tabs()` itself - see current_
                # tab_label_state's own comment above for why a gr.State
                # directly inside a Tabs block was confirmed live to
                # corrupt the rendered tab strip.
                current_settings_subtab_state = gr.State("llama-settings")
                with gr.Tabs() as settings_tabs:
                    with gr.Tab("Llama", id="llama-settings") as llama_settings_tab:
                        with gr.Group():
                            server_mode = gr.Radio(
                                choices=[
                                    (
                                        "Managed: the llama server is managed by the UI, model management enabled",
                                        "managed",
                                    ),
                                    (
                                        "External: the llama server is managed by the user",
                                        "external",
                                    ),
                                ],
                                value=cfg.server_mode,
                                label="Server mode",
                                elem_id="server-mode-radio",
                            )

                        with gr.Group(visible=(cfg.server_mode == "managed")) as managed_server_group:
                            installed_backend_text = gr.Markdown(_llama_installed_text())
                            with gr.Row():
                                llama_backend_dropdown = gr.Dropdown(
                                    choices=[(b.label, bid) for bid, b in LLAMA_BACKENDS.items()],
                                    value=LLAMA_DEFAULT_BACKEND,
                                    show_label=False, container=False, scale=3,
                                )
                                install_llama_btn = gr.Button("Install / Reinstall llama.cpp", scale=2)
                                abort_install_btn = gr.Button("Abort install", scale=1)
                            with gr.Row(visible=False) as llama_install_status_row:
                                llama_install_status_text = gr.HTML(container=False)
                            server_port = gr.Number(label="Port (managed mode)", value=cfg.server_port, precision=0)
                            with gr.Row():
                                n_gpu_layers = gr.Textbox(
                                    label="GPU layers ('auto', 'all', or an exact number)",
                                    value=cfg.n_gpu_layers,
                                )
                                context_size = gr.Number(label="Context size", value=cfg.context_size, precision=0)
                            extra_server_args = gr.Textbox(
                                label="Extra llama-server arguments", value=cfg.extra_server_args
                            )
                            with gr.Row():
                                start_llama_btn = gr.Button(
                                    "Start llama server", interactive=False, variant="secondary"
                                )
                                end_llama_btn = gr.Button("End llama server", interactive=False)
                            autostart_managed_llama = gr.Checkbox(
                                label="Autostart on app launch (if installed but not already running)",
                                value=cfg.autostart_managed_llama,
                            )

                        with gr.Group(visible=(cfg.server_mode == "external")) as external_server_group:
                            external_url = gr.Textbox(label="External server URL", value=cfg.external_url)
                            verify_external_btn = gr.Button("Verify")

                        # Shared by both mode-conditional groups above
                        # (only one of which is ever visible at a time),
                        # rather than one status box per group - Start/
                        # End/Verify never fire at the same moment anyway.
                        llama_lifecycle_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                        with gr.Group():
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

                        # Purely cosmetic (which fields are relevant to the
                        # selected mode) - server_mode itself is still the
                        # one source of truth for which mode is active;
                        # this never needs to be read back, only pushed to.
                        server_mode.change(
                            lambda mode: (gr.update(visible=mode == "managed"), gr.update(visible=mode == "external")),
                            [server_mode],
                            [managed_server_group, external_server_group],
                        )
                        # Both chain the same immediate refresh
                        # save_settings_btn already uses (see its own
                        # comment) - Start/End change server state
                        # directly, so waiting up to 8s for the next
                        # reachability_timer tick to reflect it would be
                        # the exact same staleness that fix already
                        # addressed for settings saves. current_tab_label_
                        # state is "Settings" here too (these buttons only
                        # exist on that tab), so push_to_settings is
                        # always a no-op, not a real redirect.
                        start_llama_btn.click(_start_managed_llama_ui, [], [llama_lifecycle_infotext]).then(
                            _refresh_reachability_ui, [current_tab_label_state],
                            [
                                models_tab, single_tab, batch_tab, start_llama_btn, end_llama_btn, main_tabs,
                                current_tab_label_state,
                            ],
                        )
                        end_llama_btn.click(_end_managed_llama_ui, [], [llama_lifecycle_infotext]).then(
                            _refresh_reachability_ui, [current_tab_label_state],
                            [
                                models_tab, single_tab, batch_tab, start_llama_btn, end_llama_btn, main_tabs,
                                current_tab_label_state,
                            ],
                        )
                        verify_external_btn.click(_verify_external_ui, [external_url], [llama_lifecycle_infotext])

                        _llama_install_status_outputs = [
                            llama_install_status_row, llama_install_status_text,
                            llama_lifecycle_infotext, installed_backend_text,
                        ]
                        install_llama_btn.click(
                            _llama_install_start_ui, [llama_backend_dropdown], [llama_lifecycle_infotext]
                        ).then(_llama_install_status_ui, [], _llama_install_status_outputs)
                        abort_install_btn.click(
                            _llama_install_abort_ui, [], [llama_lifecycle_infotext]
                        ).then(_llama_install_status_ui, [], _llama_install_status_outputs)

                        # Models tab's "Manage llama server" button - wired here,
                        # not next to its own declaration, because it targets
                        # settings_tabs/current_settings_subtab_state, neither
                        # of which exists yet at that earlier point in the
                        # layout (both are declared just above, inside this
                        # same Settings tab).
                        restart_server_btn.click(
                            _goto_llama_settings_ui, [],
                            [main_tabs, settings_tabs, current_settings_subtab_state, current_tab_label_state],
                        )

                    with gr.Tab("Hydra", id="hydra-settings") as hydra_settings_tab:
                        gr.Markdown(
                            "Hydra 3.5 (RedRocket) is an optional second-stage e621 tag "
                            "classifier - if loaded, its tags are appended after the VLM's "
                            "caption on every Single-image/Batch/Review caption. Loading "
                            "competes with llama-server for VRAM, so it's never automatic - "
                            "install its dependencies, download its model, then Load it "
                            "explicitly below (or enable autoload for next launch)."
                        )
                        hydra_status_md = gr.Markdown(_hydra_status_text())
                        with gr.Group():
                            with gr.Row():
                                hydra_install_btn = gr.Button("Install Hydra dependencies", scale=2)
                                hydra_download_btn = gr.Button("Download Hydra model (~1GB)", scale=2)
                            hydra_install_output = gr.Textbox(
                                label="Install output", visible=False, lines=10, max_lines=10, interactive=False,
                            )
                            with gr.Row():
                                hydra_load_btn = gr.Button("Load Hydra model")
                                hydra_unload_btn = gr.Button("Unload Hydra model")
                            hydra_autoload_model = gr.Checkbox(
                                label="Autoload on app launch (if installed, downloaded, and enabled)",
                                value=cfg.hydra_autoload_model,
                            )
                            hydra_lifecycle_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                        with gr.Group():
                            hydra_enabled = gr.Checkbox(
                                label="Enable Hydra tagging (appends tags after the VLM caption)",
                                value=cfg.hydra_enabled,
                            )
                            hydra_device = gr.Radio(
                                choices=[("CUDA", "cuda"), ("CPU", "cpu")],
                                value=cfg.hydra_device,
                                label="Device (only takes effect after an Unload + Load)",
                            )
                            _hydra_confidence_default, _hydra_threshold_default = _hydra_metric_to_sliders(
                                cfg.hydra_metric
                            )
                            with gr.Row():
                                hydra_confidence = gr.Slider(
                                    minimum=0.1, maximum=1.5, step=0.05,
                                    value=_hydra_confidence_default,
                                    label="Confidence",
                                    info=(
                                        "Per-tag strictness. Low = looser, more tags "
                                        "(favors catching everything). High = stricter, "
                                        "fewer but more certain tags (favors precision)."
                                    ),
                                )
                                hydra_threshold = gr.Slider(
                                    minimum=0.0, maximum=0.5, step=0.05,
                                    value=_hydra_threshold_default,
                                    label="Threshold",
                                    info=(
                                        "Safety floor, independent of Confidence: never let "
                                        "a tag through with a measured track record worse "
                                        "than this (e.g. 0.1 = wrong no more than 9 times "
                                        "out of 10 on Hydra's own validation data)."
                                    ),
                                )
                            hydra_implications = gr.Dropdown(
                                choices=[
                                    "preserve", "inherit", "constrain", "enforce", "remove",
                                    "constrain-remove", "enforce-inherit", "enforce-constrain",
                                    "enforce-remove", "off",
                                ],
                                value=cfg.hydra_implications,
                                label="Implications mode",
                            )
                            with gr.Row():
                                hydra_exclude_categories = gr.Textbox(
                                    label="Exclude categories (space-separated)", value=cfg.hydra_exclude_categories,
                                )
                                hydra_exclude_tags = gr.Textbox(
                                    label="Exclude tags (space-separated)", value=cfg.hydra_exclude_tags,
                                )
                            hydra_max_tags = gr.Number(
                                label="Max tags appended (0 = no cap)", value=cfg.hydra_max_tags, precision=0,
                            )

                        with gr.Row(visible=False) as hydra_download_status_row:
                            hydra_download_status_text = gr.HTML(container=False)

                        hydra_install_btn.click(_hydra_install_start_ui, [], [hydra_lifecycle_infotext]).then(
                            _hydra_install_status_ui, [],
                            [hydra_install_output, hydra_lifecycle_infotext, hydra_status_md],
                        )
                        hydra_download_btn.click(_hydra_download_model_ui, [], [hydra_lifecycle_infotext])
                        hydra_load_btn.click(
                            _load_hydra_model_ui, [], [hydra_lifecycle_infotext, hydra_status_md]
                        )
                        hydra_unload_btn.click(
                            _unload_hydra_model_ui, [], [hydra_lifecycle_infotext, hydra_status_md]
                        )

                    with gr.Tab("Image resizing", id="image-resizing-settings") as image_resizing_settings_tab:
                        with gr.Group():
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

                    with gr.Tab("Captioning defaults", id="captioning-defaults-settings") as captioning_defaults_settings_tab:
                        with gr.Group():
                            trigger_word = gr.Textbox(label="Default trigger word", value=cfg.trigger_word)
                            with gr.Row():
                                overwrite_existing = gr.Checkbox(
                                    label="Overwrite existing captions by default", value=cfg.overwrite_existing
                                )
                                recursive_batch = gr.Checkbox(
                                    label="Recursive batch by default", value=cfg.recursive_batch
                                )

                    # id="debug-settings" - the one fixed, never-gated
                    # fallback landing spot for _fallback_tab_safety_ui
                    # (see its own comment): unlike the other four
                    # sub-tabs above (all disabled while a caption is
                    # running - see _settings_gating_ui) and unlike the
                    # top-level Debug tab further down (conditionally
                    # hidden by debug_tab_enabled), this sub-tab is just a
                    # checkbox with no live effect of its own, so there's
                    # never a reason to gate it.
                    with gr.Tab("Debug", id="debug-settings"):
                        with gr.Group():
                            debug_tab_enabled = gr.Checkbox(
                                label="Enable Debuglog tab (requires app restart)",
                                value=cfg.debug_tab_enabled,
                            )

                with gr.Row():
                    save_settings_btn = gr.Button("Save settings", variant="primary")
                    restart_app_btn = gr.Button("Restart app")
                settings_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

                settings_inputs = [
                    server_mode, server_port, external_url,
                    n_gpu_layers, context_size, extra_server_args, autostart_managed_llama,
                    resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
                    prompt_template, temperature, top_p, max_tokens, request_timeout,
                    trigger_word, overwrite_existing, recursive_batch, debug_tab_enabled,
                    hydra_enabled, hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
                    hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_autoload_model,
                ]
                save_settings_btn.click(save_settings_ui, settings_inputs, [settings_infotext]).then(
                    # Saving can change server_mode (or anything else
                    # check_status/_cached_controllable/_cached_reachable
                    # read), and the Models/Single-image/Batch-processing
                    # tabs' interactive state would otherwise be stale for
                    # up to 8s until reachability_timer's next tick -
                    # refresh it immediately instead. current_tab_label_
                    # state is "Settings" here (this click can only happen
                    # from the Settings tab), so the push-to-Settings
                    # branch in _refresh_reachability_ui is always a no-op
                    # in this context, not a real redirect.
                    _refresh_reachability_ui, [current_tab_label_state],
                    [
                        models_tab, single_tab, batch_tab, start_llama_btn, end_llama_btn, main_tabs,
                        current_tab_label_state,
                    ],
                )
                restart_app_btn.click(restart_app_ui, [], [])

            with gr.Tab("Debuglog", visible=cfg.debug_tab_enabled):
                gr.Markdown(f"**Python debug log** (also written to `{PY_LOG_PATH}`)")
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
            review_recaption_btn, review_recaption_col, review_interrupt_col, review_interrupt_btn,
            review_prev_btn, review_next_btn, review_table,
            review_dir, review_browse_btn, review_scan_btn,
        ]
        # Switching to any of these tabs remounts their Columns back to
        # whatever visibility was declared at Blocks-build time, not the
        # latest server-pushed state - Gradio's own quirk, not something
        # we did. status_timer already self-corrects that within 2s (see
        # _update_ui_status' docstring), but re-running it right on
        # tab-select too makes the correction immediate instead of a
        # brief, harmless flash of both Run and Interrupt at once.
        single_tab.select(_update_ui_status, [], _run_interrupt_btns)
        batch_tab.select(_update_ui_status, [], _run_interrupt_btns)
        review_tab.select(_update_ui_status, [], _run_interrupt_btns)
        # Same remount quirk applied to the Models tab's own table: a
        # background download finishing while the user is on another tab
        # gets picked up by _download_triggered_refresh_ui within 2s (its
        # own docstring), but that server-side update is discarded the
        # moment the user switches back to Models, same as the Columns
        # above - so re-run a real rescan right on re-entry too instead of
        # leaving a stale count until a manual Refresh click.
        models_tab.select(models_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
        # Tracks which Settings sub-tab was genuinely clicked into, as
        # opposed to whatever settings_tabs is currently displaying (see
        # _on_settings_subtab_select's own docstring for why those two
        # things need to be kept separate).
        settings_tabs.select(_on_settings_subtab_select, [], [current_settings_subtab_state])
        # Redirects to Debug on entering Settings while the sub-tabs are
        # disabled - see _settings_subtab_entry_ui's own docstring.
        settings_tab.select(_settings_subtab_entry_ui, [current_settings_subtab_state], [settings_tabs])

        # Tracks which top-level tab is currently selected (declared up by
        # main_tabs' own definition, see the comment there) purely so
        # _refresh_reachability_ui can tell whether the user is on the
        # Models tab at the moment it needs to be disabled (and if so,
        # push them off it) without disturbing anyone on any other tab -
        # see that function's docstring.
        main_tabs.select(_on_main_tab_select, [], [current_tab_label_state])

        _settings_gated_tabs = [
            llama_settings_tab, hydra_settings_tab, image_resizing_settings_tab, captioning_defaults_settings_tab,
        ]

        status_timer = gr.Timer(2.0)
        status_timer.tick(get_status_text, [], [status_bar])
        status_timer.tick(_update_ui_status, [], _run_interrupt_btns)
        status_timer.tick(_settings_gating_ui, [], _settings_gated_tabs)
        status_timer.tick(
            _fallback_tab_safety_ui, [current_tab_label_state], [main_tabs, settings_tabs, current_tab_label_state]
        )
        status_timer.tick(_download_status_ui, [], [download_status_row, download_status_text])
        status_timer.tick(_download_triggered_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
        status_timer.tick(_llama_install_status_ui, [], _llama_install_status_outputs)
        status_timer.tick(
            _hydra_install_status_ui, [], [hydra_install_output, hydra_lifecycle_infotext, hydra_status_md]
        )
        status_timer.tick(_download_status_ui, [], [hydra_download_status_row, hydra_download_status_text])
        if cfg.debug_tab_enabled:
            status_timer.tick(get_python_debug_text, [], [debug_python_box])
            status_timer.tick(get_llama_debug_text, [], [debug_llama_box])

        # Separate, slower timer for _refresh_reachability_cache specifically
        # - it's a real network call (external mode), unlike everything else
        # status_timer drives, so it doesn't need or want a 2s cadence. Its
        # result is picked up by the very next status_timer tick afterward
        # (_update_ui_status reads the cache, doesn't recompute it)
        # for Single/Batch/Recaption's own Run/Recaption BUTTON gating; the
        # Models/Single-image/Batch-processing TABS' own interactive state
        # is set directly below instead, since gating those also needs to
        # know (and occasionally override) which tab is currently selected
        # - see _refresh_reachability_ui's docstring.
        reachability_timer = gr.Timer(8.0)
        reachability_timer.tick(
            _refresh_reachability_ui, [current_tab_label_state],
            [
                models_tab, single_tab, batch_tab, start_llama_btn, end_llama_btn, main_tabs,
                current_tab_label_state,
            ],
        )

        demo.load(get_status_text, [], [status_bar])
        demo.load(models_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
        demo.load(_download_status_ui, [], [download_status_row, download_status_text])
        demo.load(_llama_install_status_ui, [], _llama_install_status_outputs)
        demo.load(
            _hydra_install_status_ui, [], [hydra_install_output, hydra_lifecycle_infotext, hydra_status_md]
        )
        demo.load(_download_status_ui, [], [hydra_download_status_row, hydra_download_status_text])
        demo.load(_settings_gating_ui, [], _settings_gated_tabs)
        # _update_ui_status reads the reachability cache rather than
        # checking fresh (see its own docstring) - chained after (not a
        # parallel demo.load, which it used to be) specifically so it
        # always sees a just-refreshed cache, not whatever the optimistic
        # default happened to still be if this beat _startup_reachability_
        # ui's real (and slower - a live network call) check to the punch.
        # That race was real: Single/Batch could show enabled on a fresh
        # page load against an unreachable server, correcting only once
        # status_timer's first 2s tick came around, if at all. Autostart
        # (if enabled) chains after that, deliberately - the initial UI
        # renders in its correct not-yet-running state first, then
        # autostart is a background upgrade on top of it, never a
        # substitute for landing correctly in the first place.
        demo.load(
            _startup_reachability_ui, [],
            [main_tabs, models_tab, single_tab, batch_tab, start_llama_btn, end_llama_btn, current_tab_label_state],
        ).then(_update_ui_status, [], _run_interrupt_btns).then(
            _autostart_managed_llama_ui, [],
            [
                models_tab, single_tab, batch_tab, start_llama_btn, end_llama_btn, main_tabs,
                current_tab_label_state,
            ],
        ).then(
            # Deliberately last in this chain, after llama's own autostart -
            # see _autoload_hydra_model_ui's own docstring for why the
            # ordering matters (Hydra's autoload is meant to be tested
            # against whatever VRAM llama-server's autostart already
            # claimed, not before it).
            _autoload_hydra_model_ui, [], [hydra_lifecycle_infotext, hydra_status_md],
        )

    return demo


UI_PORT = 7901


def main() -> None:
    _setup_debug_logging()
    atexit.register(_stop_managed)
    demo = build_app()
    demo.queue()
    demo.launch(server_port=UI_PORT, footer_links=[], css=ALL_CSS)


if __name__ == "__main__":
    main()
