"""Which llama-server this app process manages (_session/_session_lock),
how to get a client for it, and the cached server-reachability state that
gates Single/Batch/Recaption and the Models tab's "Set as active model".

Gradio-free, like every non-settings_ui module in a backend: functions
that need the current config take it as `cfg` (the GUI passes
ui.state.current_cfg), and the reachability transition tracker returns
the popup text instead of showing it - ui/reachability.py does the
notify()."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from core.config import AppConfig

from .client import ClientError, LlamaClient
from .models import resolve_selection
from .server import MANAGED_HOST, ServerError, ServerStatus, check_status, is_healthy, resolve_server

log = logging.getLogger(__name__)

_session = {"managed": None, "client": None, "base_url": None}
# Guards every read/decide/write of _session - without it, two requests
# racing in at once (e.g. a batch run and a single-image caption, both
# arriving before any server is up yet) could both decide no server exists
# and each start their own, stomping on each other's _session writes and
# possibly killing one process mid-launch via the other's stop_managed().
# Reentrant (not a plain Lock) because get_client() calls stop_managed()
# internally while already holding it - a plain Lock would deadlock there.
_session_lock = threading.RLock()


def stop_managed() -> None:
    """The reset below always runs, even if .stop() itself raises (e.g.
    ManagedServer.stop()'s second process.wait(10) - after the kill() -
    has no exception handling of its own, so a process that's unusually
    slow to actually exit could propagate a TimeoutExpired here). Without
    the try/except, that would skip the reset entirely and leave _session
    pointing at a stale ManagedServer/base_url - and since
    is_server_managed_by_us() only checks whether _session matches a
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


# Registered from app.py's main() (not here, at module level) - this module gets
# imported directly by scratch/test scripts too (this project's normal
# testing approach), and an unconditional atexit.register(stop_managed)
# at import time meant any such throwaway script that happened to populate
# _session (e.g. by calling get_client() directly while testing) would kill
# a real, wanted llama-server the moment the script's own process exited
# normally - stop_managed() has no way to tell "a real app instance is
# shutting down" apart from "some process that imported this module
# exited", so the fix is to only ever register the hook in the former
# case. See setup_debug_logging()'s docstring (ui.tab_debuglog) for the same lesson
# learned earlier this session, for a lower-stakes case (log truncation
# rather than killing a real process).


# ------------------------------------------------------- Server reachability
#
# "Can Single/Batch/Review's Recaption actually work right now" - a single,
# always-live condition (not a one-time first-run check), reused by
# app.py's _update_ui_status() to gate those three actions the same way
# "something else is running" already gates them, and by the startup check
# (demo.load, near the bottom of app.py's build_app) to decide whether to land on
# the Single-image tab or on Settings with an explanation.
#
# "managed" mode has no implicit lazy-start anymore - starting the server
# is always an explicit action (the "Start llama server" button, or
# autostart at launch - see backends/llama/settings_ui.py),
# never a side effect of clicking Caption. So "reachable" here means
# exactly one thing for both modes: is something actually answering a
# health check right now. "Installed but not started yet" is just as
# unusable as "not installed at all" from this section's point of view -
# see server.ServerStatus.installed for how the two get told apart
# for status-text/button purposes elsewhere. "external" mode has no local
# fallback we control at all, so it always needs a real live health check.
#
# The mode-level categorization (see server.ServerStatus) lives in
# server.py's check_status(),
# which never starts or stops anything, unlike resolve_server(). It's a
# real network call though (is_healthy), so it's cached here and
# refreshed on its own slower timer (see reachability_timer in app.py's
# build_app) rather than the fast 2s status_timer that drives everything else -
# no reason to hit the network that often just to keep a button's
# interactive state current.
#
# Whether the Models tab's "Set as active model" would have any effect is a
# stricter, session-aware question check_status() can't answer on its own
# (it has no idea which server, if any, THIS session started) - see
# set_active_blocked() below.

_reachability_lock = threading.RLock()
_reachability_cache = ServerStatus(reachable=True, installed=True)  # optimistic default until the first real check


def refresh_reachability_cache(cfg: AppConfig) -> ServerStatus:
    global _reachability_cache
    result = check_status(cfg)
    with _reachability_lock:
        _reachability_cache = result
    return result


def cached_reachable() -> bool:
    with _reachability_lock:
        return _reachability_cache.reachable


def cached_status() -> ServerStatus:
    """The last refresh_reachability_cache() result, no network call."""
    with _reachability_lock:
        return _reachability_cache


def has_managed_process() -> bool:
    """Whether this session currently owns a llama-server process at all
    (regardless of which URL) - the Debuglog tab's "is there a log of ours
    to show" check."""
    return _session.get("managed") is not None


def is_server_managed_by_us(base_url: str) -> bool:
    """True if THIS session started (or is otherwise responsible for
    stopping) whatever's currently answering at base_url - what the status
    bar calls "running" rather than a healthy-but-unowned server. Session-
    local only; doesn't know about anything from a previous app run (see
    server.py's docstring on why that's on purpose)."""
    managed = _session.get("managed")
    connected = _session.get("client") is not None and _session.get("base_url") == base_url
    return connected and managed is not None


def managed_base_url(cfg: AppConfig) -> str:
    return f"http://{MANAGED_HOST}:{cfg.server_port}"


def set_active_blocked(cfg: AppConfig) -> Optional[str]:
    """Why the Models tab's "Set as active model" can't have any effect
    right now (None if it can) - the only part of the Models tab that
    depends on the server; downloads always work. External mode never
    loads what we pick. Neither does a running managed-mode server this
    session didn't start (leftover from a previous run, or started by
    hand): it might be running any model at all, and there's no reliable
    way to tell, so it's treated like external mode. Not running yet, or
    llama.cpp not installed yet, is fine - the next start uses the pick."""
    if cfg.server_mode == "external":
        return (
            "Set as active only applies to a llama-server this app starts - in external mode, "
            "the server you run decides which model it loads."
        )
    with _reachability_lock:
        status = _reachability_cache
    if status.reachable and not is_server_managed_by_us(managed_base_url(cfg)):
        return (
            "A llama-server this app didn't start is running on the managed port - it may have "
            "any model loaded, so Set as active has no effect until it's stopped."
        )
    return None


_last_known_reachable: Optional[bool] = None  # None = no real check has completed yet
_down_is_crash = False  # persists for a whole down-streak, not just its first tick - see update_reachability_tracking
_expected_stop = False  # set by stop_managed() itself, consumed by update_reachability_tracking - see both docstrings


def update_reachability_tracking(status: ServerStatus, cfg: AppConfig) -> Optional[str]:
    """Updates _last_known_reachable/_down_is_crash and returns exactly one
    warning-popup message (the caller shows it) for a genuinely NEW,
    UNEXPECTED transition into unreachable -
    never a repeat every poll while it stays that way, and never at all
    for a transition stop_managed() itself caused (see _expected_stop -
    stopping a server we manage is never a crash by definition, whether
    that's the explicit End button, a mode switch, or a settings-save
    that needs to restart it; stop_managed() is the only thing that ever
    calls ManagedServer.stop(), so any call to it is inherently
    deliberate). Must run before status_label() on the same poll, since
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
        return None
    managed_installed = cfg.server_mode == "managed" and status.installed
    if was_reachable:
        expected = _expected_stop
        _expected_stop = False
        _down_is_crash = not expected
        if _down_is_crash:
            if managed_installed:
                return "The managed llama-server stopped responding - it may have crashed. Restart it from Settings."
            return f"No captioning server is reachable - {status.reason} Set it up on the Settings tab."
    elif was_reachable is None and not managed_installed:
        return f"No captioning server is reachable - {status.reason} Set it up on the Settings tab."
    # was_reachable is False (steady-state down), or None+managed_installed
    # (fresh idle) - no popup either way.
    return None


def status_label(cfg: AppConfig, reachable: bool, installed: bool) -> str:
    """The status-bar's server-state label. Decoupled from ServerStatus
    (just the two fields it actually needs) since its one caller,
    get_status_text(), does its own fresh/uncached health check rather
    than reading the reachability cache - see that function's own
    comment on why the status bar's "Model: X" needs to always be fresh,
    not cached. Reads _down_is_crash (module state, maintained by
    update_reachability_tracking() on the separate 8s reachability-cache
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


def get_client(cfg: AppConfig) -> LlamaClient:
    with _session_lock:
        if cfg.server_mode == "external":
            desired_base = cfg.external_url.rstrip("/")
        else:
            desired_base = managed_base_url(cfg)

        if _session["client"] is not None and _session["base_url"] == desired_base:
            log.debug("get_client(): reusing existing client for %s", desired_base)
            return _session["client"]

        model_path = mmproj_path = None
        if cfg.server_mode != "external":
            model_path, mmproj_path, error = resolve_selection(cfg)
            if error:
                log.warning("get_client(): %s", error)
                raise ServerError(error)

        stop_managed()
        log.debug("get_client(): resolving server (mode=%s)", cfg.server_mode)
        base_url, managed = resolve_server(cfg, model_path, mmproj_path)
        _session["managed"] = managed
        _session["client"] = LlamaClient(base_url, timeout=cfg.request_timeout)
        _session["base_url"] = base_url
        return _session["client"]


def try_start_managed_llama(cfg: AppConfig) -> tuple[bool, str]:
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
        get_client(cfg)
    except ServerError as exc:
        return False, f"Couldn't start llama-server: {exc}"
    refresh_reachability_cache(cfg)
    return True, "llama-server is running."


def running_client(cfg: AppConfig) -> LlamaClient:
    """A client for whatever server cfg points at, if one answers - never
    starts one, unlike get_client(). What the pipeline's caption() uses:
    starting the server is start_pipeline's job, not a caption's."""
    base_url = display_base_url(cfg)
    if not is_healthy(base_url):
        raise ClientError(f"llama-server isn't running at {base_url}.")
    return LlamaClient(base_url, timeout=cfg.request_timeout)


def display_base_url(cfg: AppConfig) -> str:
    if cfg.server_mode == "external":
        return cfg.external_url.rstrip("/")
    return managed_base_url(cfg)
