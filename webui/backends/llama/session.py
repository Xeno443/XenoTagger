"""Which llama-server this app process manages (_session/_session_lock)
and how to get a client for it.

Gradio-free, like every non-settings_ui module in a backend: functions
that need the current config take it as `cfg`."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from core.config import AppConfig

from .client import ClientError, LlamaClient
from .models import resolve_selection
from .server import MANAGED_HOST, ServerError, is_healthy, resolve_server

log = logging.getLogger(__name__)

_session = {"managed": None, "client": None, "base_url": None}
# Guards every read/decide/write of _session - without it, two calls
# racing in at once could both decide no server exists and each start
# their own, stomping on each other's _session writes and possibly
# killing one process mid-launch via the other's stop_managed().
# Reentrant (not a plain Lock) because get_client() calls stop_managed()
# internally while already holding it - a plain Lock would deadlock there.
_session_lock = threading.RLock()


def stop_managed() -> None:
    """Stops the llama-server this session started, if any - never one it
    didn't. The reset below always runs, even if .stop() itself raises
    (e.g. a process unusually slow to exit after kill()): our bookkeeping
    of what we're responsible for shouldn't stay wrong just because
    killing it was slow - a stale entry would make anything else later
    answering on that port look like ours."""
    with _session_lock:
        if _session["managed"] is not None:
            try:
                _session["managed"].stop()
            except Exception:
                log.exception("Error stopping managed llama-server - clearing session anyway")
        _session["managed"] = None
        _session["client"] = None
        _session["base_url"] = None


# atexit registration happens in app.py's main() (via Backend.on_exit),
# never at import time here - scratch/test scripts import this module
# too, and an import-time hook would kill a real, wanted llama-server the
# moment such a script exited.


def has_managed_process() -> bool:
    """Whether this session currently owns a llama-server process at all
    (regardless of which URL) - the Debuglog tab's "is there a log of ours
    to show" check."""
    return _session.get("managed") is not None


def is_server_managed_by_us(base_url: str) -> bool:
    """True if THIS session started (and so may stop) whatever's answering
    at base_url. Session-local only; doesn't know about anything from a
    previous app run (see server.py's docstring on why that's on purpose)."""
    managed = _session.get("managed")
    connected = _session.get("client") is not None and _session.get("base_url") == base_url
    return connected and managed is not None


def managed_base_url(cfg: AppConfig) -> str:
    return f"http://{MANAGED_HOST}:{cfg.server_port}"


def display_base_url(cfg: AppConfig) -> str:
    if cfg.server_mode == "external":
        return cfg.external_url.rstrip("/")
    return managed_base_url(cfg)


def set_active_blocked(cfg: AppConfig) -> Optional[str]:
    """Why the Models tab's "Set as active model" can't have any effect
    right now (None if it can). External mode never loads what we pick.
    Neither does a running managed-mode server this session didn't start
    (leftover from a previous run, or started by hand): it might be
    running any model at all. Not running yet, or llama.cpp not installed
    yet, is fine - the next pipeline start uses the pick."""
    if cfg.server_mode == "external":
        return (
            "Set as active only applies to a llama-server this app starts - in external mode, "
            "the server you run decides which model it loads."
        )
    base_url = managed_base_url(cfg)
    if is_healthy(base_url) and not is_server_managed_by_us(base_url):
        return (
            "A llama-server this app didn't start is running on the managed port - it may have "
            "any model loaded, so Set as active has no effect until it's stopped."
        )
    return None


def get_client(cfg: AppConfig) -> LlamaClient:
    """A client for the server cfg points at - in managed mode starting
    one if nothing answers (blocking until ready). Only the pipeline's
    load() calls this (via try_start_managed_llama); captions use
    running_client(), which never starts anything."""
    with _session_lock:
        desired_base = display_base_url(cfg)
        # Reused only while it still answers - a crashed server must not
        # count as running.
        if _session["client"] is not None and _session["base_url"] == desired_base and is_healthy(desired_base):
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
    """(ok, message) - Backend.load()'s implementation: makes sure a
    server answers for cfg (managed: starts one; external: checks it)."""
    try:
        get_client(cfg)
    except ServerError as exc:
        return False, f"Couldn't start llama-server: {exc}"
    if cfg.server_mode == "external":
        return True, "External llama-server is reachable."
    return True, "llama-server is running."


def running_client(cfg: AppConfig) -> LlamaClient:
    """A client for whatever server cfg points at, if one answers - never
    starts one, unlike get_client(). What the pipeline's caption() uses:
    starting the server is start_pipeline's job, not a caption's."""
    base_url = display_base_url(cfg)
    if not is_healthy(base_url):
        raise ClientError(f"llama-server isn't running at {base_url}.")
    return LlamaClient(base_url, timeout=cfg.request_timeout)
