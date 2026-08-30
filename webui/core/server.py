"""llama-server process lifecycle.

Two modes, chosen via AppConfig.server_mode:
  - "managed": if something already answers at MANAGED_HOST:server_port,
    use it and never touch it. Otherwise start llama-server.exe ourselves
    (always bound to MANAGED_HOST - a subprocess this app spawns and talks
    to itself has no reason to bind anywhere else) and stop it again when
    we're done.
  - "external": never manage a process, always talk to external_url
    (which may point at another machine entirely) - this app has no
    business touching anything there, ever.

ManagedServer.stop()'s terminate()-then-kill() is not just cleanup - it's
also the only reliable way anywhere in this app to interrupt a request
that's already in flight. llama-server itself has open upstream bugs
where it doesn't reliably notice an HTTP client disconnecting and keeps
generating regardless, so closing our end of the connection alone can't
be trusted to actually stop server-side work; killing the OS process is
unconditional and doesn't depend on llama-server cooperating. See
app.py's Operation-tracking section (_operation_interrupt_click,
_operation_force_abort) for where this gets invoked as an abort
mechanism, not just at shutdown.

Only ever manages a process we ourselves started (resolve_server returns
managed_server=None whenever the server was already running, in either
mode) - this module will never kill something it didn't launch, on this
machine or, in external mode, especially not one that may be running on
a completely different machine.

(An earlier version of this module tracked spawned PIDs on disk to
recognize and re-adopt an orphaned llama-server.exe across app restarts -
removed after concluding the real-world trigger for that orphaning was
too narrow to justify the added complexity: closing the app's console
window already reliably kills everything attached to it, including
llama-server.exe, so the only case that actually left an orphan behind
was something surgically killing just the Python process without
touching the window - not normal usage. See check_status() below, which
stayed, for the separate reachability-gating concern.)
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .config import AppConfig

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LLAMA_SERVER_EXE = ROOT_DIR / "llama" / "llama-server.exe"
LOG_PATH = ROOT_DIR / "webui" / "logs" / "llama-server.log"

# Managed mode always binds/talks to localhost - it's a subprocess this
# app itself spawns and is the only intended client of, so there's no
# real use case for a user-configurable host there (unlike external mode,
# which legitimately may point at a different machine).
MANAGED_HOST = "127.0.0.1"

log = logging.getLogger(__name__)


class ServerError(RuntimeError):
    pass


@dataclass
class ServerStatus:
    """Cheap, read-only categorization for UI gating - never starts or
    stops anything, unlike resolve_server(). reachable: is something
    actually answering a health check right now - the only thing that
    gates Single/Batch/Recaption. There's no more "not running yet, but
    installed, so count it as reachable anyway" case: starting a managed
    server is now always an explicit action (a button, or autostart at
    launch), never an implicit side effect of clicking Caption, so
    "installed but idle" is just as unusable right now as "not installed
    at all" - both are reachable=False.

    controllable: can the Models tab's model/mmproj selection have any
    actual effect at the *mode* level - True for managed mode once
    llama.cpp is installed (whether or not anything's running yet), False
    for external mode (never ours to tell what to load) and for managed
    mode with nothing installed (nothing to ever control). app.py further
    narrows this by session ownership when reachable is True (see its
    _cached_controllable): a healthy managed-mode server this app didn't
    start itself might be running any model at all, which this module has
    no way to know.

    installed: managed mode only - is llama-server.exe present on disk.
    Not meaningful for external mode (always False there). Exists so a
    caller can tell "not installed" apart from "installed but idle"
    without another network call - both are reachable=False, but they
    need different status text/button behavior (see app.py)."""

    reachable: bool
    controllable: bool
    installed: bool
    reason: str = ""


def is_healthy(base_url: str, timeout: float = 1.5) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_loaded_model_name(base_url: str, timeout: float = 1.5) -> Optional[str]:
    """Asks the server what model it actually has loaded, via /props.
    Useful when we didn't start the server ourselves (external mode, or
    someone else's already-running server) and so don't otherwise know.
    """
    try:
        resp = requests.get(f"{base_url}/props", timeout=timeout)
        if resp.status_code != 200:
            log.debug("/props at %s returned HTTP %s", base_url, resp.status_code)
            return None
        model_path = resp.json().get("model_path")
        return Path(model_path).name if model_path else None
    except requests.RequestException as exc:
        log.debug("/props at %s unreachable: %s", base_url, exc)
        return None


class ManagedServer:
    """A llama-server process we started and are responsible for stopping."""

    def __init__(self, base_url: str, process: subprocess.Popen):
        self.base_url = base_url
        self.process = process

    def stop(self) -> None:
        if self.process.poll() is not None:
            log.debug("Managed llama-server (pid %s) already exited", self.process.pid)
            return
        log.info("Stopping managed llama-server (pid %s)", self.process.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("llama-server (pid %s) didn't terminate gracefully - killing it", self.process.pid)
            self.process.kill()
            self.process.wait(timeout=10)


def _start_process(cfg: AppConfig, model_path: Path, mmproj_path: Path) -> subprocess.Popen:
    if not LLAMA_SERVER_EXE.exists():
        raise ServerError(
            f"llama-server.exe not found at {LLAMA_SERVER_EXE}. "
            f"Install it from Settings -> Llama in the GUI, or run "
            f"`tag-cli.cmd --install-llama <backend>` (see --help)."
        )

    argv = [
        str(LLAMA_SERVER_EXE),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--host", MANAGED_HOST,
        "--port", str(cfg.server_port),
        "-ngl", str(cfg.n_gpu_layers),
        "-c", str(cfg.context_size),
        "--parallel", "1",
    ]
    if cfg.extra_server_args.strip():
        argv.extend(shlex.split(cfg.extra_server_args))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    log.info("Starting llama-server: %s", " ".join(argv))
    return subprocess.Popen(
        argv, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(LLAMA_SERVER_EXE.parent)
    )


def resolve_server(
    cfg: AppConfig,
    model_path: Optional[Path],
    mmproj_path: Optional[Path],
    ready_timeout: float = 120.0,
) -> tuple[str, Optional[ManagedServer]]:
    """Returns (base_url, managed_server). managed_server is None if we
    should not stop it (either it was already running, or mode is external).
    """
    if cfg.server_mode == "external":
        base_url = cfg.external_url.rstrip("/")
        if not is_healthy(base_url):
            log.warning("External server mode but nothing answering at %s", base_url)
            raise ServerError(f"No server responding at {base_url}")
        log.debug("Using external server at %s", base_url)
        return base_url, None

    base_url = f"http://{MANAGED_HOST}:{cfg.server_port}"
    if is_healthy(base_url):
        log.info("Reusing already-running server at %s", base_url)
        return base_url, None

    if model_path is None or mmproj_path is None:
        log.warning("resolve_server: no valid model/mmproj to start llama-server with")
        raise ServerError("No valid model/mmproj selected to start llama-server with.")

    wait_start = time.monotonic()
    process = _start_process(cfg, model_path, mmproj_path)
    deadline = wait_start + ready_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.warning("llama-server (pid %s) exited early with code %s", process.pid, process.returncode)
            raise ServerError(
                f"llama-server exited early (code {process.returncode}); "
                f"see {LOG_PATH}"
            )
        if is_healthy(base_url):
            log.info("llama-server ready after %.1fs (pid %s)", time.monotonic() - wait_start, process.pid)
            return base_url, ManagedServer(base_url, process)
        time.sleep(0.5)

    log.warning("llama-server (pid %s) did not become ready within %.0fs - killing it", process.pid, ready_timeout)
    process.terminate()
    raise ServerError(f"llama-server did not become ready within {ready_timeout}s")


def check_status(cfg: AppConfig) -> ServerStatus:
    """Cheap, read-only categorization - never starts or stops anything
    (unlike resolve_server, which this deliberately doesn't call). Meant
    for UI gating: polled periodically and at app startup to decide
    whether Single/Batch/Recaption should be enabled, and whether the
    Models tab's model/mmproj selection means anything."""
    if cfg.server_mode == "external":
        base_url = cfg.external_url.rstrip("/")
        if is_healthy(base_url):
            return ServerStatus(reachable=True, controllable=False, installed=False)
        return ServerStatus(
            reachable=False, controllable=False, installed=False,
            reason=f"No server responding at {base_url}.",
        )

    base_url = f"http://{MANAGED_HOST}:{cfg.server_port}"
    if is_healthy(base_url):
        return ServerStatus(reachable=True, controllable=True, installed=True)
    if LLAMA_SERVER_EXE.exists():
        return ServerStatus(
            reachable=False, controllable=True, installed=True,
            reason="llama-server isn't running yet.",
        )
    return ServerStatus(
        reachable=False, controllable=False, installed=False,
        reason="llama.cpp isn't installed yet.",
    )
