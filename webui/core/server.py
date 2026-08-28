"""llama-server process lifecycle.

Two modes, chosen via AppConfig.server_mode:
  - "auto": if something already answers at server_host:server_port, use it
    and never touch it. Otherwise start llama-server.exe ourselves and stop
    it again when we're done.
  - "external": never manage a process, always talk to external_url
    (which may point at another machine entirely).

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
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from .config import AppConfig

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LLAMA_SERVER_EXE = ROOT_DIR / "llama-cuda" / "llama-server.exe"
LOG_PATH = ROOT_DIR / "webui" / "logs" / "llama-server.log"

log = logging.getLogger(__name__)


class ServerError(RuntimeError):
    pass


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
            f"Run setup-tagger.cmd first."
        )

    argv = [
        str(LLAMA_SERVER_EXE),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--host", cfg.server_host,
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

    base_url = f"http://{cfg.server_host}:{cfg.server_port}"
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
