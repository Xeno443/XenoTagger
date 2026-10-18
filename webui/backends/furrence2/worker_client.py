"""The app side of the Furrence2 worker (worker.py): starts it, sends it
one request at a time, stops it. A module-level singleton like Hydra's
classifier - one worker per app process.

The worker exits on its own when its stdin closes, which also happens if
this process dies without calling stop(), so it can't outlive the app.
Its stderr (transformers warnings, load time, per-request failures) goes
to LOG_PATH, overwritten on every start like llama-server's log.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from core.config import WEBUI_DIR

from .install import LIB_DIR, MODEL_DIR, deps_installed, model_downloaded

log = logging.getLogger(__name__)

WORKER_PATH = Path(__file__).resolve().parent / "worker.py"
LOG_PATH = WEBUI_DIR / "logs" / "furrence2-worker.log"
READY_TIMEOUT = 300.0  # importing torch + loading the model; slow disks are slow
REQUEST_TIMEOUT = 120.0  # one caption takes ~1s on a GPU, minutes on CPU at worst


class Furrence2Error(RuntimeError):
    pass


@dataclass
class Furrence2Result:
    caption: str
    prompt_tokens: int
    tags_used: int  # prompt tags actually kept (general/species/meta, after trimming)
    length: int  # the "length: N" the prompt asked for
    seconds: float


@dataclass
class Furrence2Status:
    deps_installed: bool
    model_downloaded: bool
    running: bool
    device: Optional[str] = None
    # Set when a worker this session started exited on its own.
    crashed: bool = False


_lock = threading.Lock()  # one request (or start/stop) at a time
_process: Optional[subprocess.Popen] = None
_lines: "queue.Queue[Optional[str]]" = queue.Queue()  # worker stdout; None = EOF
_device: Optional[str] = None
_crashed = False


def status() -> Furrence2Status:
    global _crashed
    running = _process is not None and _process.poll() is None
    if _process is not None and not running:
        _crashed = True
    return Furrence2Status(
        deps_installed=deps_installed(), model_downloaded=model_downloaded(),
        running=running, device=_device if running else None, crashed=_crashed and not running,
    )


def _reader(stream, lines: queue.Queue) -> None:
    for line in io.TextIOWrapper(stream, encoding="utf-8"):
        lines.put(line)
    lines.put(None)


def _receive(timeout: float) -> dict:
    try:
        line = _lines.get(timeout=timeout)
    except queue.Empty:
        raise Furrence2Error(f"Furrence2 worker didn't answer within {timeout:.0f}s") from None
    if line is None:
        raise Furrence2Error(f"Furrence2 worker exited - see {LOG_PATH}")
    return json.loads(line)


def _kill_locked() -> None:
    global _process, _device
    if _process is None:
        return
    if _process.poll() is None:
        try:
            _process.stdin.close()  # the worker's normal exit signal
        except OSError:
            pass
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("Furrence2 worker (pid %s) didn't exit - killing it", _process.pid)
            _process.kill()
            _process.wait(timeout=10)
    _process = None
    _device = None


def start(device: str = "cuda") -> str:
    """Blocking until the model is loaded. Returns the device it runs on;
    raises Furrence2Error with the worker's own reason on failure.
    No-op if already running."""
    global _process, _lines, _device, _crashed
    with _lock:
        if _process is not None and _process.poll() is None:
            return _device or device
        _kill_locked()
        if not deps_installed():
            raise Furrence2Error("Furrence2's dependencies aren't installed - use Install in Settings -> Furrence2.")
        if not model_downloaded():
            raise Furrence2Error("Furrence2's model isn't downloaded - download it on the Models tab.")

        argv = [
            sys.executable, "-I", str(WORKER_PATH),
            "--lib", str(LIB_DIR), "--model-dir", str(MODEL_DIR), "--device", device,
        ]
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Starting Furrence2 worker: %s", " ".join(argv))
        with open(LOG_PATH, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_file,
                cwd=str(WORKER_PATH.parent),
            )
        _process, _lines, _crashed = process, queue.Queue(), False
        threading.Thread(target=_reader, args=(process.stdout, _lines), daemon=True).start()
        try:
            reply = _receive(READY_TIMEOUT)
        except Furrence2Error:
            _kill_locked()
            raise
        if not reply.get("ready"):
            _kill_locked()
            raise Furrence2Error(f"Furrence2 worker failed to load: {reply.get('error', reply)}")
        _device = reply["device"]
        log.info("Furrence2 worker ready on %s (pid %s)", _device, process.pid)
        return _device


def stop() -> None:
    """No-op if not running."""
    global _crashed
    with _lock:
        if _process is not None:
            log.info("Stopping Furrence2 worker (pid %s)", _process.pid)
        _kill_locked()
        _crashed = False


def caption(
    image: Union[str, Path, bytes],
    tags: list[tuple[str, str]],
    seed: str = "",
    length_min: int = 150,
    length_max: int = 200,
    prompt: Optional[str] = None,
) -> Furrence2Result:
    """tags: (e621 name with underscores, category) pairs, best first.
    seed shuffles their prompt order reproducibly (the image's file name,
    so a re-caption gets the same prompt). prompt, if given, replaces the
    built one (tests only). Raises Furrence2Error if the worker isn't
    running, fails the request, or hangs - a hung worker is killed."""
    request: dict = {"tags": [list(t) for t in tags], "seed": seed, "length_min": length_min, "length_max": length_max}
    if isinstance(image, bytes):
        request["image_b64"] = base64.b64encode(image).decode("ascii")
    else:
        request["image_path"] = str(image)
    if prompt is not None:
        request["prompt"] = prompt

    with _lock:
        if _process is None or _process.poll() is not None:
            raise Furrence2Error("Furrence2 isn't running.")
        try:
            _process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            _process.stdin.flush()
            reply = _receive(REQUEST_TIMEOUT)
        except (OSError, Furrence2Error) as exc:
            log.error("Furrence2 request failed, stopping the worker: %s", exc)
            _kill_locked()
            raise Furrence2Error(str(exc)) from exc
    if "error" in reply:
        raise Furrence2Error(f"Furrence2 failed on this image: {reply['error']}")
    return Furrence2Result(
        caption=reply["caption"], prompt_tokens=reply["prompt_tokens"], tags_used=reply["tags_used"],
        length=reply["length"], seconds=reply["seconds"],
    )
