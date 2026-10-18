"""Backend load ordering and VRAM coexistence, generic over core.backend's
Backend interface - core/ never imports backends/, so callers pass in
the backend list (backends.BACKENDS). Used by core.pipeline's
start_pipeline, the only thing that loads backends.

The one rule both functions encode: "adaptive" backends (llama-server's
n_gpu_layers="auto") size themselves to whatever VRAM is free when they
start, "fixed" ones (Hydra, Furrence2) are all-or-nothing. So fixed
backends get first pick - by loading first, and if one is loaded while an
adaptive backend this app owns is already running, by temporarily
stopping it, loading, then restarting it so it resizes around what the
fixed one claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from .backend import Backend
from .config import AppConfig


def load_order(backends: Iterable[Backend]) -> list[Backend]:
    """Fixed backends first, adaptive last, otherwise registry order."""
    backends = list(backends)
    return [b for b in backends if b.vram != "adaptive"] + [b for b in backends if b.vram == "adaptive"]


@dataclass
class LoadStep:
    phase: str  # "stopping", "loading", "restarting", or "done"
    message: str
    level: str = "info"  # "info" or "warning" - only meaningful on "done"


def load_with_coexistence(target: Backend, backends: Iterable[Backend], cfg: AppConfig) -> Iterator[LoadStep]:
    """Loads `target`, stopping and afterwards restarting every running
    adaptive backend this app owns if `target` is a fixed one that needs
    VRAM right now. A generator of progress steps so a UI can show each
    one as it happens; the last step is always phase "done", carrying the
    final message and level. Never touches a backend that isn't
    is_loaded() - an external or foreign server is left alone.

    Callers make sure no caption job is running - stopping a server out
    from under an in-flight request would break it."""
    paused: list[Backend] = []
    if target.vram == "fixed" and target.competes_for_vram(cfg):
        paused = [b for b in backends if b is not target and b.vram == "adaptive" and b.is_loaded(cfg)]

    for other in paused:
        yield LoadStep("stopping", f"Stopping {other.label} to free VRAM for {target.label}...")
        other.unload(cfg)

    yield LoadStep("loading", f"Loading {target.label} model...")
    ok, message = target.load(cfg)
    level = "info" if ok else "warning"

    for other in paused:
        yield LoadStep("restarting", f"{message} Restarting {other.label}...")
        started, start_message = other.load(cfg)
        message = f"{message} {start_message}" if started else f"{message} Restart failed: {start_message}"
        if not started:
            level = "warning"

    yield LoadStep("done", message, level)
