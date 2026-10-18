"""The pipeline's state as the UI shows it, in one cached snapshot:
whether Single/Batch are usable, the status bar's "Pipeline" segment, the
traffic lights on Settings -> Pipeline, and the one-shot toast when a
started backend stops unexpectedly.

refresh() does the real checks (llama's is a health request, its setup
check scans the model folder), so the frame calls it on its slower 8s
timer and right after Start/Stop; everything on the 2s timer reads
snapshot() instead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from backends import BACKENDS
from core import pipeline
from ui import state
from ui.notify import notify

GREEN, YELLOW, RED = "🟢", "🟡", "🔴"


@dataclass
class Snapshot:
    state: str = "stopped"  # core.pipeline's: stopped/starting/started/stopping
    usable: bool = False  # started and every backend ready - Single/Batch enabled
    not_ready: list[str] = field(default_factory=list)  # labels of crashed backends
    # Saved settings differ from what the running pipeline started with.
    restart_needed: bool = False
    lights: dict[str, str] = field(default_factory=dict)  # backend id -> light
    unavailable: set[str] = field(default_factory=set)  # backend ids with a setup problem


_lock = threading.Lock()
_snapshot = Snapshot()
_announced: set[str] = set()  # crashed backends already toasted


def snapshot() -> Snapshot:
    with _lock:
        return _snapshot


def refresh() -> Snapshot:
    global _snapshot, _announced
    cfg = state.current_cfg
    st = pipeline.status()
    plan = st.plan if st.state == "started" else None
    running = {b.id for b in plan.backends} if plan else set()

    lights: dict[str, str] = {}
    not_ready: list[str] = []
    unavailable: set[str] = set()
    for b in BACKENDS:
        if b.id in running:
            if b.is_ready(plan.cfg):
                lights[b.id] = GREEN
            else:
                lights[b.id] = RED
                not_ready.append(b.label)
        elif b.setup_problem(cfg):
            lights[b.id] = RED
            unavailable.add(b.id)
        elif b.is_ready(cfg):  # e.g. an external llama-server that answers
            lights[b.id] = GREEN
        else:
            lights[b.id] = YELLOW

    new_crashes = set(not_ready) - _announced
    if new_crashes:
        notify(f"{', '.join(sorted(new_crashes))} stopped unexpectedly - restart the pipeline "
               "(Settings -> Pipeline).", level="warning")
    _announced = set(not_ready)

    snap = Snapshot(
        state=st.state, usable=plan is not None and not not_ready, not_ready=not_ready,
        restart_needed=plan is not None and plan.cfg != cfg, lights=lights, unavailable=unavailable,
    )
    with _lock:
        _snapshot = snap
    return snap


def status_text(snap: Snapshot) -> str:
    """The status bar's "Pipeline: ..." value."""
    if snap.state != "started":
        return snap.state
    if snap.not_ready:
        return f"error - {', '.join(snap.not_ready)} stopped"
    if snap.restart_needed:
        return "started - restart to apply settings"
    return "started"
