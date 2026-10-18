"""BuildContext - how a tab's build() hands frame-level wiring back to
app.py without needing the frame's own components.

A tab registers what it needs and app.py applies it once every tab exists:
  - on_tick(fn, inputs, outputs): add a listener to the shared 2s
    status_timer. The Timer is created after all tabs (where it always
    was) - creating it earlier would move it in the layout - so ticks are
    collected here rather than registered directly.
  - on_load(fn, inputs, outputs): add a demo.load listener.
  - after_pipeline_change(event): chain the frame's pipeline refresh
    (Single/Batch gating, status bar - app.py's _refresh_pipeline_ui)
    after `event`, for buttons that start/stop the pipeline. Its outputs
    span several tabs, which a single tab can't name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BuildContext:
    ticks: list[tuple[Callable, list, list]] = field(default_factory=list)
    loads: list[tuple[Callable, list, list]] = field(default_factory=list)
    pipeline_change_events: list[Any] = field(default_factory=list)

    def on_tick(self, fn: Callable, inputs: list, outputs: list) -> None:
        self.ticks.append((fn, inputs, outputs))

    def on_load(self, fn: Callable, inputs: list, outputs: list) -> None:
        self.loads.append((fn, inputs, outputs))

    def after_pipeline_change(self, event: Any) -> None:
        self.pipeline_change_events.append(event)
