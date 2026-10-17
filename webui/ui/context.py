"""BuildContext - how a tab's build() hands frame-level wiring back to
app.py without needing the frame's own components.

A tab registers what it needs and app.py applies it once every tab exists:
  - on_tick(fn, inputs, outputs): add a listener to the shared 2s
    status_timer. The Timer is created after all tabs (where it always
    was) - creating it earlier would move it in the layout - so ticks are
    collected here rather than registered directly.
  - on_load(fn, inputs, outputs): add a demo.load listener.
  - after_server_change(event): chain the reachability refresh
    (app.py's _refresh_reachability_ui) after `event`, for buttons that
    start/stop llama-server or change its config. Its outputs span several
    tabs, which a single tab can't name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BuildContext:
    ticks: list[tuple[Callable, list, list]] = field(default_factory=list)
    loads: list[tuple[Callable, list, list]] = field(default_factory=list)
    server_change_events: list[Any] = field(default_factory=list)

    def on_tick(self, fn: Callable, inputs: list, outputs: list) -> None:
        self.ticks.append((fn, inputs, outputs))

    def on_load(self, fn: Callable, inputs: list, outputs: list) -> None:
        self.loads.append((fn, inputs, outputs))

    def after_server_change(self, event: Any) -> None:
        self.server_change_events.append(event)
