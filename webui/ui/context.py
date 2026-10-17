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
  - on_autoload(backend_id, fn, outputs): the backend's launch-time
    autoload step. app.py chains these after the startup check in
    core.lifecycle.autoload_order (fixed-VRAM backends first). outputs may
    be REACHABILITY, meaning app.py's reachability outputs - fn must then
    return its updates in that order: (Models tab, Single tab, Batch tab,
    Llama Start button, Llama Stop button, main Tabs, current tab label).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union

REACHABILITY = "reachability"


@dataclass
class BuildContext:
    ticks: list[tuple[Callable, list, list]] = field(default_factory=list)
    loads: list[tuple[Callable, list, list]] = field(default_factory=list)
    server_change_events: list[Any] = field(default_factory=list)
    autoloads: dict[str, tuple[Callable, Union[list, str]]] = field(default_factory=dict)

    def on_tick(self, fn: Callable, inputs: list, outputs: list) -> None:
        self.ticks.append((fn, inputs, outputs))

    def on_load(self, fn: Callable, inputs: list, outputs: list) -> None:
        self.loads.append((fn, inputs, outputs))

    def after_server_change(self, event: Any) -> None:
        self.server_change_events.append(event)

    def on_autoload(self, backend_id: str, fn: Callable, outputs: Union[list, str]) -> None:
        self.autoloads[backend_id] = (fn, outputs)
