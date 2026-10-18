"""Gradio UI entry point and frame: Single-image captioning, Batch
processing, Review, Models, Settings, and an opt-in Debuglog tab. The CLI
(cli.py) is a separate, independent entry point that shares core/ and
backends/ with this file but not this file or ui/ - nothing UI-specific
belongs in core/, or in a backend outside its settings_ui.py.

This file only builds the frame: the Blocks, the top-level Tabs, the
status bar, the two timers, the launch chain (startup check, then
"Start pipeline at launch"), and the wiring that crosses tabs (Batch ->
Review, Models <-> a backend's Settings sub-tab, the pipeline refresh
that gates Single/Batch). Each tab's own components and handlers live in
ui/tab_*.py, the Settings sub-tabs in ui/settings/ and each backend's
settings_ui.py (backends/<id>/), and the UI-side state several tabs share
in ui/state.py, ui/operation.py, ui/pipeline_status.py and
ui/downloads_queue.py - see ui/__init__.py and backends/__init__.py for
the import rules that keep those acyclic. Tabs hand frame-level wiring
back through a BuildContext (ui/context.py) instead of reaching for
components they don't own.

Captioning happens only through the pipeline (core.pipeline): Settings ->
Pipeline starts it (or "Start pipeline at launch"), and Single/Batch are
enabled only while it's started and every backend in it is still up
(ui.pipeline_status). Someone on Single/Batch when that stops being true
is moved to Review.

Coordination of shared, process-wide state:
  - core.pipeline's own lock: the pipeline's state (stopped/starting/
    started/stopping) and the plan it runs.
  - `_active_operation`/`operation_lock` (ui/operation.py): is a caption
    job (single, batch, review) active, and has Interrupt been clicked.
    Also the mutual-exclusion gate - only one job at a time.
  - `current_cfg`/`config_lock` (ui/state.py): the read-modify-write of
    current_cfg + config_mod.save() in Save settings and the Models tab's
    "Set as active model", the only two places that rebind it.
  - `_session`/`_session_lock` (backends/llama/session.py): which
    llama-server this process started.

The Run/Interrupt buttons are two SEPARATE gr.Button components per tab,
not one button whose label changes: Gradio disables the triggering
component of a still-pending event client-side, and every listener's
default concurrency_limit is 1, so a second click on the button still
running the long call never reaches the server. The Interrupt button
never started the long call, so it's always clickable (same pattern as
Forge/A1111's Generate/Interrupt). On the Single tab, Caption and
Interrupt each get their own gr.Column, and it's the Column's visibility
that swaps: a Row of two plain Columns is the only shape confirmed to
align with the row above without CSS (see ui/css.py for the abandoned CSS
approach).

Button state is normally only pushed by whichever browser connection's
generator is running, so a reloaded page or a second browser tab would
show stale buttons. _update_ui_status() recomputes every tab's Run/
Interrupt state from _active_operation on the 2s status_timer and on
page load, so any view self-corrects within one tick.

restart_app_ui (ui/tab_settings.py) doesn't block on a running job: it
asks it to stop (operation_force_abort), waits for it to end, then shuts
every backend down before os.execv replaces the process.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import socket
import sys
import time

# Must be set before `import gradio` - some of its telemetry checks read
# this at import time, not just when building a Blocks instance. The ui
# package imports gradio too, so this has to come before those imports.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from backends import BACKENDS
from core import config as config_mod
from core import tag_vocab
from ui import pipeline_status, state, tab_batch, tab_debuglog, tab_models, tab_review, tab_settings, tab_single
from ui.context import BuildContext
from ui.css import ALL_CSS
from ui.operation import current_operation, operation_lock, operation_status_text
from ui.tab_debuglog import setup_debug_logging
from ui.tab_review import review_scan_ui

log = logging.getLogger("app")  # fixed name, not __name__ - which is "__main__"
                                  # when run directly (the real usage) but
                                  # "app" when imported (e.g. in tests) -
                                  # this way logging.getLogger("app") always
                                  # refers to the right logger either way.

_CAPTION_TABS = ("Single image", "Batch processing")


def _refresh_pipeline_ui(current_tab_label: str):
    """The 8s pipeline_timer and every Start/Stop/Save (ctx.after_pipeline_change):
    refreshes ui.pipeline_status' snapshot, sets Single/Batch's interactive
    state, and moves someone off them to Review when they stop being
    usable. Also updates current_tab_label_state when it moves someone -
    main_tabs.select() only fires on a real click, never on a server-pushed
    switch, so the tracked label would otherwise go stale. gr.skip()
    leaves it alone otherwise."""
    snap = pipeline_status.refresh()
    gate = gr.update(interactive=snap.usable)
    push = not snap.usable and current_tab_label in _CAPTION_TABS
    return (
        gate, gate,
        gr.update(selected="review") if push else gr.update(),
        "Review" if push else gr.skip(),
    )


def _on_main_tab_select(evt: gr.SelectData) -> str:
    """Wired to main_tabs.select() - see current_tab_label_state's own
    comment for why this is tracked at all."""
    return evt.value


def _goto_backend_settings_ui(selected):
    """The Models tab's "Open <backend> settings" button's click target -
    pure navigation to the selected row's backend's Settings sub-tab
    (selected is the Models tab's [backend_id, key] state).

    Explicitly asserts every destination (main_tabs, settings_tabs,
    current_settings_subtab_state, current_tab_label_state) rather than a
    bare gr.update() no-op on any of them - a no-op can't be trusted to
    undo a still-pending prior push to a Tabs component's selected=, and
    neither main_tabs.select() nor settings_tabs.select() reliably fire on
    a server-pushed switch like this one."""
    tab_id = f"{selected[0] if selected else BACKENDS[0].id}-settings"
    return gr.update(selected="settings"), gr.update(selected=tab_id), tab_id, "Settings"


def _goto_models_ui_for(backend_id: str):
    """Click target for a backend Settings page's "Show in Models": the
    Models tab, filtered to that backend. build_app chains the Models
    refresh after it - a server-pushed tab switch doesn't fire the tab's
    own .select() refresh."""
    def _goto_models_ui():
        return gr.update(selected="models"), backend_id, "Models"
    return _goto_models_ui


def _send_batch_to_review_ui(directory: str):
    """The Batch tab's "Send to Review" button - navigation + handoff of
    the directory field to the Review tab (same "explicitly assert every
    destination" reasoning as _goto_backend_settings_ui). build_app chains
    review_scan_ui after it."""
    return gr.update(selected="review"), directory, "Review"


def _startup_ui():
    """Wired to demo.load() - where a freshly loaded page lands (Single if
    the pipeline is usable, else Review) and Single/Batch's initial
    interactive state, from a real (not cached) check. Also sets
    current_tab_label_state to match, the value every later refresh
    reads."""
    snap = pipeline_status.refresh()
    gate = gr.update(interactive=snap.usable)
    target, label = ("single", "Single image") if snap.usable else ("review", "Review")
    return gr.update(selected=target), gate, gate, label


def _fallback_tab_safety_ui(current_tab_label: str):
    """Cheap, cache-only 2s safety net: someone sitting on Single/Batch
    while the cached snapshot says they're unusable gets moved to Review,
    in case _refresh_pipeline_ui missed a case."""
    if current_tab_label in _CAPTION_TABS and not pipeline_status.snapshot().usable:
        return gr.update(selected="review"), "Review"
    return gr.update(), gr.skip()


def _update_ui_status():
    """Ground-truth Run/Interrupt appearance for all three caption tabs,
    recomputed fresh from _active_operation every call - see the module
    docstring. Each tab computes its own share (tab_*.run_controls_update);
    this concatenates them in _run_interrupt_btns' order: Single's,
    Batch's, then Review's. Run buttons are also disabled while the
    pipeline isn't usable (cached snapshot - this runs every 2s)."""
    ready = pipeline_status.snapshot().usable
    with operation_lock:
        op = current_operation()
        return (
            *tab_single.run_controls_update(op, ready),
            *tab_batch.run_controls_update(op, ready),
            *tab_review.run_controls_update(op, ready),
        )


# ------------------------------------------------------------------ Status bar

def get_status_text() -> str:
    """Version/Python/Gradio, the pipeline's state, the active operation.
    Per-backend state lives on Settings -> Pipeline (its lights)."""
    segments = [
        ("Version", config_mod.app_version()),
        ("Python", platform.python_version()),
        ("Gradio", gr.__version__),
        ("Pipeline", pipeline_status.status_text(pipeline_status.snapshot())),
        ("Status", operation_status_text()),
    ]
    return " &nbsp;·&nbsp; ".join(f"{label}: {value}" for label, value in segments)


def build_app() -> gr.Blocks:
    ctx = BuildContext()

    with gr.Blocks(title="XenoTagger", analytics_enabled=False) as demo:
        gr.Markdown("# XenoTagger - LoRA dataset captioning")
        # Plain markdown link, not raw HTML - Gradio's own Markdown
        # renderer already rewrites any real (non "#...") link to
        # target="_blank" rel="noopener noreferrer" client-side.
        gr.Markdown(
            "Check the "
            "[readme](https://github.com/Xeno443/XenoTagger/blob/main/readme/README.md)!"
        )

        # Declared up front, outside main_tabs: a gr.State as a direct
        # child of `with gr.Tabs()` was confirmed live to corrupt the
        # rendered tab strip, even though Gradio's validation allows it.
        current_tab_label_state = gr.State("Review")

        with gr.Tabs(selected="review") as main_tabs:
            single = tab_single.build(ctx)
            batch = tab_batch.build(ctx)
            review = tab_review.build(ctx)
            models = tab_models.build(ctx)
            settings = tab_settings.build(ctx)
            tab_debuglog.build(ctx)

        status_bar = gr.Markdown(get_status_text(), elem_id="status-bar")
        # Order must match _update_ui_status' concatenation.
        _run_interrupt_btns = [*single.run_controls, *batch.run_controls, *review.run_controls]
        # _refresh_pipeline_ui's outputs, in its return order.
        _pipeline_outputs = [single.tab, batch.tab, main_tabs, current_tab_label_state]

        # ---- Cross-tab wiring - each needs components from more than one tab.

        batch.send_review_btn.click(
            _send_batch_to_review_ui, [batch.dir], [main_tabs, review.dir, current_tab_label_state]
        ).then(review_scan_ui, [review.dir], review.nav_outputs)
        _goto_settings_outputs = [
            main_tabs, settings.settings_tabs, settings.current_subtab_state, current_tab_label_state,
        ]
        models.settings_btn.click(_goto_backend_settings_ui, [models.selected_state], _goto_settings_outputs)
        for backend_id, handle in settings.backends.items():
            models_btn = getattr(handle, "models_btn", None)
            if models_btn is None:
                continue
            models_btn.click(
                _goto_models_ui_for(backend_id), [], [main_tabs, models.filter, current_tab_label_state]
            ).then(tab_models.models_refresh_ui, models.refresh_inputs, models.scan_outputs)
        for event in ctx.pipeline_change_events:
            event.then(_refresh_pipeline_ui, [current_tab_label_state], _pipeline_outputs).then(
                get_status_text, [], [status_bar]
            )

        # Switching to any of these tabs remounts their Columns back to
        # their build-time visibility (Gradio's own quirk) - re-running
        # _update_ui_status on tab select makes the correction immediate
        # instead of a brief flash of both Run and Interrupt at once.
        single.tab.select(_update_ui_status, [], _run_interrupt_btns)
        batch.tab.select(_update_ui_status, [], _run_interrupt_btns)
        review.tab.select(_update_ui_status, [], _run_interrupt_btns)

        # Tracks which top-level tab is selected, so _refresh_pipeline_ui
        # can tell whether someone is on Single/Batch when they become
        # unusable without disturbing anyone on any other tab.
        main_tabs.select(_on_main_tab_select, [], [current_tab_label_state])

        status_timer = gr.Timer(2.0)
        status_timer.tick(get_status_text, [], [status_bar])
        status_timer.tick(_update_ui_status, [], _run_interrupt_btns)
        status_timer.tick(_fallback_tab_safety_ui, [current_tab_label_state], [main_tabs, current_tab_label_state])
        for fn, inputs, outputs in ctx.ticks:
            status_timer.tick(fn, inputs, outputs)

        # Slower timer for the real checks behind the snapshot (a health
        # request per server, a model-folder scan) - everything on the 2s
        # timer reads the cached snapshot instead.
        pipeline_timer = gr.Timer(8.0)
        pipeline_timer.tick(_refresh_pipeline_ui, [current_tab_label_state], _pipeline_outputs)

        demo.load(get_status_text, [], [status_bar])
        for fn, inputs, outputs in ctx.loads:
            demo.load(fn, inputs, outputs)
        # The page lands correctly first, then "Start pipeline at launch"
        # (a no-op after the first page load of this app run) runs as a
        # background upgrade on top, then the gating follows its outcome.
        demo.load(
            _startup_ui, [], [main_tabs, single.tab, batch.tab, current_tab_label_state],
        ).then(_update_ui_status, [], _run_interrupt_btns).then(
            settings.pipeline.autostart, [], settings.pipeline.status_outputs,
        ).then(_refresh_pipeline_ui, [current_tab_label_state], _pipeline_outputs).then(
            get_status_text, [], [status_bar]
        )

    return demo


UI_PORT = 7901


def _ui_port_in_use(wait: float = 3.0) -> bool:
    """Whether something is already listening on UI_PORT, checked before
    anything else - so a second instance stops with a plain message
    instead of Gradio's traceback, and before setup_debug_logging()
    truncates the running instance's log file.

    A connect test, not a bind test: a plain bind can fail on a port that
    only has leftover TIME_WAIT connections from a just-closed instance
    (Linux), which is exactly Restart app's case. Retries for `wait`
    seconds, because Restart app's os.execv starts the new process while
    the old one may still be letting go of the port."""
    deadline = time.monotonic() + wait
    while True:
        try:
            # A listener answers within a millisecond; a refused connect
            # on Windows otherwise takes ~2s of retries to fail.
            with socket.create_connection(("127.0.0.1", UI_PORT), timeout=0.2):
                pass
        except OSError:
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.5)


def main() -> None:
    if _ui_port_in_use():
        print(
            f"\nPort {UI_PORT} is already in use - XenoTagger is probably already running "
            f"(open http://127.0.0.1:{UI_PORT} in your browser), or another program is using that port. "
            "Close it and start again.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    setup_debug_logging()
    for backend in BACKENDS:
        atexit.register(backend.on_exit)
    demo = build_app()
    demo.queue()
    demo.launch(
        server_port=UI_PORT, footer_links=[], css=ALL_CSS, inbrowser=True,
        head=tag_vocab.build_autocomplete_head(state.current_cfg),
    )


if __name__ == "__main__":
    main()
