"""Settings tab: the sub-tab container, operation gating of the sub-tabs,
and the shared Save settings / Restart app row.

Sub-tabs: Pipeline (ui/settings/pipeline.py) first, then one per backend
in registry order (backends.BACKENDS - each backend's settings_ui module,
loaded by Backend.settings_ui_module), then Debug. Every sub-tab module
provides:
  - build(ctx) -> handle with tab, tab_label, tab_id, gated (disabled
    while a caption job runs) and save_inputs (its fields Save reads);
    a backend's may add models_btn (ui.model_status' "Show in Models",
    wired by app.py)
  - collect_settings(*values) -> dict of AppConfig updates, taking
    save_inputs' values in the same order
  - load_values(cfg) -> save_inputs' values from cfg, applied on page load:
    Gradio otherwise shows the values from when the app started, so a
    browser reload after a Save would show (and a second Save would write
    back) the old ones. A handle's load_outputs, if set, replaces
    save_inputs as the target (Pipeline adds its re-render triggers).
and optionally:
  - abort_background() - stop background work before Restart app, e.g.
    llama's install worker

Save never touches a started pipeline - it keeps the settings it started
with (core.pipeline) - but says "restart to apply" when something changed.
Pipeline's Start/Restart buttons save first, wired here because this
module owns Save.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import gradio as gr

from backends import BACKENDS
from core import config as config_mod
from core import pipeline
from ui import state
from ui.context import BuildContext
from ui.downloads_queue import download_abort_all
from ui.notify import notify
from ui.operation import operation_force_abort, settings_subtabs_disabled, wait_for_operation_to_end
from ui.settings import debug as settings_debug
from ui.settings import pipeline as settings_pipeline

log = logging.getLogger("app")  # fixed name, see app.py

# Filled by build(), in sub-tab order: (module, the handle its build()
# returned). Module-level rather than closed over so the handlers below
# stay plain, importable functions; build_app() runs once per process.
_subtabs: list[tuple[ModuleType, SimpleNamespace]] = []

# gr.SelectData.value for a Tab is ALWAYS its label, never its id -
# confirmed via gradio/layouts/tabs.py's own EVENTS docstring ("value
# referring to the label of the Tab"), and via a live debug-log capture
# that showed "last_chosen=Hydra" (the label) instead of the expected
# "hydra-settings" (the id). id is otherwise only used for the selected=
# kwarg when PUSHING a tab switch (gr.Tab's own docstring: "required if
# you wish to control the selected tab from a predict function") - so
# without this table, a value read off a real click could never be fed
# back into gr.update(selected=...) and have it actually match anything.
# Filled by build() from each sub-tab handle's tab_label/tab_id - every
# sub-tab needs an explicit id to be a valid push target at all.
_SETTINGS_SUBTAB_ID_BY_LABEL: dict[str, str] = {}


def _settings_gating_ui():
    """Disables the operation-sensitive Settings sub-tabs' own
    clickability (see settings_subtabs_disabled) while a Single/Batch/
    Recaption job is actively running, so a job's settings can't change
    out from under it mid-flight. The Debug sub-tab is deliberately
    excluded (gated=False) - its one checkbox already needs a full app
    restart to take effect regardless (see its own comment), so there's
    nothing it could disrupt by staying editable. Deliberately does NOT
    touch the fields inside those sub-tabs, only the tabs' own
    interactive= - see _settings_subtab_entry_ui below for what actually
    protects someone already sitting on one of these when a job starts (a
    redirect on next entry, not disabling the fields themselves)."""
    update = gr.update(interactive=not settings_subtabs_disabled())
    return tuple(update for _, handle in _subtabs if handle.gated)


def _on_settings_subtab_select(evt: gr.SelectData):
    """Wired to settings_tabs.select() - tracks the sub-tab the user
    actually, genuinely clicked into (current_settings_subtab_state) as
    a stable id (see _SETTINGS_SUBTAB_ID_BY_LABEL above - evt.value
    itself is a label, not usable directly), separately from whatever
    settings_tabs is currently DISPLAYING. Read by
    _settings_subtab_entry_ui below.

    Also guards against recording "debug-settings" while the sub-tabs
    are disabled: it's the only place a real click could land during
    that window anyway (everything else is unclickable), so there's
    nothing informative to record, and gr.skip() leaves whatever was
    genuinely there before (e.g. "hydra-settings") untouched. This
    guard's real necessity - i.e. whether a select event ever actually
    fires here that ISN'T a genuine click, such as a side effect of
    _settings_subtab_entry_ui's own redirect - was never independently
    confirmed; it's kept as cheap, harmless insurance regardless, now
    that it's at least comparing against the right kind of value (an id,
    not a label) to ever have a chance of matching."""
    tab_id = _SETTINGS_SUBTAB_ID_BY_LABEL[evt.value]
    disabled = settings_subtabs_disabled()
    if tab_id == "debug-settings" and disabled:
        log.debug("settings subtab select: label=%s id=%s disabled=%s -> IGNORED", evt.value, tab_id, disabled)
        return gr.skip()
    log.debug("settings subtab select: label=%s id=%s disabled=%s -> RECORDED", evt.value, tab_id, disabled)
    return tab_id


def _settings_subtab_entry_ui(last_chosen: str):
    """Wired to settings_tab.select() - the moment the user enters the
    Settings tab (from anywhere else), not on a timer: confirmed live
    that someone already sitting on a sub-tab when a job starts is left
    alone (matches _settings_gating_ui only disabling the tabs' own
    clickability, not evicting an existing occupant), so there's nothing
    for this to correct except at the moment of a fresh entry.

    Always explicitly asserts a destination - either "debug-settings" or
    last_chosen - never a bare gr.update() no-op: a no-op was confirmed
    live NOT to reliably undo a still-pending prior push to a Tabs
    component's selected= (see git history/session notes for the
    original bug this replaced), so "leave it alone and hope it already
    reverted" isn't trustworthy here regardless of which branch applies.

    Because this recomputes settings_subtabs_disabled() fresh on every
    single entry rather than remembering "did I force this last time",
    there's no flag to clear and nothing that can go stale: if the job
    ended 5 seconds ago or 5 minutes ago, disabled is simply False now,
    and the else branch below re-shows last_chosen - whatever that
    genuinely was, Debug included if that's really where the user left
    off - with no separate revert-to-Llama special case needed."""
    disabled = settings_subtabs_disabled()
    if disabled and last_chosen != "debug-settings":
        log.debug("settings subtab entry: last_chosen=%s disabled=%s -> pushing debug-settings", last_chosen, disabled)
        return gr.update(selected="debug-settings")
    log.debug("settings subtab entry: last_chosen=%s disabled=%s -> pushing %s", last_chosen, disabled, last_chosen)
    return gr.update(selected=last_chosen)


def restart_app_ui() -> None:
    """Replaces this process with a fresh one via os.execv - the standard
    Python self-restart trick, so newly-saved settings.json values (or a
    just-toggled Debug tab) take effect without anyone needing a terminal.
    Never returns: the process image is gone the moment execv succeeds, so
    first force-abort any running caption job and wait for it to notice,
    abort in-flight/queued downloads and each sub-tab's own background
    work (abort_background - e.g. a llama.cpp install), all in-memory only
    and won't survive this regardless, then let every backend release
    what execv wouldn't clean up (Backend.shutdown - llama-server's child
    process, Hydra's CUDA context; execv skips atexit entirely).
    """
    log.info("User triggered app restart from Settings")
    operation_force_abort()
    wait_for_operation_to_end()
    download_abort_all()
    for module, _ in _subtabs:
        abort_background = getattr(module, "abort_background", None)
        if abort_background is not None:
            abort_background()
    cfg = state.current_cfg
    for backend in BACKENDS:
        backend.shutdown(cfg)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _save(values) -> bool:
    """Applies and saves every sub-tab's values (each collect_settings()
    gets its own slice of `values`, in sub-tab order) on top of the
    current config - so fields no form covers (model_name/mmproj_name,
    set from the Models tab) carry over unchanged. Returns whether
    anything changed."""
    updates: dict = {}
    i = 0
    for module, handle in _subtabs:
        n = len(handle.save_inputs)
        updates.update(module.collect_settings(*values[i:i + n]))
        i += n

    with state.config_lock:
        old_cfg = state.current_cfg
        state.current_cfg = replace(old_cfg, **updates)
        config_mod.save(state.current_cfg)
    log.info("Settings saved (%s)", "changed" if state.current_cfg != old_cfg else "unchanged")
    return state.current_cfg != old_cfg


RESTART_TO_APPLY = "Settings saved. Restart the pipeline to apply them."


def save_settings_ui(*values) -> str:
    """Save settings' click target. A started pipeline keeps running with
    the settings it started with - if this changed anything, say so."""
    if _save(values) and pipeline.status().state == "started":
        notify(RESTART_TO_APPLY, level="warning")
        return RESTART_TO_APPLY
    return "Settings saved."


def save_quietly_ui(*values) -> str:
    """Save before Pipeline's Start/Restart - no "restart to apply"."""
    _save(values)
    return "Settings saved."


def build(ctx: BuildContext):
    _subtabs.clear()
    _SETTINGS_SUBTAB_ID_BY_LABEL.clear()
    modules = [settings_pipeline] + [importlib.import_module(b.settings_ui_module) for b in BACKENDS] + [
        settings_debug,
    ]

    with gr.Tab("Settings", id="settings") as settings_tab:
        # Declared here, before settings_tabs opens, not as a
        # direct child of `with gr.Tabs()` itself - see current_
        # tab_label_state's own comment in app.py for why a gr.State
        # directly inside a Tabs block was confirmed live to
        # corrupt the rendered tab strip. Defaults to the first sub-tab.
        current_settings_subtab_state = gr.State("pipeline-settings")
        with gr.Tabs() as settings_tabs:
            for module in modules:
                _subtabs.append((module, module.build(ctx)))

        with gr.Row():
            save_settings_btn = gr.Button("Save settings", variant="primary")
            restart_app_btn = gr.Button("Restart app")
        settings_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # _save slices these back apart per sub-tab, in this order.
        settings_inputs = [component for _, handle in _subtabs for component in handle.save_inputs]
        # The pipeline refresh after Save puts "restart to apply" into
        # the status bar right away instead of on the next 8s tick.
        ctx.after_pipeline_change(save_settings_btn.click(save_settings_ui, settings_inputs, [settings_infotext]))
        restart_app_btn.click(restart_app_ui, [], [])

        # Pipeline's Start/Restart: save pending changes first.
        pipeline_handle = _subtabs[0][1]
        for button, run in ((pipeline_handle.start_btn, settings_pipeline.start_ui),
                            (pipeline_handle.restart_btn, settings_pipeline.restart_ui)):
            ctx.after_pipeline_change(
                button.click(save_quietly_ui, settings_inputs, [settings_infotext])
                .then(run, [], pipeline_handle.status_outputs)
            )

    # Every sub-tab's fields from the saved config on page load - see the
    # module docstring.
    for module, handle in _subtabs:
        ctx.on_load(
            lambda module=module: module.load_values(state.current_cfg), [],
            getattr(handle, "load_outputs", handle.save_inputs),
        )

    _SETTINGS_SUBTAB_ID_BY_LABEL.update({handle.tab_label: handle.tab_id for _, handle in _subtabs})

    # Tracks which Settings sub-tab was genuinely clicked into, as
    # opposed to whatever settings_tabs is currently displaying (see
    # _on_settings_subtab_select's own docstring for why those two
    # things need to be kept separate).
    settings_tabs.select(_on_settings_subtab_select, [], [current_settings_subtab_state])
    # Redirects to Debug on entering Settings while the sub-tabs are
    # disabled - see _settings_subtab_entry_ui's own docstring.
    settings_tab.select(_settings_subtab_entry_ui, [current_settings_subtab_state], [settings_tabs])

    _settings_gated_tabs = [handle.tab for _, handle in _subtabs if handle.gated]
    ctx.on_tick(_settings_gating_ui, [], _settings_gated_tabs)
    ctx.on_load(_settings_gating_ui, [], _settings_gated_tabs)

    return SimpleNamespace(
        tab=settings_tab,
        settings_tabs=settings_tabs,
        current_subtab_state=current_settings_subtab_state,
        pipeline=pipeline_handle,
        # Backend sub-tab handles by backend id.
        backends={backend.id: handle for backend, (_, handle) in zip(BACKENDS, _subtabs[1:])},
    )
