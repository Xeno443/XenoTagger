"""Settings tab: the sub-tab container, operation gating of the sub-tabs,
and the shared Save settings / Restart app row.

Sub-tabs come first from the backend registry (backends.BACKENDS - each
backend's settings_ui module, loaded by Backend.settings_ui_module), then
the non-backend ones in ui/settings/. Every sub-tab module provides:
  - build(ctx) -> handle with tab, tab_label, tab_id, gated (disabled
    while a caption job runs) and save_inputs (its fields Save reads);
    a backend's may add models_btn (ui.model_status' "Show in Models",
    wired by app.py)
  - collect_settings(*values) -> dict of AppConfig updates, taking
    save_inputs' values in the same order
and optionally:
  - after_save(old_cfg, new_cfg) -> text to append to "Settings saved."
    (or None) - reacting to a saved change, e.g. llama's server restart
  - abort_background() - stop background work before Restart app, e.g.
    llama's install worker
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
from ui import state
from ui.context import BuildContext
from ui.downloads_queue import download_abort_all
from ui.operation import operation_force_abort, settings_subtabs_disabled, wait_for_operation_to_end
from ui.settings import captioning_defaults as settings_captioning_defaults
from ui.settings import debug as settings_debug
from ui.settings import image_resizing as settings_image_resizing

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


def save_settings_ui(*values) -> str:
    """Save settings' click target. values are every sub-tab's
    save_inputs, concatenated in sub-tab order (see build()); each
    sub-tab's collect_settings() gets its own slice and returns its
    AppConfig updates, applied on top of the current config - so fields
    no form covers (model_name/mmproj_name, set from the Models tab)
    carry over unchanged. Then each sub-tab module's optional
    after_save(old, new) reacts to what changed (llama: stop/restart the
    managed server on a mode or process-arg change) and can append to the
    returned message."""
    updates: dict = {}
    i = 0
    for module, handle in _subtabs:
        n = len(handle.save_inputs)
        updates.update(module.collect_settings(*values[i:i + n]))
        i += n

    old_cfg = state.current_cfg
    with state.config_lock:
        state.current_cfg = replace(state.current_cfg, **updates)
        new_cfg = state.current_cfg
        config_mod.save(new_cfg)
    log.info(
        "Settings saved: server_mode=%s model=%s ngl=%s",
        new_cfg.server_mode, new_cfg.model_name or "(none)", new_cfg.n_gpu_layers,
    )

    notes = []
    for module, _ in _subtabs:
        after_save = getattr(module, "after_save", None)
        if after_save is not None and (note := after_save(old_cfg, new_cfg)):
            notes.append(note)
    return " ".join(["Settings saved.", *notes])


def build(ctx: BuildContext):
    _subtabs.clear()
    _SETTINGS_SUBTAB_ID_BY_LABEL.clear()
    modules = [importlib.import_module(b.settings_ui_module) for b in BACKENDS] + [
        settings_image_resizing, settings_captioning_defaults, settings_debug,
    ]

    with gr.Tab("Settings", id="settings") as settings_tab:
        # Declared here, before settings_tabs opens, not as a
        # direct child of `with gr.Tabs()` itself - see current_
        # tab_label_state's own comment in app.py for why a gr.State
        # directly inside a Tabs block was confirmed live to
        # corrupt the rendered tab strip. Defaults to the first sub-tab.
        current_settings_subtab_state = gr.State(f"{BACKENDS[0].id}-settings")
        with gr.Tabs() as settings_tabs:
            for module in modules:
                _subtabs.append((module, module.build(ctx)))

        with gr.Row():
            save_settings_btn = gr.Button("Save settings", variant="primary")
            restart_app_btn = gr.Button("Restart app")
        settings_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # save_settings_ui slices these back apart per sub-tab, in this order.
        settings_inputs = [component for _, handle in _subtabs for component in handle.save_inputs]
        # Saving can change server_mode (or anything else
        # check_status/cached_reachable read), and the
        # Single-image/Batch-processing tabs' interactive state
        # would otherwise be stale for
        # up to 8s until reachability_timer's next tick -
        # refresh it immediately instead. current_tab_label_
        # state is "Settings" here (this click can only happen
        # from the Settings tab), so the push-to-Settings
        # branch in _refresh_reachability_ui is always a no-op
        # in this context, not a real redirect.
        ctx.after_server_change(save_settings_btn.click(save_settings_ui, settings_inputs, [settings_infotext]))
        restart_app_btn.click(restart_app_ui, [], [])

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
        # Backend sub-tab handles by backend id.
        backends={backend.id: handle for backend, (_, handle) in zip(BACKENDS, _subtabs)},
    )
