"""Models tab: every backend's models in one table (Backend.list_models),
with a Backend column and a Show filter, up to two dropdowns for the
selected row's choices (Backend.model_options - llama's quant/mmproj),
and one shared download queue (ui.downloads_queue).

Generic over the registry: nothing here knows a specific backend. The
tab is always enabled - downloads work in every server mode. Only "Set as
active model" can be refused, by the backend (Backend.set_active_problem,
e.g. llama in external mode) or because a caption job is running.

The action button's label and its click both come from _action_mode(),
recomputed from the dropdowns' current values every time, so the label
and what a click does can't disagree:
  - "Download" when anything the selection needs isn't on disk
    (Backend.model_downloads), queued in the background.
  - "Set as active model" when it's all on disk and the backend has
    several models to choose from (Backend.selectable_models). Clicking
    a row only shows it; only this button commits (switching llama models
    means a full server restart + multi-GB reload, too expensive for a
    stray click).
  - "Downloaded" (disabled) when there's nothing to choose (Hydra's one
    model).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import gradio as gr

from backends import BACKENDS
from core import config as config_mod
from core import pipeline
from core.backend import Backend
from ui import state
from ui.context import BuildContext
from ui.notify import notify
from ui.downloads_queue import (
    download_abort_all, download_consume_needs_refresh, download_enqueue, download_status_ui,
)
from ui.operation import operation_blocked_by

log = logging.getLogger("app")  # fixed name, see app.py

ALL = "all"
NA = "N/A"
# Dropdowns under the table - Gradio can't add components at runtime, so a
# fixed number of slots, each shown/relabeled per selected row.
OPTION_SLOTS = 2

_ACTION_LABELS = {
    "set_active": "Set as active model",
    "download": "Download",
    "downloaded": "Downloaded",
    "none": "Set as active model",
}


def _model_backends() -> list[Backend]:
    return [b for b in BACKENDS if b.has_models]


def _backend(selected) -> Optional[Backend]:
    """selected (models_selected_state) is [backend_id, row key] or None."""
    if not selected:
        return None
    return next((b for b in _model_backends() if b.id == selected[0]), None)


def _clean(value: Optional[str]) -> Optional[str]:
    return None if value in (None, "", NA) else value


def _option_updates(backend: Optional[Backend], key: Optional[str]) -> tuple[list, list]:
    """(one update per dropdown slot, the value each ends up showing - None
    for N/A or hidden). An option without choices shows a disabled "N/A"
    rather than an empty choices list: a Dropdown whose value isn't
    reliably cleared can be left showing a stale value against empty
    choices, which Gradio then rejects ("Value: X is not in the list of
    choices: []"). Slots the backend doesn't use are hidden, but get the
    same N/A choice so no stale value survives in them either."""
    options = backend.model_options(state.current_cfg, key) if backend is not None else []
    if len(options) > OPTION_SLOTS:
        log.warning("%s returned %d model options, only %d shown", backend.id, len(options), OPTION_SLOTS)
    updates, values = [], []
    for i in range(OPTION_SLOTS):
        if i >= len(options):
            updates.append(gr.update(visible=False, choices=[NA], value=NA, interactive=False))
            values.append(None)
            continue
        option = options[i]
        if not option.choices:
            updates.append(gr.update(visible=True, label=option.label, choices=[NA], value=NA, interactive=False))
            values.append(None)
            continue
        choice_values = [v for _, v in option.choices]
        value = option.value if option.value in choice_values else choice_values[0]
        updates.append(gr.update(
            visible=True, label=option.label, choices=option.choices, value=value, interactive=True,
        ))
        values.append(value)
    return updates, values


def _action_mode(backend: Optional[Backend], key: Optional[str], values: list) -> tuple[str, Optional[str]]:
    """(mode, why "none") - see the module docstring."""
    if backend is None:
        return "none", "No model selected - click a row in the table first."
    cfg = state.current_cfg
    if backend.model_downloads(cfg, key, values):
        return "download", None
    if not backend.selectable_models:
        return "downloaded", None
    problem = backend.set_active_problem(cfg, key, values)
    return ("none", problem) if problem else ("set_active", None)


def _action_button_update(mode: str):
    return gr.update(value=_ACTION_LABELS[mode], interactive=mode in ("download", "set_active"))


def _settings_button_update(backend: Optional[Backend]):
    if backend is None:
        return gr.update(value="Open settings", interactive=False)
    return gr.update(value=f"Open {backend.label} settings", interactive=True)


def _view_updates(selected) -> tuple[tuple, Optional[str]]:
    """((slot updates..., action button, settings button), why "none")."""
    backend = _backend(selected)
    key = selected[1] if backend is not None else None
    option_updates, values = _option_updates(backend, key)
    mode, problem = _action_mode(backend, key, values)
    return (*option_updates, _action_button_update(mode), _settings_button_update(backend)), problem


def models_refresh_ui(filter_id: str, selected=None):
    """Rescans every backend's models and rebuilds the table. Keeps the
    row the user is viewing (`selected`) if it's still listed - matters
    most for the refresh a finished background download triggers, which
    shouldn't yank the view away mid-queueing. Otherwise picks the active
    model of a backend with several to choose from (llama's), then any
    active one, then - when filtered to one backend - its first row.

    Returns models_scan_outputs: table, rows state, selected state, the
    dropdown slots, action button, settings button, infotext."""
    cfg = state.current_cfg
    rows = []
    for backend in _model_backends():
        if filter_id in (None, ALL, backend.id):
            rows.extend((backend, row) for row in backend.list_models(cfg))

    table = [["★" if r.active else "", b.label, r.name, r.source, r.local] for b, r in rows]
    rows_state = [[b.id, r.key, r.name, r.note] for b, r in rows]
    keys = [[b.id, r.key] for b, r in rows]
    if not selected or list(selected) not in keys:
        selected = (
            next(([b.id, r.key] for b, r in rows if r.active and b.selectable_models), None)
            or next(([b.id, r.key] for b, r in rows if r.active), None)
            or (keys[0] if keys and filter_id not in (None, ALL) else None)
        )
    view, _ = _view_updates(selected)
    status = f"{len(rows)} model(s) listed." if rows else "No models to show."
    return table, rows_state, selected, *view, status


# models_refresh_ui()'s shape, all no-ops - for _download_triggered_refresh_ui
# ticks where nothing finished downloading.
_MODELS_REFRESH_NOOP = tuple(gr.update() for _ in range(6 + OPTION_SLOTS))


def _download_triggered_refresh_ui(filter_id: str, selected=None):
    """On the 2s status_timer: refreshes the tab the moment a queued
    download finishes, so a newly downloaded file is selectable without a
    Refresh click (with several queued, the first one can be set active
    while the rest keep downloading). Just a flag check when nothing
    finished."""
    if not download_consume_needs_refresh():
        return _MODELS_REFRESH_NOOP
    return models_refresh_ui(filter_id, selected)


def download_abort_all_ui(filter_id: str, selected=None):
    """Also refreshes the table right away - downloads that finished before
    the abort should show up now, not on the next tick. (restart_app_ui
    calls download_abort_all() directly - no UI left to refresh there.)"""
    download_abort_all()
    return (*models_refresh_ui(filter_id, selected), *download_status_ui())


def models_table_select_ui(rows_state: list, evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not rows_state or not (0 <= row < len(rows_state)):
        return (gr.update(),) * (4 + OPTION_SLOTS)
    backend_id, key, name, note = rows_state[row]
    selected = [backend_id, key]
    view, problem = _view_updates(selected)
    text = f"Viewing {name}" + (f" - {note}" if note else "") + "." + (f" {problem}" if problem else "")
    return selected, *view, text


def models_selection_change_ui(selected, *option_values):
    """Wired to every dropdown slot's .change() - recomputes the action
    button for the values now showing. A value can be a split second stale
    while a row switch reprograms the dropdowns (see allow_custom_value at
    their definition); the backend treats a value that isn't one of the
    selected row's as nothing selected, and the .change() that follows
    with the new values corrects the button."""
    backend = _backend(selected)
    key = selected[1] if backend is not None else None
    mode, _ = _action_mode(backend, key, [_clean(v) for v in option_values])
    return _action_button_update(mode)


def models_action_ui(filter_id: str, selected, *option_values):
    """The action button's click: decides from the current values what it
    means (the label may be a moment stale) and does it. Returns
    models_scan_outputs + the download status row."""
    noop = (gr.update(),) * (5 + OPTION_SLOTS)  # every scan output but the infotext
    backend = _backend(selected)
    if backend is None:
        return (*noop, "No model selected - click a row in the table first.", *download_status_ui())
    key = selected[1]
    values = [_clean(v) for v in option_values]
    mode, problem = _action_mode(backend, key, values)

    if mode == "download":
        items = [replace(item, source=backend.label) for item in backend.model_downloads(state.current_cfg, key, values)]
        added = download_enqueue(items)
        if added:
            labels = ", ".join(item.label for item in added)
            log.info("Models tab: queued for download (%s): %s", backend.id, labels)
            message = f"Queued for download: {labels}."
        else:
            message = "Already queued or downloading."
        return (*noop, message, *download_status_ui())

    if mode == "set_active":
        blocked = operation_blocked_by()
        if blocked:
            return (*noop, f"Can't change the active model right now - {blocked}", *download_status_ui())
        with state.config_lock:
            updates, message = backend.set_active(state.current_cfg, key, values)
            if updates is not None:
                state.current_cfg = replace(state.current_cfg, **updates)
                config_mod.save(state.current_cfg)
        if updates is None:
            return (*noop, message, *download_status_ui())
        log.info("Models tab: %s set active: %s", backend.id, updates)
        # A started pipeline keeps the model it started with.
        if pipeline.status().state == "started":
            notify("Active model saved. Restart the pipeline to load it.", level="warning")
        # Star column and dropdowns follow the new config.
        refreshed = models_refresh_ui(filter_id, selected)
        return (*refreshed[:-1], message, *download_status_ui())

    if mode == "downloaded":
        message = "Already downloaded - nothing to do here."
    else:
        message = problem or "Nothing to do for this selection."
    return (*noop, message, *download_status_ui())


def _intro_markdown() -> str:
    notes = "\n".join(f"- {b.label}: {b.models_help}" for b in _model_backends() if b.models_help)
    return config_mod.MODELS_TAB_INTRO + ("\n\n" + notes if notes else "")


def build(ctx: BuildContext):
    with gr.Tab("Models", id="models") as models_tab:
        gr.Markdown(_intro_markdown())
        models_filter = gr.Dropdown(
            choices=[("All", ALL)] + [(b.label, b.id) for b in _model_backends()],
            value=ALL, label="Show", interactive=True,
        )
        models_table = gr.Dataframe(
            headers=["A", "Backend", "Model", "Source", "Local"], datatype=["str"] * 5,
            interactive=False, row_count=(0, "dynamic"), buttons=[],
        )
        with gr.Row():
            # allow_custom_value=True is load-bearing, not cosmetic: it
            # skips Gradio's strict "submitted value must be in the
            # current choices" check, which otherwise crashes the whole
            # request when a row switch reprograms these dropdowns'
            # choices while a .change() from the previous row is still in
            # flight with its old value - a framework-level check that
            # runs before any of our code. Our handlers already treat an
            # unrecognized value as "nothing selected".
            option_dropdowns = [
                gr.Dropdown(label=f"Option {i + 1}", interactive=False, allow_custom_value=True)
                for i in range(OPTION_SLOTS)
            ]
        with gr.Row():
            models_action_btn = gr.Button(_ACTION_LABELS["none"], variant="primary", interactive=False)
            models_settings_btn = gr.Button("Open settings", interactive=False)
            refresh_models_btn = gr.Button("Refresh")

        # One row for the one queue, whichever backend queued what.
        # "---" is plain CommonMark (a real <hr>), no CSS needed.
        gr.Markdown("---")
        with gr.Row(visible=False) as download_status_row:
            download_status_text = gr.HTML(container=False, scale=4)
            download_abort_btn = gr.Button("Abort all downloads", scale=1)

        models_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # [backend_id, key, name, note] per table row, in table order.
        models_rows_state = gr.State([])
        # [backend_id, key] of the row being viewed, or None.
        models_selected_state = gr.State(None)

        models_scan_outputs = [
            models_table, models_rows_state, models_selected_state,
            *option_dropdowns, models_action_btn, models_settings_btn, models_infotext,
        ]
        refresh_inputs = [models_filter, models_selected_state]
        download_outputs = [download_status_row, download_status_text]

        refresh_models_btn.click(models_refresh_ui, refresh_inputs, models_scan_outputs)
        # .input, not .change: the frame sets the filter from a Settings
        # page's "Show in Models" and refreshes itself right after.
        models_filter.input(models_refresh_ui, refresh_inputs, models_scan_outputs)
        models_table.select(
            models_table_select_ui, [models_rows_state],
            [models_selected_state, *option_dropdowns, models_action_btn, models_settings_btn, models_infotext],
        )
        models_action_btn.click(
            models_action_ui, [*refresh_inputs, *option_dropdowns], [*models_scan_outputs, *download_outputs],
        )
        download_abort_btn.click(download_abort_all_ui, refresh_inputs, [*models_scan_outputs, *download_outputs])
        for dropdown in option_dropdowns:
            dropdown.change(models_selection_change_ui, [models_selected_state, *option_dropdowns], [models_action_btn])

    # Gradio remounts a tab's contents to their build-time state when you
    # switch back into it, discarding updates pushed while it was hidden
    # (e.g. a download finishing) - rescan on every entry.
    models_tab.select(models_refresh_ui, refresh_inputs, models_scan_outputs)

    ctx.on_tick(download_status_ui, [], download_outputs)
    ctx.on_tick(_download_triggered_refresh_ui, refresh_inputs, models_scan_outputs)
    ctx.on_load(models_refresh_ui, refresh_inputs, models_scan_outputs)
    ctx.on_load(download_status_ui, [], download_outputs)

    return SimpleNamespace(
        tab=models_tab,
        filter=models_filter,
        selected_state=models_selected_state,
        settings_btn=models_settings_btn,
        refresh_inputs=refresh_inputs,
        scan_outputs=models_scan_outputs,
    )
