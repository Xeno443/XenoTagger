"""Models tab: the VLM (GGUF) model browser and the Hydra model section,
sharing one download queue (ui.downloads_queue)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gradio as gr

from core import config as config_mod
from core import hydra_classifier
from core.config import AppConfig
from core.downloads import DownloadItem
from core.hydra_install import (
    MODEL_PATH as HYDRA_MODEL_PATH, MODEL_SIZE_BYTES as HYDRA_MODEL_SIZE_BYTES, MODEL_URL as HYDRA_MODEL_URL,
)
from core.models import (
    IGNORED_SUBSTRINGS, MODELS_DIR, ModelGroup, format_size, group_models,
    load_curated_models, merge_curated, scan_all,
)
from ui import state
from ui.context import BuildContext
from ui.downloads_queue import (
    download_abort_all, download_consume_needs_refresh, download_enqueue, download_status_ui,
)
from ui.llama_session import cached_controllable

log = logging.getLogger("app")  # fixed name, see app.py


def _hydra_models_table_rows() -> list[list[str]]:
    """Single-row table for the Models tab's "Hydra model" section -
    deliberately mirrors _models_table_rows()' 4-column shape (A/Model/
    Source/Quants) even though most of it doesn't really apply to a lone
    safetensors file, purely so the two sections look like variations on
    one format rather than two different conventions side by side."""
    downloaded = hydra_classifier.status().model_downloaded
    return [["", "RedRocket-Hydra", "Curated", "1/1" if downloaded else "0/1"]]


def _hydra_download_model_ui():
    added = download_enqueue([
        DownloadItem(
            url=HYDRA_MODEL_URL, dest_path=HYDRA_MODEL_PATH,
            label="hydra-3.5.safetensors", size_bytes=HYDRA_MODEL_SIZE_BYTES,
        )
    ])
    message = (
        "Already downloaded, or already queued." if not added
        else "Queued hydra-3.5.safetensors for download (~1GB)."
    )
    row_u, text_u = download_status_ui()
    return message, row_u, text_u


# Mirrors models_refresh_ui()'s 7-value return shape (Models tab section
# below) - the "nothing changed, don't touch anything" return for
# _download_triggered_refresh_ui when no download has completed since the
# last tick, so a full models_refresh_ui() re-scan only actually runs
# when something on disk might really have changed.
_MODELS_REFRESH_NOOP = tuple(gr.update() for _ in range(7))


def _download_triggered_refresh_ui(selected_folder: Optional[str] = None):
    """Wired to the same 2s status_timer as the download-status row -
    auto-refreshes the Models tab the moment a queued download actually
    finishes, so a newly-downloaded quant/mmproj becomes selectable
    without a manual Refresh click. Matters when several items are
    queued: the user can set the first one that finishes active while
    the rest keep downloading in the background, instead of waiting for
    the whole queue. Cheap when nothing changed - just a lock + flag
    check, not a re-scan.

    selected_folder (models_selected_folder_state) is passed straight
    through to models_refresh_ui so this auto-triggered refresh doesn't
    yank the view back to whatever's currently ACTIVE (or to nothing, if
    nothing is) - it's specifically an auto-refresh firing on its own
    schedule, possibly while the user is mid-download-queueing on a
    model that isn't active yet, so preserving whatever row they're
    actually looking at matters more here than for a manual Refresh
    click."""
    if not download_consume_needs_refresh():
        return _MODELS_REFRESH_NOOP
    return models_refresh_ui(selected_folder)


def download_abort_all_ui(selected_folder: Optional[str] = None):
    """Also force-refreshes the Models tab, unlike a plain completed-
    download tick: if several items were queued and some already
    finished before this abort, those are real, already-complete
    downloads that need to show up right away, not whenever the next
    tick happens to notice. Not used by restart_app_ui() (ui.tab_settings), which calls
    download_abort_all() directly - the process is about to be replaced
    there, so refreshing a UI that's about to vanish would be pointless.

    selected_folder is passed through to models_refresh_ui for the same
    reason as _download_triggered_refresh_ui above - don't discard
    whatever row the user's currently viewing just because this refresh
    was forced rather than the flag-driven kind."""
    download_abort_all()
    row_update, text_update = download_status_ui()
    return (*models_refresh_ui(selected_folder), row_update, text_update)


# ---------------------------------------------------------------- Models tab
#
# One row per model DIRECTORY (core.models.group_models), not one row per
# quant file - a folder's several quants and mmproj precisions become the
# two dropdowns' choices for whichever row is currently selected, not
# separate top-level rows each. Clicking a table row only loads that
# group's options into the dropdowns for viewing/editing - it does NOT
# immediately become the active model (switching models means a full
# llama-server restart + multi-GB reload on the next request, too
# expensive to trigger from a stray click); only the explicit "Set as
# active model" button commits a choice, writing straight to
# settings.json the same way Settings' Save button already does for
# everything else. That button's own label/behavior is itself decided by
# _action_mode_for_selection(): "Set as active model" when both the
# selected quant and mmproj are already local, "Download" when either
# isn't - see ui.downloads_queue for what a "Download"
# click actually schedules. Its click handler (models_action_ui) always
# recomputes this fresh from the dropdowns' actual current values, so
# clicking is correct regardless of what the button currently displays.
#
# The label also updates live as you change either dropdown within a row
# (models_selection_change_ui, wired to both dropdowns' .change() below) -
# this previously crashed the whole request ("Value: X is not in the list
# of choices: [...]"): switching table rows reprograms a dropdown's
# choices AND value in the same event, and a .change() listener on that
# same dropdown could fire mid-flight with the value it held a moment
# ago, which Gradio's Dropdown.preprocess() then validated against the
# already-replaced choices list and rejected - a framework-level check
# that runs before any of our own code, so no amount of defensive Python
# here could have caught it. Fixed at the actual source: both dropdowns
# now have allow_custom_value=True (see their definitions below), which
# turns "value not in choices" from a hard error into a harmless pass-
# through - our own handlers already treat an unrecognized value as
# "nothing usable selected."

_ACTION_BTN_LABELS = {"set_active": "Set as active model", "download": "Download", "disabled": "Set as active model"}


def _action_mode_for_selection(
    group: Optional[ModelGroup], quant_value: Optional[str], mmproj_value: Optional[str], controllable: bool = True
) -> str:
    """"set_active" (both selections already local), "download" (either
    isn't), or "disabled" (nothing valid selected, or controllable is
    False) - single source of truth for both the action button's label
    (_models_dropdown_updates) and models_action_ui's click dispatch, so
    the two can never silently disagree about what a given selection
    means.

    controllable (see ui.llama_session's cached_controllable) gates the whole
    Models tab, download included: if the active connection isn't one
    this app manages, there's no server the user could ever point at
    something downloaded here either - see cached_controllable's own
    docstring for why "not ours" and "might already be running the wrong
    model" are the same condition."""
    if not controllable:
        return "disabled"
    if group is None or not quant_value or quant_value == "N/A" or mmproj_value == "N/A":
        return "disabled"
    quant_local = any(q.name == quant_value for q in group.quants)
    mmproj_local = mmproj_value is None or any(m.name == mmproj_value for m in group.mmprojs)
    return "set_active" if (quant_local and mmproj_local) else "download"


def _action_button_update(mode: str):
    return gr.update(value=_ACTION_BTN_LABELS[mode], interactive=(mode != "disabled"))


def _group_for_quant_value(groups: list[ModelGroup], quant_name: Optional[str]) -> Optional[ModelGroup]:
    """quant_name is always "<folder_name>/<stem>" (see _models_dropdown_
    updates) whether it's a local or a not-yet-downloaded curated choice -
    so the group it belongs to can always be found by folder name alone,
    without needing to know which case it is first."""
    if not quant_name or quant_name == "N/A":
        return None
    folder_name = quant_name.split("/", 1)[0]
    return next((g for g in groups if g.name == folder_name), None)


def models_selection_change_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """Wired to both dropdowns' .change() - recomputes the action button
    from _action_mode_for_selection, the same single decision function
    every other writer (_models_dropdown_updates, models_action_ui) goes
    through, so this can never disagree with them about what a given
    selection means. quant_name/mmproj_name may occasionally be a stale
    value from a split second ago (see allow_custom_value on the
    dropdowns' definitions) - _group_for_quant_value simply won't find a
    matching group for a value that no longer belongs to what's on
    screen, which correctly resolves to "disabled" here rather than
    crashing or showing something wrong."""
    group = _group_for_quant_value(groups, quant_name)
    mode = _action_mode_for_selection(group, quant_name, mmproj_name, cached_controllable())
    return _action_button_update(mode)


def _models_star(cfg: AppConfig, group: ModelGroup) -> str:
    return "★" if any(q.name == cfg.model_name for q in group.quants) else ""


def _models_source(group: ModelGroup) -> str:
    return "Curated" if group.curated else "Manual"


def _models_quant_count(group: ModelGroup) -> str:
    # "n/total" for a curated family (how many of the curated quants are
    # actually downloaded) - just a plain count for a manual one, since
    # there's no curated total to compare against.
    if group.curated:
        return f"{len(group.quants)}/{len(group.curated.quants)}"
    return str(len(group.quants))


def _models_table_rows(groups: list[ModelGroup]) -> list[list[str]]:
    cfg = state.current_cfg
    return [
        [_models_star(cfg, g), g.name, _models_source(g), _models_quant_count(g)]
        for g in groups
    ]


def _local_choice_label(name: str, path: Path) -> str:
    """Dropdown label for an already-local quant/mmproj, sized to match
    a not-yet-downloaded curated choice's "(download, 6.8 GB)" label -
    so the two look like variations on one format, not two different
    conventions, when they sit in the same dropdown."""
    try:
        return f"{name} ({format_size(path.stat().st_size)})"
    except OSError:
        return name


def _models_dropdown_updates(
    group: Optional[ModelGroup], preferred_quant: Optional[str] = None, preferred_mmproj: Optional[str] = None
):
    """(quant_update, mmproj_update) for `group` - both show "N/A" and
    stay disabled if group is None or has no quants at all (e.g. a folder
    whose main model download hasn't finished, only its mmproj has);
    mmproj alone shows "N/A" if the folder has quants but no usable
    mmproj. Deliberately never uses an empty choices=[] with value=None
    to mean "nothing selected" - a Dropdown whose value isn't reliably
    cleared by value=None can be left showing a stale previous value
    (e.g. "N/A" from a different row) against newly-empty choices, which
    Gradio then rejects outright ("Value: X is not in the list of
    choices: []"). Always giving both a real, matching single choice
    instead sidesteps that regardless of the exact clearing behavior.

    preferred_quant/preferred_mmproj select which of the group's choices
    to actually show, if given and valid - falls back to the first choice
    otherwise. Passing these matters: without it, every caller silently
    means "show the alphabetically-first quant/mmproj", which is wrong
    for both models_refresh_ui (should show whatever's ACTUALLY
    configured, cfg.model_name/mmproj_name) and models_set_active_ui
    (should show whatever was JUST committed) - either defaulting to
    "first in the list" regardless would visually contradict its own
    just-reported status text.

    Choices are (label, value) pairs so a not-yet-downloaded curated
    quant/mmproj can carry a "(download, 6.8 GB)" label while its VALUE
    stays the same "<folder>/<stem>" shape a real local one would have -
    that's what lets models_set_active_ui tell the two apart later just by
    checking whether the value matches a real ModelVariant/MmprojVariant.

    Also returns a third update, for the action button - computed here
    via _action_mode_for_selection rather than a Dropdown .change()
    listener, since this is the one place that already knows exactly
    which quant/mmproj value ends up selected after the preferred_*
    fallback logic runs, so there's no second copy of that resolution
    logic that could silently drift out of sync with this one."""
    if group is None:
        na = gr.update(choices=["N/A"], value="N/A", interactive=False)
        return na, na, _action_button_update("disabled")

    quant_pairs = [(_local_choice_label(q.name, q.model_path), q.name) for q in group.quants]
    local_quant_names = {q.name for q in group.quants}
    if group.curated:
        for cq in group.curated.quants:
            value = f"{group.folder.name}/{cq.name}"
            if value not in local_quant_names:
                quant_pairs.append((f"{cq.name} (download, {format_size(cq.size_bytes)})", value))

    mmproj_pairs = [(_local_choice_label(m.name, m.mmproj_path), m.name) for m in group.mmprojs]
    local_mmproj_names = {m.name for m in group.mmprojs}
    if group.curated:
        for cm in group.curated.mmprojs:
            value = f"{group.folder.name}/{cm.name}"
            if value not in local_mmproj_names:
                mmproj_pairs.append((f"{cm.name} (download, {format_size(cm.size_bytes)})", value))

    if not quant_pairs:
        na = gr.update(choices=["N/A"], value="N/A", interactive=False)
        return na, na, _action_button_update("disabled")
    quant_values = [v for _, v in quant_pairs]
    quant_value = preferred_quant if preferred_quant in quant_values else quant_values[0]
    quant_update = gr.update(choices=quant_pairs, value=quant_value, interactive=True)

    if not mmproj_pairs:
        mmproj_update = gr.update(choices=["N/A"], value="N/A", interactive=False)
        mmproj_value = "N/A"
    else:
        mmproj_values = [v for _, v in mmproj_pairs]
        mmproj_value = preferred_mmproj if preferred_mmproj in mmproj_values else mmproj_values[0]
        mmproj_update = gr.update(choices=mmproj_pairs, value=mmproj_value, interactive=True)

    mode = _action_mode_for_selection(group, quant_value, mmproj_value, cached_controllable())
    return quant_update, mmproj_update, _action_button_update(mode)


def models_refresh_ui(selected_folder: Optional[str] = None):
    """Rescans webui/models/, rebuilds the grouped table, and re-selects
    a group for the dropdowns - preferring `selected_folder` (the folder
    the caller was already viewing, typically models_selected_folder_
    state) if it's given and still a real group, falling back to
    whichever group is the currently active model, else nothing.

    Preserving the caller's current view like this matters most for the
    auto-triggered refresh that fires when a background download
    completes (_download_triggered_refresh_ui) - without it, finishing a
    download for a model that isn't active yet would silently reset the
    dropdowns back to whatever WAS active (or to nothing) right out from
    under the user, mid-download-queueing, regardless of who actually
    triggered this particular refresh."""
    models, mmprojs = scan_all()
    groups = merge_curated(group_models(models, mmprojs), load_curated_models())
    table = _models_table_rows(groups)

    cfg = state.current_cfg
    view_group = next((g for g in groups if str(g.folder) == selected_folder), None) if selected_folder else None
    if view_group is not None:
        is_active_group = any(q.name == cfg.model_name for q in view_group.quants)
    else:
        view_group = next((g for g in groups if any(q.name == cfg.model_name for q in g.quants)), None)
        is_active_group = view_group is not None

    quant_update, mmproj_update, action_update = _models_dropdown_updates(
        view_group,
        preferred_quant=cfg.model_name if is_active_group else None,
        preferred_mmproj=cfg.mmproj_name if is_active_group else None,
    )
    folder_key = str(view_group.folder) if view_group else None
    status = f"{len(groups)} model folder(s) found." if groups else f"No models found under {MODELS_DIR}."

    return table, groups, folder_key, quant_update, mmproj_update, action_update, status


def models_table_select_ui(groups: list[ModelGroup], evt: gr.SelectData):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not groups or not (0 <= row < len(groups)):
        return None, gr.update(), gr.update(), gr.update(), gr.update()
    group = groups[row]
    quant_update, mmproj_update, action_update = _models_dropdown_updates(group)
    if not group.quants:
        if group.curated:
            note = " - not downloaded yet; pick a quant below to see download info"
        else:
            note = " - no model file found (download incomplete?), can't be selected"
    elif not group.mmprojs:
        note = " - no mmproj found, can't be selected"
    else:
        note = ""
    return str(group.folder), quant_update, mmproj_update, action_update, f"Viewing {group.name}{note}."


def models_set_active_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """Only reached via models_action_ui when the current selection is
    fully local ("set_active" mode) - the refusal branches below are a
    defensive fallback for a stale/racing click, not the normal path for
    a not-yet-downloaded pick (that's models_download_ui's job now)."""
    noop = (gr.update(), gr.update(), gr.update())
    if not cached_controllable():
        return _models_table_rows(groups), *noop, (
            "Model selection has no effect right now - connected to a server this app doesn't control "
            "(external mode, or an unrecognized server already using the managed port)."
        )
    if not quant_name or quant_name == "N/A":
        return _models_table_rows(groups), *noop, "No model selected - click a row in the table first."

    group = next((g for g in groups if any(q.name == quant_name for q in g.quants)), None)
    if group is None:
        shown = quant_name.rsplit("/", 1)[-1]
        return _models_table_rows(groups), *noop, (
            f"\"{shown}\" isn't downloaded yet - use the Download button, not Set as active."
        )

    if mmproj_name == "N/A":
        # "N/A" is the sentinel _models_dropdown_updates uses specifically
        # for "this folder has no usable mmproj" - not a real empty/auto
        # value, so it must never be written as if it meant that (would
        # silently commit an unselectable model, only failing later at
        # server-start time instead of refusing here where the reason is
        # actually known).
        return _models_table_rows(groups), *noop, "This model has no mmproj - can't be set as active."
    if mmproj_name and not any(m.name == mmproj_name for m in group.mmprojs):
        shown = mmproj_name.rsplit("/", 1)[-1]
        return _models_table_rows(groups), *noop, (
            f"\"{shown}\" isn't downloaded yet - use the Download button, not Set as active."
        )
    mmproj_value = "" if mmproj_name is None else mmproj_name
    with state.config_lock:
        state.current_cfg = replace(state.current_cfg, model_name=quant_name, mmproj_name=mmproj_value)
        config_mod.save(state.current_cfg)
    log.info("Models tab: set active model to %s (mmproj=%s)", quant_name, mmproj_value or "(auto)")
    mmproj_note = f"mmproj: {mmproj_value}" if mmproj_value else "mmproj: auto-pick largest"

    # Rebuild the dropdowns fresh from the group that was just committed,
    # rather than leaving whatever was on screen before untouched - that
    # stale-state gap (a disabled "N/A" mmproj dropdown surviving into a
    # later action that assumes real choices) is what actually caused the
    # "value N/A not in choices []" error. `group` is already the right
    # one, found above.
    quant_update, mmproj_update, action_update = _models_dropdown_updates(
        group, preferred_quant=quant_name, preferred_mmproj=mmproj_name
    )
    return (
        _models_table_rows(groups),
        quant_update, mmproj_update, action_update,
        f"Active model set to {quant_name} ({mmproj_note}). "
        "Click \"Restart server connection\" below to actually load it.",
    )


def models_download_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """Only reached via models_action_ui when the current selection has
    something not yet local ("download" mode). Queues whichever of
    quant/mmproj isn't on disk yet - possibly both - and leaves the
    dropdowns/table alone, since nothing about what's locally available
    has actually changed yet."""
    noop = (gr.update(), gr.update(), gr.update())
    if not quant_name or quant_name == "N/A":
        return _models_table_rows(groups), *noop, "No model selected - click a row in the table first."

    group = _group_for_quant_value(groups, quant_name)
    if group is None or group.curated is None:
        return _models_table_rows(groups), *noop, "Nothing curated to download for this selection."

    to_queue: list[DownloadItem] = []
    if not any(q.name == quant_name for q in group.quants):
        stem = quant_name.split("/", 1)[1]
        cq = next((c for c in group.curated.quants if c.name == stem), None)
        if cq:
            to_queue.append(DownloadItem(
                url=cq.url, dest_path=group.folder / Path(cq.url).name,
                label=cq.name, size_bytes=cq.size_bytes,
            ))
    if mmproj_name not in (None, "N/A") and not any(m.name == mmproj_name for m in group.mmprojs):
        stem = mmproj_name.split("/", 1)[1]
        cm = next((c for c in group.curated.mmprojs if c.name == stem), None)
        if cm:
            to_queue.append(DownloadItem(
                url=cm.url, dest_path=group.folder / Path(cm.url).name,
                label=cm.name, size_bytes=cm.size_bytes,
            ))

    if not to_queue:
        return _models_table_rows(groups), *noop, "Already downloaded - nothing to queue."

    added = download_enqueue(to_queue)
    if not added:
        return _models_table_rows(groups), *noop, "Already queued or downloading."
    labels = ", ".join(i.label for i in added)
    log.info("Models tab: queued for download: %s", labels)
    return _models_table_rows(groups), *noop, f"Queued for download: {labels}."


def models_action_ui(groups: list[ModelGroup], quant_name: str, mmproj_name: str):
    """The single "Set as active model"/"Download" button's click target -
    decides which of the two the current selection actually means (same
    logic _models_dropdown_updates used to pick the button's label, via
    _action_mode_for_selection) and dispatches to it, then refreshes the
    download-status row too in case this action changed the queue."""
    group = _group_for_quant_value(groups, quant_name)
    mode = _action_mode_for_selection(group, quant_name, mmproj_name, cached_controllable())
    if mode == "download":
        table_u, quant_u, mmproj_u, action_u, status = models_download_ui(groups, quant_name, mmproj_name)
    else:
        table_u, quant_u, mmproj_u, action_u, status = models_set_active_ui(groups, quant_name, mmproj_name)
    row_u, text_u = download_status_ui()
    return table_u, quant_u, mmproj_u, action_u, status, row_u, text_u


def build(ctx: BuildContext):
    with gr.Tab("Models", interactive=False) as models_tab:
        gr.Markdown("### VLM model")
        gr.Markdown(
            config_mod.MODELS_TAB_INTRO.format(ignored_substrings=", ".join(IGNORED_SUBSTRINGS))
        )
        models_table = gr.Dataframe(
            headers=["A", "Model", "Source", "Quants"], datatype=["str", "str", "str", "str"],
            interactive=False, row_count=(0, "dynamic"), buttons=[],
        )
        with gr.Row():
            # allow_custom_value=True is load-bearing, not cosmetic: it
            # skips Gradio's strict "submitted value must be in the
            # current choices" check, which otherwise crashes the whole
            # request when a table-row switch reprograms these dropdowns'
            # choices at the same moment a .change() event from the old
            # row is still in flight (see the Models tab handlers' own
            # comment block below for the full story). Our own handlers
            # already treat an unrecognized value as "not selected" -
            # this only removes Gradio's redundant, crash-prone copy of
            # that same check.
            models_quant_dropdown = gr.Dropdown(label="Quant", interactive=False, allow_custom_value=True)
            models_mmproj_dropdown = gr.Dropdown(label="mmproj", interactive=False, allow_custom_value=True)
        with gr.Row():
            models_action_btn = gr.Button("Set as active model", variant="primary", interactive=False)
            restart_server_btn = gr.Button("Manage llama server")
            refresh_models_btn = gr.Button("Refresh")

        # Deliberately minimal for now, mirroring the VLM section's
        # look (same 4 columns) without its "set as active model"
        # machinery - there's only ever one Hydra model, no
        # quant/mmproj pair to pick between, so a click-to-select
        # row + dropdowns would have nothing real to do. Refresh
        # here is literally the same handler as the VLM section's
        # own Refresh (redundant today - this row doesn't need a
        # live rescan of webui/models/ - kept only for UX
        # consistency with a "there's a Refresh button on this
        # section too" expectation).
        #
        # Not wrapped in a gr.Group() (nor is the VLM section above) -
        # tried once and reverted, see ui/css.py's own note on
        # gr.Group() (also in this project's CLAUDE.md gotchas):
        # it merges mismatched-height siblings - a Markdown block, a
        # Dataframe, and Rows of buttons here - into one stretched
        # grey slab instead of leaving each its own distinct look,
        # confirmed live. Group is only safe for genuinely uniform
        # stacked rows (see Settings' own Groups), not a mixed
        # section like this one.
        #
        # "---" is plain CommonMark (a real <hr>, not a custom rule) -
        # gr.Markdown already renders it, no CSS needed.
        gr.Markdown("---")
        gr.Markdown("### Hydra model")
        hydra_models_table = gr.Dataframe(
            headers=["A", "Model", "Source", "Quants"], datatype=["str", "str", "str", "str"],
            interactive=False, row_count=(1, "fixed"), buttons=[],
            value=_hydra_models_table_rows(),
        )
        with gr.Row():
            hydra_models_download_btn = gr.Button("Download Hydra model (~1GB)")
            hydra_models_manage_btn = gr.Button("Manage Hydra")
            hydra_models_refresh_btn = gr.Button("Refresh")

        # Single shared row for the one underlying download queue -
        # the VLM and Hydra sections above both feed the same
        # _download_queue/_download_worker (see ui.downloads_queue),
        # so there is only ever one thing actually
        # downloading at a time regardless of which section queued
        # it. Two separate status rows (one per section) used to
        # each mirror the same global _download_current, which just
        # showed identical, section-mismatched content in both
        # places (e.g. the Hydra row showing a VLM download's
        # progress) - a single row below both sections is the
        # honest reflection of "one queue, one download bar, one
        # abort button".
        gr.Markdown("---")
        with gr.Row(visible=False) as download_status_row:
            download_status_text = gr.HTML(container=False, scale=4)
            download_abort_btn = gr.Button("Abort all downloads", scale=1)

        models_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # See this tab's own handler-function docstrings above for what
        # each of these two actually holds.
        models_groups_state = gr.State([])
        models_selected_folder_state = gr.State(None)

        _models_scan_outputs = [
            models_table, models_groups_state, models_selected_folder_state,
            models_quant_dropdown, models_mmproj_dropdown, models_action_btn, models_infotext,
        ]
        refresh_models_btn.click(models_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
        hydra_models_download_btn.click(
            _hydra_download_model_ui, [], [models_infotext, download_status_row, download_status_text]
        )
        hydra_models_refresh_btn.click(
            models_refresh_ui, [models_selected_folder_state], _models_scan_outputs
        )
        models_table.select(
            models_table_select_ui,
            [models_groups_state],
            [
                models_selected_folder_state, models_quant_dropdown, models_mmproj_dropdown,
                models_action_btn, models_infotext,
            ],
        )
        models_action_btn.click(
            models_action_ui,
            [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
            [
                models_table, models_quant_dropdown, models_mmproj_dropdown, models_action_btn, models_infotext,
                download_status_row, download_status_text,
            ],
        )
        download_abort_btn.click(
            download_abort_all_ui,
            [models_selected_folder_state],
            [*_models_scan_outputs, download_status_row, download_status_text],
        )
        models_quant_dropdown.change(
            models_selection_change_ui,
            [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
            [models_action_btn],
        )
        models_mmproj_dropdown.change(
            models_selection_change_ui,
            [models_groups_state, models_quant_dropdown, models_mmproj_dropdown],
            [models_action_btn],
        )

    # Gradio remounts a tab's contents to their build-time state when you
    # switch back into it: a background download finishing while the user
    # is on another tab gets picked up by _download_triggered_refresh_ui
    # within 2s (its own docstring), but that server-side update is
    # discarded the moment the user switches back to Models - so re-run a
    # real rescan right on re-entry too instead of leaving a stale count
    # until a manual Refresh click.
    models_tab.select(models_refresh_ui, [models_selected_folder_state], _models_scan_outputs)

    ctx.on_tick(download_status_ui, [], [download_status_row, download_status_text])
    ctx.on_tick(_download_triggered_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
    ctx.on_load(models_refresh_ui, [models_selected_folder_state], _models_scan_outputs)
    ctx.on_load(download_status_ui, [], [download_status_row, download_status_text])

    return SimpleNamespace(
        tab=models_tab,
        manage_llama_btn=restart_server_btn,
        manage_hydra_btn=hydra_models_manage_btn,
    )
