"""Settings tab: the sub-tab container, operation gating of the sub-tabs,
and the shared Save settings / Restart app row."""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import gradio as gr

from core import config as config_mod
from core import hydra_classifier
from core.config import AppConfig
from ui import state
from ui.context import BuildContext
from ui.downloads_queue import download_abort_all
from ui.llama_session import stop_managed, try_start_managed_llama
from ui.operation import operation_force_abort, settings_subtabs_disabled, wait_for_operation_to_end
from ui.settings import captioning_defaults as settings_captioning_defaults
from ui.settings import debug as settings_debug
from ui.settings import hydra as settings_hydra
from ui.settings import image_resizing as settings_image_resizing
from ui.settings import llama as settings_llama
from ui.settings.hydra import hydra_sliders_to_metric
from ui.settings.llama import llama_install_abort_all

log = logging.getLogger("app")  # fixed name, see app.py


def _settings_gating_ui():
    """Disables the operation-sensitive Settings sub-tabs' own
    clickability (see settings_subtabs_disabled) while a Single/Batch/
    Recaption job is actively running, so a job's settings can't change
    out from under it mid-flight. The Debug sub-tab is deliberately
    excluded - its one checkbox already needs a full app restart to take
    effect regardless (see its own comment), so there's nothing it could
    disrupt by staying editable. Deliberately does NOT touch the fields
    inside those sub-tabs, only the tabs' own interactive= - see
    _settings_subtab_entry_ui below for what actually protects someone
    already sitting on one of these when a job starts (a redirect on
    next entry, not disabling the fields themselves)."""
    update = gr.update(interactive=not settings_subtabs_disabled())
    return update, update, update, update


# gr.SelectData.value for a Tab is ALWAYS its label, never its id -
# confirmed via gradio/layouts/tabs.py's own EVENTS docstring ("value
# referring to the label of the Tab"), and via a live debug-log capture
# that showed "last_chosen=Hydra" (the label) instead of the expected
# "hydra-settings" (the id). id is otherwise only used for the selected=
# kwarg when PUSHING a tab switch (gr.Tab's own docstring: "required if
# you wish to control the selected tab from a predict function") - so
# without this table, a value read off a real click could never be fed
# back into gr.update(selected=...) and have it actually match anything.
# Three of the five sub-tabs (Hydra, Image resizing, Captioning
# defaults) never had an id at all until this fix, which independently
# meant they could never have been a valid push target either way - see
# their own gr.Tab(...) declarations.
_SETTINGS_SUBTAB_ID_BY_LABEL = {
    "Llama": "llama-settings",
    "Hydra": "hydra-settings",
    "Image resizing": "image-resizing-settings",
    "Captioning defaults": "captioning-defaults-settings",
    "Debug": "debug-settings",
}


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
    stop our own managed llama-server first (execv skips atexit entirely),
    and abort any in-flight/queued downloads or llama.cpp install too -
    all in-memory only and won't survive this regardless, better to say
    so cleanly. Also unload Hydra if loaded - unlike llama-server (a
    separate OS process stop_managed kills outright), Hydra's model
    lives in THIS process's own CUDA context, which execv does not
    clean up on its own (it replaces the process image but skips normal
    interpreter/CUDA teardown) - explicit unload() first avoids leaking
    VRAM into the replaced process.
    """
    log.info("User triggered app restart from Settings")
    operation_force_abort()
    wait_for_operation_to_end()
    download_abort_all()
    llama_install_abort_all()
    stop_managed()
    hydra_classifier.unload()
    os.execv(sys.executable, [sys.executable] + sys.argv)


def save_settings_ui(
    server_mode, server_port, external_url,
    n_gpu_layers, context_size, extra_server_args, autostart_managed_llama,
    resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
    prompt_template, temperature, top_p, max_tokens, request_timeout,
    trigger_word, overwrite_existing, debug_tab_enabled,
    hydra_enabled, hydra_device, hydra_confidence, hydra_threshold, hydra_implications,
    hydra_exclude_categories, hydra_exclude_tags, hydra_max_tags, hydra_autoload_model,
    hydra_tag_vocab_path,
) -> str:
    """Per-category behavior on save (see the design discussion this was
    built from): image resizing/captioning defaults/prompt-template-bundle
    fields are just written through, live immediately, no process
    interaction - already true by construction, nothing special needed
    for them below. Server-process-affecting fields (mode, port, GPU
    layers, context size, extra args) each need a specific reaction:
    switching to external stops any managed server we own; switching to
    managed never auto-starts (the user must do that explicitly, even
    with autostart on - autostart is for the *next app launch*, not this
    save); changing the process args while staying managed stops the
    current one (a no-op if we don't own it - see stop_managed) and,
    only if autostart is enabled, immediately starts a fresh one with the
    new settings."""
    old_cfg = state.current_cfg
    # model_name/mmproj_name are deliberately NOT settable from this form -
    # they live entirely in the Models tab now (models_set_active_ui), so
    # carry forward whatever's already configured rather than defaulting
    # to empty just because this form has no field for them.
    with state.config_lock:
        state.current_cfg = AppConfig(
            server_mode=server_mode,
            server_port=int(server_port),
            external_url=external_url,
            model_name=state.current_cfg.model_name,
            mmproj_name=state.current_cfg.mmproj_name,
            n_gpu_layers=n_gpu_layers.strip() or "auto",
            context_size=int(context_size),
            extra_server_args=extra_server_args,
            autostart_managed_llama=bool(autostart_managed_llama),
            resize_enabled=bool(resize_enabled),
            resize_target_mp=float(resize_target_mp),
            snap_enabled=bool(snap_enabled),
            snap_multiple=int(snap_multiple),
            prompt_template=prompt_template,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            request_timeout=int(request_timeout),
            trigger_word=trigger_word,
            overwrite_existing=bool(overwrite_existing),
            debug_tab_enabled=bool(debug_tab_enabled),
            hydra_enabled=bool(hydra_enabled),
            hydra_device=hydra_device,
            hydra_metric=hydra_sliders_to_metric(float(hydra_confidence), float(hydra_threshold)),
            hydra_implications=hydra_implications,
            hydra_exclude_categories=hydra_exclude_categories,
            hydra_exclude_tags=hydra_exclude_tags,
            hydra_max_tags=int(hydra_max_tags),
            hydra_autoload_model=bool(hydra_autoload_model),
            hydra_tag_vocab_path=hydra_tag_vocab_path,
        )
        new_cfg = state.current_cfg
        config_mod.save(new_cfg)
    log.info(
        "Settings saved: server_mode=%s model=%s ngl=%s",
        new_cfg.server_mode, new_cfg.model_name or "(none)", new_cfg.n_gpu_layers,
    )

    switched_to_external = old_cfg.server_mode != "external" and new_cfg.server_mode == "external"
    switched_to_managed = old_cfg.server_mode == "external" and new_cfg.server_mode == "managed"
    process_args_changed = (
        old_cfg.server_port, old_cfg.n_gpu_layers, old_cfg.context_size, old_cfg.extra_server_args
    ) != (
        new_cfg.server_port, new_cfg.n_gpu_layers, new_cfg.context_size, new_cfg.extra_server_args
    )

    if switched_to_external:
        stop_managed()
        message = "Settings saved. Switched to external mode."
    elif switched_to_managed:
        message = (
            "Settings saved. Switched to managed mode - click \"Start llama server\" "
            "in this tab when you're ready (autostart applies on the next app launch, not now)."
        )
    elif new_cfg.server_mode == "managed" and process_args_changed:
        stop_managed()
        if new_cfg.autostart_managed_llama:
            _, start_message = try_start_managed_llama()
            message = f"Settings saved. {start_message}"
        else:
            message = "Settings saved. Click \"Start llama server\" to apply the new server settings."
    else:
        message = "Settings saved."

    return message


def build(ctx: BuildContext):
    with gr.Tab("Settings", id="settings") as settings_tab:
        # Declared here, before settings_tabs opens, not as a
        # direct child of `with gr.Tabs()` itself - see current_
        # tab_label_state's own comment in app.py for why a gr.State
        # directly inside a Tabs block was confirmed live to
        # corrupt the rendered tab strip.
        current_settings_subtab_state = gr.State("llama-settings")
        with gr.Tabs() as settings_tabs:
            llama = settings_llama.build(ctx)
            hydra = settings_hydra.build(ctx)
            image_resizing = settings_image_resizing.build(ctx)
            captioning_defaults = settings_captioning_defaults.build(ctx)
            debug = settings_debug.build(ctx)

        with gr.Row():
            save_settings_btn = gr.Button("Save settings", variant="primary")
            restart_app_btn = gr.Button("Restart app")
        settings_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # Must match save_settings_ui's parameter order exactly.
        settings_inputs = [
            llama.server_mode, llama.server_port, llama.external_url,
            llama.n_gpu_layers, llama.context_size, llama.extra_server_args, llama.autostart_managed_llama,
            image_resizing.resize_enabled, image_resizing.resize_target_mp,
            image_resizing.snap_enabled, image_resizing.snap_multiple,
            llama.prompt_template, llama.temperature, llama.top_p, llama.max_tokens, llama.request_timeout,
            captioning_defaults.trigger_word, captioning_defaults.overwrite_existing, debug.debug_tab_enabled,
            hydra.hydra_enabled, hydra.hydra_device, hydra.hydra_confidence, hydra.hydra_threshold,
            hydra.hydra_implications, hydra.hydra_exclude_categories, hydra.hydra_exclude_tags,
            hydra.hydra_max_tags, hydra.hydra_autoload_model, hydra.hydra_tag_vocab_path,
        ]
        # Saving can change server_mode (or anything else
        # check_status/cached_controllable/cached_reachable
        # read), and the Models/Single-image/Batch-processing
        # tabs' interactive state would otherwise be stale for
        # up to 8s until reachability_timer's next tick -
        # refresh it immediately instead. current_tab_label_
        # state is "Settings" here (this click can only happen
        # from the Settings tab), so the push-to-Settings
        # branch in _refresh_reachability_ui is always a no-op
        # in this context, not a real redirect.
        ctx.after_server_change(save_settings_btn.click(save_settings_ui, settings_inputs, [settings_infotext]))
        restart_app_btn.click(restart_app_ui, [], [])

    # Tracks which Settings sub-tab was genuinely clicked into, as
    # opposed to whatever settings_tabs is currently displaying (see
    # _on_settings_subtab_select's own docstring for why those two
    # things need to be kept separate).
    settings_tabs.select(_on_settings_subtab_select, [], [current_settings_subtab_state])
    # Redirects to Debug on entering Settings while the sub-tabs are
    # disabled - see _settings_subtab_entry_ui's own docstring.
    settings_tab.select(_settings_subtab_entry_ui, [current_settings_subtab_state], [settings_tabs])

    _settings_gated_tabs = [llama.tab, hydra.tab, image_resizing.tab, captioning_defaults.tab]
    ctx.on_tick(_settings_gating_ui, [], _settings_gated_tabs)
    ctx.on_load(_settings_gating_ui, [], _settings_gated_tabs)

    return SimpleNamespace(
        tab=settings_tab,
        settings_tabs=settings_tabs,
        current_subtab_state=current_settings_subtab_state,
        llama=llama,
        hydra=hydra,
    )
