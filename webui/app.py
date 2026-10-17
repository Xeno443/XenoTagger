"""Gradio UI entry point and frame: Single-image captioning, Batch
processing, Review, Models, Settings, and an opt-in Debuglog tab. The CLI
(cli.py) is a separate, independent entry point that shares core/ and
backends/ with this file but not this file or ui/ - nothing UI-specific
belongs in core/, or in a backend outside its settings_ui.py.

This file only builds the frame: the Blocks, the top-level Tabs, the
status bar, the two timers, the launch chain (each backend's autoload, in
core.lifecycle.autoload_order), and the wiring that crosses tabs (Batch ->
Review, Models <-> a backend's Settings sub-tab, the reachability refresh
that gates Single/Batch). Each tab's own components and handlers live in
ui/tab_*.py, the Settings sub-tabs in each backend's settings_ui.py
(backends/<id>/) and in ui/settings/, and the UI-side state several tabs
share in ui/state.py, ui/operation.py, ui/reachability.py and
ui/downloads_queue.py - see ui/__init__.py and backends/__init__.py for
the import rules that keep those acyclic. Tabs hand frame-level wiring
back through a BuildContext (ui/context.py) instead of reaching for
components they don't own.

Three coordination mechanisms guard shared, process-wide state that
several tabs (and Settings' restart actions) can touch concurrently - the
first two are threading.RLock, not a plain Lock, because each has a
function that calls another lock-holding function of its own while
already holding the lock (get_client() calls stop_managed(); a plain
Lock would deadlock there):

  - `_session`/`_session_lock` (backends/llama/session.py): which llama-server
    (if any) we're currently managing. Guards against two requests racing
    in before any server is up yet (e.g. a batch run and a single-image
    caption both arriving at once) each deciding independently that no
    server exists and each starting their own.

  - `_active_operation`/`operation_lock` (ui/operation.py): is a
    long-running job (single-image, batch or review recaption) currently
    active, and what stage of being interrupted is it in. This is also the
    mutual-exclusion gate that stops them from running at once, since
    they all ultimately share the one llama-server connection.

  - `current_cfg`/`config_lock` (ui/state.py; plain Lock - neither writer
    calls back into itself or the other while holding it): guards the
    read-modify-write of current_cfg + config_mod.save() in
    save_settings_ui() and the Models tab's models_action_ui() ("Set as
    active model"), the only two places that mutate current_cfg wholesale. Without it, two overlapping saves
    could each read a stale current_cfg and the second write would
    silently clobber fields the first one just changed.

The Run/Interrupt buttons are two SEPARATE gr.Button components per tab,
not one button whose label changes - this was tried first and doesn't
work: Gradio disables the triggering component of a still-pending event
client-side, and defaults every event listener's server-side
concurrency_limit to 1, so a second click on the button that's still
running the long captioning/batch call either never reaches the server
or queues silently behind it - never actually interrupts anything. The
fix (and the reason it's structured as two buttons, mirroring how Forge/
A1111 do this in modules/ui_toprow.py: Generate, Interrupt, and Skip are
independent gr.Button()s there too) is that the Interrupt button was
never the one that started the long call, so it's never "busy" and is
always immediately clickable - its own click handler is fast and
non-blocking (flip a flag; on the second click, kill the managed
llama-server process - see backends/llama/server.py's ManagedServer.stop for why a
hard kill is the only reliable way to actually interrupt an in-flight
request). First click requests a graceful stop (batch: finish the
current image, then stop before the next one; single-image has no next
item to gracefully stop before, so this click is intentionally a no-op
buffer against a stray double-click); second click force-aborts.

On the Single-image tab specifically, Caption and Interrupt each get
their OWN gr.Column (single_run_col/single_interrupt_col) rather than
sharing one - swapping which Column is visible, not the buttons inside
them. This is a layout fix, not just the concurrency one above: Gradio
gives every gr.Row() its own independent flex computation with no shared
alignment across separate Rows (it's not a table), so the row above
(image | caption box) and this button row only ever lined up because both
happened to be simple two-Column shapes computing the same coincidental
50/50 split. A CSS-based single-Column stack (Interrupt absolutely/grid
positioned over Caption) was tried first and worked functionally, but the
Column's own default padding/gap - which a bare Button in the original
layout never had - broke that coincidental alignment, through several
rounds of trying to CSS-patch it back. Two separate Columns, each either
visible or not, sidesteps the whole problem: at any moment exactly two of
the three Columns (Caption, Interrupt, Save caption) are visible, always
that same plain "Row of two Columns" shape confirmed to align correctly -
whichever two they happen to be. See ui/css.py's own note on the
abandoned CSS approach for the full story.

Because button state is normally only pushed by whichever specific
browser connection's generator is actively running (run_single_ui /
run_batch_ui's own yields), a reloaded page or a second browser tab would
otherwise show stale "idle" buttons even while something is genuinely
running elsewhere. _update_ui_status() closes that gap: it
recomputes both tabs' Run/Interrupt appearance fresh from
_active_operation on every call, and is wired into the same 2-second
status_timer that already refreshes the status bar, plus demo.load - so
any view self-corrects within one tick instead of staying wrong
indefinitely.

restart_app_ui (ui/tab_settings.py) doesn't block just because something's
running, nor does it silently kill it out from under whatever's in flight
- it force-aborts (skipping the two-click grace period; see
operation_force_abort) and actually waits for that operation to finish
noticing and clean up (wait_for_operation_to_end) before touching the
server. This matters
because os.execv replaces the entire process with zero chance for any
cleanup to run afterward - waiting first means an interrupted operation
gets an observable, handled failure instead of just vanishing without a
trace. The Models tab's "Open <backend> settings" button is pure
navigation to that backend's Settings sub-tab - actually stopping/
starting llama-server only ever happens from the Llama sub-tab's
Start/End buttons, which are already correctly gated (disabled mid-job,
see _settings_gating_ui) so there's no force-abort race for a plain
redirect to need to guard against.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform

# Must be set before `import gradio` - some of its telemetry checks read
# this at import time, not just when building a Blocks instance. The ui
# package imports gradio too, so this has to come before those imports.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from backends import BACKENDS
from backends.llama.session import cached_reachable, refresh_reachability_cache
from core import config as config_mod
from core import tag_vocab
from core.lifecycle import autoload_order
from ui import state, tab_batch, tab_debuglog, tab_models, tab_review, tab_settings, tab_single
from ui.context import REACHABILITY, BuildContext
from ui.css import ALL_CSS
from ui.operation import current_operation, operation_lock, operation_status_text
from ui.reachability import tab_gating_updates, track_reachability_transitions
from ui.tab_debuglog import setup_debug_logging
from ui.tab_review import review_scan_ui

log = logging.getLogger("app")  # fixed name, not __name__ - which is "__main__"
                                  # when run directly (the real usage) but
                                  # "app" when imported (e.g. in tests) -
                                  # this way logging.getLogger("app") always
                                  # refers to the right logger either way.


def _refresh_reachability_ui(current_tab_label: str):
    """Wired to reachability_timer.tick(). Refreshes the cache (picked up
    by the next status_timer tick for Single/Batch/Recaption's own Run/
    Recaption BUTTON gating, see the cache's own comment above) and
    separately updates two TABS themselves: Single-image/Batch processing
    (disabled unless reachable at all - captioning there is pointless the
    moment nothing can answer). If the user is on one of these when it becomes disabled,
    pushes them off to Settings rather than leaving them stranded on a tab
    they can no longer switch back into. Never touches main_tabs'
    selection otherwise, so it can't yank anyone off Review mid-task -
    Review's own filesystem-only nav doesn't need a server at all (see
    _update_ui_status' docstring), so its tab stays reachable regardless;
    only its Recaption button is gated, same as Single/Batch's Run
    buttons.

    Also updates current_tab_label_state to "Settings" whenever it
    redirects there - main_tabs.select() (which is what normally keeps
    that state current, see _on_main_tab_select) only fires on an actual
    user click, never on a server-pushed gr.update(selected=...) like the
    one this function itself issues. Without this, the tracked label
    would silently go stale the moment THIS function redirects someone,
    and every later poll would keep reading the pre-redirect tab -
    confirmed live: this exact gap was yanking users from Settings->Llama
    to Settings->Debug a few seconds later, via _fallback_tab_safety_ui
    still seeing the stale "Single image"/"Batch processing" label and
    concluding they were still stuck on a disabled tab. gr.skip() leaves
    it untouched on ticks that don't redirect."""
    cfg = state.current_cfg
    status = refresh_reachability_cache(cfg)
    track_reachability_transitions(cfg, status)
    push_to_settings = not status.reachable and current_tab_label in ("Single image", "Batch processing")
    tabs_update = gr.update(selected="settings") if push_to_settings else gr.update()
    label_update = "Settings" if push_to_settings else gr.skip()
    return (*tab_gating_updates(cfg, status), tabs_update, label_update)


def _on_main_tab_select(evt: gr.SelectData) -> str:
    """Wired to main_tabs.select() - see current_tab_label_state's own
    comment for why this is tracked at all."""
    return evt.value


def _goto_backend_settings_ui(selected):
    """The Models tab's "Open <backend> settings" button's click target -
    pure navigation to the selected row's backend's Settings sub-tab
    (selected is the Models tab's [backend_id, key] state), no action of
    its own.

    Explicitly asserts every destination (main_tabs, settings_tabs,
    current_settings_subtab_state, current_tab_label_state) rather than a
    bare gr.update() no-op on any of them - _refresh_reachability_ui's own
    docstring covers why a no-op can't be trusted to self-correct a
    still-pending prior push to a Tabs component's selected=, and
    current_tab_label_state/current_settings_subtab_state both need to
    reflect this redirect immediately since neither main_tabs.select() nor
    settings_tabs.select() reliably fire on a server-pushed switch like
    this one (see their own comments)."""
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
    """The Batch tab's "Send to Review" button - pure navigation + handoff
    of the directory field to the Review tab (same "explicitly assert
    every destination" reasoning as _goto_backend_settings_ui above -
    main_tabs.select() doesn't reliably fire on a server-pushed switch
    like this one). Wired in build_app's cross-tab section, once both
    tabs exist. Chained with .then(review_scan_ui, ...) there, same
    two-step "set the field, then scan" shape review_browse_btn's own
    click already uses - there's no success/failure signal from a batch
    run worth reflecting here (see the button's own interactive= gating:
    a valid directory is the only thing checked, not whether the last
    batch run in it actually succeeded)."""
    return gr.update(selected="review"), directory, "Review"


def _startup_reachability_ui():
    """Wired to demo.load() - decides which tab a freshly-loaded page
    lands on, and the Single-image/Batch-processing tabs' initial
    interactive state (see _refresh_reachability_ui, which this shares its
    per-tab logic with). A real (not cached) check, since the optimistic
    default cache value would otherwise land everyone on Single-image once
    with everything enabled, even on a machine where nothing's been set up
    yet. Also sets current_tab_label_state to match wherever it actually
    lands - the very first value driving every later poll, before any
    real main_tabs.select() has ever fired (see _refresh_reachability_ui's
    docstring for why this matters and what silently goes wrong without
    it)."""
    cfg = state.current_cfg
    status = refresh_reachability_cache(cfg)
    track_reachability_transitions(cfg, status)
    tab_target, tab_label = ("single", "Single image") if status.reachable else ("settings", "Settings")
    return (gr.update(selected=tab_target), *tab_gating_updates(cfg, status), tab_label)


def _fallback_tab_safety_ui(current_tab_label: str):
    """Cheap, cache-only 2s-timer safety net: if the user is currently
    sitting on a tab the cached state says is disabled, push them to
    Settings -> Debug (the one sub-tab that's never gated on anything -
    see its own comment) regardless of why. A catch-all in case the
    primary per-transition push logic in _refresh_reachability_ui/
    _startup_reachability_ui missed a case - reads the same caches those
    already refresh, no network call of its own, safe to run this often.
    Also updates current_tab_label_state to "Settings" when it redirects,
    same reasoning as _refresh_reachability_ui - a server-pushed tab
    switch doesn't fire main_tabs.select() on its own."""
    if current_tab_label in ("Single image", "Batch processing") and not cached_reachable():
        return gr.update(selected="settings"), gr.update(selected="debug-settings"), "Settings"
    return gr.update(), gr.update(), gr.skip()


def _update_ui_status():
    """Ground-truth Run/Interrupt appearance for all three tabs, recomputed
    fresh from _active_operation every call - not just whatever a single
    running generator last pushed. Wired into the periodic status timer
    and page load below, so a reloaded page, a second browser tab, or a
    missed/delayed update all self-correct within one tick instead of
    showing stale state indefinitely.

    All three use column-visibility swapping (single_run_col <->
    single_interrupt_col, etc. - see the Single-image tab and app.py's own
    module docstring for why: confirmed live that a Row of two plain
    gr.Column()s aligns correctly with a neighboring row with zero CSS, so
    toggling which two of N Columns are visible reuses that proven shape
    instead of hand-tuned CSS. Batch/Review's own rows have no neighboring
    row to misalign against, but use the same pattern anyway for
    consistency).

    Review recaptioning is its own "review" operation kind, distinct from
    "single" even though it calls the exact same caption_image() - if it
    shared the "single" kind, the Single-image tab's own Interrupt would
    light up whenever Review (not it) was the one running, which would be
    backwards. A third kind gives it its own correct run/interrupt state
    while still mutually excluding against the other two for free (the
    existing "blocked if op.kind != kind" check needs no changes to cover
    a third kind).

    review_recaption_btn's own interactive state follows the same
    "blocked by something else" pattern single_run_btn/batch_run_btn
    already use - recaptioning genuinely can't run concurrently with
    Single/Batch (all three share the one llama-server connection), so
    it needs to be visibly disabled the same way, not just refuse with a
    message after being clicked.

    All three Run/Recaption buttons are ALSO blocked when no server is
    reachable right now (cached_reachable(), see the Server reachability
    section above) - starting a caption request when there's nothing to
    answer it is exactly as pointless as starting a second operation
    while one's already running, so it's gated the same way. This is why
    the check is cached rather than computed fresh here: it involves a
    real network call for external mode, and this function runs on every
    2s status_timer tick.

    Review's OTHER nav (prev/next/table/dir/browse/scan), though, is
    pure filesystem/UI - scanning or browsing a folder doesn't touch the
    llama-server at all, so there's no real conflict with a Single/Batch
    job running elsewhere. It's disabled only while REVIEW ITSELF is
    recaptioning (op.kind == "review"), not whenever anything anywhere
    is active - browsing a different, already-tagged folder while a
    Batch job elsewhere is still running is a normal, useful thing to
    want to do.

    Each tab computes its own share (tab_*.run_controls_update, mostly
    ui.operation.run_controls_update); this just concatenates them in the
    order of _run_interrupt_btns in build_app - Single's, Batch's, then
    Review's run_controls.
    """
    reachable = cached_reachable()
    with operation_lock:
        op = current_operation()
        return (
            *tab_single.run_controls_update(op, reachable),
            *tab_batch.run_controls_update(op, reachable),
            *tab_review.run_controls_update(op, reachable),
        )


# ------------------------------------------------------------------ Status bar

def get_status_text() -> str:
    """Version/Python/Gradio, then each backend's own segments in registry
    order (Backend.status_bar_segments), then the active operation."""
    cfg = state.current_cfg
    segments = [
        ("Version", config_mod.app_version()),
        ("Python", platform.python_version()),
        ("Gradio", gr.__version__),
    ]
    for backend in BACKENDS:
        segments.extend(backend.status_bar_segments(cfg))
    segments.append(("Status", operation_status_text()))
    return " &nbsp;·&nbsp; ".join(f"{label}: {value}" for label, value in segments)


def build_app() -> gr.Blocks:
    ctx = BuildContext()

    with gr.Blocks(title="XenoTagger", analytics_enabled=False) as demo:
        gr.Markdown("# XenoTagger - LoRA dataset captioning")
        # Plain markdown link, not raw HTML - Gradio's own Markdown
        # renderer already rewrites any real (non "#...") link to
        # target="_blank" rel="noopener noreferrer" client-side, so this
        # opens in a new browser tab with no extra markup needed.
        gr.Markdown(
            "Check the "
            "[readme](https://github.com/Xeno443/XenoTagger/blob/main/readme/README.md)!"
        )

        # Declared up front, outside main_tabs, before any tab builds. Must NOT
        # be declared as a direct child of `with gr.Tabs()` even though
        # gr.State has no DOM footprint of its own and Gradio's own
        # validation explicitly allows it there (see Tabs.__exit__) -
        # confirmed live that doing so still corrupts the rendered tab
        # strip (labels duplicating/shifting on click), so it's declared
        # here instead, before main_tabs even opens.
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
        # Every output of the reachability refresh - Single/Batch tab
        # gating, the Llama sub-tab's Start/Stop, and the push-to-Settings
        # redirect. Shared by reachability_timer, the autostart chain, and
        # every button a tab registered via ctx.after_server_change.
        # Order is the contract ui.context.REACHABILITY documents.
        llama_ui = settings.backends["llama"]
        _reachability_outputs = [
            single.tab, batch.tab, llama_ui.start_btn, llama_ui.end_btn, main_tabs, current_tab_label_state,
        ]

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
        for event in ctx.server_change_events:
            event.then(_refresh_reachability_ui, [current_tab_label_state], _reachability_outputs)

        # Switching to any of these tabs remounts their Columns back to
        # whatever visibility was declared at Blocks-build time, not the
        # latest server-pushed state - Gradio's own quirk, not something
        # we did. status_timer already self-corrects that within 2s (see
        # _update_ui_status' docstring), but re-running it right on
        # tab-select too makes the correction immediate instead of a
        # brief, harmless flash of both Run and Interrupt at once.
        single.tab.select(_update_ui_status, [], _run_interrupt_btns)
        batch.tab.select(_update_ui_status, [], _run_interrupt_btns)
        review.tab.select(_update_ui_status, [], _run_interrupt_btns)

        # Tracks which top-level tab is currently selected (declared up by
        # main_tabs' own definition, see the comment there) purely so
        # _refresh_reachability_ui can tell whether the user is on
        # Single/Batch at the moment it needs to be disabled (and if so,
        # push them off it) without disturbing anyone on any other tab -
        # see that function's docstring.
        main_tabs.select(_on_main_tab_select, [], [current_tab_label_state])

        status_timer = gr.Timer(2.0)
        status_timer.tick(get_status_text, [], [status_bar])
        status_timer.tick(_update_ui_status, [], _run_interrupt_btns)
        status_timer.tick(
            _fallback_tab_safety_ui, [current_tab_label_state],
            [main_tabs, settings.settings_tabs, current_tab_label_state],
        )
        for fn, inputs, outputs in ctx.ticks:
            status_timer.tick(fn, inputs, outputs)

        # Separate, slower timer for refresh_reachability_cache specifically
        # - it's a real network call (external mode), unlike everything else
        # status_timer drives, so it doesn't need or want a 2s cadence. Its
        # result is picked up by the very next status_timer tick afterward
        # (_update_ui_status reads the cache, doesn't recompute it)
        # for Single/Batch/Recaption's own Run/Recaption BUTTON gating; the
        # Single-image/Batch-processing TABS' own interactive state
        # is set directly below instead, since gating those also needs to
        # know (and occasionally override) which tab is currently selected
        # - see _refresh_reachability_ui's docstring.
        reachability_timer = gr.Timer(8.0)
        reachability_timer.tick(_refresh_reachability_ui, [current_tab_label_state], _reachability_outputs)

        demo.load(get_status_text, [], [status_bar])
        for fn, inputs, outputs in ctx.loads:
            demo.load(fn, inputs, outputs)
        # _update_ui_status reads the reachability cache rather than
        # checking fresh (see its own docstring) - chained after (not a
        # parallel demo.load, which it used to be) specifically so it
        # always sees a just-refreshed cache, not whatever the optimistic
        # default happened to still be if this beat _startup_reachability_
        # ui's real (and slower - a live network call) check to the punch.
        # That race was real: Single/Batch could show enabled on a fresh
        # page load against an unreachable server, correcting only once
        # status_timer's first 2s tick came around, if at all. Each
        # backend's autoload (if enabled) chains after that, deliberately -
        # the initial UI renders in its correct not-yet-running state
        # first, then autoload is a background upgrade on top of it, never
        # a substitute for landing correctly in the first place. Backends
        # run in core.lifecycle.autoload_order: fixed-VRAM ones (Hydra)
        # before adaptive ones (llama-server's n_gpu_layers="auto" adapts
        # around whatever's left, not the other way around).
        launch_chain = demo.load(
            _startup_reachability_ui, [],
            [main_tabs, single.tab, batch.tab, llama_ui.start_btn, llama_ui.end_btn, current_tab_label_state],
        ).then(_update_ui_status, [], _run_interrupt_btns)
        for backend in autoload_order(BACKENDS):
            if backend.id not in ctx.autoloads:
                continue
            fn, outputs = ctx.autoloads[backend.id]
            launch_chain = launch_chain.then(fn, [], _reachability_outputs if outputs == REACHABILITY else outputs)

    return demo


UI_PORT = 7901


def main() -> None:
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
