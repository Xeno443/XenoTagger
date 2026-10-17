"""Gradio UI entry point and frame: Single-image captioning, Batch
processing, Review, Models, Settings, and an opt-in Debuglog tab. The CLI
(cli.py) is a separate, independent entry point that shares core/ with
this file but not this file or ui/ - nothing UI-specific belongs in core/.

This file only builds the frame: the Blocks, the top-level Tabs, the
status bar, the two timers, the launch chain (Hydra autoload, llama
autostart), and the wiring that crosses tabs (Batch -> Review, Models ->
Settings, the reachability refresh that gates Single/Batch/Models). Each
tab's own components and handlers live in ui/tab_*.py, the Settings
sub-tabs in ui/settings/, and the UI-side state several tabs share in
ui/state.py, ui/operation.py, ui/llama_session.py and
ui/downloads_queue.py - see ui/__init__.py for the import rules that keep
those acyclic. Tabs hand frame-level wiring back through a BuildContext
(ui/context.py) instead of reaching for components they don't own.

Three coordination mechanisms guard shared, process-wide state that
several tabs (and Settings' restart actions) can touch concurrently - the
first two are threading.RLock, not a plain Lock, because each has a
function that calls another lock-holding function of its own while
already holding the lock (get_client() calls stop_managed(); a plain
Lock would deadlock there):

  - `_session`/`_session_lock` (ui/llama_session.py): which llama-server
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
    save_settings_ui() and models_set_active_ui(), the only two places
    that mutate current_cfg wholesale. Without it, two overlapping saves
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
llama-server process - see core/server.py's ManagedServer.stop for why a
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
trace. The Models tab's own "Manage llama server" button used to have an
in-place equivalent (restart_server_ui, force-abort + stop + refresh
right there) but is now pure navigation to Settings -> Llama - actually
stopping/starting only ever happens from that sub-tab's Start/End
buttons, which are already correctly gated (disabled mid-job, see
_settings_gating_ui) so there's no force-abort race left for a plain
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

from core import config as config_mod
from core import hydra_classifier
from core import tag_vocab
from core.config import AppConfig
from core.server import LLAMA_SERVER_EXE, get_loaded_model_name, is_healthy
from ui import state, tab_batch, tab_debuglog, tab_models, tab_review, tab_settings, tab_single
from ui.context import BuildContext
from ui.css import ALL_CSS
from ui.llama_session import (
    cached_controllable, cached_reachable, cached_status, display_base_url, note_controllable_transition,
    refresh_reachability_cache, status_label, stop_managed, try_start_managed_llama, update_reachability_tracking,
)
from ui.notify import notify
from ui.operation import current_operation, operation_lock, operation_status_text
from ui.settings.hydra import load_hydra_with_llama_coexistence
from ui.settings.llama import llama_lifecycle_button_updates
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
    separately updates three TABS themselves: Models (disabled unless
    controllable - see cached_controllable) and Single-image/Batch
    processing (disabled unless reachable at all - captioning there is
    pointless, not just "not ours to redirect", the moment nothing can
    answer). If the user is on one of these when it becomes disabled,
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
    status = refresh_reachability_cache()
    controllable = cached_controllable()
    note_controllable_transition(controllable, cfg, status)
    update_reachability_tracking(status, cfg)
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = llama_lifecycle_button_updates(cfg, status)
    push_to_settings = (not controllable and current_tab_label == "Models") or (
        not status.reachable and current_tab_label in ("Single image", "Batch processing")
    )
    tabs_update = gr.update(selected="settings") if push_to_settings else gr.update()
    label_update = "Settings" if push_to_settings else gr.skip()
    return (
        models_tab_update, single_batch_update, single_batch_update,
        start_btn_update, end_btn_update, tabs_update, label_update,
    )


def _on_main_tab_select(evt: gr.SelectData) -> str:
    """Wired to main_tabs.select() - see current_tab_label_state's own
    comment for why this is tracked at all."""
    return evt.value


def _goto_llama_settings_ui():
    """The Models tab's "Manage llama server" button's click target -
    pure navigation to Settings -> Llama, no server action of its own
    (see this module's own docstring for why that's safe: the Start/End
    buttons there are already gated off while a job's running, so this
    can't be used to bypass anything the way an in-place restart-from-
    Models-tab could).

    Explicitly asserts every destination (main_tabs, settings_tabs,
    current_settings_subtab_state, current_tab_label_state) rather than a
    bare gr.update() no-op on any of them - _refresh_reachability_ui's own
    docstring covers why a no-op can't be trusted to self-correct a
    still-pending prior push to a Tabs component's selected=, and
    current_tab_label_state/current_settings_subtab_state both need to
    reflect this redirect immediately since neither main_tabs.select() nor
    settings_tabs.select() reliably fire on a server-pushed switch like
    this one (see their own comments)."""
    return gr.update(selected="settings"), gr.update(selected="llama-settings"), "llama-settings", "Settings"


def _goto_hydra_settings_ui():
    """The Models tab's "Manage Hydra" button's click target - same pure
    navigation as _goto_llama_settings_ui above, to Settings -> Hydra
    instead."""
    return gr.update(selected="settings"), gr.update(selected="hydra-settings"), "hydra-settings", "Settings"


def _send_batch_to_review_ui(directory: str):
    """The Batch tab's "Send to Review" button - pure navigation + handoff
    of the directory field to the Review tab (same "explicitly assert
    every destination" reasoning as _goto_llama_settings_ui above -
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
    lands on, and the Models/Single-image/Batch-processing tabs' initial
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
    status = refresh_reachability_cache()
    controllable = cached_controllable()
    note_controllable_transition(controllable, cfg, status)
    update_reachability_tracking(status, cfg)
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = llama_lifecycle_button_updates(cfg, status)
    tab_target, tab_label = ("single", "Single image") if status.reachable else ("settings", "Settings")
    return (
        gr.update(selected=tab_target), models_tab_update, single_batch_update, single_batch_update,
        start_btn_update, end_btn_update, tab_label,
    )


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
    controllable = cached_controllable()
    reachable = cached_reachable()
    on_disabled_models = current_tab_label == "Models" and not controllable
    on_disabled_single_batch = current_tab_label in ("Single image", "Batch processing") and not reachable
    if on_disabled_models or on_disabled_single_batch:
        return gr.update(selected="settings"), gr.update(selected="debug-settings"), "Settings"
    return gr.update(), gr.update(), gr.skip()


def _autostart_managed_llama_ui():
    """Chained onto demo.load AFTER the initial UI has already rendered
    with tabs correctly enabled/disabled (see the demo.load chain below) -
    autostart is a background convenience layered on top of that correct
    initial state, never a substitute for it. Only actually does anything
    for managed mode, installed, not already running, and the checkbox
    on - exactly "state 2" (see the state table this was designed
    against), the one case a user would otherwise have to click Start for
    themselves. Silent no-op in every other case (external mode, nothing
    installed, already running, or autostart off) - no popup for "didn't
    autostart", only for actually doing something. Reads the cache
    _startup_reachability_ui (immediately before this in the same chain)
    already refreshed - no extra network call just to decide whether to
    bother. Also sets current_tab_label_state to "Single image" when it
    redirects there - same reasoning as the other tab-pushing functions
    above (a server-pushed switch doesn't fire main_tabs.select() on its
    own; see _refresh_reachability_ui's docstring for the bug this class
    of fix addresses)."""
    cfg = state.current_cfg
    noop = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.skip())
    if cfg.server_mode != "managed" or not cfg.autostart_managed_llama:
        return noop
    status = cached_status()
    if not status.installed or status.reachable:
        return noop
    notify("Starting managed llama-server automatically (autostart is enabled)...")
    ok, _ = try_start_managed_llama()
    if not ok:
        return noop
    status = cached_status()
    controllable = cached_controllable()
    models_tab_update = gr.update(interactive=controllable)
    single_batch_update = gr.update(interactive=status.reachable)
    start_btn_update, end_btn_update = llama_lifecycle_button_updates(cfg, status)
    return (
        models_tab_update, single_batch_update, single_batch_update,
        start_btn_update, end_btn_update, gr.update(selected="single"), "Single image",
    )


def _autoload_hydra_model_ui():
    """Chained onto the demo.load chain BEFORE _autostart_managed_llama_ui
    (deliberately reordered - see AppConfig.hydra_autoload_model's own
    docstring) - so that when both autostart-llama and autoload-Hydra are
    enabled, Hydra gets first pick of a clean, fully-free VRAM budget and
    llama-server's own n_gpu_layers="auto" (which gracefully adapts to
    reduced free VRAM, unlike Hydra's own all-or-nothing load) sizes
    itself around whatever's left. Silent no-op unless hydra_enabled,
    hydra_autoload_model, deps+model are both present, and nothing is
    already loaded - same "no popup for didn't-autostart" convention as
    _autostart_managed_llama_ui itself.

    The actual load - including the stop-managed-llama/restart dance,
    now moot here in the normal case since nothing's started yet this
    early in a fresh launch, but still a real safety net for an edge
    case like an orphaned managed process surviving an app crash - is
    delegated to load_hydra_with_llama_coexistence() (see its own
    docstring), the same shared logic the manual Load button uses, so
    correctness doesn't depend on this function's own ordering assumption
    staying true forever."""
    cfg = state.current_cfg
    noop = (gr.update(), gr.update())
    if not (cfg.hydra_enabled and cfg.hydra_autoload_model):
        yield noop
        return
    st = hydra_classifier.status()
    if not (st.deps_installed and st.model_downloaded) or st.loaded:
        yield noop
        return
    notify("Loading Hydra model automatically (autoload is enabled)...")
    yield from load_hydra_with_llama_coexistence()


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

def _hydra_footer_label(cfg: AppConfig) -> str:
    """Compact one-word(ish) Hydra state for the status bar - see
    _hydra_status_text() for the fuller sentence-form version shown on
    the Hydra sub-tab itself; this is deliberately terser, matching the
    llama-server segment's own "running"/"idle"/etc. brevity."""
    if not cfg.hydra_enabled:
        return "disabled"
    st = hydra_classifier.status()
    if st.loaded:
        return f"loaded ({st.device})"
    if not st.deps_installed or not st.model_downloaded:
        return "not set up"
    return "enabled, not loaded"


def get_status_text() -> str:
    cfg = state.current_cfg
    base_url = display_base_url()
    healthy = is_healthy(base_url)
    installed = LLAMA_SERVER_EXE.exists()  # cheap, local - no extra network call
    # Always the server's own answer, never our config - our selection is
    # just what we'd ask it to load next, not necessarily what's actually
    # loaded right now (e.g. external mode, or a server someone else picked).
    model = get_loaded_model_name(base_url) if healthy else None
    model = model or "n/a"
    return (
        f"Version: {config_mod.app_version()} &nbsp;·&nbsp; "
        f"Python: {platform.python_version()} &nbsp;·&nbsp; "
        f"Gradio: {gr.__version__} &nbsp;·&nbsp; "
        f"llama: {status_label(cfg, healthy, installed)} &nbsp;·&nbsp; "
        f"Model: {model} &nbsp;·&nbsp; "
        f"Hydra: {_hydra_footer_label(cfg)} &nbsp;·&nbsp; "
        f"Status: {operation_status_text()}"
    )


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
        # Every output of the reachability refresh - Models/Single/Batch tab
        # gating, the Llama sub-tab's Start/Stop, and the push-to-Settings
        # redirect. Shared by reachability_timer, the autostart chain, and
        # every button a tab registered via ctx.after_server_change.
        _reachability_outputs = [
            models.tab, single.tab, batch.tab, settings.llama.start_btn, settings.llama.end_btn, main_tabs,
            current_tab_label_state,
        ]

        # ---- Cross-tab wiring - each needs components from more than one tab.

        batch.send_review_btn.click(
            _send_batch_to_review_ui, [batch.dir], [main_tabs, review.dir, current_tab_label_state]
        ).then(review_scan_ui, [review.dir], review.nav_outputs)
        _goto_settings_outputs = [
            main_tabs, settings.settings_tabs, settings.current_subtab_state, current_tab_label_state,
        ]
        models.manage_llama_btn.click(_goto_llama_settings_ui, [], _goto_settings_outputs)
        models.manage_hydra_btn.click(_goto_hydra_settings_ui, [], _goto_settings_outputs)
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
        # _refresh_reachability_ui can tell whether the user is on the
        # Models tab at the moment it needs to be disabled (and if so,
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
        # Models/Single-image/Batch-processing TABS' own interactive state
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
        # status_timer's first 2s tick came around, if at all. Autostart
        # (if enabled) chains after that, deliberately - the initial UI
        # renders in its correct not-yet-running state first, then
        # autostart is a background upgrade on top of it, never a
        # substitute for landing correctly in the first place.
        demo.load(
            _startup_reachability_ui, [],
            [
                main_tabs, models.tab, single.tab, batch.tab, settings.llama.start_btn, settings.llama.end_btn,
                current_tab_label_state,
            ],
        ).then(_update_ui_status, [], _run_interrupt_btns).then(
            # Deliberately BEFORE llama's own autostart below - see
            # _autoload_hydra_model_ui's own docstring for why the
            # ordering matters (Hydra gets first pick of free VRAM;
            # llama-server's n_gpu_layers="auto" adapts around whatever's
            # left, not the other way around).
            _autoload_hydra_model_ui, [], [settings.hydra.lifecycle_infotext, settings.hydra.status_md],
        ).then(_autostart_managed_llama_ui, [], _reachability_outputs)

    return demo


UI_PORT = 7901


def main() -> None:
    setup_debug_logging()
    atexit.register(stop_managed)
    demo = build_app()
    demo.queue()
    demo.launch(
        server_port=UI_PORT, footer_links=[], css=ALL_CSS, inbrowser=True,
        head=tag_vocab.build_autocomplete_head(state.current_cfg),
    )


if __name__ == "__main__":
    main()
