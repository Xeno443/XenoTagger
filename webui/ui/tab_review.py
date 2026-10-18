"""Review tab."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd
from PIL import Image as PILImage

from core.batch import (
    ISSUE_SUFFIX, NLP_SUFFIX, TAGS_SUFFIX, ReviewItem, review_item_status, scan_review_status,
)
from ui.context import BuildContext
from ui.dialogs import browse_directory_ui
from ui.operation import current_operation, interrupt_button_ui, operation_lock
from ui.operation import run_controls_update as run_controls_update_for

log = logging.getLogger("app")  # fixed name, see app.py

# ---------------------------------------------------------------- Review tab
#
# Browse a directory's images and captions side by side, edit inline, and
# re-run captioning on just the one currently shown - no automated bad-
# caption detection (see status column instead), matching how TagGUI and
# similar tools handle this: a human looks, a human decides.
#
# State lives in two gr.State components (Gradio has no server-side
# per-session storage otherwise): review_items_state (list[ReviewItem],
# the current directory's scan_review_status() snapshot) and
# review_index_state (int, which item is currently shown, -1 if none).
# No separate "loaded baseline" state - _review_maybe_save compares
# against a fresh read of each file's current on-disk content instead,
# same statelessness as the rest of this module.
#
# Recaption is its own "review" operation kind, so the Single tab's own
# Interrupt doesn't light up while Review is the one running, and it
# still mutually excludes against Single/Batch. It's parked until Review
# moves onto the pipeline's piece files - see RECAPTION_PARKED.


def _review_status_table(items: list[ReviewItem], current_index: int = -1):
    """Builds review_table's value, highlighting current_index's row (the
    image currently loaded in the Caption/Tags fields) via a pandas
    Styler - gr.Dataframe(value=...) natively accepts a Styler (checked
    its constructor signature: `pd.DataFrame | Styler | ...`), so this
    needs no custom CSS (see this project's CSS house rule). Uses
    --table-row-focus, the same theme variable Gradio's own Dataframe
    already applies to a hovered row's background, so the highlight
    matches whatever theme/light-dark mode is active automatically rather
    than a hardcoded color."""
    df = pd.DataFrame(
        [[item.path.name, item.status] for item in items], columns=["File", "Status"],
    )
    if not (0 <= current_index < len(df)):
        return df
    return df.style.apply(
        lambda row: (
            ["background-color: var(--table-row-focus)"] * len(row)
            if row.name == current_index else [""] * len(row)
        ),
        axis=1,
    )


def _review_tags_enabled(item: ReviewItem) -> bool:
    """Whether the Tags field should be interactive for this item.

    The real switch isn't "does this item currently show split fields" -
    it's whether a .txt.nlp/.txt.tags sidecar has ever been committed to
    disk for it (by a batch run, or an earlier Review save - see
    _review_maybe_save's own docstring for the full two-phase model this
    mirrors). Tags is disabled only for a pure legacy/flat caption: .txt
    exists and neither sidecar ever has - editing that caption always just
    rewrites .txt directly, never spins up a lone .txt.nlp on its own.
    Recomputed fresh wherever needed rather than cached, same
    statelessness as scan_review_status() itself."""
    txt_path = item.path.with_suffix(".txt")
    has_txt = txt_path.exists()
    has_nlp = Path(f"{txt_path}{NLP_SUFFIX}").exists()
    has_tags = Path(f"{txt_path}{TAGS_SUFFIX}").exists()
    return (has_nlp or has_tags) or not has_txt


def _review_load(items: list[ReviewItem], index: int):
    """Returns (image, caption_text, tags_text, tags_enabled) for
    items[index], or (None, "", "", True) if index is out of range.
    Caption prefers .txt.nlp content (falling back to .txt when there's no
    .txt.nlp to draw the prose-only part from); Tags is .txt.tags content,
    or empty if there is none."""
    if not items or not (0 <= index < len(items)):
        return None, "", "", True
    item = items[index]
    try:
        with PILImage.open(item.path) as img:
            image = img.copy()
    except OSError as exc:
        log.warning("Review: could not load image %s: %s", item.path, exc)
        image = None

    txt_path = item.path.with_suffix(".txt")
    nlp_path = Path(f"{txt_path}{NLP_SUFFIX}")
    tags_path = Path(f"{txt_path}{TAGS_SUFFIX}")
    has_txt = txt_path.exists()
    has_nlp = nlp_path.exists()
    has_tags = tags_path.exists()

    def _read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Review: could not read %s: %s", p, exc)
            return ""

    if has_nlp:
        caption = _read(nlp_path)
    elif has_txt:
        caption = _read(txt_path)
    else:
        caption = ""
    tags_text = _read(tags_path) if has_tags else ""
    tags_enabled = (has_nlp or has_tags) or not has_txt

    return image, caption, tags_text, tags_enabled


def _review_maybe_save(items: list[ReviewItem], index: int, current_caption: str, current_tags: str) -> None:
    """Auto-save on navigate-away. Never writes an empty file, and never
    deletes/blanks an existing one just because a field was cleared -
    clearing text to retype it should never delete a caption by accident.

    Two-phase model keyed on whether a .txt.nlp/.txt.tags sidecar already
    exists ("committed") - not the same question as whether Tags is
    currently enabled (see _review_tags_enabled):

    - Not yet committed (no .txt.nlp, no .txt.tags exist yet): the two
      fields aren't independent yet. Only if BOTH are non-empty is a real
      split born now - .txt.nlp/.txt.tags get created alongside .txt. If
      only one is non-empty, nothing sidecar-shaped is created - that
      content goes straight into .txt alone (this is also exactly what
      happens when editing a plain flat caption, since Tags is disabled -
      and so always empty - whenever .txt exists with no sidecar yet).
    - Already committed: each field is independent - a non-empty, changed
      field updates its own sidecar; an empty field is left alone. .txt
      gets resynced from whatever's currently non-empty in either field
      whenever anything changed - which, since a missing .txt's "current
      content" reads as "", also naturally covers a Sidecar-only item (no
      .txt yet) picking one up the first time it's touched, with no
      separate case needed for that.
    """
    if not items or not (0 <= index < len(items)):
        return
    item = items[index]
    txt_path = item.path.with_suffix(".txt")
    nlp_path = Path(f"{txt_path}{NLP_SUFFIX}")
    tags_path = Path(f"{txt_path}{TAGS_SUFFIX}")
    has_txt = txt_path.exists()
    has_nlp = nlp_path.exists()
    has_tags = tags_path.exists()
    committed = has_nlp or has_tags

    cap = current_caption or ""
    tags = current_tags or ""
    cap_nonempty = cap.strip() != ""
    tags_nonempty = tags.strip() != ""

    def _read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""

    wrote = False
    if committed or (cap_nonempty and tags_nonempty):
        if cap_nonempty:
            old_nlp = _read(nlp_path) if has_nlp else ""
            if cap != old_nlp:
                nlp_path.write_text(cap, encoding="utf-8")
                wrote = True
        if tags_nonempty:
            old_tags = _read(tags_path) if has_tags else ""
            if tags != old_tags:
                tags_path.write_text(tags, encoding="utf-8")
                wrote = True

    combined = "\n".join(text for text, nonempty in ((cap, cap_nonempty), (tags, tags_nonempty)) if nonempty)
    if combined:
        old_txt = _read(txt_path) if has_txt else ""
        if combined != old_txt:
            txt_path.write_text(combined, encoding="utf-8")
            wrote = True

    if wrote:
        Path(f"{txt_path}{ISSUE_SUFFIX}").unlink(missing_ok=True)
        item.status = review_item_status(item.path)
        log.info("Review: saved caption for %s", item.path)


def _review_position_text(items: list[ReviewItem], index: int, prefix: str = "") -> str:
    if not items:
        return "No images found."
    if not (0 <= index < len(items)):
        return f"{len(items)} image(s) found."
    item = items[index]
    return f"{prefix}{index + 1}/{len(items)} - {item.path.name} ({item.status})"


def review_scan_ui(directory_str: str):
    directory = Path(directory_str) if directory_str else None
    if not directory or not directory.is_dir():
        log.warning("Review: not a directory: %s", directory_str)
        return f"Not a directory: {directory_str}", [], -1, None, "", gr.update(value="", interactive=True), _review_status_table([])

    items = scan_review_status(directory)
    log.info("Review: scanned %s - %d image(s)", directory, len(items))
    if not items:
        return (
            f"No images found in {directory}", items, -1, None, "",
            gr.update(value="", interactive=True), _review_status_table(items),
        )

    image, caption, tags, tags_enabled = _review_load(items, 0)
    return (
        _review_position_text(items, 0),
        items, 0,
        image, caption, gr.update(value=tags, interactive=tags_enabled),
        _review_status_table(items, 0),
    )


def review_clear_ui(items: list[ReviewItem], index: int, current_caption: str, current_tags: str):
    """Resets the whole Review tab - directory field, scanned table, and
    the currently loaded item - back to its just-opened state. Saves any
    pending edit on the current item first, same as navigating away via
    Prev/Next/the table."""
    _review_maybe_save(items, index, current_caption, current_tags)
    return (
        "", _review_position_text([], -1), [], -1, None, "",
        gr.update(value="", interactive=True), _review_status_table([]),
    )


def review_prev_ui(items: list[ReviewItem], index: int, current_caption: str, current_tags: str):
    _review_maybe_save(items, index, current_caption, current_tags)
    new_index = max(0, index - 1) if items else -1
    image, caption, tags, tags_enabled = _review_load(items, new_index)
    return (
        _review_position_text(items, new_index), items, new_index,
        image, caption, gr.update(value=tags, interactive=tags_enabled), _review_status_table(items, new_index),
    )


def review_next_ui(items: list[ReviewItem], index: int, current_caption: str, current_tags: str):
    _review_maybe_save(items, index, current_caption, current_tags)
    new_index = min(len(items) - 1, index + 1) if items else -1
    image, caption, tags, tags_enabled = _review_load(items, new_index)
    return (
        _review_position_text(items, new_index), items, new_index,
        image, caption, gr.update(value=tags, interactive=tags_enabled), _review_status_table(items, new_index),
    )


def review_table_select_ui(
    items: list[ReviewItem], index: int, current_caption: str, current_tags: str, evt: gr.SelectData,
):
    with operation_lock:
        op = current_operation()
        if op is not None and op.kind == "review":
            return tuple(gr.update() for _ in range(7))
    _review_maybe_save(items, index, current_caption, current_tags)
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    image, caption, tags, tags_enabled = _review_load(items, row)
    return (
        _review_position_text(items, row), items, row,
        image, caption, gr.update(value=tags, interactive=tags_enabled), _review_status_table(items, row),
    )


# Mirrors RUN_IDLE/RUN_BUSY from the Single-image tab - reused here
# since Review's nav controls (prev/next/table/dir/browse/scan) need the
# exact same "disabled while a 'single'-kind operation is running
# anywhere" treatment as part of the requirement that recaptioning
# disables the rest of this tab's navigation, not just morph its own
# button. See run_controls_update below and app.py's _update_ui_status.
#
# review_table's own slot is deliberately always a no-op gr.update(),
# never interactive=True/False - it's built interactive=False at
# construction and must stay that way, since _review_status_table()
# hands it a pandas Styler for the current-row highlight and Gradio's
# Dataframe.postprocess silently discards a Styler (with a console
# warning) whenever interactive is True. Blocking a row click during an
# active recaption is instead handled server-side in
# review_table_select_ui, not via this client-side flag.
_REVIEW_NAV_BUSY = (
    gr.update(interactive=False), gr.update(interactive=False), gr.update(),
    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
    gr.update(interactive=False),
)
_REVIEW_NAV_IDLE = (
    gr.update(interactive=True), gr.update(interactive=True), gr.update(),
    gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
    gr.update(interactive=True),
)
# 3 (recaption_col, interrupt_col, interrupt_btn) + 7 (_REVIEW_NAV_BUSY/IDLE
# width - prev/next/table/dir/browse/scan/clear) = 10 - must match
# running_state/idle_state's length below exactly, so it's a named constant
# rather than a repeated magic number.
_REVIEW_STATE_NOOP = tuple(gr.update() for _ in range(10))


# Recaption is parked until Review moves onto the pipeline's piece files
# (v0.9 step 3, part 6): it ran the old llama+Hydra caption path, which the
# pipeline replaced. The button stays (disabled, see run_controls_update).
RECAPTION_PARKED = "Recaption comes back with the Review update (it moves onto the pipeline's files)."


def review_recaption_ui(items: list[ReviewItem], index: int, current_caption: str):
    yield gr.update(), gr.update(), RECAPTION_PARKED, *_REVIEW_STATE_NOOP


def interrupt_review_ui():
    return interrupt_button_ui("review")

def build(ctx: BuildContext):
    with gr.Tab("Review", id="review") as review_tab:
        with gr.Row():
            review_dir = gr.Textbox(label="Directory of images", scale=4)
            with gr.Column(scale=1, min_width=120):
                review_browse_btn = gr.Button("Browse...")
                review_scan_btn = gr.Button("Scan")
                review_clear_btn = gr.Button("Clear")

        with gr.Row(equal_height=True):
            review_prev_btn = gr.Button("←", scale=1, min_width=60)
            review_image = gr.Image(
                label="Image", interactive=False, show_label=False, buttons=[], scale=8, height=480,
            )
            review_next_btn = gr.Button("→", scale=1, min_width=60)

        with gr.Row():
            with gr.Column() as review_recaption_col:
                review_recaption_btn = gr.Button("Recaption")
            # Separate component from review_recaption_btn on purpose,
            # same reasoning as every other Interrupt in this app - see
            # the Operation tracking section.
            with gr.Column(visible=False) as review_interrupt_col:
                review_interrupt_btn = gr.Button("Interrupt", variant="stop")

        review_caption = gr.Textbox(label="Caption", lines=4, interactive=True)
        # Always visible now - interactive= (not visible=) is what
        # toggles based on _review_tags_enabled(), since a flat
        # legacy caption still has nothing to show here, just
        # nothing to edit either. A plain Textbox, not a
        # gr.Dropdown - a Dropdown with the full e621 vocabulary as
        # its `choices` was confirmed unusably slow (a known,
        # unfixed Gradio limitation, not a tuning problem - see the
        # elem_id below). elem_id is what lets
        # static/tag_autocomplete.js (injected via
        # core.tag_vocab.build_autocomplete_head, see
        # demo.launch(head=...) in app.py) attach its own hand-rolled
        # popup to this field specifically - review_caption
        # deliberately has no elem_id and is never touched by it.
        review_tags = gr.Textbox(
            label="Tags", lines=2, interactive=True,
            elem_id="review_tags_box",
        )

        review_table = gr.Dataframe(
            headers=["File", "Status"], datatype=["str", "str"],
            interactive=False, row_count=(0, "dynamic"), buttons=[],
        )

        review_infotext = gr.Textbox(show_label=False, container=False, interactive=False)

        # See the Review tab's handler-function docstrings above for
        # what each of these two actually holds.
        review_items_state = gr.State([])
        review_index_state = gr.State(-1)

        _review_nav_outputs = [
            review_infotext, review_items_state, review_index_state,
            review_image, review_caption, review_tags, review_table,
        ]
        review_browse_btn.click(browse_directory_ui, [review_dir], [review_dir]).then(
            review_scan_ui, [review_dir], _review_nav_outputs
        )
        review_scan_btn.click(review_scan_ui, [review_dir], _review_nav_outputs)
        review_clear_btn.click(
            review_clear_ui,
            [review_items_state, review_index_state, review_caption, review_tags],
            [review_dir, *_review_nav_outputs],
        )
        review_prev_btn.click(
            review_prev_ui,
            [review_items_state, review_index_state, review_caption, review_tags],
            _review_nav_outputs,
        )
        review_next_btn.click(
            review_next_ui,
            [review_items_state, review_index_state, review_caption, review_tags],
            _review_nav_outputs,
        )
        review_table.select(
            review_table_select_ui,
            [review_items_state, review_index_state, review_caption, review_tags],
            _review_nav_outputs,
        )
        review_recaption_btn.click(
            review_recaption_ui,
            [review_items_state, review_index_state, review_caption],
            [
                review_caption, review_tags, review_infotext,
                review_recaption_col, review_interrupt_col, review_interrupt_btn,
                review_prev_btn, review_next_btn, review_table,
                review_dir, review_browse_btn, review_scan_btn, review_clear_btn,
            ],
        )
        review_interrupt_btn.click(interrupt_review_ui, [], [review_interrupt_btn])

    return SimpleNamespace(
        tab=review_tab,
        run_controls=[
            review_recaption_btn, review_recaption_col, review_interrupt_col, review_interrupt_btn,
            review_prev_btn, review_next_btn, review_table,
            review_dir, review_browse_btn, review_scan_btn, review_clear_btn,
        ],
        dir=review_dir,
        nav_outputs=_review_nav_outputs,
    )


def run_controls_update(op, ready: bool) -> tuple:
    """Updates for build()'s run_controls - the shared Recaption/Interrupt
    four (see operation.run_controls_update), plus the nav controls, which
    are disabled only while Review itself is recaptioning. Recaption is
    parked (always disabled) - see RECAPTION_PARKED."""
    review_running = op is not None and op.kind == "review"
    review_nav = _REVIEW_NAV_BUSY if review_running else _REVIEW_NAV_IDLE
    return (*run_controls_update_for("review", op, False), *review_nav)
