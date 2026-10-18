"""Settings -> Pipeline: what runs (one box per backend - a traffic light,
Run, and for llama its variants), the Output settings, and Start/Stop/
Restart. The first Settings sub-tab. core.pipeline does the work; this is
only its page.

Start and Restart save pending Settings changes first (wired in
ui.tab_settings, which owns Save) so the pipeline never runs stale
settings by surprise. A started pipeline keeps the settings it started
with; a Save while it runs only says "restart to apply" (toast + status
bar).

Lights come from ui.pipeline_status' snapshot (refreshed on the frame's
8s timer and after Start/Stop): 🟢 ready, 🟡 set up but not running, 🔴
something missing (Run disabled) or stopped unexpectedly.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import replace
from types import SimpleNamespace

import gradio as gr

from backends import BACKENDS
from core import pipeline
from core.backend import Backend
from core.config import AppConfig
from ui import pipeline_status, state
from ui.context import BuildContext
from ui.notify import notify
from ui.operation import operation_blocked_by

log = logging.getLogger("app")  # fixed name, see app.py

# Backends whose box has a variant list, in registry order.
_VARIANT_BACKENDS = [b for b in BACKENDS if b.role == "captioner" and b.variant_label]
_versions = itertools.count(1)  # fresh values for the variant lists' re-render triggers

_SAMPLE_TAGS = ["anthro", "wolf", "solo", "standing", "forest", "smiling", "looking_at_viewer", "grey_fur"]
_SAMPLE_CAPTIONS = [
    "A grey wolf stands in a forest, smiling.",
    "A grey anthropomorphic wolf with a bushy tail stands upright in a sunlit forest clearing.",
    "An anthro wolf with grey fur standing among trees, smiling at the viewer.",
]


def _what_it_does(b: Backend) -> str:
    if b.role == "tagger":
        return "Produces tags."
    if b.variant_label:
        text = "Writes one caption per checked variant."
        if b.takes_tags == "optional":
            text += " A prompt containing {tags} gets the tagger's tags."
        return text
    return "Writes one caption." + (" Needs a tagger's tags." if b.takes_tags == "required" else "")


def _header(b: Backend, snap: pipeline_status.Snapshot) -> str:
    lines = [f"### {snap.lights.get(b.id, '')} {b.label}", _what_it_does(b)]
    if b.has_models:
        lines.append(b.model_status(state.current_cfg))
    return "  \n".join(lines)


def _status_text() -> str:
    st = pipeline.status()
    snap = pipeline_status.snapshot()
    text = f"**Pipeline: {pipeline_status.status_text(snap) if st.state == snap.state else st.state}**"
    if st.message:
        text += f"  \n{st.message}"
    if snap.restart_needed:
        text += "  \nSettings were saved since it started - Restart to apply them."
    return text


def _button_updates():
    """(Start, Stop, Restart) interactive, by pipeline state."""
    s = pipeline.status().state
    return (
        gr.update(interactive=s == "stopped"),
        gr.update(interactive=s == "started"),
        gr.update(interactive=s == "started"),
    )


def _updates_from_ui(start_at_launch, *values) -> dict:
    """AppConfig updates from the page's inputs (see save_inputs' order)."""
    n_run, n_var = len(BACKENDS), len(_VARIANT_BACKENDS)
    runs, variants = values[:n_run], values[n_run:n_run + n_var]
    trigger_on, trigger_word, tags_mode, separator, style = values[n_run + n_var:]
    return dict(
        pipeline_start_at_launch=bool(start_at_launch),
        pipeline_run=[b.id for b, on in zip(BACKENDS, runs) if on],
        pipeline_variants={b.id: [dict(v) for v in vs] for b, vs in zip(_VARIANT_BACKENDS, variants)},
        pipeline_trigger_on=bool(trigger_on),
        trigger_word=trigger_word or "",
        pipeline_tags_mode=tags_mode,
        pipeline_tag_separator=separator,
        pipeline_tag_style=style,
    )


def collect_settings(*values) -> dict:
    return _updates_from_ui(*values)


def load_values(cfg: AppConfig) -> tuple:
    """save_inputs' values from cfg (page load - see ui.tab_settings),
    plus fresh values for the variant lists' re-render triggers."""
    return (
        cfg.pipeline_start_at_launch,
        *(b.id in cfg.pipeline_run for b in BACKENDS),
        *(pipeline.variants_for(cfg, b) for b in _VARIANT_BACKENDS),
        cfg.pipeline_trigger_on, cfg.trigger_word, cfg.pipeline_tags_mode,
        cfg.pipeline_tag_separator, cfg.pipeline_tag_style,
        *(next(_versions) for _ in _VARIANT_BACKENDS),
    )


def _preview_ui(*values):
    """(problems, preview) for the page's current, unsaved values."""
    cfg = replace(state.current_cfg, **_updates_from_ui(*values))
    plan = pipeline.make_plan(cfg, BACKENDS)
    problems = "  \n".join(f"⚠ {p}" for p in plan.problems)
    captions = [_SAMPLE_CAPTIONS[i % len(_SAMPLE_CAPTIONS)] for i, _ in enumerate(plan.captioners)]
    preview = pipeline.render(cfg, captions, _SAMPLE_TAGS if plan.tagger else None, cfg.trigger_word)
    return problems, preview


def _progress_ui(gen, toast: bool = True):
    """Streams a start/stop generator into (status line, buttons), with a
    toast on the outcome."""
    for p in gen:
        yield (_status_text(), *_button_updates())
        if p.done and toast:
            notify(p.message, level="info" if p.ok else "warning")
    yield (_status_text(), *_button_updates())


def start_ui():
    if blocked := operation_blocked_by():
        yield (blocked, *_button_updates())
        return
    yield from _progress_ui(pipeline.start_pipeline(state.current_cfg, BACKENDS))


def stop_ui():
    if blocked := operation_blocked_by():
        yield (blocked, *_button_updates())
        return
    yield from _progress_ui(pipeline.stop_pipeline())


def restart_ui():
    if blocked := operation_blocked_by():
        yield (blocked, *_button_updates())
        return
    yield from _progress_ui(pipeline.stop_pipeline(), toast=False)
    yield from _progress_ui(pipeline.start_pipeline(state.current_cfg, BACKENDS))


_autostart_done = False


def autostart_ui():
    """"Start pipeline at launch" - chained after the frame's startup check
    on page load, so only the first page load after app start acts."""
    global _autostart_done
    cfg = state.current_cfg
    if _autostart_done or not cfg.pipeline_start_at_launch or pipeline.status().state != "stopped":
        _autostart_done = True
        yield (_status_text(), *_button_updates())
        return
    _autostart_done = True
    notify("Starting the pipeline (Start pipeline at launch is on)...")
    yield from _progress_ui(pipeline.start_pipeline(cfg, BACKENDS))


def _tick_ui():
    snap = pipeline_status.snapshot()
    return (
        _status_text(), *_button_updates(),
        *(gr.update(value=_header(b, snap)) for b in BACKENDS),
        *(gr.update(interactive=b.id not in snap.unavailable) for b in BACKENDS),
    )


def _variant_list(backend: Backend, cfg: AppConfig) -> tuple[gr.State, gr.State]:
    """A backend's editable variant rows. gr.render re-runs only when
    `version` changes (add/remove/page load), not on every edit - edits
    write straight into the variants state, so typing never re-renders
    (which would drop focus)."""
    variants = gr.State(pipeline.variants_for(cfg, backend))
    version = gr.State(0)

    def _setter(i, key):
        def set_field(value, current):
            current[i][key] = value
            return current
        return set_field

    def _remover(i):
        def remove(current):
            return current[:i] + current[i + 1:], next(_versions)
        return remove

    @gr.render(inputs=[variants], triggers=[version.change])
    def _rows(current):
        for i, v in enumerate(current):
            # Not equal_height: the field grows with its text, the rest
            # of the row stays one line high.
            with gr.Row():
                on = gr.Checkbox(value=v["on"], show_label=False, container=False, scale=0, min_width=30)
                name = gr.Textbox(
                    value=v["name"], placeholder="Name", show_label=False, container=False,
                    scale=1, min_width=120,
                )
                value = gr.Textbox(
                    value=v["value"], placeholder=backend.variant_label, show_label=False,
                    container=False, scale=4, lines=1, max_lines=8,
                )
                remove = gr.Button("Remove", scale=0, min_width=80)
            on.input(_setter(i, "on"), [on, variants], [variants])
            name.input(_setter(i, "name"), [name, variants], [variants])
            value.input(_setter(i, "value"), [value, variants], [variants])
            remove.click(_remover(i), [variants], [variants, version])

    add = gr.Button("Add variant", size="sm")
    add.click(
        lambda current: (current + [dict(on=True, name=f"variant{len(current) + 1}", value="")], next(_versions)),
        [variants], [variants, version],
    )
    return variants, version


def build(ctx: BuildContext):
    cfg = state.current_cfg
    snap = pipeline_status.snapshot()
    with gr.Tab("Pipeline", id="pipeline-settings") as pipeline_settings_tab:
        gr.Markdown(
            "What runs when you caption, and how its output is put together into "
            "the .txt. The tagger runs first, so captioners can use its tags. "
            "Start loads everything that's set to Run (and saves pending changes "
            "first); Single and Batch work while the pipeline is started."
        )
        status_md = gr.Markdown(_status_text())
        with gr.Row():
            start_btn = gr.Button("Start pipeline", variant="primary")
            stop_btn = gr.Button("Stop pipeline")
            restart_btn = gr.Button("Restart pipeline")
        start_at_launch = gr.Checkbox(label="Start pipeline at launch", value=cfg.pipeline_start_at_launch)
        problems_md = gr.Markdown()

        headers, runs, variant_states, versions = [], [], [], []
        for b in BACKENDS:
            with gr.Column(variant="panel"):
                headers.append(gr.Markdown(_header(b, snap)))
                runs.append(gr.Checkbox(
                    label="Run", value=b.id in cfg.pipeline_run, interactive=b.id not in snap.unavailable,
                ))
                if b in _VARIANT_BACKENDS:
                    variants, version = _variant_list(b, cfg)
                    variant_states.append(variants)
                    versions.append(version)

        gr.Markdown("### Output")
        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True):
                trigger_on = gr.Checkbox(label="Trigger word at the start of every line", value=cfg.pipeline_trigger_on)
                trigger_word = gr.Textbox(
                    label="Trigger word", value=cfg.trigger_word,
                    info="Prefills the trigger field on Single and Batch.",
                )
        with gr.Column(variant="panel"):
            with gr.Row(equal_height=True):
                tags_mode = gr.Radio(
                    choices=[
                        ("On their own line", "own"),
                        ("Appended to every caption", "append"),
                        ("Not written (only fed to captioners)", "none"),
                    ],
                    value=cfg.pipeline_tags_mode, label="Tag handling", scale=3,
                )
                separator = gr.Textbox(label="Tag separator", value=cfg.pipeline_tag_separator, scale=1, min_width=100)
                style = gr.Radio(
                    choices=[("Spaces (grey fur)", "spaces"), ("Underscores (grey_fur)", "underscores")],
                    value=cfg.pipeline_tag_style, label="Tag style", scale=2,
                )
        preview = gr.Textbox(label="Preview (made-up captions and tags)", lines=4, interactive=False)

        save_inputs = [start_at_launch, *runs, *variant_states, trigger_on, trigger_word, tags_mode, separator, style]
        for component in save_inputs:
            component.change(_preview_ui, save_inputs, [problems_md, preview])

        status_outputs = [status_md, start_btn, stop_btn, restart_btn]
        ctx.after_pipeline_change(stop_btn.click(stop_ui, [], status_outputs))

    ctx.on_tick(_tick_ui, [], [*status_outputs, *headers, *runs])
    ctx.on_load(_tick_ui, [], [*status_outputs, *headers, *runs])
    ctx.on_load(_preview_ui, save_inputs, [problems_md, preview])

    return SimpleNamespace(
        tab=pipeline_settings_tab,
        tab_label="Pipeline",
        tab_id="pipeline-settings",
        gated=True,
        save_inputs=save_inputs,
        # load_values() also bumps the variant lists' re-render triggers.
        load_outputs=[*save_inputs, *versions],
        # ui.tab_settings wires these two after Save (see the docstring).
        start_btn=start_btn,
        restart_btn=restart_btn,
        status_outputs=status_outputs,
        # app.py chains this after its startup check (see autostart_ui).
        autostart=autostart_ui,
    )
