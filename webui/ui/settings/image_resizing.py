"""Settings -> Image resizing."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui import state
from ui.context import BuildContext


def collect_settings(resize_enabled, resize_target_mp, snap_enabled, snap_multiple) -> dict:
    return dict(
        resize_enabled=bool(resize_enabled),
        resize_target_mp=float(resize_target_mp),
        snap_enabled=bool(snap_enabled),
        snap_multiple=int(snap_multiple),
    )


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Image resizing", id="image-resizing-settings") as image_resizing_settings_tab:
        with gr.Group():
            resize_enabled = gr.Checkbox(
                label="Downscale oversized images before sending to the model",
                value=cfg.resize_enabled,
            )
            resize_target_mp = gr.Slider(
                0.1, 4.0, value=cfg.resize_target_mp, step=0.1,
                label="Target resolution (megapixels)",
            )
            with gr.Row():
                snap_enabled = gr.Checkbox(
                    label="Scale to multiple of", value=cfg.snap_enabled,
                )
                snap_multiple = gr.Number(
                    value=cfg.snap_multiple, precision=0, show_label=False,
                )

    return SimpleNamespace(
        tab=image_resizing_settings_tab,
        tab_label="Image resizing",
        tab_id="image-resizing-settings",
        gated=True,
        save_inputs=[resize_enabled, resize_target_mp, snap_enabled, snap_multiple],
    )
