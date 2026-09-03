# XenoTagger user manual

XenoTagger captions images for LoRA dataset preparation. It sends each
image to a vision-capable language model (VLM) running on
[llama.cpp](https://github.com/ggml-org/llama.cpp) and writes the
resulting caption to a `.txt` file next to the image. An optional second
pass through a RedRocket Hydra e621 tag classifier can append grounded
tags to the same caption.

This manual documents every tab and settings page of the web interface
(`run.cmd`) using the application's **default configuration** - the
state of a freshly installed copy that has not been configured yet.
Most screenshots are taken from such a fresh install; the Debug
sub-tab, the Debuglog tab, and External server mode do not yet have one
and show a broken image link until added.

## Where to start

New to XenoTagger? Follow **[First run walkthrough](firstrun.md)** -
it walks through installing a model backend, selecting a vision model,
running a first caption, and (optionally) enabling Hydra tagging, in
order.

Otherwise, use the reference below to jump to a specific tab.

## Interface reference

The application window is a row of top-level tabs, with a status bar
fixed to the bottom of every tab showing server state, the active
model, Hydra state, and the current operation. The status bar refreshes
automatically every two seconds.

| Tab | Purpose |
|---|---|
| [Single image](tab-single-image.md) | Caption one image at a time and inspect the result before saving. |
| [Batch processing](tab-batch-processing.md) | Caption every image in a folder in one run. |
| [Review](tab-review.md) | Browse a folder's existing captions, edit them, and re-run individual images. |
| [Models](tab-models.md) | Select, download, or refresh the vision model and the Hydra model. |
| Settings | Configuration, split into five sub-tabs (below). |
| [Debuglog](tab-debuglog.md) | Raw application and llama-server log output. Hidden until enabled in Settings. |

### Settings sub-tabs

| Sub-tab | Purpose |
|---|---|
| [Llama](tab-settings-llama.md) | Server connection mode, backend install, model-server parameters, and prompt/generation settings. |
| [Hydra](tab-settings-hydra.md) | Hydra tag classifier install, load state, and tagging behavior. |
| [Image resizing](tab-settings-image-resizing.md) | Downscaling images before they are sent to the vision model. |
| [Captioning defaults](tab-settings-captioning-defaults.md) | Default trigger word and overwrite behavior for new captioning jobs. |
| [Debug](tab-settings-debug.md) | Enables the Debuglog tab. |

## Related references

- [Batch captioning sidecar & overwrite FAQ](faq-caption.md) - exactly what happens to `.txt`/`.txt.nlp`/`.txt.tags` files during a batch run and in Review.
- [Hydra confidence & threshold FAQ](faq-hydra-setting.md) - what the Confidence and Threshold sliders on the Hydra settings page do.
- [Hydra implications FAQ](faq-hydra-implications.md) - how the Implications mode setting resolves nested e621 tag families.
- [Linux compatibility FAQ](faq-linux-considerations.md) - which parts of the app are Windows-specific.
- [Reinstalling the Python environment](faq-reinstall-python.md)

---

[Next: First run walkthrough →](firstrun.md)
