# XenoTagger

A vibe-coded Gradio + CLI tool for captioning images for LoRA dataset preparation.
Point it at a folder of images and it writes a `.txt` caption next to
each one, using a local (or remote) llama.cpp vision model. An optional
second pass with a RedRocket Hydra e621 tag classifier can fold in
grounded e621-style tags alongside the caption.

## Install

Tested on Windows; may work on Linux too. The setup scripts and the
in-app llama.cpp installer are Windows-specific, but the core app has no
platform-specific code — running against an external, self-managed
llama-server should work fine.

A prepackaged release zip with everything already set up is planned for
later. Until then, get the code one of these ways:

- Download [`setup-portable.bat`](https://github.com/Xeno443/XenoTagger/blob/main/setup-portable.bat) into an empty folder and run it. It installs a portable Python and Git, fetches the rest of the app from GitHub, and finishes setup automatically.
- Download the whole repo as  a Zip using GitHub's ["Download ZIP"](https://github.com/Xeno443/XenoTagger/archive/refs/heads/main.zip) button, extract it, then run a setup script (see below).

- If you already have Git installed on your system, do a standard `git clone`
  ```
  git clone https://github.com/Xeno443/XenoTagger.git
  cd XenoTagger
  ```

Then run one of the following setup scripts:

### Portable environment

```
setup-portable.bat
```

This downloads its own Python and Git into `system\`, without affecting
any system-wide installation. If the app code isn't there yet, it fetches that too before finishing.

### Using your own Python

Requires Python and git already installed and available on PATH.

```
setup-venv.bat
```

This creates a `.venv\` using the systemwide Python.

## Launch

```
run.cmd
```

(GUI) or `cli.cmd` (headless batch captioning). Both auto-detect
whichever setup you used above.

## First launch

XenoTagger requires a llama.cpp build and a vision-capable GGUF model to
caption images. Both can be installed from within the app, from the
Models and Settings tabs, after the first launch.

## Models

The Models tab offers a curated selection of vision models for download,
including Gemma, Qwen3-VL, JoyCaption, and huihui-ai's abliterated
lineage.

Any GGUF vision model can be used: place the model file and its matching
`mmproj` file in a folder under `webui/models/`, then refresh the Models
tab to detect it.

## System requirements

- Windows 10/11 (tested); Linux is untested but expected to work with an
  external llama-server
- An Nvidia or AMD GPU is recommended (CUDA and ROCm builds are
  available). CPU-only operation is supported but slow.
- 8GB VRAM is the practical floor — llama.cpp's partial offload handles
  the rest — with 12–16GB more comfortable. Additional VRAM allows larger
  vision models at higher quality.
- Sufficient disk space for models — vision GGUFs commonly run several GB
  each.
- The optional Hydra tag classifier downloads its own ~1GB model and a
  PyTorch install on demand; it is not required for basic captioning.
