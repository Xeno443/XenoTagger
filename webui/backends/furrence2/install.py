"""Furrence2's dependencies and model files.

Two separate Python environments are involved, because Furrence2's
vendored Florence-2 code only runs on transformers 4.46.3, which needs
huggingface_hub<1.0 - and Gradio needs >=1.16, so it can never be
installed next to the app:
  - SYSTEM_DEPS go into whichever interpreter runs this app
    (sys.executable - system\\python, or a user's .venv), like Hydra's:
    torch (cu130) for the worker, safetensors + numpy for the fp16
    conversion below, which runs in the app process.
  - LIB_REQUIREMENTS go into LIB_DIR via `pip install --target`. Only the
    worker process (worker.py) ever puts LIB_DIR on its sys.path, ahead
    of everything else, so its transformers/huggingface_hub never meet
    Gradio's. torch isn't a transformers dependency, so the worker still
    uses the one from SYSTEM_DEPS. pip's "ERROR: ... gradio requires
    huggingface-hub>=1.16, but you have 0.36" during this install is
    expected and harmless: it compares the target folder against the
    app's packages, which is exactly the clash the separate folder avoids
    (pip still exits 0, system\\python is untouched).

LIB_DIR sits at the repo root next to llama/ (gitignored as /furrence2/,
anchored so it doesn't also match webui/backends/furrence2/), outside
system\\ so a .venv install works the same.

The model repo ships fp32 weights (3.3GB) and no code or tokenizer. The
download lands as model.fp32.safetensors; convert_to_fp16() (the
download's after_download step) writes the fp16 model.safetensors the
worker loads (1.6GB) and deletes the fp32 file. The llama backend's GGUF
scanner only globs *.gguf, so this folder never shows up as a llama row.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from core.config import MODELS_DIR, ROOT_DIR
from core.downloads import DownloadItem

log = logging.getLogger(__name__)

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
SYSTEM_DEPS = ["torch>=2.6.0,<2.13.0", "safetensors", "numpy"]

LIB_DIR = ROOT_DIR / "furrence2" / "lib"
TRANSFORMERS_VERSION = "4.46.3"
LIB_REQUIREMENTS = [f"transformers=={TRANSFORMERS_VERSION}", "einops"]

MODEL_REPO_URL = "https://huggingface.co/lodestone-horizon/furrence2-large/resolve/main"
MODEL_DIR = MODELS_DIR / "lodestone-horizon-furrence2-large"
CONFIG_PATH = MODEL_DIR / "config.json"
CONFIG_SIZE_BYTES = 2_445
FP32_PATH = MODEL_DIR / "model.fp32.safetensors"
FP32_SIZE_BYTES = 3_291_921_348
WEIGHTS_PATH = MODEL_DIR / "model.safetensors"  # fp16, what the worker loads


def deps_installed() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and (LIB_DIR / f"transformers-{TRANSFORMERS_VERSION}.dist-info").is_dir()
    )


def model_downloaded() -> bool:
    return CONFIG_PATH.exists() and WEIGHTS_PATH.exists()


def _run_pip(args: list[str], on_output: Optional[Callable[[str], None]]) -> bool:
    argv = [sys.executable, "-m", "pip", "install", *args]
    log.info("Running: %s", " ".join(argv))
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(ROOT_DIR),
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        log.debug("pip: %s", line)
        if on_output:
            on_output(line)
    returncode = process.wait()
    if returncode != 0:
        log.warning("pip install failed (exit code %s)", returncode)
    return returncode == 0


def install_deps(on_output: Optional[Callable[[str], None]] = None) -> bool:
    """SYSTEM_DEPS into this interpreter, then LIB_REQUIREMENTS into
    LIB_DIR (--upgrade replaces an older --target install in place).
    Returns True if both pip runs succeeded."""
    if not _run_pip([*SYSTEM_DEPS, "--extra-index-url", TORCH_INDEX_URL], on_output):
        return False
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    if not _run_pip(["--target", str(LIB_DIR), "--upgrade", *LIB_REQUIREMENTS], on_output):
        return False
    log.info("Furrence2 dependencies installed")
    return True


def convert_to_fp16(fp32_path: Path) -> None:
    """after_download step: fp32 -> fp16 model.safetensors, one tensor at
    a time, then deletes the fp32 file. Plain numpy, so it runs in the app
    process without transformers. The "pt" format metadata is what
    transformers' from_pretrained checks for."""
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    log.info("Converting %s to fp16 ...", fp32_path.name)
    tensors = {}
    with safe_open(str(fp32_path), framework="np") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            tensors[key] = t.astype(np.float16) if t.dtype == np.float32 else t
    tmp_path = WEIGHTS_PATH.with_name(WEIGHTS_PATH.name + ".tmp")
    save_file(tensors, str(tmp_path), metadata={"format": "pt"})
    tmp_path.replace(WEIGHTS_PATH)
    fp32_path.unlink()
    log.info("Wrote %s, deleted the fp32 file", WEIGHTS_PATH)


def download_items() -> list[DownloadItem]:
    """What's still missing, for the Models tab and the CLI installer."""
    items = []
    if not CONFIG_PATH.exists():
        items.append(DownloadItem(
            url=f"{MODEL_REPO_URL}/config.json", dest_path=CONFIG_PATH,
            label="config.json", size_bytes=CONFIG_SIZE_BYTES,
        ))
    if not WEIGHTS_PATH.exists():
        items.append(DownloadItem(
            url=f"{MODEL_REPO_URL}/model.safetensors", dest_path=FP32_PATH,
            label="model.safetensors (fp32, converted to fp16 after download)",
            size_bytes=FP32_SIZE_BYTES, after_download=convert_to_fp16,
        ))
    return items
