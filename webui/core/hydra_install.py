"""Installs the (optional, heavy) Python dependencies Hydra's classifier
needs - torch, pyvips, einops, safetensors, numpy - into this app's own
system\\python, and downloads the hydra-3.5.safetensors model weight.
Mirrors core.llama_install's spirit (heavy/optional, installed on demand
via an in-app button or a CLI flag, never baked into webui/requirements.txt)
but is much simpler: no GitHub-release-asset resolution needed, just a
pip install and one fixed-URL file download.

deps_installed() checks importlib.util.find_spec("torch") rather than a
marker file written after a successful install - always reflects reality
even if the user manually pip-uninstalls something afterward, unlike a
marker that could drift out of sync with what's actually on disk.

The model weight lands under core.models' own MODELS_DIR, in a
"RedRocket-Hydra" folder - reusing webui/models/'s existing gitignore
treatment rather than adding a new top-level directory. models.py's
scanner only globs *.gguf per folder, so a folder holding only a
.safetensors is invisible to it - no interference with the Models tab.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from .downloads import DownloadItem, download_one
from .models import MODELS_DIR

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = ROOT_DIR / "system" / "python" / "python.exe"

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
DEPS = ["torch>=2.6.0,<2.13.0", "pyvips[binary]", "einops", "safetensors", "numpy"]

MODEL_DIR = MODELS_DIR / "RedRocket-Hydra"
MODEL_FILENAME = "hydra-3.5.safetensors"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_URL = "https://huggingface.co/RedRocket/Hydra/resolve/main/models/hydra-3.5.safetensors"
# Confirmed via HydraTagger-portable's own research: publicly downloadable
# without an HF token (302 redirect to their CDN; the X-HF-Warning header
# on the response is just a rate-limit notice, not a block).
MODEL_SIZE_BYTES = 1_064_526_448


class InstallError(RuntimeError):
    pass


def deps_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def model_downloaded() -> bool:
    return MODEL_PATH.exists()


def install_deps(on_output: Optional[Callable[[str], None]] = None) -> bool:
    """Runs `python.exe -m pip install <DEPS> --extra-index-url <cu130>`
    via system\\python itself, streaming pip's own stdout/stderr lines to
    on_output as they arrive (pip's own progress text is what's worth
    showing here, not a byte counter - unlike llama_install's zip
    downloads). Returns True on a zero exit code."""
    if not PYTHON_EXE.exists():
        raise InstallError(f"{PYTHON_EXE} not found - is the portable environment set up?")

    argv = [
        str(PYTHON_EXE), "-m", "pip", "install",
        *DEPS, "--extra-index-url", TORCH_INDEX_URL,
    ]
    log.info("Installing Hydra dependencies: %s", " ".join(argv))
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
        log.warning("Hydra dependency install failed (exit code %s)", returncode)
        return False
    log.info("Hydra dependencies installed successfully")
    return True


def download_model(
    should_abort: Callable[[], bool],
    on_progress: Optional[Callable[[int], None]] = None,
) -> bool:
    """Downloads hydra-3.5.safetensors into MODEL_PATH. Returns True on a
    completed download, False if should_abort() cut it short - same
    convention as core.downloads.download_one itself."""
    item = DownloadItem(
        url=MODEL_URL, dest_path=MODEL_PATH, label=MODEL_FILENAME, size_bytes=MODEL_SIZE_BYTES,
    )
    return download_one(item, should_abort, on_progress)
