"""Installs llama-server.exe (+ matching CUDA runtime, where applicable)
from ggml-org/llama.cpp's GitHub releases - the in-app replacement for
what used to be a hand-run, CUDA-13.3-only setup-tagger.cmd.

llama.cpp ships nightly-numbered builds ("b10656", "b10700", ...) rather
than semver releases, and there's no stable "latest" pointer asset to
fetch - the newest build is simply whichever release GitHub's releases
list returns first (newest-first by creation date), so resolve_latest_
build_tag() queries that live instead of hardcoding a build number that
would only get staler over time.

Reuses core.downloads' DownloadItem/download_one for the actual byte
transfer - a llama.cpp release zip is just as much a (url, dest_path,
label, size_bytes) as a curated model file, so there's no reason to
duplicate that streaming/progress logic here.

Every backend's install lands in the same canonical INSTALL_DIR
(server.py's LLAMA_SERVER_EXE always points there) - switching backends
means wiping and reinstalling, not keeping several side by side. That
mirrors this app's general preference for one obvious piece of state
over configurable multiplicity, and avoids several-GB of stale unused
backends silently accumulating on disk. MARKER_FILENAME is written only
after a fully successful install, specifically so a failed/aborted
attempt never leaves the UI claiming a backend is installed when it
might not actually work.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

from .downloads import DownloadItem, download_one

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp"
REQUEST_TIMEOUT = 15

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INSTALL_DIR = ROOT_DIR / "llama"
MARKER_FILENAME = ".llama_backend.json"


class InstallError(RuntimeError):
    pass


@dataclass
class BackendDef:
    label: str
    asset_stub: str  # matches "llama-<tag>-bin-win-<asset_stub>-x64.zip"
    cudart_stub: Optional[str] = None  # matches "cudart-llama-bin-win-<cudart_stub>-x64.zip"


# Dict order is display order; callers use this directly for a dropdown's
# choices. "cuda-13.3" is the default selection shown in the UI/CLI - not
# an auto-detected guess, just the newest/most broadly applicable choice.
BACKENDS: dict[str, BackendDef] = {
    "cuda-13.3": BackendDef("CUDA 13.3 (newest, most Nvidia GPUs)", "cuda-13.3", cudart_stub="cuda-13.3"),
    "cuda-12.4": BackendDef("CUDA 12.4 (older Nvidia GPUs/drivers)", "cuda-12.4", cudart_stub="cuda-12.4"),
    "rocm-7.14": BackendDef("ROCm 7.14 (AMD GPUs)", "rocm-7.14"),
    "cpu": BackendDef("CPU only (no GPU)", "cpu"),
}

DEFAULT_BACKEND = "cuda-13.3"


@dataclass
class InstallAsset:
    url: str
    filename: str
    size_bytes: int


@dataclass
class InstallPlan:
    backend_id: str
    build_tag: str
    main_asset: InstallAsset
    cudart_asset: Optional[InstallAsset]

    @property
    def total_bytes(self) -> int:
        total = self.main_asset.size_bytes
        if self.cudart_asset:
            total += self.cudart_asset.size_bytes
        return total


def resolve_latest_build_tag() -> str:
    resp = requests.get(f"{GITHUB_API}/releases", params={"per_page": 1}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    releases = resp.json()
    if not releases:
        raise InstallError("GitHub returned no llama.cpp releases.")
    tag = releases[0]["tag_name"]
    log.info("Latest llama.cpp build: %s", tag)
    return tag


def plan_install(backend_id: str) -> InstallPlan:
    backend = BACKENDS.get(backend_id)
    if backend is None:
        raise InstallError(f"Unknown backend: {backend_id}")

    build_tag = resolve_latest_build_tag()
    resp = requests.get(f"{GITHUB_API}/releases/tags/{build_tag}", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    assets_by_name = {a["name"]: a for a in resp.json().get("assets", [])}

    main_name = f"llama-{build_tag}-bin-win-{backend.asset_stub}-x64.zip"
    main_raw = assets_by_name.get(main_name)
    if main_raw is None:
        raise InstallError(
            f"Expected asset \"{main_name}\" not found in llama.cpp release {build_tag} - "
            f"the release layout may have changed upstream."
        )
    main_asset = InstallAsset(url=main_raw["browser_download_url"], filename=main_name, size_bytes=main_raw["size"])

    cudart_asset = None
    if backend.cudart_stub:
        cudart_name = f"cudart-llama-bin-win-{backend.cudart_stub}-x64.zip"
        cudart_raw = assets_by_name.get(cudart_name)
        if cudart_raw is None:
            raise InstallError(
                f"Expected asset \"{cudart_name}\" not found in llama.cpp release {build_tag} - "
                f"the release layout may have changed upstream."
            )
        cudart_asset = InstallAsset(
            url=cudart_raw["browser_download_url"], filename=cudart_name, size_bytes=cudart_raw["size"]
        )

    return InstallPlan(backend_id=backend_id, build_tag=build_tag, main_asset=main_asset, cudart_asset=cudart_asset)


def install(
    plan: InstallPlan,
    should_abort: Callable[[], bool],
    on_progress: Optional[Callable[[int], None]] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> bool:
    """Downloads + extracts `plan` into INSTALL_DIR, wiping any previous
    install first. Returns True on a completed install, False if
    should_abort() cut it short (mirrors download_one's own convention).
    on_progress is called with cumulative bytes written across both the
    main and cudart downloads (not per-file) - simplest thing that gives
    a single sensible progress bar. on_phase, if given, is called with a
    short phase name ("downloading", "extracting") for status text."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="xenotagger-llama-install-"))

    assets = [plan.main_asset] + ([plan.cudart_asset] if plan.cudart_asset else [])
    downloaded_paths: list[Path] = []
    bytes_before_current = 0

    try:
        if on_phase:
            on_phase("downloading")
        for asset in assets:
            dest = tmp_dir / asset.filename

            def combined_progress(written: int, _base=bytes_before_current) -> None:
                if on_progress:
                    on_progress(_base + written)

            item = DownloadItem(url=asset.url, dest_path=dest, label=asset.filename, size_bytes=asset.size_bytes)
            completed = download_one(item, should_abort, combined_progress)
            if not completed:
                log.info("llama.cpp install aborted during download of %s", asset.filename)
                return False
            downloaded_paths.append(dest)
            bytes_before_current += asset.size_bytes

        if should_abort():
            log.info("llama.cpp install aborted before extraction")
            return False

        if on_phase:
            on_phase("extracting")
        if INSTALL_DIR.exists():
            shutil.rmtree(INSTALL_DIR)
        INSTALL_DIR.mkdir(parents=True)
        for zip_path in downloaded_paths:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(INSTALL_DIR)

        marker = {"backend_id": plan.backend_id, "label": BACKENDS[plan.backend_id].label, "build_tag": plan.build_tag}
        (INSTALL_DIR / MARKER_FILENAME).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        log.info("llama.cpp installed: %s (%s) into %s", plan.backend_id, plan.build_tag, INSTALL_DIR)
        return True
    finally:
        for path in downloaded_paths:
            path.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def installed_info() -> Optional[dict]:
    marker_path = INSTALL_DIR / MARKER_FILENAME
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", marker_path, exc)
        return None
