"""Headless, unattended batch captioning - shares all its core logic
(core/pipeline.py, core/batch.py's run_batch) with the Gradio GUI; this
file only adds argparse plumbing and console/file logging on top, nothing
here duplicates what's in core/.

Settings come from the same webui/config/settings.json the GUI reads and
writes (core/config.py), including the pipeline (what runs, variants,
Output). Each run starts the pipeline, captions the folder, and stops it
again - the CLI's own "lazy start", since there's nothing to click
headless. Most flags below (--overwrite, --model, --mmproj) are optional
per-run overrides layered on top of whatever's saved, not a separate
configuration.

Example:
    system\\python\\python.exe webui\\cli.py --dir D:\\dataset\\images --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from core import config as config_mod
from core import pipeline
from core.batch import run_batch
from core.downloads import download_one
from backends import BACKENDS as REGISTRY
from backends.hydra.install import (
    MODEL_SIZE_BYTES as HYDRA_MODEL_SIZE_BYTES,
    download_model as hydra_download_model,
    install_deps as hydra_install_deps,
)
from backends.furrence2 import install as furrence2_install
from backends.llama.install import BACKENDS, DEFAULT_BACKEND, install, plan_install
from backends.llama.models import scan_model_variants


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install-llama", nargs="?", const=DEFAULT_BACKEND, choices=list(BACKENDS), default=None,
        help=f"Install llama-server.exe for the given backend and exit (default: {DEFAULT_BACKEND}). "
             f"Choices: {', '.join(BACKENDS)}",
    )
    parser.add_argument(
        "--install-hydra", action="store_true",
        help="Install Hydra's Python dependencies (torch + friends) and download its "
             "model weight (~1GB), then exit. Whether it runs is the pipeline's "
             "setting (Settings -> Pipeline, or pipeline_run in settings.json).",
    )
    parser.add_argument(
        "--install-furrence2", action="store_true",
        help="Install Furrence2's dependencies (torch, plus transformers 4.46 into "
             "furrence2/lib) and download its model (3.3GB, converted to fp16), then exit.",
    )
    parser.add_argument("--dir", required=False, help="Directory of images to caption")
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--trigger-word", default=None)
    parser.add_argument("--model", default=None, help="Model variant name, e.g. 'my-model/Q4_K_M'")
    parser.add_argument("--mmproj", default=None, help="mmproj variant name; default auto-picks the largest in the model's folder")
    parser.add_argument("--log-file", default=None, help="Also log to this file")
    parser.add_argument(
        "--debuglog", default=None,
        help="Write detailed troubleshooting info (server lifecycle, request "
             "timing/tokens, cache/resolution decisions) to this file. "
             "Independent of -v/--verbose - if omitted, none of this is "
             "collected at all, only the normal status messages go to console.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def setup_logging(log_file: str | None, debuglog: str | None, verbose: bool) -> None:
    console_level = logging.DEBUG if verbose else logging.INFO
    plain_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root = logging.getLogger()
    # Deliberately NOT touching the root logger's level - only our own
    # namespaces ("core", "backends", "cli") get elevated below. Every third-party
    # library's logger (urllib3, httpcore, asyncio, ...) inherits from root
    # when it has no level of its own, so leaving root alone keeps them all
    # at their own quiet default instead of flooding output with unrelated
    # chatter the moment DEBUG is wanted for our own code.
    our_level = logging.DEBUG if (verbose or debuglog) else logging.INFO
    logging.getLogger("core").setLevel(our_level)
    logging.getLogger("backends").setLevel(our_level)
    logging.getLogger("cli").setLevel(our_level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(plain_formatter)
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(console_level)
        file_handler.setFormatter(plain_formatter)
        root.addHandler(file_handler)

    if debuglog:
        Path(debuglog).parent.mkdir(parents=True, exist_ok=True)
        debug_handler = logging.FileHandler(debuglog, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(debug_handler)


def _install_llama(log: logging.Logger, backend_id: str) -> int:
    """Synchronous install (no worker thread needed for a one-shot CLI
    invocation) - logs progress at INFO on phase changes and roughly
    every 10% of the download, then returns 0/1. Shares backends.llama.install
    with the GUI's Settings -> Llama installer, so both stay in sync."""
    try:
        plan = plan_install(backend_id)
    except Exception as exc:
        log.error("Could not resolve a download plan for backend %r: %s", backend_id, exc)
        return 1

    log.info(
        "Installing llama.cpp %s (%s), %s ...",
        plan.build_tag, BACKENDS[backend_id].label, plan.main_asset.filename,
    )

    last_logged_pct = -1

    def on_progress(written: int) -> None:
        nonlocal last_logged_pct
        pct = int(written * 100 / max(plan.total_bytes, 1))
        if pct >= last_logged_pct + 10:
            last_logged_pct = pct
            log.info("Downloading: %d%%", pct)

    def on_phase(phase: str) -> None:
        log.info("%s ...", phase.capitalize())

    try:
        completed = install(plan, should_abort=lambda: False, on_progress=on_progress, on_phase=on_phase)
    except Exception as exc:
        log.error("Install failed: %s", exc)
        return 1

    if not completed:
        log.error("Install did not complete.")
        return 1

    log.info("llama.cpp installed successfully.")
    return 0


def _install_hydra(log: logging.Logger) -> int:
    """Synchronous (no worker thread needed for a one-shot CLI invocation) -
    pip-installs Hydra's deps then downloads its model weight, sharing
    backends.hydra.install with the GUI's Settings -> Hydra installer/
    downloader so both stay in sync."""
    log.info("Installing Hydra dependencies (torch + friends - this downloads several GB) ...")
    if not hydra_install_deps(on_output=lambda line: log.info("pip: %s", line)):
        log.error("Hydra dependency install failed.")
        return 1

    log.info("Downloading Hydra model (~1GB) ...")
    last_logged_pct = -1

    def on_progress(written: int) -> None:
        nonlocal last_logged_pct
        pct = int(written * 100 / max(HYDRA_MODEL_SIZE_BYTES, 1))
        if pct >= last_logged_pct + 10:
            last_logged_pct = pct
            log.info("Downloading: %d%%", pct)

    if not hydra_download_model(should_abort=lambda: False, on_progress=on_progress):
        log.error("Hydra model download did not complete.")
        return 1

    log.info("Hydra dependencies and model installed successfully.")
    return 0


def _install_furrence2(log: logging.Logger) -> int:
    """Same shape as _install_hydra, sharing backends.furrence2.install
    with the GUI."""
    log.info("Installing Furrence2 dependencies ...")
    if not furrence2_install.install_deps(on_output=lambda line: log.info("pip: %s", line)):
        log.error("Furrence2 dependency install failed.")
        return 1

    for item in furrence2_install.download_items():
        log.info("Downloading %s ...", item.label)
        last_logged_pct = -1

        def on_progress(written: int) -> None:
            nonlocal last_logged_pct
            pct = int(written * 100 / max(item.size_bytes, 1))
            if pct >= last_logged_pct + 10:
                last_logged_pct = pct
                log.info("Downloading: %d%%", pct)

        try:
            if not download_one(item, should_abort=lambda: False, on_progress=on_progress):
                log.error("Furrence2 model download did not complete.")
                return 1
        except Exception as exc:
            log.error("Furrence2 model download failed: %s", exc)
            return 1

    log.info("Furrence2 dependencies and model installed successfully.")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.debuglog, args.verbose)
    log = logging.getLogger("cli")

    if args.install_llama:
        return _install_llama(log, args.install_llama)

    if args.install_hydra:
        return _install_hydra(log)

    if args.install_furrence2:
        return _install_furrence2(log)

    if not args.dir:
        log.error("--dir is required (unless using --install-llama, --install-hydra or --install-furrence2)")
        return 2

    cfg = config_mod.load()
    if args.overwrite is not None:
        cfg.overwrite_existing = args.overwrite

    if args.model is not None:
        cfg.model_name = args.model
    if args.mmproj is not None:
        cfg.mmproj_name = args.mmproj

    directory = Path(args.dir)
    if not directory.is_dir():
        log.error("Not a directory: %s", directory)
        return 2

    plan = pipeline.make_plan(cfg, REGISTRY)
    if plan.problems:
        for problem in plan.problems:
            log.error("%s", problem)
        if any(b.id == "llama" for b in plan.backends) and cfg.server_mode != "external":
            available = ", ".join(m.name for m in scan_model_variants() if m.valid) or "(none found)"
            log.error("Valid llama models (--model): %s", available)
        return 2

    for progress in pipeline.start_pipeline(cfg, REGISTRY):
        log.info("%s", progress.message)
        if progress.done and not progress.ok:
            return 1

    # should_stop reaches llama's per-token check, and not_ready() makes a
    # health request - so it's only actually asked every few seconds.
    crash_check = {"at": 0.0, "crashed": False}

    def crashed() -> bool:
        now = time.monotonic()
        if now - crash_check["at"] > 5:
            crash_check.update(at=now, crashed=bool(pipeline.not_ready()))
        return crash_check["crashed"]

    def on_progress(i, total, path, status, text, detail):
        line = f"[{i}/{total}] {status}: {path.name}"
        if text:
            line += f" -> {text}"
        if detail:
            line += f" ({detail})"
        log.info("%s", line)

    try:
        result = run_batch(
            pipeline.running_plan(), directory,
            trigger_word=args.trigger_word, overwrite=cfg.overwrite_existing,
            progress_cb=on_progress, should_stop=crashed,
        )
    finally:
        for progress in pipeline.stop_pipeline():
            log.info("%s", progress.message)

    log.info("Done: %d captioned, %d skipped, %d failed", result.done, result.skipped, result.failed)
    for path, err in result.errors:
        log.error("  %s: %s", path.name, err)
    if result.conflicts:
        log.warning("Skipped (another image has the same name): %s", ", ".join(p.name for p in result.conflicts))
    return 0 if result.failed == 0 and not result.aborted else 1

if __name__ == "__main__":
    sys.exit(main())
