"""Headless, unattended batch captioning.

Example:
    system\\python\\python.exe webui\\cli.py --dir D:\\dataset\\images --recursive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core import config as config_mod
from core.client import LlamaClient
from core.models import resolve_selection, scan_model_variants
from core.server import ServerError, resolve_server


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory of images to caption")
    parser.add_argument("--recursive", action="store_true", default=None)
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
    # namespaces ("core", "cli") get elevated below. Every third-party
    # library's logger (urllib3, httpcore, asyncio, ...) inherits from root
    # when it has no level of its own, so leaving root alone keeps them all
    # at their own quiet default instead of flooding output with unrelated
    # chatter the moment DEBUG is wanted for our own code.
    our_level = logging.DEBUG if (verbose or debuglog) else logging.INFO
    logging.getLogger("core").setLevel(our_level)
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


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.debuglog, args.verbose)
    log = logging.getLogger("cli")

    cfg = config_mod.load()
    if args.recursive is not None:
        cfg.recursive_batch = args.recursive
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

    model_path = mmproj_path = None
    if cfg.server_mode != "external":
        model_path, mmproj_path, sel_error = resolve_selection(cfg)
        if sel_error:
            available = ", ".join(m.name for m in scan_model_variants() if m.valid) or "(none found)"
            log.error("%s Valid models: %s", sel_error, available)
            return 2

    try:
        base_url, managed = resolve_server(cfg, model_path, mmproj_path)
    except ServerError as exc:
        log.error("Server error: %s", exc)
        return 1

    log.info("Using llama-server at %s", base_url)
    client = LlamaClient(base_url, timeout=cfg.request_timeout)

    from core.batch import run_batch

    def on_progress(i, total, path, status, caption):
        if caption:
            log.info("[%d/%d] %s: %s -> %s", i, total, status, path.name, caption)
        else:
            log.info("[%d/%d] %s: %s", i, total, status, path.name)

    try:
        result = run_batch(
            directory,
            client,
            cfg,
            recursive=cfg.recursive_batch,
            overwrite=cfg.overwrite_existing,
            trigger_word=args.trigger_word,
            progress_cb=on_progress,
        )
    finally:
        if managed:
            managed.stop()

    log.info(
        "Done: %d captioned, %d skipped, %d failed",
        result.processed, result.skipped, result.failed,
    )
    for path, err in result.errors:
        log.error("  %s: %s", path, err)

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
