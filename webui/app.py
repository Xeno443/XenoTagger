"""Gradio UI: Single-image captioning, Batch processing, Models, Settings,
and an opt-in Debug tab. The CLI (cli.py) is a separate, independent
entry point that shares core/ with this file but not this file itself -
nothing UI-specific belongs in core/.

Two coordination mechanisms guard shared, process-wide state that both
tabs (and Settings' restart actions) can touch concurrently - both are
threading.RLock, not a plain Lock, because each has a function that calls
another lock-holding function of its own while already holding the lock
(get_client() calls _stop_managed(); a plain Lock would deadlock there):

  - `_session`/`_session_lock`: which llama-server (if any) we're
    currently managing. Guards against two requests racing in before any
    server is up yet (e.g. a batch run and a single-image caption both
    arriving at once) each deciding independently that no server exists
    and each starting their own.

  - `_active_operation`/`_operation_lock` ("Operation tracking" section):
    is a long-running job (single-image or batch) currently active, and
    what stage of being interrupted is it in. This is also the mutual-
    exclusion gate that stops single-image and batch from running at
    once, since both ultimately share the one llama-server connection.

The Run/Interrupt buttons are two SEPARATE gr.Button components per tab,
not one button whose label changes - this was tried first and doesn't
work: Gradio disables the triggering component of a still-pending event
client-side, and defaults every event listener's server-side
concurrency_limit to 1, so a second click on the button that's still
running the long captioning/batch call either never reaches the server
or queues silently behind it - never actually interrupts anything. The
fix (and the reason it's structured as two buttons, mirroring how Forge/
A1111 do this in modules/ui_toprow.py: Generate, Interrupt, and Skip are
independent gr.Button()s there too) is that the Interrupt button was
never the one that started the long call, so it's never "busy" and is
always immediately clickable - its own click handler is fast and
non-blocking (flip a flag; on the second click, kill the managed
llama-server process - see core/server.py's ManagedServer.stop for why a
hard kill is the only reliable way to actually interrupt an in-flight
request). First click requests a graceful stop (batch: finish the
current image, then stop before the next one; single-image has no next
item to gracefully stop before, so this click is intentionally a no-op
buffer against a stray double-click); second click force-aborts.

Because button state is normally only pushed by whichever specific
browser connection's generator is actively running (run_single_ui /
run_batch_ui's own yields), a reloaded page or a second browser tab would
otherwise show stale "idle" buttons even while something is genuinely
running elsewhere. _operation_button_states() closes that gap: it
recomputes both tabs' Run/Interrupt appearance fresh from
_active_operation on every call, and is wired into the same 2-second
status_timer that already refreshes the status bar, plus demo.load - so
any view self-corrects within one tick instead of staying wrong
indefinitely.

Restart (both restart_server_ui and restart_app_ui) doesn't block just
because something's running, nor does it silently kill it out from under
whatever's in flight - it force-aborts (skipping the two-click grace
period; see _operation_force_abort) and actually waits for that
operation to finish noticing and clean up (_wait_for_operation_to_end)
before touching the server. This matters most for restart_app_ui, whose
os.execv replaces the entire process with zero chance for any cleanup to
run afterward - waiting first means an interrupted operation gets an
observable, handled failure instead of just vanishing without a trace.
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# Must be set before `import gradio` - some of its telemetry checks read
# this at import time, not just when building a Blocks instance.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from core import config as config_mod
from core.batch import find_images, run_batch
from core.captioner import caption_image
from core.client import ClientError, LlamaClient
from core.config import AppConfig
from core.models import IGNORED_SUBSTRINGS, ModelVariant, MmprojVariant, resolve_selection, scan_all
from core.server import LOG_PATH, ManagedServer, ServerError, get_loaded_model_name, is_healthy, resolve_server
from ui_css import ALL_CSS

log = logging.getLogger("app")  # fixed name, not __name__ - which is "__main__"
                                  # when run directly (the real usage) but
                                  # "app" when imported (e.g. in tests) -
                                  # this way logging.getLogger("app") always
                                  # refers to the right logger either way.

current_cfg: AppConfig = config_mod.load()

_session = {"managed": None, "client": None, "base_url": None}
# Guards every read/decide/write of _session - without it, two requests
# racing in at once (e.g. a batch run and a single-image caption, both
# arriving before any server is up yet) could both decide no server exists
# and each start their own, stomping on each other's _session writes and
# possibly killing one process mid-launch via the other's _stop_managed().
# Reentrant (not a plain Lock) because get_client() calls _stop_managed()
# internally while already holding it - a plain Lock would deadlock there.
_session_lock = threading.RLock()

# Debug tab: captures log output from our own Python code (core.* modules
# all log via the standard `logging` module, but nothing was ever attached
# to see it before this) into a bounded in-memory buffer.
_PY_LOG_BUFFER: deque[str] = deque(maxlen=2000)


class _BufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _PY_LOG_BUFFER.append(self.format(record))


if current_cfg.debug_tab_enabled:
    _log_handler = _BufferLogHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_log_handler)
    # Deliberately NOT touching the root logger's level. Elevating only our
    # own namespaces to DEBUG (rather than the root, which every library's
    # logger inherits from when it has no level of its own) means Gradio's
    # httpx/httpcore/asyncio internals - and anything else - stay at their
    # own quiet default instead of flooding this with unrelated chatter.
    logging.getLogger("core").setLevel(logging.DEBUG)
    logging.getLogger("app").setLevel(logging.DEBUG)


def _stop_managed() -> None:
    with _session_lock:
        if _session["managed"] is not None:
            _session["managed"].stop()
        _session["managed"] = None
        _session["client"] = None
        _session["base_url"] = None


atexit.register(_stop_managed)


# ------------------------------------------------------ Operation tracking
#
# Single source of truth for "is something long-running active, and what".
# The Single-image and Batch tabs' Run buttons double as Interrupt buttons
# once running (same Generate/Interrupt pattern most Stable Diffusion
# WebUIs use): first click while running asks for a graceful stop (batch:
# finish the current image, then stop before the next one; single-image
# has no queue to gracefully stop before, so this click is just a
# deliberate one-click buffer against an accidental double-click), second
# click hard-aborts by killing the managed llama-server outright - the
# only actually reliable way to stop an in-flight request. (llama-server
# itself has open upstream bugs where it doesn't reliably notice a
# disconnected client and keeps generating regardless - closing our end of
# the connection alone can't be trusted to stop it; see ManagedServer.stop
# in core/server.py for the real kill.)
#
# This also acts as the mutual-exclusion gate for the race discussed
# above _session_lock: only one kind of operation ("single" or "batch")
# may be active at a time, since both ultimately share the one
# llama-server connection.

@dataclass
class _Operation:
    kind: str  # "single" or "batch"
    label: str  # human-readable, shown in the status bar
    stop_requested: bool = False
    abort_requested: bool = False


_active_operation: Optional[_Operation] = None
_operation_lock = threading.RLock()


def _operation_blocked_by() -> Optional[str]:
    """None if nothing is currently running. Otherwise a message naming
    what's active - covers both a different kind of operation and an
    accidental duplicate start of the same kind (the Run button disables
    itself while busy, but this is a second line of defense, e.g. against
    a stray click landing just before that update reaches the browser)."""
    with _operation_lock:
        op = _active_operation
        if op is not None:
            return f"{op.label} is currently running - interrupt it from its own tab first."
        return None


def _operation_start(kind: str, label: str) -> None:
    global _active_operation
    with _operation_lock:
        _active_operation = _Operation(kind=kind, label=label)


def _operation_end() -> None:
    global _active_operation
    with _operation_lock:
        _active_operation = None


def _operation_interrupt_click(kind: str) -> str:
    """Call when the Run/Interrupt button is clicked while `kind` is
    already the active operation. Returns "stopping" on the first click
    (graceful - just sets a flag the running loop checks), "aborting" on
    the second (hard - kills the server, which is done outside the lock
    since _stop_managed() acquires its own _session_lock)."""
    with _operation_lock:
        op = _active_operation
        if op is None or op.kind != kind:
            return "idle"
        if not op.stop_requested:
            op.stop_requested = True
            log.info("%s: stop requested (graceful)", op.label)
            return "stopping"
        op.abort_requested = True
        log.info("%s: abort requested (killing server)", op.label)
    _stop_managed()
    return "aborting"


def _operation_should_stop(kind: str) -> bool:
    with _operation_lock:
        op = _active_operation
        return op is not None and op.kind == kind and op.stop_requested


def _operation_status_text() -> str:
    with _operation_lock:
        op = _active_operation
        if op is None:
            return "idle"
        if op.abort_requested:
            return f"{op.label} (aborting)"
        if op.stop_requested:
            return f"{op.label} (stopping)"
        return op.label


def _operation_button_states():
    """Ground-truth Run/Interrupt button appearance for both tabs,
    recomputed fresh from _active_operation every call - not just
    whatever a single running generator last pushed. Wired into the
    periodic status timer and page load below, so a reloaded page, a
    second browser tab, or a missed/delayed update all self-correct
    within one tick instead of showing stale button state indefinitely.
    Returns (single_run, single_interrupt, batch_run, batch_interrupt).
    """
    with _operation_lock:
        op = _active_operation
        kinds_state = []
        for kind in ("single", "batch"):
            if op is None:
                kinds_state.append((True, False, "Interrupt", True))  # run_ok, show_interrupt, label, interrupt_ok
            elif op.kind != kind:
                kinds_state.append((False, False, "Interrupt", True))
            elif op.abort_requested:
                kinds_state.append((False, True, "Interrupting...", False))
            elif op.stop_requested:
                kinds_state.append((False, True, "Interrupt (click again to abort now)", True))
            else:
                kinds_state.append((False, True, "Interrupt", True))

    updates = []
    for run_interactive, show_interrupt, label, interrupt_interactive in kinds_state:
        updates.append(gr.update(interactive=run_interactive))
        updates.append(gr.update(visible=show_interrupt, value=label, variant="stop", interactive=interrupt_interactive))
    return tuple(updates)


def _operation_force_abort() -> None:
    """Immediately hard-aborts whatever's running, if anything - skips the
    two-click grace period (for Restart, which needs the operation gone
    now, not eventually). No-op if nothing is running."""
    with _operation_lock:
        op = _active_operation
        if op is None:
            return
        op.stop_requested = True
        op.abort_requested = True
        log.info("%s: force-aborted (restart requested)", op.label)
    _stop_managed()


def _wait_for_operation_to_end(timeout: float = 30.0) -> None:
    """Blocks until the active operation's own thread notices the abort
    and calls _operation_end() - killing the server (above) should make
    its in-flight request fail almost immediately, so this is normally
    quick; the timeout is just a safety net against hanging forever."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _operation_lock:
            if _active_operation is None:
                return
        time.sleep(0.2)
    log.warning("Timed out waiting for the running operation to stop before restart")


def get_client(cfg: AppConfig) -> LlamaClient:
    with _session_lock:
        if cfg.server_mode == "external":
            desired_base = cfg.external_url.rstrip("/")
        else:
            desired_base = f"http://{cfg.server_host}:{cfg.server_port}"

        if _session["client"] is not None and _session["base_url"] == desired_base:
            log.debug("get_client(): reusing existing client for %s", desired_base)
            return _session["client"]

        model_path = mmproj_path = None
        if cfg.server_mode != "external":
            model_path, mmproj_path, error = resolve_selection(cfg)
            if error:
                log.warning("get_client(): %s", error)
                raise ServerError(error)

        _stop_managed()
        log.debug("get_client(): resolving server (mode=%s)", cfg.server_mode)
        base_url, managed = resolve_server(cfg, model_path, mmproj_path)
        _session["managed"] = managed
        _session["client"] = LlamaClient(base_url, timeout=cfg.request_timeout)
        _session["base_url"] = base_url
        return _session["client"]


def restart_server_ui() -> str:
    log.info("User cleared the server connection from Settings")
    # If a single-image or batch operation is currently running, force-abort
    # it now (rather than block this action) and wait for that to actually
    # finish before touching the server out from under it - see the
    # Operation tracking section above for why a hard kill is the only
    # reliable way to interrupt an in-flight request.
    _operation_force_abort()
    _wait_for_operation_to_end()
    _stop_managed()
    return "Server connection cleared. It will (re)connect/(re)start on the next request."


def restart_app_ui() -> None:
    """Replaces this process with a fresh one via os.execv - the standard
    Python self-restart trick, so newly-saved settings.json values (or a
    just-toggled Debug tab) take effect without anyone needing a terminal.
    Never returns: the process image is gone the moment execv succeeds, so
    stop our own managed llama-server first (execv skips atexit entirely).
    """
    log.info("User triggered app restart from Settings")
    _operation_force_abort()
    _wait_for_operation_to_end()
    _stop_managed()
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------- Single tab

# The Run/Caption button just disables while busy - it never needs a
# second click to do anything, so it can't get stuck behind Gradio's
# per-event concurrency limit. Interrupting is a SEPARATE button (see
# interrupt_single_ui below) with its own fast, non-generator click
# handler, so it's never "the busy component" and is always clickable
# immediately - this is the same split Forge/A1111 use (modules/
# ui_toprow.py: Generate, Interrupt, and Skip are three independent
# gr.Button()s, not one button that changes what its own click does).
_RUN_IDLE = gr.update(interactive=True)
_RUN_BUSY = gr.update(interactive=False)
_INTERRUPT_HIDDEN = gr.update(visible=False, value="Interrupt", variant="stop", interactive=True)
_INTERRUPT_SHOWN = gr.update(visible=True, value="Interrupt", variant="stop", interactive=True)


def run_single_ui(image_path, trigger_word_override: str):
    blocked = _operation_blocked_by()
    if blocked:
        yield gr.update(), blocked, gr.update(), gr.update()
        return

    if not image_path:
        yield "", "Please choose an image first.", gr.update(), gr.update()
        return

    _operation_start("single", "Single-image captioning")
    try:
        log.info("Single-image caption requested: %s", image_path)
        cfg = current_cfg
        base_url = _display_base_url()
        already_up = is_healthy(base_url)
        if cfg.server_mode == "external":
            yield "", ("Processing..." if already_up else "Connecting to external server..."), _RUN_BUSY, _INTERRUPT_SHOWN
        else:
            yield "", ("Processing..." if already_up else "Starting server (loading model)..."), _RUN_BUSY, _INTERRUPT_SHOWN

        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Single-image caption: server error: %s", exc)
            yield "", f"Server error: {exc}", _RUN_IDLE, _INTERRUPT_HIDDEN
            return

        if not already_up:
            yield "", "Processing...", gr.update(), gr.update()

        try:
            caption, result = caption_image(
                image_path, client, cfg, trigger_word=trigger_word_override
            )
        except ClientError as exc:
            log.warning("Single-image caption failed for %s: %s", image_path, exc)
            yield "", f"Captioning failed: {exc}", _RUN_IDLE, _INTERRUPT_HIDDEN
            return

        speed = f", {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""

        if result.truncated:
            # llama-server clamps actual generation to whatever fits in
            # (context_size - prompt_tokens), even if that's below the
            # requested max_tokens. If it stopped short of max_tokens, raising
            # max_tokens further won't help - context_size is the real ceiling.
            if result.completion_tokens < cfg.max_tokens:
                advice = (
                    f"prompt used {result.prompt_tokens} tokens, leaving no room "
                    f"in the {cfg.context_size}-token context for the requested "
                    f"{cfg.max_tokens} — raise Context size in Settings"
                )
            else:
                advice = "raise Max tokens in Settings for a full caption"
            status = (
                f"Finished in {result.elapsed_s:.1f}s, but CUT OFF at {result.completion_tokens} "
                f"tokens{speed} ({advice})"
            )
        else:
            status = f"Finished in {result.elapsed_s:.1f}s ({result.completion_tokens} tokens{speed})"
        if result.resize_note:
            status = f"Resized {result.resize_note}. {status}"
        yield caption, status, _RUN_IDLE, _INTERRUPT_HIDDEN
    finally:
        _operation_end()


def interrupt_single_ui():
    action = _operation_interrupt_click("single")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level above webui/


def _is_gradio_temp_upload(path: Path) -> bool:
    """True if `path` is a copy Gradio made in its own temp cache rather
    than the real file - which is always true for a browser drag-and-drop
    or click-to-upload, since browsers never expose a local file's real
    absolute path to the page. gr.Image gives us that temp copy's path,
    not the original's, so we can't just save next to it in that case.
    """
    import tempfile

    gradio_temp = Path(os.environ.get("GRADIO_TEMP_DIR") or (Path(tempfile.gettempdir()) / "gradio"))
    try:
        path.resolve().relative_to(gradio_temp.resolve())
        return True
    except ValueError:
        return False


def clear_single_result_ui() -> tuple[str, str]:
    """Wired to the image box's own .change() - fires on upload, clear, or
    replace, so a stale caption/status from a previous image never lingers
    next to a new or now-empty image slot.
    """
    return "", ""


def save_single_ui(image_path, caption_text: str) -> str:
    if not image_path:
        return "No image loaded."
    if not caption_text.strip():
        return "Nothing to save."

    src = Path(image_path)
    if not _is_gradio_temp_upload(src):
        txt_path = src.with_suffix(".txt")
        txt_path.write_text(caption_text, encoding="utf-8")
        log.info("Saved caption to %s", txt_path)
        return f"Saved to {txt_path}"

    log.debug("save_single_ui(): %s is a Gradio temp upload - showing save dialog", src)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    save_path = filedialog.asksaveasfilename(
        initialdir=str(PROJECT_ROOT),
        initialfile=src.with_suffix(".txt").name,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    root.destroy()

    if not save_path:
        log.info("Save caption cancelled by user")
        return "Save cancelled."
    Path(save_path).write_text(caption_text, encoding="utf-8")
    log.info("Saved caption to %s", save_path)
    return f"Saved to {save_path}"


# ----------------------------------------------------------------- Batch tab

def browse_directory_ui(current_value: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = current_value if current_value and Path(current_value).is_dir() else None
    selected = filedialog.askdirectory(initialdir=initial) if initial else filedialog.askdirectory()
    root.destroy()
    if selected:
        log.debug("browse_directory_ui(): user picked %s", selected)
    return selected or current_value


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def run_batch_ui(directory_str, recursive, overwrite, trigger_word_override):
    blocked = _operation_blocked_by()
    if blocked:
        yield blocked, "", None, "", gr.update(), gr.update()
        return

    log.info("Batch requested: %s (recursive=%s, overwrite=%s)", directory_str, recursive, overwrite)
    directory = Path(directory_str) if directory_str else None
    if not directory or not directory.is_dir():
        log.warning("Batch: not a directory: %s", directory_str)
        yield f"Not a directory: {directory_str}", "", None, "", gr.update(), gr.update()
        return

    _operation_start("batch", "Batch captioning")
    try:
        cfg = current_cfg
        base_url = _display_base_url()
        already_up = is_healthy(base_url)
        if cfg.server_mode == "external":
            yield ("Processing..." if already_up else "Connecting to external server..."), "", None, "", _RUN_BUSY, _INTERRUPT_SHOWN
        else:
            yield ("Processing..." if already_up else "Starting server (loading model)..."), "", None, "", _RUN_BUSY, _INTERRUPT_SHOWN

        try:
            client = get_client(cfg)
        except ServerError as exc:
            log.warning("Batch: server error: %s", exc)
            yield f"Server error: {exc}", "", None, "", _RUN_IDLE, _INTERRUPT_HIDDEN
            return

        if not already_up:
            yield "Processing...", "", None, "", gr.update(), gr.update()

        images = find_images(directory, recursive=recursive)
        if not images:
            log.warning("Batch: no images found in %s", directory)
            yield "No images found in that directory.", "", None, "", _RUN_IDLE, _INTERRUPT_HIDDEN
            return

        q: "queue.Queue" = queue.Queue()
        result_holder = {}

        def progress_cb(i, total, path, status, caption, resize_note):
            q.put((i, total, path, status, caption, resize_note))

        def worker():
            try:
                result_holder["result"] = run_batch(
                    directory, client, current_cfg,
                    recursive=recursive,
                    overwrite=overwrite,
                    trigger_word=trigger_word_override,
                    progress_cb=progress_cb,
                    should_stop=lambda: _operation_should_stop("batch"),
                )
            except Exception as exc:  # noqa: BLE001 - surface to UI, don't crash app
                result_holder["error"] = str(exc)
            q.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        start_time = time.monotonic()
        thread.start()

        last_file = ""
        last_image = None
        last_caption = ""
        while True:
            item = q.get()
            if item is None:
                break
            i, total, path, status, caption, resize_note = item

            elapsed = time.monotonic() - start_time
            avg = elapsed / i if i else 0.0
            eta = avg * (total - i)
            status_text = f"{i}/{total} processed · avg {avg:.1f}s/image · ETA {_format_duration(eta)}"
            if resize_note:
                status_text += f" · resized {resize_note}"

            last_file = path.name
            try:
                # Gradio refuses to serve a raw path outside its CWD/temp dir
                # (InvalidPathError) - batch directories can be anywhere on
                # disk, so hand it decoded image data instead of a path at all.
                with PILImage.open(path) as img:
                    last_image = img.copy()
            except OSError as exc:
                log.warning("Could not load preview for %s: %s", path, exc)
                last_image = None
            if status == "ok":
                last_caption = caption or ""
            elif status == "skipped":
                last_caption = "(skipped - caption already exists)"
            elif status == "truncated":
                last_caption = "(truncated - not saved, see .txt.issue)"
            else:
                last_caption = "(failed - see .txt.issue)"
            yield status_text, last_file, last_image, last_caption, gr.update(), gr.update()

        thread.join()
        if "error" in result_holder:
            yield f"Unexpected error: {result_holder['error']}", last_file, last_image, last_caption, _RUN_IDLE, _INTERRUPT_HIDDEN
        else:
            r = result_holder["result"]
            total_elapsed = time.monotonic() - start_time
            verb = "Aborted" if r.aborted else "Done"
            summary = (
                f"{verb}: {r.processed} captioned, {r.truncated} truncated, "
                f"{r.skipped} skipped, {r.failed} failed "
                f"in {_format_duration(total_elapsed)}"
            )
            yield summary, last_file, last_image, last_caption, _RUN_IDLE, _INTERRUPT_HIDDEN
    finally:
        _operation_end()


def interrupt_batch_ui():
    action = _operation_interrupt_click("batch")
    if action == "aborting":
        return gr.update(value="Interrupting...", interactive=False)
    return gr.update(value="Interrupt (click again to abort now)")


# -------------------------------------------------------------- Settings tab

def _model_choices(models: list[ModelVariant]) -> list[str]:
    return [m.name for m in models]


def _mmproj_choices(
    model_name: str, models: list[ModelVariant], mmprojs: list[MmprojVariant]
) -> list[str]:
    """mmproj choices for the given model's folder, plus a leading '' (auto)."""
    model = next((m for m in models if m.name == model_name), None)
    if model is None:
        return [""]
    return [""] + [m.name for m in mmprojs if m.folder == model.folder]


def on_model_change_ui(model_name: str):
    models, mmprojs = scan_all()
    return gr.update(choices=_mmproj_choices(model_name, models, mmprojs), value="")


def save_settings_ui(
    server_mode, server_host, server_port, external_url,
    model_name, mmproj_name, n_gpu_layers, context_size, extra_server_args,
    resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
    prompt_template, temperature, top_p, max_tokens, request_timeout,
    trigger_word, overwrite_existing, recursive_batch, debug_tab_enabled,
) -> str:
    global current_cfg
    current_cfg = AppConfig(
        server_mode=server_mode,
        server_host=server_host,
        server_port=int(server_port),
        external_url=external_url,
        model_name=model_name or "",
        mmproj_name=mmproj_name or "",
        n_gpu_layers=n_gpu_layers.strip() or "auto",
        context_size=int(context_size),
        extra_server_args=extra_server_args,
        resize_enabled=bool(resize_enabled),
        resize_target_mp=float(resize_target_mp),
        snap_enabled=bool(snap_enabled),
        snap_multiple=int(snap_multiple),
        prompt_template=prompt_template,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        request_timeout=int(request_timeout),
        trigger_word=trigger_word,
        overwrite_existing=bool(overwrite_existing),
        recursive_batch=bool(recursive_batch),
        debug_tab_enabled=bool(debug_tab_enabled),
    )
    log.info(
        "Settings saved: server_mode=%s model=%s ngl=%s",
        current_cfg.server_mode, current_cfg.model_name or "(none)", current_cfg.n_gpu_layers,
    )
    config_mod.save(current_cfg)
    return "Settings saved. Use \"Restart server connection\" if you changed the model or server settings, or restart the app if you changed the Debug tab setting."


# ---------------------------------------------------------------- Models tab

def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _models_table(models: list[ModelVariant]) -> str:
    rows = [[m.name, "yes" if m.valid else "no", m.architecture or "", m.error or ""] for m in models]
    return _md_table(["Model", "Valid", "Architecture", "Error"], rows)


def _mmprojs_table(mmprojs: list[MmprojVariant]) -> str:
    rows = [[m.name, "yes" if m.valid else "no", m.error or ""] for m in mmprojs]
    return _md_table(["mmproj", "Valid", "Error"], rows)


def refresh_models_ui(current_model_name: str):
    models, mmprojs = scan_all()
    return (
        _models_table(models),
        _mmprojs_table(mmprojs),
        gr.update(choices=_model_choices(models)),
        gr.update(choices=_mmproj_choices(current_model_name, models, mmprojs)),
    )


# ------------------------------------------------------------------ Status bar

def _display_base_url() -> str:
    cfg = current_cfg
    if cfg.server_mode == "external":
        return cfg.external_url.rstrip("/")
    return f"http://{cfg.server_host}:{cfg.server_port}"


def _server_status_text(base_url: str, healthy: bool) -> str:
    # Just "managed" (we started it) vs "remote" (already running somewhere
    # - local-but-not-ours, external mode, doesn't matter which) - the
    # adjacent "Model: n/a" already covers unhealthy/crashed, so a third
    # state for that here would just be saying the same thing twice.
    cfg = current_cfg
    if cfg.server_mode == "external":
        return "remote"

    managed = _session.get("managed")
    connected = _session.get("client") is not None and _session.get("base_url") == base_url
    if connected and managed is not None:
        return "managed"
    return "remote" if healthy else "stopped"


def get_status_text() -> str:
    base_url = _display_base_url()
    healthy = is_healthy(base_url)
    # Always the server's own answer, never our config - our selection is
    # just what we'd ask it to load next, not necessarily what's actually
    # loaded right now (e.g. external mode, or a server someone else picked).
    model = get_loaded_model_name(base_url) if healthy else None
    model = model or "n/a"
    return (
        f"Python {platform.python_version()} &nbsp;·&nbsp; "
        f"Gradio {gr.__version__} &nbsp;·&nbsp; "
        f"llama-server: {_server_status_text(base_url, healthy)} &nbsp;·&nbsp; "
        f"Model: {model} &nbsp;·&nbsp; "
        f"{_operation_status_text()}"
    )


# ---------------------------------------------------------------- Debug tab

def get_python_debug_text() -> str:
    # Newest first - the boxes don't reliably autoscroll on periodic
    # Timer.tick updates (Gradio's autoscroll is gated on streaming/
    # generating state, not plain polled value replacement), so the latest
    # line needs to be visible without scrolling.
    return "\n".join(reversed(_PY_LOG_BUFFER)) or "(no log output yet)"


def get_llama_debug_text() -> str:
    if _session.get("managed") is None:
        return "n/a"
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        return "\n".join(reversed(text.splitlines())) if text else "(empty)"
    except OSError as exc:
        return f"(could not read {LOG_PATH}: {exc})"


def clear_debug_ui() -> tuple[str, str]:
    _PY_LOG_BUFFER.clear()
    if _session.get("managed") is not None:
        try:
            LOG_PATH.write_text("", encoding="utf-8")
        except OSError:
            pass
    return get_python_debug_text(), get_llama_debug_text()


def build_app() -> gr.Blocks:
    cfg = current_cfg
    models, mmprojs = scan_all()

    with gr.Blocks(title="NLPtagger", analytics_enabled=False) as demo:
        gr.Markdown("# NLPtagger — LoRA dataset captioning")

        with gr.Tab("Single image"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    single_image = gr.Image(
                        type="filepath", label="Image", sources=["upload"]
                    )
                with gr.Column():
                    single_trigger = gr.Textbox(
                        label="Trigger word",
                        value=cfg.trigger_word,
                    )
                    single_caption = gr.Textbox(
                        label="Caption", lines=20, interactive=True
                    )
            with gr.Row():
                single_run_btn = gr.Button("Caption", variant="primary")
                # Separate from single_run_btn on purpose - see the
                # Operation tracking section: an Interrupt that shares the
                # same button/event as the long-running call would sit
                # queued behind it and never actually reach the server
                # while captioning is in flight.
                single_interrupt_btn = gr.Button("Interrupt", variant="stop", visible=False)
                single_save_btn = gr.Button("Save caption", interactive=False)
            single_status = gr.Textbox(show_label=False, container=False, interactive=False)

            single_run_btn.click(
                run_single_ui,
                [single_image, single_trigger],
                [single_caption, single_status, single_run_btn, single_interrupt_btn],
            )
            single_interrupt_btn.click(interrupt_single_ui, [], [single_interrupt_btn])
            single_save_btn.click(
                save_single_ui, [single_image, single_caption], [single_status]
            )
            single_caption.change(
                lambda text: gr.update(interactive=bool(text.strip())),
                [single_caption], [single_save_btn],
            )
            single_image.change(clear_single_result_ui, [], [single_caption, single_status])

        with gr.Tab("Batch processing"):
            with gr.Row():
                batch_dir = gr.Textbox(label="Directory of images", scale=4)
                batch_browse_btn = gr.Button("Browse...", scale=1)

            with gr.Row():
                batch_trigger = gr.Textbox(
                    label="Trigger word",
                    value=cfg.trigger_word,
                )
                with gr.Column():
                    batch_recursive = gr.Checkbox(label="Recursive", value=cfg.recursive_batch)
                    batch_overwrite = gr.Checkbox(
                        label="Overwrite existing captions", value=cfg.overwrite_existing
                    )

            with gr.Row():
                batch_run_btn = gr.Button("Run batch", variant="primary")
                batch_interrupt_btn = gr.Button("Interrupt", variant="stop", visible=False)

            with gr.Row():
                batch_last_image = gr.Image(
                    label="Preview", interactive=False, show_label=False,
                    buttons=[],
                )
                with gr.Column():
                    batch_last_file = gr.Textbox(label="Last file processed", interactive=False)
                    batch_last_caption = gr.Textbox(label="Last caption created", interactive=False)

            batch_status = gr.Textbox(show_label=False, container=False, interactive=False)

            batch_browse_btn.click(browse_directory_ui, [batch_dir], [batch_dir])
            batch_run_btn.click(
                run_batch_ui,
                [batch_dir, batch_recursive, batch_overwrite, batch_trigger],
                [batch_status, batch_last_file, batch_last_image, batch_last_caption, batch_run_btn, batch_interrupt_btn],
            )
            batch_interrupt_btn.click(interrupt_batch_ui, [], [batch_interrupt_btn])

        with gr.Tab("Models"):
            gr.Markdown(
                "Every `.gguf` under a `webui/models/<folder>/` is picked up: "
                "files with \"mmproj\" in the name are projectors, everything "
                "else is a selectable model quant (pick which mmproj pairs "
                "with it in Settings). Files matching "
                f"`{', '.join(IGNORED_SUBSTRINGS)}` "
                "are ignored (e.g. speculative-decoding draft models)."
            )
            gr.Markdown("**Models**")
            models_table = gr.Markdown(_models_table(models))
            gr.Markdown("**mmproj files**")
            mmprojs_table = gr.Markdown(_mmprojs_table(mmprojs))
            refresh_models_btn = gr.Button("Refresh")

        with gr.Tab("Settings"):
            with gr.Group():
                gr.Markdown("### llama-server")
                server_mode = gr.Radio(
                    choices=[
                        (
                            "Auto - use an already-running server if there is one; "
                            "otherwise start llama-server.exe and stop it again when done",
                            "auto",
                        ),
                        (
                            "External - always connect to a server you start yourself, "
                            "possibly on a different machine",
                            "external",
                        ),
                    ],
                    value=cfg.server_mode,
                    label="Server mode",
                )
                with gr.Row():
                    server_host = gr.Textbox(label="Host (auto mode)", value=cfg.server_host)
                    server_port = gr.Number(label="Port (auto mode)", value=cfg.server_port, precision=0)
                external_url = gr.Textbox(
                    label="External server URL (external mode)", value=cfg.external_url
                )
                with gr.Row():
                    model_name = gr.Dropdown(
                        label="Model",
                        choices=_model_choices(models),
                        value=cfg.model_name or None,
                    )
                    mmproj_name = gr.Dropdown(
                        label="mmproj (blank = auto-pick largest in model's folder)",
                        choices=_mmproj_choices(cfg.model_name, models, mmprojs),
                        value=cfg.mmproj_name or "",
                    )
                with gr.Row():
                    n_gpu_layers = gr.Textbox(
                        label="GPU layers ('auto', 'all', or an exact number)",
                        value=cfg.n_gpu_layers,
                    )
                    context_size = gr.Number(label="Context size", value=cfg.context_size, precision=0)
                extra_server_args = gr.Textbox(
                    label="Extra llama-server arguments", value=cfg.extra_server_args
                )
                restart_server_btn = gr.Button("Restart server connection")

            with gr.Group():
                gr.Markdown("### Generation")
                prompt_template = gr.Textbox(
                    label="Prompt template", value=cfg.prompt_template, lines=4
                )
                with gr.Row():
                    temperature = gr.Slider(0.0, 2.0, value=cfg.temperature, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=cfg.top_p, label="Top-p")
                    max_tokens = gr.Number(label="Max tokens", value=cfg.max_tokens, precision=0)
                request_timeout = gr.Number(
                    label="Request timeout (seconds)", value=cfg.request_timeout, precision=0
                )

            with gr.Group():
                gr.Markdown("### Image resizing")
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

            with gr.Group():
                gr.Markdown("### Captioning defaults")
                trigger_word = gr.Textbox(label="Default trigger word", value=cfg.trigger_word)
                with gr.Row():
                    overwrite_existing = gr.Checkbox(
                        label="Overwrite existing captions by default", value=cfg.overwrite_existing
                    )
                    recursive_batch = gr.Checkbox(
                        label="Recursive batch by default", value=cfg.recursive_batch
                    )

            with gr.Group():
                gr.Markdown("### Debug")
                debug_tab_enabled = gr.Checkbox(
                    label="Enable Debug tab (requires app restart)",
                    value=cfg.debug_tab_enabled,
                )

            save_settings_btn = gr.Button("Save settings", variant="primary")
            restart_app_btn = gr.Button("Restart app")
            settings_status = gr.Textbox(label="Status", interactive=False)

            settings_inputs = [
                server_mode, server_host, server_port, external_url,
                model_name, mmproj_name, n_gpu_layers, context_size, extra_server_args,
                resize_enabled, resize_target_mp, snap_enabled, snap_multiple,
                prompt_template, temperature, top_p, max_tokens, request_timeout,
                trigger_word, overwrite_existing, recursive_batch, debug_tab_enabled,
            ]
            save_settings_btn.click(save_settings_ui, settings_inputs, [settings_status])
            restart_server_btn.click(restart_server_ui, [], [settings_status])
            restart_app_btn.click(restart_app_ui, [], [])
            model_name.change(on_model_change_ui, [model_name], [mmproj_name])

        with gr.Tab("Debug", visible=cfg.debug_tab_enabled):
            gr.Markdown("**Python debug log**")
            debug_python_box = gr.Textbox(
                lines=18, max_lines=18, interactive=False, show_label=False,
                value=get_python_debug_text(),
            )
            gr.Markdown("**llama-server output** (n/a unless started by this app)")
            debug_llama_box = gr.Textbox(
                lines=18, max_lines=18, interactive=False, show_label=False,
                value=get_llama_debug_text(),
            )
            debug_clear_btn = gr.Button("Clear")
            debug_clear_btn.click(clear_debug_ui, [], [debug_python_box, debug_llama_box])

        refresh_models_btn.click(
            refresh_models_ui, [model_name],
            [models_table, mmprojs_table, model_name, mmproj_name],
        )

        status_bar = gr.Markdown(get_status_text(), elem_id="status-bar")
        _run_interrupt_btns = [single_run_btn, single_interrupt_btn, batch_run_btn, batch_interrupt_btn]
        status_timer = gr.Timer(2.0)
        status_timer.tick(get_status_text, [], [status_bar])
        status_timer.tick(_operation_button_states, [], _run_interrupt_btns)
        if cfg.debug_tab_enabled:
            status_timer.tick(get_python_debug_text, [], [debug_python_box])
            status_timer.tick(get_llama_debug_text, [], [debug_llama_box])
        demo.load(get_status_text, [], [status_bar])
        demo.load(_operation_button_states, [], _run_interrupt_btns)

    return demo


UI_PORT = 7901


def main() -> None:
    demo = build_app()
    demo.queue()
    demo.launch(server_port=UI_PORT, footer_links=[], css=ALL_CSS)


if __name__ == "__main__":
    main()
