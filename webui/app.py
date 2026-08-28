"""Gradio UI: single-image captioning, batch captioning, settings, models."""

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
from dataclasses import asdict
from pathlib import Path

from PIL import Image as PILImage

# Must be set before `import gradio` - some of its telemetry checks read
# this at import time, not just when building a Blocks instance.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from core import config as config_mod
from core.batch import find_images, run_batch
from core.captioner import apply_trigger_word, caption_image
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
    if _session["managed"] is not None:
        _session["managed"].stop()
    _session["managed"] = None
    _session["client"] = None
    _session["base_url"] = None


atexit.register(_stop_managed)


def get_client(cfg: AppConfig) -> LlamaClient:
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
    _stop_managed()
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------- Single tab

def run_single_ui(image_path, trigger_word_override: str):
    if not image_path:
        yield "", "Please choose an image first."
        return

    log.info("Single-image caption requested: %s", image_path)
    cfg = current_cfg
    base_url = _display_base_url()
    already_up = is_healthy(base_url)
    if cfg.server_mode == "external":
        yield "", "Processing..." if already_up else "Connecting to external server..."
    else:
        yield "", "Processing..." if already_up else "Starting server (loading model)..."

    try:
        client = get_client(cfg)
    except ServerError as exc:
        log.warning("Single-image caption: server error: %s", exc)
        yield "", f"Server error: {exc}"
        return

    if not already_up:
        yield "", "Processing..."

    try:
        result = client.caption(
            image_path,
            prompt=cfg.prompt_template,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
    except ClientError as exc:
        log.warning("Single-image caption failed for %s: %s", image_path, exc)
        yield "", f"Captioning failed: {exc}"
        return

    word = cfg.trigger_word if trigger_word_override is None else (trigger_word_override or cfg.trigger_word)
    caption = apply_trigger_word(result.content, word)
    speed = f", {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second else ""

    if result.truncated:
        status = (
            f"Finished in {result.elapsed_s:.1f}s, but CUT OFF at {result.completion_tokens} "
            f"tokens{speed} (hit the Max tokens limit — raise it in Settings for a full caption)"
        )
    else:
        status = f"Finished in {result.elapsed_s:.1f}s ({result.completion_tokens} tokens{speed})"
    yield caption, status


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


_batch_stop_event = threading.Event()


def abort_batch_ui() -> str:
    log.info("Batch abort requested by user")
    _batch_stop_event.set()
    return "Stopping after the current image finishes..."


def run_batch_ui(directory_str, recursive, overwrite, trigger_word_override):
    log.info("Batch requested: %s (recursive=%s, overwrite=%s)", directory_str, recursive, overwrite)
    directory = Path(directory_str) if directory_str else None
    if not directory or not directory.is_dir():
        log.warning("Batch: not a directory: %s", directory_str)
        yield f"Not a directory: {directory_str}", "", None, ""
        return

    cfg = current_cfg
    base_url = _display_base_url()
    already_up = is_healthy(base_url)
    if cfg.server_mode == "external":
        yield "Processing..." if already_up else "Connecting to external server...", "", None, ""
    else:
        yield "Processing..." if already_up else "Starting server (loading model)...", "", None, ""

    try:
        client = get_client(cfg)
    except ServerError as exc:
        log.warning("Batch: server error: %s", exc)
        yield f"Server error: {exc}", "", None, ""
        return

    if not already_up:
        yield "Processing...", "", None, ""

    images = find_images(directory, recursive=recursive)
    if not images:
        log.warning("Batch: no images found in %s", directory)
        yield "No images found in that directory.", "", None, ""
        return

    q: "queue.Queue" = queue.Queue()
    result_holder = {}
    _batch_stop_event.clear()

    def progress_cb(i, total, path, status, caption):
        q.put((i, total, path, status, caption))

    def worker():
        try:
            result_holder["result"] = run_batch(
                directory, client, current_cfg,
                recursive=recursive,
                overwrite=overwrite,
                trigger_word=trigger_word_override or None,
                progress_cb=progress_cb,
                should_stop=_batch_stop_event.is_set,
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
        i, total, path, status, caption = item

        elapsed = time.monotonic() - start_time
        avg = elapsed / i if i else 0.0
        eta = avg * (total - i)
        status_text = f"{i}/{total} processed · avg {avg:.1f}s/image · ETA {_format_duration(eta)}"

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
        else:
            last_caption = "(failed - see console log)"
        yield status_text, last_file, last_image, last_caption

    thread.join()
    if "error" in result_holder:
        yield f"Unexpected error: {result_holder['error']}", last_file, last_image, last_caption
    else:
        r = result_holder["result"]
        total_elapsed = time.monotonic() - start_time
        verb = "Aborted" if r.aborted else "Done"
        summary = (
            f"{verb}: {r.processed} captioned, {r.skipped} skipped, {r.failed} failed "
            f"in {_format_duration(total_elapsed)}"
        )
        yield summary, last_file, last_image, last_caption


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
    cfg = current_cfg
    if cfg.server_mode == "external":
        return f"external ({base_url}, {'reachable' if healthy else 'unreachable'})"

    managed = _session.get("managed")
    connected = _session.get("client") is not None and _session.get("base_url") == base_url

    if connected and managed is not None:
        alive = managed.process.poll() is None
        return f"running, managed by us, pid {managed.process.pid}" if alive else "crashed (was managed by us)"
    if healthy:
        return "running (not started by us)"
    return "stopped"


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
        f"Model: {model}"
    )


# ---------------------------------------------------------------- Debug tab

def get_python_debug_text() -> str:
    return "\n".join(_PY_LOG_BUFFER) or "(no log output yet)"


def get_llama_debug_text() -> str:
    if _session.get("managed") is None:
        return "n/a"
    try:
        return LOG_PATH.read_text(encoding="utf-8", errors="replace") or "(empty)"
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
                        label="Trigger word (optional, overrides default)",
                        value=cfg.trigger_word,
                    )
                    single_caption = gr.Textbox(
                        label="Caption", lines=20, interactive=True
                    )
            with gr.Row():
                single_run_btn = gr.Button("Caption", variant="primary")
                single_save_btn = gr.Button("Save caption", interactive=False)
            single_status = gr.Textbox(show_label=False, container=False, interactive=False)

            single_run_btn.click(
                run_single_ui, [single_image, single_trigger], [single_caption, single_status]
            )
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
                    label="Trigger word (optional, overrides default)",
                    value=cfg.trigger_word,
                )
                with gr.Column():
                    batch_recursive = gr.Checkbox(label="Recursive", value=cfg.recursive_batch)
                    batch_overwrite = gr.Checkbox(
                        label="Overwrite existing captions", value=cfg.overwrite_existing
                    )

            with gr.Row():
                batch_run_btn = gr.Button("Run batch", variant="primary")
                batch_abort_btn = gr.Button("Abort")

            with gr.Row():
                batch_last_image = gr.Image(label="Preview", interactive=False, show_label=False)
                with gr.Column():
                    batch_last_file = gr.Textbox(label="Last file processed", interactive=False)
                    batch_last_caption = gr.Textbox(label="Last caption created", interactive=False)

            batch_status = gr.Textbox(show_label=False, container=False, interactive=False)

            batch_browse_btn.click(browse_directory_ui, [batch_dir], [batch_dir])
            batch_run_btn.click(
                run_batch_ui,
                [batch_dir, batch_recursive, batch_overwrite, batch_trigger],
                [batch_status, batch_last_file, batch_last_image, batch_last_caption],
            )
            batch_abort_btn.click(abort_batch_ui, [], [batch_status])

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
                lines=18, interactive=False, show_label=False,
                value=get_python_debug_text(),
            )
            gr.Markdown("**llama-server output** (n/a unless started by this app)")
            debug_llama_box = gr.Textbox(
                lines=18, interactive=False, show_label=False,
                value=get_llama_debug_text(),
            )
            debug_clear_btn = gr.Button("Clear")
            debug_clear_btn.click(clear_debug_ui, [], [debug_python_box, debug_llama_box])

        refresh_models_btn.click(
            refresh_models_ui, [model_name],
            [models_table, mmprojs_table, model_name, mmproj_name],
        )

        status_bar = gr.Markdown(get_status_text(), elem_id="status-bar")
        status_timer = gr.Timer(2.0)
        status_timer.tick(get_status_text, [], [status_bar])
        if cfg.debug_tab_enabled:
            status_timer.tick(get_python_debug_text, [], [debug_python_box])
            status_timer.tick(get_llama_debug_text, [], [debug_llama_box])
        demo.load(get_status_text, [], [status_bar])

    return demo


UI_PORT = 7901


def main() -> None:
    demo = build_app()
    demo.queue()
    demo.launch(server_port=UI_PORT, footer_links=[], css=ALL_CSS)


if __name__ == "__main__":
    main()
