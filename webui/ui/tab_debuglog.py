"""Debuglog tab (opt-in via Settings -> Debug) and the Python log capture
behind it."""

from __future__ import annotations

import logging
from collections import deque

import gradio as gr

from backends.llama.server import LOG_PATH
from backends.llama.session import has_managed_process
from core import config as config_mod
from ui import state
from ui.context import BuildContext

log = logging.getLogger("app")  # fixed name, see app.py

# Debug tab: captures log output from our own Python code (core.*/backends.* modules
# all log via the standard `logging` module, but nothing was ever attached
# to see it before this) into a bounded in-memory buffer, AND a plain text
# file - the buffer is what the Debug tab's textbox polls, the file is for
# anything that can't/didn't have the tab open when something happened
# (including, deliberately, a coding assistant working on this repo, which
# has no way to peek into this process's own live memory - unlike the
# textbox, a file is something that can just be read after the fact).
_PY_LOG_BUFFER: deque[str] = deque(maxlen=2000)
PY_LOG_PATH = config_mod.WEBUI_DIR / "logs" / "app.log"


class _BufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _PY_LOG_BUFFER.append(self.format(record))


def setup_debug_logging() -> None:
    """Attaches the buffer + file handlers below, if debug_tab_enabled -
    called once from app.py's main(), deliberately NOT at module import time. This
    module gets imported directly by ad-hoc test scripts too (this
    project's normal testing approach this session, e.g. `import app`
    from a scratch script to exercise a handler function without a live
    server) - attaching a mode="w" FileHandler at import time meant every
    such import truncated PY_LOG_PATH out from under whatever the
    actually-running app process had already written to it. Scoping
    this to the real entrypoint keeps a plain `import app` side-effect-
    free on disk."""
    if not state.current_cfg.debug_tab_enabled:
        return
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    buffer_handler = _BufferLogHandler()
    buffer_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(buffer_handler)

    # Truncated fresh on every app start, matching backends.llama.server's own
    # LOG_PATH (llama-server.log) convention - this is meant as a live
    # window into the CURRENT run, not an accumulating history across
    # restarts (which, this session, happen often during iteration).
    PY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(PY_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)

    # Deliberately NOT touching the root logger's level. Elevating only our
    # own namespaces to DEBUG (rather than the root, which every library's
    # logger inherits from when it has no level of its own) means Gradio's
    # httpx/httpcore/asyncio internals - and anything else - stay at their
    # own quiet default instead of flooding this with unrelated chatter.
    # "backends" covers every backend module's __name__ logger
    # (backends.llama.server etc.).
    logging.getLogger("core").setLevel(logging.DEBUG)
    logging.getLogger("backends").setLevel(logging.DEBUG)
    logging.getLogger("app").setLevel(logging.DEBUG)


def get_python_debug_text() -> str:
    # Newest first - the boxes don't reliably autoscroll on periodic
    # Timer.tick updates (Gradio's autoscroll is gated on streaming/
    # generating state, not plain polled value replacement), so the latest
    # line needs to be visible without scrolling.
    return "\n".join(reversed(_PY_LOG_BUFFER)) or "(no log output yet)"


def get_llama_debug_text() -> str:
    if not has_managed_process():
        return "n/a"
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        return "\n".join(reversed(text.splitlines())) if text else "(empty)"
    except OSError as exc:
        return f"(could not read {LOG_PATH}: {exc})"


def clear_debug_ui() -> tuple[str, str]:
    _PY_LOG_BUFFER.clear()
    try:
        PY_LOG_PATH.write_text("", encoding="utf-8")
    except OSError:
        pass
    if has_managed_process():
        try:
            LOG_PATH.write_text("", encoding="utf-8")
        except OSError:
            pass
    return get_python_debug_text(), get_llama_debug_text()


def build(ctx: BuildContext):
    cfg = state.current_cfg
    with gr.Tab("Debuglog", visible=cfg.debug_tab_enabled):
        gr.Markdown(f"**Python debug log** (also written to `{PY_LOG_PATH}`)")
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

    if cfg.debug_tab_enabled:
        ctx.on_tick(get_python_debug_text, [], [debug_python_box])
        ctx.on_tick(get_llama_debug_text, [], [debug_llama_box])
