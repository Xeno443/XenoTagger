"""Model backends - one self-contained package per model type, each a
core.backend.Backend subclass instance. BACKENDS is the registry; its
order is the Settings sub-tab order.

Import rules - keep the graph acyclic and the CLI Gradio-free:
  - A backend imports core/ freely, and its own package's modules.
  - A backend never imports another backend's modules. Cross-backend
    behavior (e.g. Hydra's load stopping/restarting llama-server) goes
    through core.lifecycle and this registry instead.
  - Only <backend>/settings_ui.py may import Gradio or ui/ (the shared
    modules - state, notify, operation, context, dialogs, reachability,
    downloads_queue, model_status - never a tab_*). Nothing else imports settings_ui;
    the GUI loads it by Backend.settings_ui_module.
  - core/ never imports backends/ - except core.captioner/core.batch,
    which still call llama and Hydra directly until the v0.9 step-3
    caption-pipeline rewrite moves them behind the Backend interface.
  - Loggers use __name__ ("backends.<id>.<module>"); the GUI and CLI
    raise "backends" to DEBUG alongside "core", never the root logger.
"""

from __future__ import annotations

from core.backend import Backend

from .hydra import BACKEND as HYDRA
from .llama import BACKEND as LLAMA

BACKENDS: list[Backend] = [LLAMA, HYDRA]


def get(backend_id: str) -> Backend:
    return next(b for b in BACKENDS if b.id == backend_id)
