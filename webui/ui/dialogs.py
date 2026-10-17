"""tkinter file/folder pickers shared by several tabs. Only work when the
browser and Python run on the same machine."""

from __future__ import annotations

import logging
from pathlib import Path

from core import config as config_mod

log = logging.getLogger("app")  # fixed name, see app.py


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


def browse_tag_vocab_file_ui(current_value: str) -> str:
    """Same shape as browse_directory_ui, but for picking a single tag
    vocabulary CSV file - defaults into webui/tags/ (where the curated
    e621.csv already ships) rather than the current file's own directory,
    unless current_value already points at a real file."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    current_path = Path(current_value) if current_value else None
    initial = str(current_path.parent) if current_path and current_path.is_file() else str(config_mod.WEBUI_DIR / "tags")
    selected = filedialog.askopenfilename(
        initialdir=initial, filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    if selected:
        log.debug("browse_tag_vocab_file_ui(): user picked %s", selected)
    return selected or current_value
