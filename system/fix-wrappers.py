"""Fix distlib-style console-script .exe wrappers (Scripts\\*.exe) whose
baked-in python.exe path no longer exists, e.g. after the environment
they were installed into got moved to a new location.

A wrapper's shebang can be either an absolute path (the old-style,
non-relocatable form some pip builds still write) or a bare "python.exe"
that resolves via PATH (what a recent-enough pip writes, and what this
project's own environment.bat already puts on PATH before anything
runs). A bare shebang is already location-independent and is left
untouched. An absolute path that still resolves is also left untouched
(nothing moved). Only an absolute path that no longer points at a real
file gets rewritten - to bare "python.exe", the form that survives any
future move too, not just this one.

Usage: fix-wrappers.py <Scripts-dir>
"""
import sys
from pathlib import Path

ZIP_MAGIC = b"PK\x03\x04"
SHEBANG_PREFIX = b"#!"
FIXED_SHEBANG = SHEBANG_PREFIX + b"python.exe"


def patch_one(exe_path: Path) -> str:
    data = exe_path.read_bytes()
    zip_start = data.find(ZIP_MAGIC)
    if zip_start == -1:
        return "skip (no embedded zip found, not a distlib-style launcher)"

    shebang_start = data.rfind(SHEBANG_PREFIX, 0, zip_start)
    if shebang_start == -1:
        return "skip (no shebang line found before zip data)"

    newline = data.find(b"\n", shebang_start, zip_start)
    if newline == -1:
        return "skip (shebang line not newline-terminated)"

    old_shebang = data[shebang_start + len(SHEBANG_PREFIX):newline]
    old_path = old_shebang.decode("utf-8", "replace")

    if "\\" not in old_path and "/" not in old_path:
        return f"skip (relative reference '{old_path}', resolved via PATH)"

    if Path(old_path).is_file():
        return "skip (already points at a real file)"

    patched = data[:shebang_start] + FIXED_SHEBANG + data[newline:]
    exe_path.write_bytes(patched)
    return f"fixed (was broken): {old_path} -> python.exe"


def main():
    scripts_dir = Path(sys.argv[1])

    if not scripts_dir.is_dir():
        return

    for exe_path in sorted(scripts_dir.glob("*.exe")):
        result = patch_one(exe_path)
        print(f"{exe_path.name}: {result}")


if __name__ == "__main__":
    main()
