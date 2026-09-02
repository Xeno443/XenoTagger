# Linux compatibility FAQ

Notes from an audit of `webui/app.py`, `webui/cli.py`, and `webui/core/*.py`
(done 2026-09-02, prompted by writing the README's system requirements),
answering a specific question: if someone on Linux installed Python
themselves, ran `pip install -r webui/requirements.txt`, and launched
`python webui/app.py` / `python webui/cli.py` directly - bypassing the
`.bat`/`.cmd` setup and launcher scripts entirely, which are obviously
Windows-only convenience wrappers and don't count for this question -
would anything in the actual Python code fail or misbehave on Linux?

## Short answer

Nothing in the core app hard-fails on Linux. No `platform.system()`/
`sys.platform`/`os.name` checks gate anything anywhere in `app.py`,
`cli.py`, or `core/`. Process management (`terminate()`/`kill()`/`wait()`),
path handling (`pathlib` throughout, no hardcoded separators), and even
the app-restart mechanism (`os.execv`) are all POSIX-native to begin with
- Windows is the platform emulating those, not the other way around.

The two real gaps are both scoped to XenoTagger's **managed** llama-server
features specifically - **external** mode (pointing XenoTagger at a
llama-server you build and run yourself) is unaffected by either.

## Gap 1: managed server only looks for `llama-server.exe`

`core/server.py:56`:

```python
LLAMA_SERVER_EXE = ROOT_DIR / "llama" / "llama-server.exe"
```

`_start_process()` (`core/server.py:152-177`) checks `LLAMA_SERVER_EXE.exists()`
and raises if it's missing (`core/server.py:153-155`), then launches exactly
that path. A Linux llama.cpp build is normally just named `llama-server`
(no extension) - it won't be found under this hardcoded name. This
doesn't crash Python; `Path.exists()` and `subprocess.Popen` don't care
about extensions on Linux, it's simply that the managed start path is
looking for a file that won't exist by that name unless someone renames
their Linux binary to literally end in `.exe`.

Everything else in this file is clean: `ManagedServer.stop()`
(`core/server.py:138-149`) uses plain `terminate()` then `kill()`, and
health/status detection is plain HTTP (`requests.get(.../health)`) - both
fully cross-platform.

## Gap 2: the in-app llama.cpp installer only fetches Windows release assets

`core/llama_install.py:60-61`:

```python
asset_stub: str  # matches "llama-<tag>-bin-win-<asset_stub>-x64.zip"
cudart_stub: Optional[str] = None  # matches "cudart-llama-bin-win-<cudart_stub>-x64.zip"
```

...and the names actually built from those at `core/llama_install.py:120`
and `:131`:

```python
main_name = f"llama-{build_tag}-bin-win-{backend.asset_stub}-x64.zip"
cudart_name = f"cudart-llama-bin-win-{backend.cudart_stub}-x64.zip"
```

ggml-org/llama.cpp's GitHub releases also publish Ubuntu/macOS build
assets, but `plan_install()`/`install()` never look for them - only
`bin-win-*.zip` patterns. On Linux this doesn't crash either; it just
raises a clean `InstallError` ("Expected asset ... not found") since the
filename never matches anything in the release. The in-app installer
(Settings → Llama, and `cli.py --install-llama`) simply can't produce a
working Linux install today - extending `BackendDef`/`plan_install()` to
also match Linux asset names would be the fix, not yet done.

## Everything else checked and found clean

- **`core/hydra_install.py`** - uses `sys.executable` (not a hardcoded
  interpreter path), plain `subprocess.Popen`, `pathlib` throughout. Its
  comments mention `system\python` but that's prose describing the
  portable-env convention, not logic that branches on it.
- **`core/config.py`** - fully `pathlib`-based, no hardcoded separators,
  no platform checks.
- **`core/hydra_classifier.py`** - no platform-specific code at all.
- **`webui/app.py`'s `os.execv` restart** (`app.py:1216`, see the
  docstring at `app.py:1195`) - POSIX-standard, not a Windows trick
  despite the "standard Python self-restart trick" phrasing; works
  identically on Linux.
- **`webui/app.py`'s three `tkinter.filedialog` call sites**
  (`app.py:1385-1440`, save/browse-directory/browse-file) - `tkinter`
  ships cross-platform and works on Linux given `python3-tk` plus an
  X/Wayland display. The existing "only works when the browser and
  Python process are on the same machine" limitation (see CLAUDE.md's
  Gradio gotchas) applies identically on both platforms - a headless
  Linux server with no `DISPLAY` would fail the same way a remote Windows
  install already does, it's not a new Linux-specific problem.
- **`webui/cli.py`** - the only Windows-flavored text is in `--help`
  strings/docstrings (e.g. `system\python\python.exe`, `llama-server.exe`
  as example paths) - cosmetic only, no logic branches on it.

## Practical takeaway for a Linux user today

- Build or obtain your own `llama-server` binary, run it yourself, and
  point XenoTagger at it in **external** server mode. This path is fully
  supported - nothing in `core/client.py`, `core/captioner.py`, or the
  request/response handling has any platform dependency at all.
- The **managed** server mode and the in-app llama.cpp installer are
  Windows-only until gaps 1 and 2 above are addressed - not a fundamental
  limitation, just work not yet done.
- Hydra (`core/hydra_classifier.py`, `core/hydra_install.py`) works
  identically either way - it's pure Python + torch, no process
  management involved.
