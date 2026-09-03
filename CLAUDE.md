# XenoTagger

Gradio + CLI Python app that captions images for LoRA dataset creation,
using a local (or remote) llama.cpp multimodal vision server. Point it at
a folder of images, it writes a `.txt` caption sidecar per image.

Standalone single-branch repo (`https://github.com/Xeno443/XenoTagger.git`).
Ignore stray references elsewhere to worktrees or `portable-env-<branch>`
- not this repo's structure.

## Layout

- `webui/app.py` - Gradio GUI (Single-image, Batch, Review, Models,
  Settings, opt-in Debuglog). Module docstring covers the concurrency
  model (`_session_lock`/`_operation_lock`/`_config_lock`) and Gradio
  layout workarounds - read before touching button/tab wiring.
- `webui/cli.py` - headless batch captioning. Shares `core/batch.py`/
  `core/captioner.py` with the GUI, same `settings.json`. No UI code in
  `core/`.
- `webui/core/` - framework-agnostic shared logic:
  - `config.py` - `AppConfig` dataclass, `load()`/`save()`.
  - `server.py` - llama-server lifecycle (`managed` spawn/own vs.
    `external` URL-only), `check_status()`.
  - `client.py` - llama-server chat API calls, image preprocessing.
  - `captioner.py` - shared "caption one image" entry point (Single,
    Batch, CLI). `caption_image()` is the Hydra hook point.
  - `batch.py` - directory batch loop. Stateless - `.txt` existing is
    the only "done" record.
  - `models.py` - discovers/classifies local GGUF models.
  - `downloads.py` - background curated-model downloader.
  - `llama_install.py` - installs llama-server.exe from ggml-org/llama.cpp
    releases into `llama/` (gitignored). CUDA 13.3 (default)/12.4, ROCm
    7.14, CPU - resolved against GitHub's current release listing, not a
    hardcoded build. Settings → Llama, or `cli.py --install-llama`.
  - `hydra_classifier.py` - in-process Hydra 3.5 (RedRocket) e621 tag
    classifier. Lazy singleton (`load()`/`unload()`/`classify()`/
    `status()`), not a managed server.
  - `hydra_install.py` - installs Hydra's deps + `hydra-3.5.safetensors`.
    Settings → Hydra, or `cli.py --install-hydra`.
- `webui/vendor/rr_hydra/` - vendored inference-only source from
  RedRocket's `hydra` package (renamed to avoid colliding with PyPI's
  `hydra-core`). Imports are absolute (`from vendor.rr_hydra...`), never
  relative - `core/` is a top-level package.
- `webui/config/` - `settings.json` (gitignored), `models_source.json`
  (checked in), `models_cache.json` (gitignored).
- `readme/` - deep-dive/FAQ docs. New ones go here, not repo root.
  - `faq-hydra-implications.md` - how `remove`/`constrain-remove`/
    `enforce-remove` resolve nested e621 tag families.
  - `faq-hydra-setting.md` - what the Confidence/Threshold sliders
    (`hydra_metric`) do.
  - `faq-caption.md` - sidecar-adoption case table (see below).
  - `faq-linux-considerations.md` - what's actually Windows-specific.
    Managed server only finds `llama-server.exe`; installer only fetches
    Windows assets. External server mode has no platform dependency.
- `run.cmd` / `cli.cmd` - launch GUI / CLI, auto-detect portable env vs.
  `.venv`.
- `setup-portable.bat` / `environment.bat` - build/activate portable
  Python+Git under `system\git\`/`system\python\` (gitignored; `system\`
  itself is tracked, holds `fix-wrappers.py`), install requirements.
- `update.bat` - `git pull` (or reset+pull), refresh requirements. No
  `.git`? Adopts the folder as a checkout (`git init` + remote + fetch +
  `reset --hard`) instead of requiring an empty dir.
- `setup-venv.bat` - `.venv\` alternative for systemwide git/Python.
  Doesn't touch `system\`. No version pins in requirements.txt, so
  `hydra_install.py` targets `sys.executable`, not a hardcoded path.

## VRAM / hardware constraint

- Target: 8-16GB consumer GPUs, often with several GB already used by
  desktop/browser/editor.
- `n_gpu_layers="auto"` (partial CPU offload) is the default, not a
  fallback - must work down to 8GB.
- Dev machine: RTX 5080, 16GB, ~9.5-10.5GB actually free. One data
  point, not the floor.
- huihui-ai's abliteration lineage is the reliable refusal-free choice.
  Other uploaders' "abliterated" tags have been confirmed to still
  refuse - check lineage back to huihui-ai.

## Server lifecycle

- Modes: `"managed"` (spawn/own) / `"external"` (URL only). Managed
  binds `127.0.0.1`, port **8901**.
- No implicit lazy-start from the UI - explicit Start button or
  once-at-launch autostart checkbox only. CLI's `get_client()`/
  `resolve_server()` still lazy-start (no button to click headless).
- "End llama server" gated on session ownership
  (`_is_server_managed_by_us`) - never kills a server we didn't start.
- Status: N/A / idle / error / running (managed) / unreachable /
  connected (external). idle = never started; error = was running, then
  stopped unexpectedly (edge-detect vs. previous poll). A deliberate
  "End" click is never labeled error (`_expected_stop`).
- Settings sub-tabs: Llama/Hydra/Image resizing/Captioning defaults/
  Debug. The four non-Debug sub-tabs get `interactive=` disabled during
  a caption job (`_settings_gating_ui`) - **tab-level only, not
  per-field**. Entering Settings while disabled redirects to Debug,
  restores the real sub-tab once re-enabled
  (`current_settings_subtab_state`) - known-strange edge case, see
  Open/deferred.
- Save Settings: prompt-template applies immediately; switch to external
  kills an owned managed server; switch to managed never auto-starts;
  server-arg changes kill (if owned) + optionally restart. Guarded by
  `_config_lock`.

## Hydra 3.5 second-stage tag classifier

Appends Hydra's e621 tags after the VLM caption on their own line
(`"<trigger>, <caption>\ntag1, tag2, ..."`) - grounds NSFW/explicit
tagging the VLM is weak at.

- In-process (pure Python/torch), not a managed server.
- Load/Unload is explicit (button + independent autoload-on-launch
  checkbox), never implicit - `n_gpu_layers="auto"` can claim most/all
  VRAM first, so loading Hydra after risks OOM. `hydra_enabled` only
  gates *use* of an already-loaded model, never triggers a load.
- Failures degrade gracefully everywhere (`captioner.py`, `cli.py`) -
  catches `HydraError`, keeps VLM-only caption, never fails the job.
- Heavy deps on demand, not in base requirements.txt (mirrors
  `llama_install.py`). Weight (~1GB) in
  `webui/models/RedRocket-Hydra/`, ignored by the GGUF scanner.
- `hydra_implications` defaults to `"remove"` (collapse to most specific
  tag per family), not `"inherit"` (whole-chain propagation ~doubled tag
  counts in testing). `constrain-remove`/`enforce-remove` rejected -
  see `readme/faq-hydra-implications.md`.
- `hydra_metric` defaults to `"f0.5@0.1"`, not `"f1.0@0.1"` - best
  tag-count/contradiction tradeoff in testing. Exposed as two sliders
  (Confidence `f<beta>`, Threshold `@<min_precision>`) via a direct
  linear mapping (`_hydra_metric_to_sliders`/`_hydra_sliders_to_metric`
  in `app.py`), not upstream's non-linear curve. `csi<weight>@<precision>`
  still works hand-edited into settings.json, just not in the UI.
- `exclusive-groups`/`aliases`/per-category-prefix rewriting are
  permanently out of scope - don't re-propose without the user raising
  it. `hydra_exclude_categories` defaults to
  `"artist copyright meta rating lore"`.

### Hydra/llama VRAM coexistence

llama's `"auto"` adapts to reduced VRAM; Hydra's load doesn't -
whichever claims VRAM first wins.

- `_load_hydra_with_llama_coexistence()` (`app.py`) - shared by manual
  Load and startup autoload. If Hydra is on CUDA and a managed server we
  own is running: stop it, load Hydra, restart the server so `"auto"`
  resizes around it. Skipped if Hydra is on CPU or a caption job is
  active.
- Startup order: Hydra autoload runs *before* llama autostart, so Hydra
  gets first pick of clean VRAM.

## Status/notification architecture

- **status bar** - single global `status_bar` (server health, loaded
  model, active op), polled every 2s, shown on every tab.
- **infotext bar** - per-tab textboxes, written only by that tab's own
  handlers. Kept separate so live-progress writers don't stomp one-off
  action feedback.
- `_notify(text, level)` - the one place a popup toast fires. `"error"`
  maps to a `gr.Warning`-styled toast, not `gr.Error` (see gotcha
  below). Log-only messages call `log.<level>()` directly.
- Background work (batch, downloads, installs) fires a one-shot
  `_notify()` on completion/failure, not just a silent infotext update -
  so finishing doesn't require staying on the tab. Downloads/installs run
  off-thread, so they queue an announcement the next 2s poll relays.
- Captioning shows live per-image stage in its tab's infotext bar (not
  status bar). `caption_image()` takes `on_stage(text)`
  (`"captioning"`/`"tagging (Hydra)"`), threaded from `run_batch()`.
  Single-image/Review got the same background-thread + `queue.Queue`
  treatment Batch already had, so there's something to yield from
  mid-call.

## Batch sidecar adoption

`run_batch()` adopts a pre-existing `.txt.nlp`/`.txt.tags` sidecar
instead of clobbering it - only the missing half gets a real model call.
Full table in `readme/faq-caption.md`:

- `.txt.tags` only → VLM runs, Hydra skipped (tags trusted as-is).
- `.txt.nlp` only → VLM skipped (caption reused verbatim), Hydra runs if
  enabled.
- Both exist → zero model calls, `.txt` synthesized directly.
- `overwrite` checked → bypasses all of this, same as pre-existing `.txt`.

`caption_image()`'s `existing_caption` param skips the VLM call and
reuses that text, still falling through to Hydra.

## Wrapper-path self-heal

A relocated portable install can leave `system\python\Scripts\*.exe`
wrappers pointing at a dead `python.exe` path.

- `system\fix-wrappers.py` (tracked) rewrites any broken non-bare
  shebang to bare `python.exe` (not the current absolute path - would
  need re-fixing again next move). Bare/still-resolving shebangs left
  alone. Runs silently on every launch via `environment.bat`.
- WinPython's bundled pip already writes bare shebangs (patched
  `distlib/scripts.py`) - but only until PyPI publishes a newer pip than
  WinPython's patched build, at which point `pip install --upgrade pip`
  silently pulls the unpatched version.
- `system\fix-pip-shebang.py` (standalone/manual, not wired in - see
  Open/deferred) restores that one-line patch if it ever regresses.
  Parses with `ast`, only touches the exact recognized assignment,
  refuses on anything unrecognized.
- `.venv` (`setup-venv.bat`) isn't covered by either script.

## House rules

- **No venv inside the portable-env path, ever.** Everything installs
  into `system\python`. Never call `system\python\Scripts\*.exe`
  directly - always `system\python\python.exe -m pip ...` /
  `python.exe script.py`. `setup-venv.bat`'s `.venv\` is a separate,
  explicit alternative - fine on its own, just don't nest one inside
  the portable-env workflow.
- **No custom CSS unless a built-in Gradio option was checked and ruled
  out first.** `webui/ui_css.py` has the only two rules, each commented
  with what was ruled out.
- **Never elevate the root logger to DEBUG** - only
  `logging.getLogger("core")` and `"app"` (fixed name).
- Review `git status`/`git diff` before committing.

## Gradio gotchas (don't re-discover these)

- `gr.SelectData.value` for a `Tab` is always its label, never `id`.
  Every `Tab` targeted by `gr.update(selected=...)` needs an explicit
  `id=`.
- A bare `gr.update()` doesn't self-correct a prior `Tabs.selected=`
  push - always assert the destination explicitly, both directions.
- `.select()` fires only on genuine user click, not a server push -
  except a nested `Tabs` inside a `Tab` being switched into has, at
  least once, fired it anyway (not root-caused).
- Leaving and returning to a top-level tab remounts nested `Column`s to
  build-time `visible=`, not latest server state.
- `gr.Group()` gives ugly stretched/gapped layouts with mismatched-height
  siblings - only use for uniform-height stacked rows.
- `gr.Image` output won't serve a raw file path outside CWD/temp - load
  via PIL into memory instead.
- `os.execv` restart on Windows spawns an untracked PID - look up
  whatever's bound to port 7901, don't trust a known task ID.
- `tkinter` dialogs only work when browser and Python are the same
  machine.
- `gr.Error` only displays if actually `raise`d, and raising aborts the
  function. `_notify()` maps `"error"` to `gr.Warning`-styled, not
  `gr.Error` - most callers are 2s-poll handlers that can't afford an
  abort.

## Open / deferred

- `.venv` relocation - not covered by self-heal. Needs its own checks
  (`pyvenv.cfg`'s base-interpreter path, etc.) before deciding
  patch-in-place vs. recreate. Not started.
- `fix-pip-shebang.py` not wired into `environment.bat` - deferred
  pending user check-in.
- Broad UI-logic refactor, not scoped. Much of the app predates the
  status-bar/infotext-bar/`_notify` consolidation. User: "no more lazy
  start and ui elements enabled/disabled redesign" - don't let old
  implicit-lazy-start patterns creep back in; the `interactive=True/False`
  pattern itself is a redesign candidate.
- Settings sub-tab redirect has a known "strange effect," not diagnosed
  - nested-`Tabs`-fires-`.select()` gotcha is the leading suspect.
- Llama tab's download progress bar vs. Hydra's - visual match deferred
  by user, don't start without it being raised again.
- Smaller: "unsaved Settings changes" indicator; brief tab-disabled
  flash before redirect on fresh load; possible rare race on overlapping
  Start/End llama clicks.
- No browser access here - UI behavior can only be verified by scripted
  simulation (`py_compile` + `build_app()` + mocked handlers). A live
  pass by the user is still the real test.
