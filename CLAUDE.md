# XenoTagger

A Gradio + CLI Python app that captions images for LoRA dataset creation,
using a local (or remote) llama.cpp multimodal vision server. Point it at
a folder of images, it writes a `.txt` caption sidecar next to each one.

This repo used to be the `portable-env` common-base toolchain (worktrees,
`main`/`webui`/etc. branches). That model is gone — XenoTagger is now a
**standalone single-branch repo**
(`https://github.com/Xeno443/XenoTagger.git`). Ignore any leftover
references elsewhere to worktrees or a `portable-env-<branch>` layout;
they describe a repo structure this one no longer has.

## Layout

- `webui/app.py` — the Gradio GUI. Single-image, Batch processing,
  Review (edit/recaption existing captions), Models (local + curated
  downloads), Settings (Llama/Hydra/Image resizing/Captioning defaults/
  Debug sub-tabs), an opt-in Debuglog tab. Its own module docstring
  documents the concurrency model (`_session_lock`/`_operation_lock`/
  `_config_lock`) and several Gradio layout workarounds in detail — read
  that before changing button/tab wiring, don't re-derive it.
- `webui/cli.py` — headless batch captioning, independent entry point.
  Shares `core/batch.py`/`core/captioner.py` with the GUI; reads/writes
  the same `webui/config/settings.json`. No UI-specific code ever
  belongs in `core/`.
- `webui/core/` — framework-agnostic logic shared by both entry points:
  - `config.py` — `AppConfig` dataclass, the single source of truth for
    every setting; `load()`/`save()` for `settings.json`.
  - `server.py` — llama-server process lifecycle (`managed` mode: we
    spawn/own it; `external` mode: we just talk to a URL) and
    `check_status()` for cheap read-only UI gating.
  - `client.py` — the one place that actually calls llama-server's
    chat API, plus in-memory image preprocessing.
  - `captioner.py` — the shared "caption one image" entry point used by
    Single-image, Batch, and the CLI alike. `caption_image()` is where
    the Hydra second-stage tag classifier hooks in (see below) —
    exactly the single hook point its own docstring anticipated.
  - `batch.py` — directory batch-captioning loop (stateless: a `.txt`
    sidecar existing is the only "already done" record).
  - `models.py` — discovers/classifies local GGUF models under
    `webui/models/`.
  - `downloads.py` — background curated-model downloader.
  - `llama_install.py` — installs llama-server.exe (+ matching CUDA
    runtime where applicable) from ggml-org/llama.cpp's GitHub releases
    into `llama/` (gitignored, backend-agnostic — one canonical
    install, not one dir per backend). Backend choices: CUDA 13.3
    (default)/12.4, ROCm 7.14, CPU — resolved against whichever build
    GitHub's releases list currently returns first, not a hardcoded
    build number. Reachable from Settings → Llama in the GUI and from
    `cli.py --install-llama <backend>`.
  - `hydra_classifier.py` — in-process Hydra 3.5 (RedRocket) e621 tag
    classifier: a lazily-populated singleton (`load()`/`unload()`/
    `classify()`/`status()`), not a second managed server — see
    "Hydra 3.5 second-stage tag classifier" below.
  - `hydra_install.py` — pip-installs Hydra's Python deps (torch +
    friends, into `system\python` like everything else) and downloads
    `hydra-3.5.safetensors` into `webui/models/RedRocket-Hydra/`.
    Reachable from Settings → Hydra and from `cli.py --install-hydra`.
- `webui/vendor/rr_hydra/` — vendored (checked-in) inference-only source
  from RedRocket's own `hydra/hydra/` package, renamed from upstream's
  `hydra` to avoid colliding with PyPI's unrelated `hydra-core`. Only
  what a synchronous single-image `classify()` call needs — not
  upstream's GUI/HTTP-service/multi-worker-dataloader code, none of
  which this app uses.
- `webui/config/` — `settings.json` (gitignored, per-machine),
  `models_source.json` (checked in — curated downloadable model list),
  `models_cache.json` (gitignored).
- `run-tagger.cmd` / `tag-cli.cmd` — launch the GUI / CLI through the
  portable environment.
- `setup-env.bat` / `environment.bat` — inherited from the portable-env
  base: build/activate the portable Python + Git toolchain under
  `system\` (gitignored), then install `webui/requirements.txt`. Still
  accurate, unrelated to the worktree model that was dropped.
  `setup-tagger.cmd` (which used to also install a hardcoded CUDA-only
  llama.cpp build) is gone — that job moved to `core/llama_install.py`
  above.

## VRAM / hardware constraint

Consumer GPUs this needs to work on commonly have 8–16GB VRAM, often
with several GB already spoken for by the desktop/browser/editor before
the app even starts — actual free VRAM at caption time can end up
smaller than a single model quant under `webui/models/`. Don't assume a
high-VRAM card: `n_gpu_layers="auto"` (partial CPU offload) is relied on
as the default behavior, not an optional fallback for edge cases, and
that needs to keep working down to 12GB and 8GB cards, not just
whatever the developer happens to have. Development/testing so far has
mainly been on an RTX 5080 (16GB, typically ~9.5–10.5GB actually free)
— treat that as one data point, not the floor.

huihui-ai's abliteration lineage has been the reliable choice for
refusal-free vision models; other uploaders' "abliterated" tags have
been confirmed to still refuse on the same content — check a model
card's lineage back to huihui-ai before trusting the tag alone.

## Major design decisions (llama-server lifecycle rewrite, 2026-08-28/30)

The server-management UX went through a full rewrite this session,
replacing an earlier implicit-lazy-start design:

- Server modes are `"managed"` / `"external"` everywhere (renamed from
  `"auto"`/`"remote"`; `config.load()` migrates an old `settings.json`
  automatically). Managed always binds `127.0.0.1` (`server.MANAGED_HOST`)
  — no more user-configurable host for it. Default managed port is
  **8901** (was 8080).
- No more implicit lazy-start from the UI: starting the managed server
  is always an explicit action — a "Start llama server" button in
  Settings→Llama, or an "autostart on launch" checkbox that only fires
  once at app startup, never mid-session. (`get_client()`/
  `resolve_server()` still lazy-start for the CLI — headless has no
  "disabled button" concept and still needs that.)
- "End llama server" is gated on **session ownership**
  (`_is_server_managed_by_us` / `_session` dict) — this app only ever
  offers to kill a process it itself started, never someone else's
  server found already running on the same port.
- Status is a 5-state label: N/A, idle, error, running (managed) /
  unreachable, connected (external) — distinguishing "fresh, never
  started" (idle) from "was running, then unexpectedly stopped" (error,
  with a crash warning) via an edge-detect on the previous poll. A
  deliberate "End llama server" click is explicitly NOT labeled "error"
  (`_expected_stop` flag, set inside `_stop_managed()` itself).
- Settings is restructured into Llama/Hydra/Image resizing/Captioning
  defaults/Debug sub-tabs. The four operation-sensitive sub-tabs (not
  Debug) get their own `interactive=` disabled while a caption job is
  running (`_settings_gating_ui`) — **deliberately tab-level only, not
  the fields inside them** (a field-level version was drafted and then
  reverted — see git history around 2026-08-30 — because it wasn't what
  was wanted). Separately, entering the Settings tab while those
  sub-tabs are disabled redirects to Debug and remembers which sub-tab
  you were genuinely on, restoring it once things re-enable
  (`_settings_subtab_entry_ui`/`_on_settings_subtab_select`,
  `current_settings_subtab_state`) — see the file for a known-strange,
  not-fully-diagnosed edge case still open below.
- Save Settings behavior differs per field category: the prompt-template
  bundle applies immediately; switching to external kills an owned
  managed server; switching to managed never auto-starts; server-process
  args changed while managed kill (if owned) and optionally restart per
  the autostart checkbox. A `_config_lock` guards the read-modify-write
  of `current_cfg` + `config_mod.save()`.
- Installing llama.cpp itself now has an in-UI installer (Settings →
  Llama, plus `cli.py --install-llama`) — see `core/llama_install.py`
  under Layout above. This was out of scope for the lifecycle-rewrite
  pass described in this section and was added afterward.

## Hydra 3.5 second-stage tag classifier (integrated 2026-08-31)

A second-stage e621 tag classifier (`webui/vendor/rr_hydra/`,
`core/hydra_classifier.py`/`core/hydra_install.py`, Settings → Hydra)
appends Hydra's own tags after the VLM's caption
(`"<trigger>, <VLM paragraph>, tag1, tag2, ..."`) to ground NSFW/explicit
tagging the VLM alone is weak at. Prototyped separately first in a
standalone `HydraTagger` repo; this is the real integration.

- **In-process, not a second managed server.** Unlike llama-server,
  Hydra is pure Python + torch running in the same interpreter as
  webui/cli — a plain lazily-populated singleton
  (`hydra_classifier.load()`/`unload()`/`classify()`/`status()`), not a
  second `ManagedServer`/port/health-check apparatus.
- **Loading is an explicit "Load Hydra model" / "Unload Hydra model"
  action in Settings → Hydra (+ an independent autoload-on-launch
  checkbox), never implicit.** `n_gpu_layers="auto"` means llama-server
  can claim most/all free VRAM at startup — loading Hydra's model
  *after* llama-server is already running could OOM or spill into slow
  shared/system memory on cards near this project's 8GB floor. This
  needed to be an observable, deliberate action rather than something
  buried inside a caption call — same reasoning `core/server.py`
  already established for llama-server itself.
  `AppConfig.hydra_enabled` only gates *using* an already-loaded model
  in `captioner.py`; it never triggers a load. When both
  `autostart_managed_llama` and `hydra_autoload_model` are enabled, the
  demo.load chain deliberately runs Hydra's autoload *after*
  llama-server's, so the real-world VRAM-contention scenario is what
  actually gets exercised at every launch.
- **Hydra failures degrade gracefully** — `captioner.py` catches
  `HydraError` (deps/model missing, not loaded, OOM, whatever) and just
  keeps the VLM-only caption; it never fails a whole caption/batch.
- **Heavy deps on demand, not in base `requirements.txt`** — mirrors
  `llama_install.py`'s own precedent. The model weight
  (`hydra-3.5.safetensors`, ~1GB) lives in `webui/models/RedRocket-Hydra/`
  (reuses `webui/models/`'s existing gitignore treatment); `core/models.py`'s
  GGUF scanner ignores it since it only globs `*.gguf`.
- **`webui/vendor/rr_hydra/`'s internal imports are absolute
  (`from vendor.rr_hydra... import ...`), never `from ..vendor...`** —
  `core/` is imported as a top-level package (no `webui` package
  umbrella), so a relative import climbing above it fails. Caught via a
  real load-a-real-model-and-classify-a-real-image smoke test, not just
  `py_compile`.

## House rules

- **No venv, ever.** Everything installs straight into `system\python`.
  Never invoke `system\python\Scripts\*.exe` directly — always
  `system\python\python.exe -m pip install ...` / `python.exe script.py`.
- **No custom CSS unless a real built-in Gradio option was checked and
  confirmed absent first.** `webui/ui_css.py` holds the only two custom
  rules, each commented with the built-in option that was ruled out.
- **Never elevate the root logger to DEBUG** — elevate only
  `logging.getLogger("core")` and the app's own `"app"` logger (fixed
  name, not `__name__`). Third-party libraries get noisy fast otherwise.
- Before committing, review `git status`/`git diff` — nothing here
  needs the old "don't touch main-originated files" branch discipline,
  that was portable-env-specific and no longer applies.

## Gradio gotchas hit in this codebase (don't re-discover these)

- `gr.SelectData.value` for a `Tab` is **always its label, never its
  `id`** — confirmed via `gradio/layouts/tabs.py`'s own docstring. An
  `id` is required to ever target a tab via `gr.update(selected=...)`;
  without one, a push silently matches nothing. Every `gr.Tab(...)`
  that's ever a `selected=` target needs an explicit `id=`.
- A bare `gr.update()` does **not** reliably self-correct/forget a prior
  real push to a `Tabs` component's `selected=` once the user navigates
  away and back — always explicitly assert the destination in both
  directions, never rely on a no-op to "leave it alone."
- `.select()` fires only on a genuine user click, never on a server-
  pushed `gr.update(selected=...)` — reliable for top-level `Tabs`, but
  a **nested** `Tabs` inside a `Tab` that's itself being switched into
  has, at least once, appeared to fire `.select()` anyway (never fully
  root-caused — see the open item below).
- Switching away from and back to a top-level tab remounts nested
  `Column`s back to their build-time-declared `visible=`, not the
  latest server-pushed state.
- `gr.Group()` produces ugly stretched/gapped layouts with mismatched-
  height siblings — only use for uniform-height stacked rows.
- `gr.Image` output components refuse to serve a raw file path outside
  the app's CWD/system temp dir — load via PIL into memory instead.
- `os.execv`-based restart on Windows spawns a fresh process under an
  untracked PID — look up whatever's bound to port 7901, don't trust a
  previously-known task ID.
- `tkinter` save/browse dialogs only work when the browser and Python
  process are on the same machine.

## Open / deferred

- **Settings sub-tab redirect has a known "strange effect," not fully
  diagnosed** — mostly works (tab-level disable while a job runs,
  redirect-to-Debug on entry, restore the real sub-tab afterward via
  `current_settings_subtab_state`), but the user flagged residual odd
  behavior after live testing on 2026-08-30. Revisit before relying on
  it further; the nested-`Tabs`-fires-`.select()`-anyway gotcha above is
  the leading suspect.
- **Nothing has been committed since `4b45cd5`** ("Gate captioning and
  Models tabs on live server reachability, restructure Settings") — the
  entire server-lifecycle rewrite above is sitting uncommitted in the
  working tree. Review and commit in logical chunks rather than one
  giant commit.
- **Model management / llama.cpp setup UI**: curated GGUF downloading
  is built (`core/downloads.py`, `models_source.json`). The same
  pattern is now applied to llama.cpp itself via `core/llama_install.py`
  (CUDA 13.3/12.4, ROCm, CPU backend picker; no auto-detect heuristic —
  the dropdown just defaults to CUDA 13.3 and the user picks otherwise).
- **Hydra tagger integration** — done, see "Hydra 3.5 second-stage tag
  classifier" above. Deliberately left out of this first pass (upstream
  supports them, just not exposed in Settings yet): exclusive-groups,
  aliases, and per-category-prefix knobs. Also still open: the actual
  `pip install torch...`/model download and a live GPU
  caption-with-Hydra-enabled test haven't been run yet in this repo —
  only a real load+classify smoke test against another repo's
  already-downloaded weights (see the section above); a live pass
  through Settings → Hydra on real hardware is still the real test,
  same caveat as the rest of this file's UI verification.
- Smaller deferred items: an "unsaved Settings changes" indicator; the
  brief tab-disabled flash before redirect on a fresh page load; a
  possible rare race between overlapping Start/End llama clicks.
- Live UI behavior in this environment can only be verified by
  scripted simulation (`py_compile` + `build_app()` + mocked event
  handlers) — there is no browser access here. Treat "verified" claims
  in git history/PR descriptions accordingly; a live pass by the user
  is still the real test.
