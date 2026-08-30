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
    Single-image, Batch, and the CLI alike (and the intended hook point
    for a future Hydra classifier pass, see below).
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
- **Hydra tagger integration — not started in this repo.** A second-
  stage Hydra 3.5 (RedRocket) e621-style tag classifier, meant to
  ground NSFW/explicit tagging the VLM alone is weak at, is prototyped
  separately in a standalone `HydraTagger` repo. `core/captioner.py`'s
  shared entry point exists partly so this only needs wiring in once.
- Smaller deferred items: an "unsaved Settings changes" indicator; the
  brief tab-disabled flash before redirect on a fresh page load; a
  possible rare race between overlapping Start/End llama clicks.
- Live UI behavior in this environment can only be verified by
  scripted simulation (`py_compile` + `build_app()` + mocked event
  handlers) — there is no browser access here. Treat "verified" claims
  in git history/PR descriptions accordingly; a live pass by the user
  is still the real test.
