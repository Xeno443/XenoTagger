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
- `readme/` — deep-dive/FAQ docs too long for a CLAUDE.md paragraph; any
  new one goes here too, not the repo root.
  - `faq-hydra-implications.md` — a worked walkthrough (concrete 4-level
    tag-hierarchy example) of how Hydra's `remove`/`constrain-remove`/
    `enforce-remove` implications modes actually resolve nested e621 tag
    families differently. The behavior isn't obvious from the option
    names alone — read before touching `hydra_implications` again.
  - `faq-hydra-setting.md` — plain-English walkthrough of Settings →
    Hydra's Confidence/Threshold sliders (the `hydra_metric` F-beta/
    min-precision pair) - what each one actually does to which tags and
    why, plus the same real-image sweep result referenced in "Hydra 3.5
    second-stage tag classifier" below.
  - `faq-caption.md` — the case table for `core.batch.run_batch()`'s
    sidecar adoption (what happens to an image with no `.txt` yet but an
    existing `.txt.nlp`/`.txt.tags`, e.g. renamed in from another tool,
    and how `overwrite` interacts with it - see "Batch sidecar adoption"
    below for the change itself), plus how the Review tab's own per-item
    load/save handles the same three files.
  - `faq-linux-considerations.md` — file:line audit of what is/isn't
    actually Windows-specific in `app.py`/`cli.py`/`core/` (excluding the
    obviously-Windows-only `.bat`/`.cmd` scripts): the managed server
    only looks for `llama-server.exe` (`core/server.py`), and the in-app
    llama.cpp installer only fetches Windows release assets
    (`core/llama_install.py`) - both managed-mode-only gaps, external
    server mode has no platform dependency at all. Read before assuming
    "Windows only" or claiming full Linux support either way.
- `run-tagger.cmd` / `tag-cli.cmd` — launch the GUI / CLI through the
  portable environment.
- `setup-env.bat` / `environment.bat` — inherited from the portable-env
  base: build/activate the portable Python + Git toolchain under
  `system\` (gitignored), then install `webui/requirements.txt`. Still
  accurate, unrelated to the worktree model that was dropped.
  `setup-tagger.cmd` (which used to also install a hardcoded CUDA-only
  llama.cpp build) is gone — that job moved to `core/llama_install.py`
  above.
- `setup-venv.bat` / `run-tagger-venv.cmd` / `tag-cli-venv.cmd`
  (added 2026-09-01) — an alternative to the portable environment for
  anyone who already has git and Python installed systemwide and would
  rather use a normal venv. `setup-venv.bat` creates `.venv\` (gitignored)
  via the systemwide `python` on PATH and installs
  `webui/requirements.txt` into it; the two `-venv.cmd` launchers mirror
  `run-tagger.cmd`/`tag-cli.cmd` but target `.venv\Scripts\python.exe`
  instead of `system\python\python.exe`. Doesn't touch `system\` at all
  — both setups coexist untouched side by side in the same checkout, and
  nothing about the portable-env path changed. `webui/requirements.txt`
  has no version pins, so a systemwide Python that's a different version
  than the portable env's pinned one (see `setup-env.bat`) is expected to
  just work. This is why `core/hydra_install.py`'s pip-install subprocess
  was changed from a hardcoded `system\python\python.exe` path to
  `sys.executable` — it now targets whichever interpreter is actually
  running the app, portable or venv, instead of always assuming the
  portable one.

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
appends Hydra's own tags after the VLM's caption, on their own line
(`"<trigger>, <VLM paragraph>\ntag1, tag2, ..."`) to ground NSFW/explicit
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
  demo.load chain runs Hydra's autoload **before** llama-server's own
  autostart (reversed 2026-09-01 — see "Hydra/llama VRAM coexistence"
  below for why and for the belt-and-suspenders mechanism that makes
  this safe regardless of ordering).
- **Hydra failures degrade gracefully** — `captioner.py` catches
  `HydraError` (deps/model missing, not loaded, OOM, whatever) and just
  keeps the VLM-only caption; it never fails a whole caption/batch. Same
  philosophy in `cli.py` (added 2026-09-01 — see "Hydra/llama VRAM
  coexistence" below): `cli.py` previously never called
  `hydra_classifier.load()` anywhere, so `hydra_enabled: true` in
  `settings.json` silently did nothing on every CLI caption; now it
  attempts a load before `resolve_server()`, logs a warning and
  continues VLM-only on failure.
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
- **`hydra_implications` defaults to `"remove"`, not `"inherit"`** — a
  real-image sweep (varying `hydra_metric` at a fixed image; see
  `readme/faq-hydra-implications.md` for the mechanism) found `inherit`'s
  whole-ancestor-chain propagation (e.g. `mammal` + `canid` + `canine` +
  `domestic dog` + `herding dog` + `pastoral dog` + `german shepherd` all
  firing together for one dog) roughly doubled the tag count at every
  tested `hydra_metric` setting versus `remove`, which collapses each
  family down to just its most specific present tag. `constrain-remove`/
  `enforce-remove` were considered and rejected: `constrain-remove`'s
  confidence-clamping has no visible effect here since `classify()`'s
  `tag_text` discards probabilities entirely and `hydra_max_tags`
  defaults to 0 (uncapped), and `enforce-remove` risks vetoing an
  otherwise-confident specific tag over an unrelated ancestor's unlucky
  score — undesirable when Hydra's whole purpose is grounding the
  specific tags the VLM is weak at.
- **`hydra_metric` defaults to `"f0.5@0.1"`, not `"f1.0@0.1"`** — same
  sweep, at `remove`: `f0.5` landed in a "complements the caption, not a
  wall of text" range (35–52 tags on the test image) without the
  contradictory tags (e.g. `male penetrating female` beside `male/male`
  content) that started reappearing above `~f0.9`.
- **Settings → Hydra exposes `hydra_metric` as two sliders — Confidence
  (`f<beta>`, 0.1–1.5) and Threshold (the `@<min_precision>` floor,
  0.0–0.5) — not a raw text field.** `_hydra_metric_to_sliders()`/
  `_hydra_sliders_to_metric()` in `app.py` convert directly (the slider
  value *is* the number in the metric string), deliberately not
  replicating upstream's non-linear slider-to-beta curve
  (`classification.py`'s `simple_slider`/`log_interval`) — more
  transparent, and it's what the actual tuning above happened against.
  This also drops `csi<weight>@<precision>` from the reachable UI
  (judged not worth a second control for; the field still accepts it if
  hand-edited into `settings.json`).

## Hydra/llama VRAM coexistence (2026-09-01)

Flagged live: `hydra_enabled: true` with the model not actually loaded
(never loaded, autoload off, or a load failed) was being silently
ignored - `captioner.py` just keeps the VLM-only caption with no
user-visible sign Hydra didn't fire. Investigating that surfaced a
second, more fundamental issue: llama-server's own `n_gpu_layers="auto"`
is the one piece in this app that gracefully adapts to reduced free VRAM
(partial CPU offload) - `hydra_classifier.load()` has no such fallback,
it's all-or-nothing onto one device. So whichever of the two claims VRAM
*first* matters: if llama's "auto" goes first and grabs most of it,
Hydra loading second is the one likely to just OOM.

Two things this drove, both in `app.py` unless noted:

- **`_load_hydra_with_llama_coexistence()`** - the one shared generator
  behind both the manual "Load Hydra model" button (`_load_hydra_model_ui`)
  and the startup autoload path (`_autoload_hydra_model_ui`), so
  correctness doesn't depend on either caller getting the details right
  independently. If `hydra_device == "cuda"` and a **managed** llama-
  server **this session owns** (`_is_server_managed_by_us` - same
  ownership gate "Stop llama server" already uses; never touches an
  external server, an orphaned process from a crashed previous run, or
  someone else's server on that port) is currently running, it's stopped
  first, Hydra loads against a clean VRAM budget, then llama-server is
  restarted via `_try_start_managed_llama()` (the same shared start
  primitive already reused by the Start button/autostart/a settings-save
  restart) so `"auto"` resizes itself around whatever Hydra actually
  claimed. Skipped entirely if `hydra_device == "cpu"` (no VRAM
  contention) or if `_operation_blocked_by()` says a caption job is
  active (killing the server mid-request would break it - in practice
  the Hydra sub-tab is already disabled while one runs, so this is a
  second-line-of-defense check, not the primary gate, same idiom
  `_operation_blocked_by()` uses everywhere else). Fires one `_notify()`
  popup on the final outcome, since a stop+restart can now take as long
  as a full llama-server reload (10-30GB), not just Hydra's own ~1GB.
- **Startup autoload order reversed**: the `demo.load` chain now runs
  `_autoload_hydra_model_ui` *before* `_autostart_managed_llama_ui`
  (previously the other way around, deliberately, specifically to
  exercise the contention case during initial Hydra integration testing -
  see the git history around 2026-08-31 for that older reasoning, now
  superseded). With both autostart-llama and autoload-Hydra enabled,
  Hydra now gets first pick of a clean VRAM budget at every launch,
  llama's "auto" adapts around it. This is an optimization on top of
  the shared function above, not a substitute for it - at true startup
  `_load_hydra_with_llama_coexistence()`'s own stop/restart branch is
  normally a no-op (nothing's started yet this early), but it's still
  live as the real safety net for the orphaned-process edge case.
- **`cli.py`** now calls `hydra_classifier.load(cfg)` (if `hydra_enabled`)
  right before `resolve_server()` - same ordering logic, no stop/restart
  dance needed since a fresh CLI process hasn't started (or connected to)
  llama-server yet at that point. A load failure logs a warning and
  continues VLM-only, matching `captioner.py`'s own degrade-gracefully
  philosophy. Doesn't help if `resolve_server()` ends up *connecting to*
  an already-running server owned by something else (e.g. the GUI) that
  already claimed VRAM before this process even started - out of scope,
  same "never touch what we don't own" boundary as everywhere else.

## Status/notification architecture (2026-08-31)

Two distinct UI concepts, named deliberately differently so they're never
confused in conversation or code:

- **status bar** — the single global `status_bar` component (Python/
  Gradio version, llama-server health, loaded model, active operation),
  polled every 2s by `status_timer` and shown on every tab.
- **infotext bar** — the per-tab textboxes (`single_infotext`,
  `batch_infotext`, `review_infotext`, `models_infotext`,
  `settings_infotext`, `llama_lifecycle_infotext`,
  `hydra_lifecycle_infotext`), each written only by its own tab's own
  handlers. Deliberately kept per-tab, not merged into one shared field —
  a merged box would have live-progress writers (e.g. batch's per-image
  yield loop) stomping on one-off action feedback (e.g. Save Settings)
  landing in the same tick.

`_notify(text, level="info"|"warning"|"error")` is the one place a
Gradio popup toast gets fired, replacing what used to be scattered direct
`gr.Info`/`gr.Warning` calls. `level` drives both the `log.<level>()`
call (auto-captured by the debug log when enabled) and the popup type -
see the `gr.Error` gotcha above for why `"error"` still pops a
`gr.Warning`-styled toast, not a red one. A message that should only be
logged, never popped, still just calls `log.<level>()` directly - `_notify`
always pops.

Long-running background work (batch captioning, curated-model/Hydra-model
downloads, llama.cpp/Hydra-deps installs) now fires a one-shot `_notify()`
popup on completion or failure, not just a silent infotext-box update -
so finding out something finished doesn't require still being on the tab
that started it (the original ask: "I start a few model downloads, let
Hydra deps install, then go run a batch - I have no idea what's happening
to the other things"). Batch fires it directly from its own generator
(already a live Gradio callback throughout the run). Downloads/installs
run on raw background threads with no Gradio request context, so each
queues a one-shot (message, level) announcement (`_download_announces`,
`_install_announce`, `_hydra_install_announce`) that the next 2s
`status_timer` poll pops and relays through `_notify` - the same
single-shot-flag idiom `_download_needs_refresh` already established, now
also used to drive a popup instead of only a silent infotext update.

Captioning now shows a live per-image stage in its own tab's **infotext
bar** (not the status bar - that was tried first and reverted; the status
bar doesn't need this level of detail, and it's the wrong field for it
per the status-bar/infotext-bar split above). `core.captioner.caption_image()`
takes an optional `on_stage(text)` callback, called with `"captioning"`
before the VLM call and `"tagging (Hydra)"` before Hydra's, threaded
through unchanged from `core.batch.run_batch()`.

Infotext boxes only update on a yield, never on an independent poll, so
showing a *live* stage change means the caller has to actually yield
when the stage changes, not just before/after the whole call. Batch
already ran `caption_image()` inside a background worker thread feeding
a `queue.Queue` that its generator drains and yields from (needed for
per-image progress) - `on_stage` there just pushes a `("stage", text)`
item onto the same queue, tagged separately from `("progress", ...)`
items so the drain loop can tell them apart. Single-image and Review
didn't have that shape at all (they called `caption_image()` directly
and only yielded once before, once after) - both got the same
background-thread-+-queue treatment added specifically so `on_stage` has
something to push into and the generator has something to yield from
mid-call.

## Batch sidecar adoption (2026-09-02)

`core.batch.run_batch()` used to only ever check whether `.txt` existed
before deciding to (re)generate an image from scratch - a pre-existing
`.txt.nlp`/`.txt.tags` sidecar with no `.txt` yet (e.g. hand-renamed in
from another captioning/tagging tool) was ignored and clobbered by a full
fresh `caption_image()` call. Now that content is adopted instead: only
whichever half is actually missing gets a real model call. Full case
table in `readme/faq-caption.md` (see Layout above); short version:

- `.txt.tags` exists, `.txt.nlp` doesn't → VLM runs as normal, but Hydra
  is skipped for that image even if `hydra_enabled` - an adopted
  `.txt.tags` is trusted as-is, never touched by a second tag source.
- `.txt.nlp` exists, `.txt.tags` doesn't → the VLM call is skipped
  entirely (the adopted caption is reused verbatim, no trigger word
  reapplied), and Hydra runs to fill in the missing tags if enabled.
- Both sidecars already exist → zero model calls, `.txt` is just
  synthesized from the two of them directly.
- `overwrite` checked bypasses all of this - every image is treated as
  fully fresh, sidecars included, exactly like it already did for a
  pre-existing `.txt`.

Implementation-wise, `core.captioner.caption_image()` gained an
`existing_caption` parameter: when given, it skips the VLM call and
reuses that text as `vlm_caption` verbatim, still falling through to
Hydra tagging as normal below it - letting `run_batch()` route the
"adopt caption, still need tags" case through the exact same call/outcome
handling as a normal run (a zero-cost stand-in `CaptionResult` keeps
`outcome.truncated`/`resize_note` readable uniformly either way), rather
than needing a separately-maintained code path. The "adopt tags, skip
Hydra" case reuses `dataclasses.replace(cfg, hydra_enabled=False)` for
just that one `caption_image()` call instead of touching `captioner.py`
again - `cfg` itself (used everywhere else, e.g. the truncation-reason
message) is left untouched.

## House rules

- **No venv, ever — for this repo's own portable-env path.** Everything
  installs straight into `system\python`. Never invoke
  `system\python\Scripts\*.exe` directly — always
  `system\python\python.exe -m pip install ...` / `python.exe script.py`.
  This is about `system\python` specifically, not a blanket objection to
  venvs everywhere: `setup-venv.bat` (see Layout above) offers an
  end-user-facing `.venv\` as an explicit alternative for anyone who'd
  rather not use the portable environment at all. Don't reintroduce a
  venv *inside* the portable-env workflow itself, e.g. as some
  in-between layer under `system\python` — that's what this rule
  actually rules out.
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
- `gr.Error` is not a fire-and-forget popup like `gr.Info`/`gr.Warning` —
  calling it just constructs an exception object; it only displays
  anything if actually `raise`d, and raising it aborts whatever function
  raised it (Gradio's own dispatcher catches it to render the modal).
  `app.py`'s `_notify()` deliberately maps its `"error"` level to a
  `gr.Warning`-styled toast for this reason, not `gr.Error` — none of
  its callers (mostly 2s-timer polling handlers) can afford to have a
  popup silently abort them mid-function.

## Open / deferred

- **Broad UI-logic refactor, not yet scoped.** Much of the app predates
  this session's status-bar/infotext-bar/`_notify` consolidation (see
  "Status/notification architecture" above) and still carries older UI
  patterns from before that. Revisit the rest of the app against the new
  logic once it's settled. User's own remark on this, verbatim: "no more
  lazy start and ui elements enabled/disabled redesign" - i.e. don't let
  the old implicit-lazy-start style creep back in anywhere during the
  refactor, and the widespread `interactive=True/False` enable/disable
  pattern (Run/Interrupt gating, Settings sub-tab gating, etc.) is itself
  a candidate for redesign here, not just something to preserve as-is.
- **Settings sub-tab redirect has a known "strange effect," not fully
  diagnosed** — mostly works (tab-level disable while a job runs,
  redirect-to-Debug on entry, restore the real sub-tab afterward via
  `current_settings_subtab_state`), but the user flagged residual odd
  behavior after live testing on 2026-08-30. Revisit before relying on
  it further; the nested-`Tabs`-fires-`.select()`-anyway gotcha above is
  the leading suspect.
- **Settings tab visual harmonization (Llama vs. Hydra sub-tabs) — done
  (2026-09-01).** Both sub-tabs now share the same shape: a top intro
  Markdown pulled from a `core.config` variable (`LLAMA_TAB_INTRO`,
  mirroring `MODELS_TAB_INTRO`'s own precedent), a management `gr.Group()`
  with side-by-side action buttons, an autostart/autoload checkbox above
  the lifecycle buttons, and a lifecycle infotext moved to the very
  bottom of the sub-tab (lands right above the shared Save settings/
  Restart app row, since both sub-tabs stack inside one `Tabs()` that
  sits above that row). Llama's install/reinstall flow now reads real
  installed-backend state (`llama_installed_info()`) to preselect the
  backend dropdown and swap the button between "Install llama.cpp"/
  "Reinstall llama.cpp"; `abort_install_btn` is now actually gated on
  install-in-progress (previously always clickable). Hydra's pip-output
  Textbox was removed — `core.hydra_install.install_deps()` already
  unconditionally `log.debug()`s every line regardless of any UI
  callback, so nothing was lost; only completion/failure surfaces to the
  UI now, via infotext + a `_notify()` popup. Hydra's "Download Hydra
  model" button/progress row moved off this sub-tab entirely, onto the
  Models tab's new "Hydra model" section (see below).
  `installed_backend_text` (the "Installed: CUDA 12.4 ... (b10701)" line)
  now sits right under the tab's intro Markdown, the same position
  `hydra_status_md` occupies on the Hydra tab.
- **Follow-up (2026-09-01, same day): the gr.Group()-squashes-buttons
  problem confirmed live on the Models tab (see below) turned out to
  also affect the Llama sub-tab's button rows** - `managed_server_group`/
  `external_server_group` were `gr.Group(visible=...)`, used purely to
  toggle a whole section's visibility together, but Group's bordered/
  connected-siblings styling was squashing "Install llama.cpp"/"Abort
  install"/"Start llama server"/"Stop llama server"/"Verify" into merged
  slabs the same way it did on Models. Fixed by swapping both to
  `gr.Column(visible=...)` - same visible= toggle behavior, no bordering
  side effect. Hydra's Install/Load/Unload button Row had its own
  dedicated (single-child) `gr.Group()` wrapper too, removed the same
  way. A plain `gr.Markdown("---")` (real CommonMark `<hr>`, no CSS) now
  separates the Models tab's two sections.
- **Follow-up #2 (2026-09-01): the Llama sub-tab's config fields (port,
  GPU layers, context size, extra args, install-status caption,
  Autostart) now sit inside their own `gr.Group()` for a shared grey
  background matching Server mode's box** - requested live after seeing
  that range sitting on the plain black background next to Server
  mode's own box. The backend-dropdown+Install+Abort Row was
  deliberately kept OUTSIDE this Group (same reasoning as the fix right
  above - a Row of buttons inside a Group loses its inter-button gaps
  and merges into one flat bar), so the grey box actually starts at the
  install-status row, not literally at the dropdown. `autostart_managed_
  llama` moved from below Extra args to right after the installed-
  backend caption (above Port) and is now only `interactive=` once
  `llama_installed_info()` is not None (there's nothing to autostart
  otherwise) - kept live via the same `_llama_install_status_ui` poll
  that already drives the Install button's label/interactive.
- **Other 2026-09-01 UI redesign items, all done alongside the Settings
  harmonization above**: recursive batch/review scanning was removed
  entirely (UI checkbox, `AppConfig.recursive_batch`, `find_images()`/
  `run_batch()`/`scan_review_status()` params, `cli.py --recursive`) —
  batch/review only ever process the given root directory now, no
  subfolder walk. Batch's Run/Interrupt buttons moved to just above
  `batch_infotext`. Review's Browse button now chains an automatic
  `review_scan_ui` call (Scan button label unchanged — typing a path by
  hand still needs a manual Scan/click, no auto-scan-on-Enter). The
  Models tab is now two plain (ungrouped) sections under a `### VLM
  model`/`### Hydra model` Markdown title each - a `gr.Group()` wrap was
  tried first and reverted live: it merged the mismatched-height
  Markdown/Dataframe/button-Row children into one stretched grey slab
  instead of leaving each its own look (the exact failure mode the
  gr.Group() gotcha below already warns about - Group is only safe for
  genuinely uniform stacked rows, confirmed again here the hard way).
  "VLM model" is the pre-existing table/dropdowns/buttons, now with
  `buttons=[]` on
  `models_table` to drop Gradio's default copy/fullscreen toolbar
  buttons — a real `gr.Dataframe(buttons=...)` param, not CSS, following
  `review_table`'s existing precedent) and a new "Hydra model" section:
  a matching 4-column one-row `gr.Dataframe` (`_hydra_models_table_rows()`),
  a Download button (moved here from Settings → Hydra, now writes to
  `models_infotext` instead of `hydra_lifecycle_infotext`), a "Manage
  Hydra" button (`_goto_hydra_settings_ui()`, mirrors the pre-existing
  `_goto_llama_settings_ui()`), and a Refresh button that's deliberately
  just the same `models_refresh_ui` handler as the VLM section's own
  Refresh — it doesn't actually rescan the Hydra row (that only happens
  on page load or after a Load/Unload/Install click chains
  `_hydra_install_status_ui`), kept intentionally minimal per explicit
  user direction ("useless for now").
- **Follow-up #3 (2026-09-01, later the same day): the Models tab's two
  download-status rows (one per section) were merged into one.** Both
  the VLM section's row and the Hydra section's own row were wired to
  the same `_download_status_ui()`/global `_download_current` state, so
  whichever download was actually active showed identically (and
  confusingly) in both places at once — e.g. the Hydra row showing a
  VLM download's progress, which live-read as "my Hydra download got
  silently cancelled" when it hadn't. There's only ever one real
  download in flight regardless of which section queued it (both share
  `_download_enqueue`/`_download_queue`/`_download_worker`), so the fix
  was to delete the duplicate `hydra_download_status_row`/
  `hydra_download_status_text` entirely and move the one real
  `download_status_row` (bar + "Abort all downloads" button) below both
  sections, behind its own `gr.Markdown("---")` divider.
  `_hydra_download_model_ui()` now also returns that row's update
  directly (mirroring `models_action_ui`'s own immediate-refresh
  pattern) instead of waiting for the next 2s poll.
- **Batch tab: new "Send to Review" button** — a third, gray
  (default-variant) button in the same Row as Run batch/Interrupt,
  `interactive=` only when `batch_dir` is a real, existing directory
  (`batch_dir.change()`-driven). There's no cheap way to know whether
  the last batch run in that directory actually succeeded — `batch.py`
  is deliberately stateless (see its own module docstring) — so path
  validity is the only check, by design. Clicking it (`_send_batch_to_
  review_ui`, wired down in the Review tab's own section since
  `review_dir`/`review_scan_ui`/`_review_nav_outputs` aren't defined yet
  that early in the file — same reason `hydra_models_manage_btn`'s click
  is wired inside the Settings tab body) switches `main_tabs` to Review
  (needed adding `id="review"` to that Tab — every `gr.update(selected=
  ...)` target needs one, see the Gradio gotcha below), copies
  `batch_dir` into `review_dir`, and chains into the same
  `review_scan_ui` scan Review's own Browse button uses.
- **Model management / llama.cpp setup UI**: curated GGUF downloading
  is built (`core/downloads.py`, `models_source.json`). The same
  pattern is now applied to llama.cpp itself via `core/llama_install.py`
  (CUDA 13.3/12.4, ROCm, CPU backend picker; no auto-detect heuristic —
  the dropdown just defaults to CUDA 13.3 and the user picks otherwise).
- **Hydra tagger integration** — done, see "Hydra 3.5 second-stage tag
  classifier" above. A real `pip install torch...`/model download plus
  live GPU classify passes (a throwaway metric-sweep script and the real
  Settings → Hydra sliders in the running GUI) have now happened against
  real images on real hardware, and `hydra_metric`/`hydra_implications`
  defaults were retuned as a result. Still deliberately left out
  (upstream supports them, just not exposed in Settings): exclusive-
  groups, aliases, and per-category-prefix knobs.
- **Llama tab's download progress bar - user-owned follow-up, not yet
  scheduled.** During the 2026-09-01 harmonization pass, whether Hydra's
  model-download progress display (on the Models tab's "Hydra model"
  section) should visually match llama.cpp's own byte-progress `<progress>`
  bar (`llama_install_status_text`, `_llama_install_status_ui`) came up
  but was explicitly deferred: "I will revisit the llama download bar,
  leave as is for now." Don't start on this without the user bringing it
  back up first.
- Smaller deferred items: an "unsaved Settings changes" indicator; the
  brief tab-disabled flash before redirect on a fresh page load; a
  possible rare race between overlapping Start/End llama clicks.
- Live UI behavior in this environment can only be verified by
  scripted simulation (`py_compile` + `build_app()` + mocked event
  handlers) — there is no browser access here. Treat "verified" claims
  in git history/PR descriptions accordingly; a live pass by the user
  is still the real test.
