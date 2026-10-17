"""Gradio UI pieces assembled by app.py (the frame): one module per
top-level tab (tab_*.py), the non-backend Settings sub-tabs under
settings/ (each backend brings its own as backends/<id>/settings_ui.py),
and the UI-side state/helpers several tabs share. Neither cli.py, core/,
nor a backend outside its settings_ui.py ever imports anything from here.

Layering: app -> ui -> backends -> core, never back up - except a
backend's settings_ui.py, which may use the shared ui modules below
(see backends/__init__.py).

Import rules - keep the graph acyclic:
  - Nothing here imports app. It runs as __main__, so `import app` would
    load a second copy of it with its own separate state.
  - state, notify, context, dialogs import no other ui module.
  - downloads_queue -> notify; reachability -> notify; model_status ->
    state, context; operation -> no ui module (it stops llama-server via
    backends.llama.session).
  - tab_*.py and settings/*.py import only the shared modules above, never
    each other - except tab_settings, which assembles settings/* and the
    backends' settings_ui modules. Wiring that crosses tabs (Batch ->
    Review, Models <-> a backend's Settings sub-tab, the reachability
    refresh) lives in app.py, which gets the components it needs from each
    build()'s returned handles.
  - current_cfg gets rebound on save - always read it as
    state.current_cfg, never `from ui.state import current_cfg`.
"""
