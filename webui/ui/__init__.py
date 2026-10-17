"""Gradio UI pieces assembled by app.py (the frame): one module per
top-level tab (tab_*.py), the Settings sub-tabs under settings/, and the
UI-side state/helpers several tabs share. Neither cli.py nor core/ ever
imports anything from here.

Import rules - keep the graph acyclic:
  - Nothing here imports app. It runs as __main__, so `import app` would
    load a second copy of it with its own separate state.
  - state, notify, context, dialogs import no other ui module.
  - downloads_queue -> notify; llama_session -> state, notify,
    downloads_queue; operation -> llama_session.
  - tab_*.py and settings/*.py import only the shared modules above, never
    each other - except tab_settings, which assembles settings/*.
    Wiring that crosses tabs (Batch -> Review, Models -> Settings, the
    reachability refresh) lives in app.py, which gets the components it
    needs from each build()'s returned handles.
  - current_cfg gets rebound on save - always read it as
    state.current_cfg, never `from ui.state import current_cfg`.
"""
