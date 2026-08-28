"""Shared captioning logic - zero Gradio dependency, used identically by
both entry points: the GUI (app.py, one level up) and the headless CLI
(cli.py, same level). Anything here needs to make sense run from either
one; anything that only makes sense for the interactive GUI belongs in
app.py instead, not here.
"""
