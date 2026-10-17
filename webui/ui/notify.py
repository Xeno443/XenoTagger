"""notify() - the one place a Gradio popup toast gets fired."""

from __future__ import annotations

import logging

import gradio as gr

log = logging.getLogger("app")  # fixed name, see app.py

# gr.Error is deliberately not in here, even though it's the "error"-
# flavored popup - unlike gr.Info/gr.Warning (plain calls that post a
# toast and let execution continue), gr.Error only displays anything
# when *raised* and allowed to propagate all the way out to Gradio's own
# dispatcher, which aborts whatever function raised it. That's the wrong
# shape for notify's callers below - none of them want firing a popup
# to also cut their own function off before it returns its normal
# outputs. So "error" still logs as a real log.error() line, it just
# surfaces as a (non-fatal) warning-styled toast rather than a red one.
_POPUP_FNS = {"info": gr.Info, "warning": gr.Warning, "error": gr.Warning}


def notify(text: str, level: str = "info") -> None:
    """The one place a Gradio popup toast gets fired - replaces what used
    to be scattered direct gr.Info/gr.Warning calls. level picks both the
    log.<level>() call (auto-captured by the debug log/file when enabled -
    see setup_debug_logging) and the matching popup type - see
    _POPUP_FNS above for why "error" still pops a warning-styled toast,
    not a red one. This always pops a toast - for a message that should
    only be logged, call log.<level>() directly instead.

    Must be called from inside a real Gradio callback (click/tick/load
    handler) - like gr.Info/Warning themselves, it needs Gradio's request
    context, so it can't be called from a raw background thread (e.g. the
    batch/download/install worker threads) - those still have to route
    through a queue/state that a polling handler relays from, the same
    pattern already used for lifecycle infotext boxes."""
    getattr(log, level)(text)
    _POPUP_FNS[level](text)
