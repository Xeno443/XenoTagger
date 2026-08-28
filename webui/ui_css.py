"""Custom CSS for app.py's Gradio UI.

Every rule here exists because Gradio has no built-in way to do the thing -
each one was only added after checking the relevant component's constructor
signature (and, where relevant, actually testing an alternative) turned up
no supported option. Default to Gradio's own props/kwargs for styling;
only add a rule here, with a comment explaining what built-in options were
ruled out and why, when there's genuinely no other way.
"""

# Gradio's own default footer ("Use via API / Built with Gradio / Settings")
# is removed via demo.launch(footer_links=[]) - a real built-in option - but
# our own replacement status bar (a plain gr.Markdown) has no equivalent
# built-in text-alignment prop anywhere in Gradio's API (checked
# gr.Markdown, gr.Row, gr.Column, and all 268 gr.themes.Base.set() theme
# variables - none exist). webui's own reference app centers its footer via
# CSS too (#footer { text-align: center; } in its style.css), which is the
# confirmation this is the standard way Gradio apps do this, not a hack.
STATUS_BAR_CSS = "#status-bar { text-align: center; }"

# NOTE: a "make the caption box fill remaining space next to the image"
# rule was attempted here and reverted - it actively broke the Single-image
# tab's layout (no browser devtools access in this environment to verify
# selectors against Gradio's actual rendered DOM before shipping them).
# Went with a plain large `lines=` value on that Textbox instead, in
# app.py - not perfect (fixed size rather than truly dynamic) but safe.

ALL_CSS = STATUS_BAR_CSS
