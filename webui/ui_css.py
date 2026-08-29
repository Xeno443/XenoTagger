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

# NOTE: centering the Models tab's "A" (active-model star) column was
# tried and reverted - `#models-table td:first-child { text-align:
# center; }`. gr.Dataframe has no per-column alignment prop (checked its
# constructor signature), so CSS looked like the only option, but its
# rendered DOM turned out to be a div-based virtualized grid, not a real
# <table>/<td> - grepped the compiled component bundle directly and
# confirmed zero <td>/<tbody> tags anywhere in it, so the selector was
# never matching anything. Finding the real hook would need live browser
# devtools (not available here), same class of limitation as the
# Interrupt-button saga below - not worth chasing further for this.

# NOTE: a "make the caption box fill remaining space next to the image"
# rule was attempted here and reverted - it actively broke the Single-image
# tab's layout (no browser devtools access in this environment to verify
# selectors against Gradio's actual rendered DOM before shipping them).
# Went with a plain large `lines=` value on that Textbox instead, in
# app.py - not perfect (fixed size rather than truly dynamic) but safe.

# NOTE: a CSS-based Forge/A1111-style Generate<->Interrupt swap was tried
# here and abandoned - stacking Caption and Interrupt in one Column via
# `position: absolute` (then CSS Grid, once absolute positioning turned
# out to collapse the Column's height) did make them swap in place with
# no extra row, but the Column's own default padding/gap - which a bare
# Button never had - kept throwing off alignment with the row above,
# through several rounds of trying to compensate for it live. Solved
# without any CSS instead: Caption and Interrupt now sit in two SEPARATE
# Columns (see single_run_col/single_interrupt_col in app.py), and it's
# each Column's own `visible` that toggles - confirmed live that a Row of
# two plain gr.Column()s (no CSS at all) aligns correctly on its own, so
# swapping which two of three Columns are shown reuses that proven shape
# instead of fighting Gradio's layout with hand-tuned CSS.

ALL_CSS = STATUS_BAR_CSS
