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
#
# font-size/color are the same story - gr.Markdown's constructor (checked)
# has no typography props at all, only structural ones (height/container/
# etc). User explicitly signed off on a CSS hack here rather than asking
# for a built-in check first.
#
# A first attempt just set font-size/color on #status-bar itself and had
# no visible effect - grepped Gradio's compiled CSS bundle
# (gradio/templates/frontend/assets/index-*.css) and found why: Markdown
# wraps its rendered text in a `.prose` element, and Gradio's own
# stylesheet has `.prose{font-size:var(--text-md)}` and
# `.prose *{color:var(--body-text-color)}` - both target the actual text
# node directly, so they win over anything set on the ancestor #status-bar
# regardless of that ID selector's specificity (a rule that directly
# matches an element always beats an inherited value from an ancestor,
# no matter how specific the ancestor's own selector is). Targeting
# `#status-bar .prose`/`#status-bar .prose *` directly gives an ID+class
# selector, which does outrank Gradio's plain class selectors. `calc(1em
# - 2px)` shrinks by a fixed 2px off whatever the inherited size actually
# is, rather than a guessed px value. color uses a theme variable (see
# below), not a hardcoded hex, so it stays correct if the theme or
# light/dark mode ever changes.
# color was originally --input-background-fill (a disabled Textbox's
# own background) - too dark/low-contrast against the app's dark theme
# to actually read once applied live (confirmed by the user, not
# guessable without a real render). Switched to
# --button-secondary-background-fill - the same variable a plain
# gr.Button("...") (e.g. "Restart app", which has no variant= override
# and so IS the default "secondary" one) already renders with, per the
# user's own follow-up ask to match that instead.
STATUS_BAR_CSS = (
    "#status-bar { text-align: center; } "
    "#status-bar .prose { font-size: calc(1em - 2px); } "
    "#status-bar .prose * { color: var(--button-secondary-background-fill); }"
)

# gr.Radio has no orientation/layout parameter (checked its constructor -
# only choices/value/type/label/etc, nothing about how the choice buttons
# lay out relative to each other); its choice buttons render inside an
# internal flex-wrap container with no exposed prop to force one per line.
# Targets the standard Gradio Radio DOM (same base component CheckboxGroup
# shares) via this Radio's own elem_id, so it can't affect any other
# Radio/CheckboxGroup in the app. Unverified against a live render (no
# browser devtools in this environment - see the Models-table NOTE below
# for the class of thing that can go wrong here).
SERVER_MODE_RADIO_CSS = "#server-mode-radio .wrap { flex-direction: column; }"

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

ALL_CSS = STATUS_BAR_CSS + "\n" + SERVER_MODE_RADIO_CSS
