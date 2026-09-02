// Tag-autocomplete popup for Review's Tags field only (#review_tags_box).
// Ported (deliberately scoped down) from a1111-sd-webui-tagcomplete's core
// mechanism: a hand-rolled floating popup over a plain textarea, filtering
// a tag list held in a plain JS array/object - not a framework-reactive
// component. That's the point: a gr.Dropdown with the same ~100k-tag
// vocabulary as its `choices` prop was confirmed unusably slow (a known,
// unfixed Gradio limitation - Dropdown renders every choice as a real
// tracked DOM element), while the reference extension handles the exact
// same CSVs smoothly because filtering here is just inert array scanning,
// and only the handful of matched results ever become real DOM nodes.
//
// Left out on purpose (present in the reference extension, not needed for
// a plain comma-separated tag field): prompt weighting, wildcards,
// embeddings/LoRAs, translations/ruby annotations, a usage-frequency DB,
// style presets. window.XT_TAG_DATA (set in a preceding inline <script> -
// see core.tag_vocab.build_autocomplete_head) is `{tags: [...], aliases:
// {alias: canonical}}`, both already space-form to match what's actually
// written to .tags files.
(function () {
    "use strict";

    const MAX_RESULTS = 20;
    const BOX_SELECTOR = "#review_tags_box textarea, #review_tags_box input[type='text']";

    function injectStyle() {
        if (document.getElementById("xt-tac-style")) return;
        const style = document.createElement("style");
        style.id = "xt-tac-style";
        // Uses Gradio's own theme CSS variables (present in every theme,
        // light or dark) rather than hardcoded colors, so this stays
        // consistent with whatever theme is active.
        style.textContent = `
            .xt-tac-popup {
                position: fixed;
                z-index: 10000;
                background: var(--background-fill-primary);
                border: 1px solid var(--border-color-primary);
                border-radius: var(--radius-lg, 8px);
                max-height: 320px;
                overflow-y: auto;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
                display: none;
            }
            .xt-tac-popup ul {
                list-style: none;
                margin: 0;
                padding: 4px 0;
            }
            .xt-tac-popup li {
                padding: 6px 12px;
                cursor: pointer;
                color: var(--body-text-color);
                white-space: nowrap;
                font-size: var(--text-md, 14px);
            }
            .xt-tac-popup li.xt-tac-selected,
            .xt-tac-popup li:hover {
                background: var(--background-fill-secondary);
            }
            .xt-tac-popup li .xt-tac-alias-of {
                color: var(--body-text-color-subdued, #999);
                font-size: 0.85em;
                margin-left: 8px;
            }
        `;
        document.head.appendChild(style);
    }

    // Finds the comma-delimited segment the cursor is currently in, with
    // its boundaries trimmed of surrounding whitespace - so splicing
    // before/after it back together never duplicates or loses spacing
    // around the comma.
    function getWordAt(text, cursor) {
        const beforeText = text.slice(0, cursor);
        const afterText = text.slice(cursor);
        let start = beforeText.lastIndexOf(",") + 1;
        let end = afterText.indexOf(",");
        end = end === -1 ? text.length : cursor + end;
        while (start < end && /\s/.test(text[start])) start++;
        while (end > start && /\s/.test(text[end - 1])) end--;
        return { start: start, end: end, word: text.slice(start, end) };
    }

    function filterTags(query) {
        if (!query) return [];
        const q = query.toLowerCase();
        const data = window.XT_TAG_DATA || { tags: [], aliases: {} };
        const results = [];
        const seen = new Set();

        for (let i = 0; i < data.tags.length && results.length < MAX_RESULTS; i++) {
            const tag = data.tags[i];
            if (!seen.has(tag) && tag.toLowerCase().includes(q)) {
                results.push({ text: tag, matchedAlias: null });
                seen.add(tag);
            }
        }
        if (results.length < MAX_RESULTS) {
            for (const alias in data.aliases) {
                if (results.length >= MAX_RESULTS) break;
                if (!alias.toLowerCase().includes(q)) continue;
                const canonical = data.aliases[alias];
                if (!seen.has(canonical)) {
                    results.push({ text: canonical, matchedAlias: alias });
                    seen.add(canonical);
                }
            }
        }
        return results;
    }

    class TagAutocomplete {
        constructor(textarea) {
            this.textarea = textarea;
            this.results = [];
            this.selectedIndex = -1;

            this.popup = document.createElement("div");
            this.popup.className = "xt-tac-popup";
            this.list = document.createElement("ul");
            this.popup.appendChild(this.list);
            document.body.appendChild(this.popup);

            textarea.addEventListener("input", () => this.onInput());
            textarea.addEventListener("keydown", (e) => this.onKeyDown(e));
            // Delayed so a suggestion's mousedown (which prevents default,
            // see render()) registers before the resulting blur hides it.
            textarea.addEventListener("blur", () => {
                setTimeout(() => this.hide(), 150);
            });
        }

        onInput() {
            const word = getWordAt(this.textarea.value, this.textarea.selectionStart).word;
            this.results = filterTags(word);
            this.selectedIndex = this.results.length > 0 ? 0 : -1;
            if (this.results.length === 0) {
                this.hide();
                return;
            }
            this.render();
            this.show();
        }

        render() {
            this.list.innerHTML = "";
            this.results.forEach((r, i) => {
                const li = document.createElement("li");
                li.textContent = r.text;
                if (r.matchedAlias) {
                    const span = document.createElement("span");
                    span.className = "xt-tac-alias-of";
                    span.textContent = "(" + r.matchedAlias + ")";
                    li.appendChild(span);
                }
                if (i === this.selectedIndex) li.classList.add("xt-tac-selected");
                li.addEventListener("mousedown", (e) => {
                    // mousedown (not click) fires before the textarea's
                    // own blur handler, and preventDefault keeps focus on
                    // the textarea so the blur-hide timer never starts.
                    e.preventDefault();
                    this.select(i);
                });
                this.list.appendChild(li);
            });
        }

        show() {
            const rect = this.textarea.getBoundingClientRect();
            this.popup.style.left = rect.left + "px";
            this.popup.style.top = rect.bottom + 4 + "px";
            this.popup.style.minWidth = rect.width + "px";
            this.popup.style.display = "block";
        }

        hide() {
            this.popup.style.display = "none";
            this.results = [];
            this.selectedIndex = -1;
        }

        isVisible() {
            return this.popup.style.display === "block";
        }

        onKeyDown(e) {
            if (!this.isVisible()) return;
            switch (e.key) {
                case "ArrowDown":
                    e.preventDefault();
                    this.selectedIndex = (this.selectedIndex + 1) % this.results.length;
                    this.render();
                    break;
                case "ArrowUp":
                    e.preventDefault();
                    this.selectedIndex = (this.selectedIndex - 1 + this.results.length) % this.results.length;
                    this.render();
                    break;
                case "Enter":
                case "Tab":
                    if (this.selectedIndex >= 0) {
                        e.preventDefault();
                        this.select(this.selectedIndex);
                    }
                    break;
                case "Escape":
                    this.hide();
                    break;
            }
        }

        select(index) {
            const result = this.results[index];
            if (!result) return;

            const value = this.textarea.value;
            const pos = getWordAt(value, this.textarea.selectionStart);
            // Extend the replaced range through one trailing comma (+
            // surrounding whitespace), if present, so we don't end up
            // with a doubled separator between the inserted tag and the
            // next one - insertText below only replaces the *selected*
            // range, it can't also edit text further along on its own.
            let end = pos.end;
            const trailingSep = value.slice(end).match(/^\s*,\s*/);
            if (trailingSep) end += trailingSep[0].length;

            const insertion = result.text + ", ";

            this.textarea.focus();
            this.textarea.setSelectionRange(pos.start, end);

            // execCommand is deprecated but still the only way to make a
            // script-driven edit land on the browser's native undo stack
            // (Ctrl+Z) - a plain .value assignment is invisible to it,
            // which is exactly why undo worked for typed text but not for
            // a popup-inserted tag. Falls back to the old manual splice +
            // dispatched input event for the rare case it's unavailable.
            const usedExecCommand =
                typeof document.execCommand === "function" &&
                document.execCommand("insertText", false, insertion);

            if (!usedExecCommand) {
                const before = value.slice(0, pos.start);
                const after = value.slice(end);
                const newValue = before + insertion + after;
                const newCursor = before.length + insertion.length;
                this.textarea.value = newValue;
                this.textarea.setSelectionRange(newCursor, newCursor);
                this.textarea.dispatchEvent(new Event("input", { bubbles: true }));
            }

            this.hide();
            this.textarea.focus();
        }
    }

    function attach() {
        document.querySelectorAll(BOX_SELECTOR).forEach((el) => {
            if (el.classList.contains("xt-tac-bound")) return;
            el.classList.add("xt-tac-bound");
            new TagAutocomplete(el);
        });
    }

    function init() {
        injectStyle();
        attach();
        // Gradio remounts nested components on some tab-switch patterns
        // (documented elsewhere in this app) - re-scanning on any DOM
        // change, guarded by the .xt-tac-bound class above, is what makes
        // re-attaching after a remount automatic rather than a one-shot
        // that silently stops working.
        new MutationObserver(() => attach()).observe(document.body, {
            childList: true,
            subtree: true,
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
