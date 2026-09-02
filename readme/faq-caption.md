# Batch captioning - sidecar & overwrite FAQ

Notes on exactly what `core.batch.run_batch()` (the Batch tab and the CLI
share this one loop, see `webui/core/batch.py`) does with each image,
depending on which of `.txt`/`.txt.nlp`/`.txt.tags` already exist next to it
and whether "Overwrite existing" is checked. Written up because the sidecar-
adoption behavior isn't obvious from the option alone, and because it
answers a real workflow: adopting captions or tags produced by another tool
into XenoTagger's combined `.txt`. Will grow as more cases come up.

## The short version

Batch only ever looks at `.txt` to decide whether an image is "done" -
that hasn't changed. What's new is what happens to a "not done" image
(`.txt` missing) when a `.txt.nlp` and/or `.txt.tags` sidecar is already
sitting there: **that content is adopted (reused) rather than blown away**,
and only whichever half is actually missing gets a fresh model call.
`overwrite` is the escape hatch - checking it means "ignore everything
already on disk, sidecars included, and regenerate this image from
scratch," exactly like it already meant for a pre-existing `.txt`.

## The case table

Given an image with no `.txt` yet (if `.txt` already exists, it's just
skipped unless `overwrite` - unchanged, no adoption logic involved):

| `.txt.nlp` exists? | `.txt.tags` exists? | What happens |
|---|---|---|
| no | no | Normal fresh run: VLM captions it, and Hydra tags it too if `hydra_enabled`. Unchanged from before adoption existed. |
| no | **yes** | VLM captions it as normal. Hydra is **skipped for this image even if `hydra_enabled`** - an adopted `.txt.tags` is trusted as-is, never touched by a second tag source. `.txt.nlp` is written from the fresh caption; `.txt.tags` is left untouched; `.txt` = fresh caption + adopted tags. |
| **yes** | no | The VLM call is skipped entirely - the existing `.txt.nlp` content is reused verbatim (no trigger word reapplied; it's assumed to already carry one if it should). If `hydra_enabled`, Hydra runs and fills in `.txt.tags`. If Hydra is off, nothing is generated at all - `.txt` is just synthesized from the adopted caption, zero model calls. |
| **yes** | **yes** | Zero model calls, period. Both sidecars already say everything needed - `.txt` is synthesized directly from the two of them (same self-heal `.txt` gets whenever it's missing but both sidecars exist). |

`overwrite` checked short-circuits all of the above: every image is treated
as the "no / no" fresh-run row, whether or not sidecars exist, and any
sidecar content already there gets overwritten by the new run's output.

## Example workflow: adopting tags from another tool

You have a dataset already tagged by another tool (pure e621-style tags,
no prose caption) and want XenoTagger's VLM to add the descriptive part,
keeping the tags you already have:

1. Rename each caption file from `image.txt` to `image.txt.tags`.
2. Run Batch with Overwrite unchecked. Hydra can be on or off in Settings -
   it won't touch these images either way, since `.txt.tags` is already
   adopted.
3. Each image ends up with `.txt` = new VLM caption + your original tags,
   `.txt.nlp` = just the new caption, `.txt.tags` = untouched.

## Example workflow: adopting captions, adding Hydra tags

You have a dataset with prose captions from another captioner (or an
earlier XenoTagger run whose `.txt.tags` you deleted) and want Hydra to
add e621 tags without re-running the VLM:

1. Rename each caption file from `image.txt` to `image.txt.nlp`.
2. Turn Hydra on in Settings, run Batch with Overwrite unchecked.
3. Each image ends up with `.txt` = your original caption + fresh Hydra
   tags, `.txt.nlp` = your original caption (rewritten verbatim, unchanged
   content), `.txt.tags` = the new Hydra output. No VLM calls are made.

## The Review tab: a separate, per-item file-existence model

Batch's adoption logic above only ever runs once, up front, across a
whole directory. The Review tab is a different thing entirely: no model
calls, just a per-item load/save cycle every time you open, edit, or
navigate away from an image, governed by its own file-existence rules
(`app.py`'s `_review_load`/`_review_maybe_save`/`_review_tags_enabled`).
Worth knowing both, since anything Batch left half-adopted eventually gets
finished off the moment you visit it in Review.

### On load (opening an item, or navigating to it)

Caption and Tags are two independent fields, populated straight from
whichever files exist - **no combining or splitting happens on load**,
only on save:

| `.txt.nlp` | `.txt.tags` | `.txt` | Caption field shows | Tags field shows | Tags editable? |
|---|---|---|---|---|---|
| no | no | no | *(empty)* | *(empty)* | yes |
| no | **yes** | no | *(empty)* | `.txt.tags` content | yes |
| **yes** | no | no | `.txt.nlp` content | *(empty)* | yes |
| **yes** | **yes** | no | `.txt.nlp` content | `.txt.tags` content | yes |
| no | no | **yes** | `.txt` content | *(empty)* | **no** |
| no | **yes** | **yes** | `.txt` content (see note) | `.txt.tags` content | yes |
| **yes** | no | **yes** | `.txt.nlp` content | *(empty)* | yes |
| **yes** | **yes** | **yes** | `.txt.nlp` content | `.txt.tags` content | yes |

Tags is only ever disabled for the pure-legacy row (`.txt` alone, no
sidecar has ever existed) - a flat caption with nothing to split out yet.
The instant *any* sidecar exists (even just one), Tags becomes editable,
whether or not `.txt` itself exists yet.

**Note on the `.txt` + `.txt.tags`-only row:** Caption falls back to
reading `.txt` only when there's no `.txt.nlp` - but if `.txt` already
holds a previously-combined string (caption + tags on their own line,
the normal shape once something's been committed), the Caption box will
show that whole combined blob, tags line included, while the Tags box
separately shows `.txt.tags` again - a real duplication on screen. This
is a genuinely unusual state to be in (normally `.txt.nlp` gets written
alongside `.txt` any time `.txt.tags` does), most likely from someone
hand-deleting just the `.txt.nlp` sidecar. Editing and saving from here
self-heals it (see below).

### On save (navigating away, or any edit committing)

Save is keyed on whether a sidecar has ever been **committed** -
`.txt.nlp` or `.txt.tags` already existing on disk - which is a different
question from whether Tags is currently *enabled* in the UI:

- **Not committed yet** (neither sidecar exists): the two fields aren't
  independent yet. Only if **both** Caption and Tags are non-empty does a
  real split get born now - `.txt.nlp` and `.txt.tags` both get created
  alongside `.txt`. If only one field is non-empty, nothing sidecar-shaped
  is created at all - that content goes straight into `.txt` alone (this
  is also exactly what happens editing a flat legacy caption, since Tags
  is disabled and so always empty there).
- **Already committed** (at least one sidecar exists): the two fields are
  fully independent. A non-empty, *changed* field updates its own
  sidecar; an empty field is left completely alone - never created, never
  blanked. `.txt` then gets resynced from whatever's currently non-empty
  in either field, any time either one changed.

Two rules apply in both cases: **an empty file is never written**, and
**clearing a field to retype it never deletes anything** - only an actual
change to a non-empty value causes a write. A successful save also clears
any stray `.txt.issue` from a prior failed/truncated batch run.

### Self-healing mismatches

A "Sidecar only" item (Batch's adoption case, or an image with a `.txt.nlp`/
`.txt.tags` but no `.txt` yet for any other reason) is, by the rules above,
already "committed." So the moment you load it and navigate away - even
with zero edits - the committed-branch save logic fires: a missing `.txt`
reads as empty "current content," differs from the non-empty combined
value, and gets written. No special case needed for it; it falls straight
out of the same two rules everything else uses. This is slower than
Batch's own up-front adoption (one item at a time, requires actually
visiting each one) but reaches the same end state.
