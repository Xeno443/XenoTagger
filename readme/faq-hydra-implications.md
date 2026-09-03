# Hydra 3.5 tag classifier - configuration FAQ

Notes on how the Hydra second-stage e621 tag classifier (`core/hydra_classifier.py`,
Settings → Hydra) actually resolves confidence thresholds, tag caps, and
near-duplicate/nested tag families (e.g. `canine` + `dog` + `dobermann` firing
together). Written up because the behavior isn't obvious from the option names
alone.

## 1. Calibration metric (confidence threshold)

`hydra_metric` (Settings → Hydra "Calibration metric", default `"f1.0@0.1"`)
isn't a probability like `0.35`. It's a formula string that tells Hydra how to
*pick* each tag's own threshold from its validation data.

Format: `(f|csi)<num>@<min_precision>`, or the literal word `"default"`.

- **`f<beta>`** - optimize an F-beta score (precision/recall balance) per tag.
  - `beta = 1.0` → F1, precision and recall weighted equally.
  - `beta > 1` (e.g. `f2.0`) → weight recall more: catch more true tags,
    tolerate more false positives.
  - `beta < 1` (e.g. `f0.5`) → weight precision more: only tag when
    confident, tolerate missing some.
- **`csi<weight>`** - Critical Success Index instead of F-beta; `weight`
  scales how harshly false positives count.
- **`@<min_precision>`** - a floor: reject any threshold that would let
  measured precision fall below this fraction on Hydra's validation set, no
  matter what score it'd otherwise get. This exists to stop a degenerate pick
  (e.g. threshold ≈ 0, "call everything positive") on labels with very few
  validation examples.
- **`default`** - a shortcut for `f1.0@0.1`, which is also XenoTagger's own
  default value.

Practical effect: raising the `@` number (e.g. `f1.0@0.3`) tightens every
tag's threshold - fewer tags, higher confidence in the ones that survive.
Lowering it loosens things. Changing `beta` shifts precision vs. recall
independently of that floor. It's recalculated on every caption (cheap), so
edits take effect on the next caption with no Hydra reload needed.

## 2. Max tag count / top-k

Already exposed: `hydra_max_tags` / Settings → Hydra "Max tags appended
(0 = no cap)". It's a flat truncate-to-top-N by probability, applied last,
after thresholding and implications have already run. It's a single global
cap across all tags, not a per-category top-k.

## 3. Implications mode - merging/collapsing nested tag families

`hydra_implications` (Settings → Hydra "Implications mode") controls whether
a family like `mammal → canine → domestic_dog → dobermann` collapses down to
just the most specific tag, or reports every level at once. It's driven by
Hydra's own e621 tag-implication graph baked into the model (`label.implies`
/ `label.implied_by`), not something built for this integration.

The three modes that end in "collapse" (`remove`, `constrain-remove`,
`enforce-remove`) all finish with the same step but differ in what happens
before it.

### The chain used in the examples below

```
mammal   (level 1, most general)
  ↑ implies
canine   (level 2)
  ↑ implies
domestic_dog   (level 3)
  ↑ implies
dobermann   (level 4, most specific)
```

"implies" always points specific → general (dobermann implies domestic_dog,
never the reverse).

### The raw model output for one image

| tag | raw probability | its own calibrated threshold | passes on its own? |
|---|---|---|---|
| dobermann | 0.95 | 0.50 | yes |
| domestic_dog | 0.90 | 0.45 | yes |
| canine | 0.25 | 0.35 | **no** |
| mammal | 0.70 | 0.20 | yes |

The model is very confident about the specific breed and separately
confident it's some kind of mammal, but its mid-chain "canine" score is
oddly low. This dip is what makes the three modes diverge.

### `remove`

**Step 1 - plain independent thresholding**, no hierarchy awareness at all.
Each tag checks only itself:
- dobermann 0.95 ≥ 0.50 → kept
- domestic_dog 0.90 ≥ 0.45 → kept
- canine 0.25 < 0.35 → **dropped**
- mammal 0.70 ≥ 0.20 → kept

Set going into collapse: `{dobermann, domestic_dog, mammal}` (canine already
gone).

**Step 2 - collapse.** Take each surviving tag and delete whatever it
`implies`, walking the full chain even through an already-missing tag:
- Start at **dobermann**: implies `domestic_dog` → delete it. Keep walking to
  what `domestic_dog` itself implies, `canine` → already absent, nothing to
  delete, but the walk doesn't stop there - it keeps climbing to whatever
  `canine` implies, `mammal` → **delete mammal too.**
- The loop also visits `domestic_dog` and `mammal` as separate starting
  points (they were in the original snapshot), but everything above them is
  already gone, so nothing more happens.

**Final result: `{dobermann: 0.95}`.** Mammal passed its own threshold fine,
but it still gets swept away - purely because it sits above dobermann in the
implication graph, even though the connecting rung (canine) had already
failed independently and was never present during collapse.

### `constrain-remove`

**Step 1** - same independent thresholding: `{dobermann: 0.95,
domestic_dog: 0.90, mammal: 0.70}` (canine still absent).

**Step 2** - before collapsing, clamp any tag showing *more* confidence than
a more general tag above it. This walk also passes straight through the
missing `canine` slot rather than stopping at it:
- Starting from **mammal** (0.70): it looks at what implies mammal
  (`canine`), then at what implies *that* (`domestic_dog`), then *that*
  (`dobermann`) - carrying mammal's 0.70 value all the way down.
  `domestic_dog` (0.90 > 0.70) gets clamped to 0.70. `dobermann` (0.95 >
  0.70) gets clamped to 0.70 too. `canine` itself is never resurrected -
  clamping only adjusts a tag that's already present, it doesn't add missing
  ones back.

Now everything reads: dobermann 0.70, domestic_dog 0.70, mammal 0.70.

**Step 3** - collapse, same mechanism as `remove`: dobermann's cascade
deletes domestic_dog, walks past absent canine, deletes mammal.

**Final result: `{dobermann: 0.70}`.** Same surviving tag as plain `remove`,
but the confidence number dropped from 0.95 to 0.70 - mammal's weaker signal
leaked all the way down through the broken link and dragged dobermann's
displayed number with it.

### `enforce-remove`

This mode doesn't start with independent thresholding - it starts from the
full, unfiltered set of raw probabilities and actively vetoes based on
failures anywhere in the chain.

**Step 1** - walk every known tag. When one fails its own threshold, delete
it **and cascade the deletion downward to everything more specific than it**
(the opposite direction from `remove`'s collapse):
- **canine** fails (0.25 < 0.35) → delete canine, then cascade to whatever's
  more specific than canine: `domestic_dog` → delete it too, then cascade
  further to whatever's more specific than that: `dobermann` → **delete it
  too.**
- **mammal** is checked independently and passes (0.70 ≥ 0.20) - nothing
  vetoes it, because the cascade only radiates outward from an actual
  failure, and nothing more general than mammal failed.

After this step, only `mammal: 0.70` is left from the whole family -
dobermann and domestic_dog are gone even though both individually scored
extremely well (0.95 and 0.90).

**Step 2** - collapse runs on whatever survived: just `{mammal: 0.70}`, and
mammal implies nothing above it, so nothing changes.

**Final result: `{mammal: 0.70}`.**

### Side by side

| mode | survives |
|---|---|
| `remove` | `dobermann: 0.95` |
| `constrain-remove` | `dobermann: 0.70` |
| `enforce-remove` | `mammal: 0.70` |

Same raw model output, three genuinely different captions: one tags the
specific breed at full confidence, one tags the same breed but with a
chastened confidence number, and one throws the specific breed away entirely
and only reports the generic "mammal" - because a single weak mid-chain
score was enough to veto everything more specific than it.

For contrast, the other implications modes go the opposite direction instead
of collapsing:
- **`inherit`** - boosts a general ancestor's
  displayed confidence up to match a specific descendant's, and force-keeps
  it in the output. Would show `mammal`, `canine`, `domestic_dog`, and
  `dobermann` all together, with the weaker ones bumped up toward 0.95.
- **`constrain`** (no `-remove`) - applies the same downward clamping as
  `constrain-remove` above, but skips the final collapse, so all four tags
  stay in the output at their clamped values.
- **`off`** - no hierarchy logic at all; whatever independently clears its
  own threshold is reported as-is (`dobermann`, `domestic_dog`, `mammal` in
  this example - the exact redundant-family clutter these modes exist to
  avoid).
