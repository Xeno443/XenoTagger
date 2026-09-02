# Hydra Confidence & Threshold FAQ

Plain-English notes on what Settings → Hydra's **Confidence** and
**Threshold** sliders actually do. Companion to `faq-hydra-implications.md`
(which covers the separate "Implications mode" setting - nested/redundant
tag families like `canine` + `dog` + `dobermann`). This one is about how an
individual tag gets included or excluded in the first place.

## The basic idea

For every possible tag (`dobermann`, `blue_fur`, whatever), Hydra computes a
confidence score. It doesn't dump every tag it has any opinion on into your
caption - each tag has a cutoff, and only tags that beat their own cutoff
get added. The two sliders are two different ways of controlling where that
cutoff sits.

## Confidence slider - how picky Hydra should be, overall

Think of it like a spam filter's sensitivity dial.

- Push it **up** → Hydra gets stricter. Fewer tags show up, but the ones
  that do are ones it's genuinely sure about. Risk: it might miss some real
  tags it wasn't quite confident enough about.
- Push it **down** → Hydra gets looser. More tags show up, including some
  it's only somewhat sure of. Risk: more wrong/junk tags sneak into your
  captions.

Under the hood this is the `beta` in an F-beta score (precision/recall
balance) - XenoTagger's slider maps directly to it (Confidence value *is*
beta, no hidden curve). Low beta weights precision (fewer, safer tags);
high beta weights recall (catch more, tolerate more noise).

## Threshold slider - a hard safety minimum, separate from the dial above

This one says: "no matter how the Confidence dial is set, never let a tag
through if, historically, it's been wrong more than X% of the time." At
`0.10`, that means: never let a tag through whose track record is worse
than "right at least 9 times out of 10." It's a backstop that catches a
specific failure case - a tag with almost no supporting examples that the
Confidence dial alone might rate too generously.

Three examples of what it actually does:

1. **A common, well-behaved tag (like "dog")** - Hydra's seen thousands of
   dogs and separates dog/not-dog cleanly. Even a loose cutoff is right
   ~95% of the time. Raising Threshold from 10% to 50% changes nothing for
   this tag - it was never close to failing.
2. **A rare, ambiguous tag** - Hydra's seen only a handful of true examples
   and its scores overlap between true/false cases. At whatever cutoff the
   Confidence dial would normally pick, it might only be right ~8% of the
   time. That's below a 10% floor, so Hydra is forced to demand a much
   higher score before showing this tag at all - it still might show up,
   just rarely, only on images it's very sure about.
3. **The same rare tag, Threshold raised to 30%** - now even the strictest
   possible cutoff for this tag can't reach 30% precision. The tag gets
   **permanently switched off** - it will never appear on any image, no
   matter how high a raw score it gets that day. This is the difference
   from Confidence: Confidence makes a tag *rarer*; a high enough Threshold
   makes it *disappear entirely*.

## How does Hydra "know" its accuracy?

It's not guessing - it's counting against an answer key.

1. **The quiz.** When the model was trained, a batch of images was set
   aside that the model never trained on - the *validation set* - and the
   correct tags for every one of those images are already known
   (human-verified e621 data).
2. **Every tag takes the quiz.** For each tag, the model scored every
   validation image, producing a list of (score, actually-has-this-tag?)
   pairs. This list is saved inside the model file itself, one per tag -
   it's not recomputed at caption time, it's already there.
3. **Try candidate cutoffs.** For a given tag, Hydra tries a bunch of
   candidate score cutoffs against that saved list.
4. **Just count.** For each candidate cutoff, it counts: of the quiz images
   that would've scored above it, how many were actually correct? That
   ratio is the measured precision at that cutoff - real counting against
   known answers, not an estimate.
5. **Threshold discards, Confidence picks.** Any candidate cutoff whose
   counted precision falls below the Threshold floor gets thrown out.
   Among what's left, the Confidence dial's beta picks the best balance of
   catching true positives vs. avoiding false ones. That winner becomes the
   tag's real cutoff for every future image.

Why this varies so much tag to tag: a well-behaved tag has a clean score
gap between true and false images, so almost any cutoff gives good
precision. A rare/confusing tag has overlapping scores - there's no clean
line to draw, so every cutoff is a bad trade-off between missing true
positives and letting in false ones. That's exactly what the Threshold
floor is built to catch.

## Why aren't low-precision tags dropped automatically?

The default floor (10%) is deliberately lenient, not an oversight. A tag
that scores 8% precision on the validation quiz isn't necessarily a bad
tag - it might just be a rare tag with very few quiz questions about it,
and small samples are noisy. A strict default floor (say 50%) would nuke a
mix of genuinely-bad tags and genuinely-fine-but-rare tags with no way to
tell which is which. So the judgment call of "how much noise am I willing
to tolerate to keep rare tags alive" is left to the user via the slider,
rather than baked in.

If your goal is tags that complement a caption without swamping it (rather
than casting the widest possible net), low-precision tags genuinely are
closer to noise for that goal - raising Threshold, in addition to raising
Confidence, is the right move.

## What we found testing against a real image

A sweep across `hydra_metric` values on one real caption image (with
`hydra_implications = "remove"`, see `faq-hydra-implications.md`) showed:

- **Threshold barely moved the tag count** (a handful of tags at most,
  across `0.1`–`0.2`) - this particular image just didn't have many
  genuinely flaky tags for the floor to catch.
- **Confidence (beta) was the real lever** - tag count ranged from ~35
  (`beta 0.3`, tight/confident) up to ~91 (`beta 1.1`, loose).
- **Contradictory tags reappeared above ~beta 0.9** - e.g. `male
  penetrating female` showing up alongside clearly `male/male` content.
  That's a concrete sign the looser end isn't just "more detail," it's
  genuine noise.
- `beta 0.5` (XenoTagger's new default) landed in a range (35–52 tags) that
  reads as complementary detail rather than a wall of tags, without the
  contradictions seen further up the range.

## The sliders in Settings → Hydra

- **Confidence**: `0.1`–`1.5`, step `0.05`, default `0.5`. Low end is
  near-pure-precision ("only the obvious stuff"); high end pushes past
  where noise/contradictions start appearing - a "go crazy" ceiling more
  than a recommended one.
- **Threshold**: `0.0`–`0.5`, step `0.05`, default `0.1`. `0` = no floor at
  all; `0.5` = demand at least 50% measured precision, aggressive enough to
  silence most rare/thin-data tags.

The slider values map directly onto the `f<beta>@<min_precision>` string
Hydra actually uses - no hidden curve, what you see is what gets sent.
