"""The one shared "process one image" entry point - see caption_image()'s
own docstring below for why this indirection exists (it's the thing that
makes the Single-image tab, the Batch tab, and the CLI all behave
identically instead of three separately-maintained call sites). The
Hydra 3.5 second-stage tag classifier (core.hydra_classifier) hooks in
here for exactly that reason - see caption_image() below.

apply_trigger_word()'s trigger_word argument distinguishes None from ""
deliberately, and callers rely on that: None means "no override was
given, fall back to cfg.trigger_word" (the CLI's default when
--trigger-word isn't passed), while an explicit "" means "no trigger word
at all for this call, even if cfg.trigger_word is set" - which is what
lets a user clear the GUI's trigger-word field to deliberately caption
without one for a single run, without having to go change the saved
default in Settings first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

from . import hydra_classifier
from .client import CaptionResult, LlamaClient
from .config import AppConfig

log = logging.getLogger(__name__)


def apply_trigger_word(caption: str, trigger_word: str) -> str:
    trigger_word = trigger_word.strip()
    if not trigger_word:
        return caption
    return f"{trigger_word}, {caption}"


def caption_image(
    image: Union[str, Path, bytes],
    client: LlamaClient,
    cfg: AppConfig,
    trigger_word: Optional[str] = None,
    on_stage: Optional[Callable[[str], None]] = None,
    existing_caption: Optional[str] = None,
) -> tuple[str, CaptionResult, str, Optional[str]]:
    """The one shared "process one image" call - used by the Single-image
    tab, the Batch tab, and the CLI alike, so every caller sees the same
    resize/trigger-word/truncation behavior. Returns (caption, result,
    vlm_caption, hydra_tags): caption has the trigger word applied and, if
    Hydra fired, its tags appended on their own line (result.content is the
    raw, pre-trigger-word, pre-Hydra text); result is the full CaptionResult
    for callers that need more detail (token counts, timing,
    result.truncated, resize info) than just the caption string;
    vlm_caption is caption with the trigger word applied but before any
    Hydra tags were appended (identical to caption when Hydra didn't fire);
    hydra_tags is Hydra's own tag_text on its own (not folded into caption)
    when it actually fired, else None - so a caller that wants to write the
    VLM caption and Hydra's tags to separate files doesn't have to re-split
    caption's combined string back apart.

    existing_caption, if given, skips the VLM call entirely and reuses this
    text verbatim as vlm_caption - no trigger word reapplied, since callers
    pass already-final text (e.g. an adopted .txt.nlp sidecar, which is
    documented to already carry the trigger word if any). Lets a caller
    that already has a caption - from a previous run, or adopted from
    another tool - still route through Hydra tagging below without paying
    for a redundant VLM call. result is then a zero-cost stand-in
    (finish_reason="stop", no real token/timing/resize data) so callers can
    keep reading outcome.truncated/resize_note uniformly either way - see
    core.batch.run_batch's sidecar-adoption logic, the only current caller
    of this parameter.

    on_stage, if given, is called with a short label ("captioning" /
    "tagging (Hydra)") right before each of this function's two possible
    model calls - this is the hook the GUI uses to show which stage a
    caption is currently in (see app.py's _operation_set_stage). Purely
    optional and side-effect-only - core/ itself has no UI, this is just
    a plain callback so it stays framework-agnostic; the CLI simply
    doesn't pass one."""
    if existing_caption is not None:
        vlm_caption = existing_caption
        caption = vlm_caption
        result = CaptionResult(
            content=existing_caption, finish_reason="stop",
            completion_tokens=0, prompt_tokens=0, elapsed_s=0.0,
        )
    else:
        if on_stage:
            on_stage("captioning")
        result = client.caption(
            image,
            prompt=cfg.prompt_template,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            resize_enabled=cfg.resize_enabled,
            resize_target_mp=cfg.resize_target_mp,
            snap_enabled=cfg.snap_enabled,
            snap_multiple=cfg.snap_multiple,
        )
        word = cfg.trigger_word if trigger_word is None else trigger_word
        if word.strip():
            log.debug("caption_image(): applying trigger word '%s'", word.strip())
        caption = apply_trigger_word(result.content, word)
        vlm_caption = caption

    hydra_tags: Optional[str] = None
    if cfg.hydra_enabled:
        if on_stage:
            on_stage("tagging (Hydra)")
        try:
            hydra_result = hydra_classifier.classify(image, cfg)
            if hydra_result.tag_text:
                hydra_tags = hydra_result.tag_text
                caption = f"{caption}\n{hydra_tags}"
        except hydra_classifier.HydraError as exc:
            log.warning("caption_image(): Hydra classification failed, continuing without it: %s", exc)

    return caption, result, vlm_caption, hydra_tags
