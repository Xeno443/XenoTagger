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
from typing import Optional, Union

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
) -> tuple[str, CaptionResult]:
    """The one shared "process one image" call - used by the Single-image
    tab, the Batch tab, and the CLI alike, so every caller sees the same
    resize/trigger-word/truncation behavior. Returns (caption, result):
    caption has the trigger word applied (result.content is the raw,
    pre-trigger-word text); result is the full CaptionResult for callers
    that need more detail (token counts, timing, result.truncated, resize
    info) than just the caption string."""
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

    if cfg.hydra_enabled:
        try:
            hydra_result = hydra_classifier.classify(image, cfg)
            if hydra_result.tag_text:
                caption = f"{caption}, {hydra_result.tag_text}"
        except hydra_classifier.HydraError as exc:
            log.warning("caption_image(): Hydra classification failed, continuing without it: %s", exc)

    return caption, result
