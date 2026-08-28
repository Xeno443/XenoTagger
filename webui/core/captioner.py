"""Turns a single image into a caption string, applying the trigger word."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .client import LlamaClient
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
) -> str:
    result = client.caption(
        image,
        prompt=cfg.prompt_template,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )
    word = cfg.trigger_word if trigger_word is None else trigger_word
    if word.strip():
        log.debug("caption_image(): applying trigger word '%s'", word.strip())
    return apply_trigger_word(result.content, word)
