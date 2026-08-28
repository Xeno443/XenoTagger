"""HTTP client for a running llama-server's OpenAI-compatible chat API,
plus the in-memory image preprocessing applied before a request goes out.

LlamaClient.caption() is the single place in the whole app that actually
talks to llama-server - core/captioner.py, core/batch.py, and the
Single-image tab (via captioner.caption_image, see core/captioner.py's
own docstring for why that indirection exists) all funnel through it, so
request-shape changes (new sampling params, a different API path) only
ever need to happen here.

CaptionResult.truncated (finish_reason == "length") is easy to
misinterpret: llama-server clamps actual generation to whatever fits in
(context_size - prompt_tokens) even when that's below the requested
max_tokens, so a truncated result does NOT always mean "raise max_tokens
would fix it" - it can equally mean the prompt (a large image can burn
thousands of tokens depending on the vision encoder) left no room at all.
Distinguish by comparing completion_tokens against the requested
max_tokens (if generation stopped short of what was asked for, the
context was the real ceiling, not the token limit) - see app.py's
run_single_ui for the actual diagnosis logic shown to the user.

_resize_for_target_mp() is why an oversized source image doesn't
necessarily crash the request-shaping budget above: when resize_enabled,
it downscales in memory (a fresh bytes buffer - the source file on disk
is only ever read, never written) to about resize_target_mp megapixels
before it's ever base64-encoded, and only ever shrinks, never upscales.
When snap_enabled, it goes a step further for LoRA-training use: the
shorter side is resized (one uniform scale factor, so nothing gets
distorted) to the nearest multiple of snap_multiple, and the longer side
is then center-cropped - never stretched - down to its own clean
multiple, which is what SDXL and similar trainers expect of their input
dimensions.

This module keeps no state between calls beyond what's set once at
LlamaClient construction (base_url, timeout) - each caption() call is
independent and safe to make from a different thread than the one that
built the client, which is how the app already uses it: a batch run's
worker thread (core/batch.py) makes its sequential run of caption() calls
on the same LlamaClient instance the GUI thread obtained via
app.py's get_client(), while the GUI thread itself is off polling a
progress queue, not calling caption() at the same time.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import requests
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300


class ClientError(RuntimeError):
    pass


@dataclass
class CaptionResult:
    content: str
    finish_reason: str  # "stop" (finished naturally) or "length" (hit max_tokens)
    completion_tokens: int
    prompt_tokens: int
    elapsed_s: float
    tokens_per_second: Optional[float] = None
    resize_note: Optional[str] = None  # e.g. "4032x3024 (12.2MP) -> 1224x912 (1.0MP)"

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _resize_for_target_mp(
    data: bytes,
    target_mp: float,
    snap_enabled: bool = True,
    snap_multiple: int = 64,
) -> tuple[bytes, Optional[str]]:
    """Downscale image bytes to about target_mp megapixels, in memory.
    Returns the original bytes unchanged (and None) if already at or under
    the target - only ever downscales, never upscales.

    If snap_enabled, also snaps both dimensions to a multiple of
    snap_multiple (what SDXL and similar trainers expect): the shorter
    side by resizing (one uniform scale factor for the whole image, so
    nothing gets stretched), the longer side by center-cropping the excess
    rather than squashing it to fit. If disabled, both sides are just
    resized straight to target_mp with no snapping/cropping."""
    with Image.open(io.BytesIO(data)) as img:
        orig_w, orig_h = img.size
        orig_mp = orig_w * orig_h / 1_000_000
        if orig_mp <= target_mp:
            return data, None

        scale = (target_mp / orig_mp) ** 0.5

        if not snap_enabled or snap_multiple <= 1:
            final_w, final_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
            result = img.convert("RGB").resize((final_w, final_h), Image.Resampling.LANCZOS)
        else:
            d = snap_multiple
            scaled_w, scaled_h = orig_w * scale, orig_h * scale

            # Resize by the shorter side, rounded to the *nearest* multiple
            # of d (hitting target_mp exactly was never the point) - a
            # single scale factor for both axes, so nothing gets distorted.
            if scaled_w <= scaled_h:
                short_target = max(d, round(scaled_w / d) * d)
                resize_scale = short_target / orig_w
            else:
                short_target = max(d, round(scaled_h / d) * d)
                resize_scale = short_target / orig_h

            resized_w = max(1, round(orig_w * resize_scale))
            resized_h = max(1, round(orig_h * resize_scale))
            resized = img.convert("RGB").resize((resized_w, resized_h), Image.Resampling.LANCZOS)

            # Crop the longer side down to its own clean multiple of d -
            # floor only (never up), since cropping can only remove pixels.
            final_w = max(d, (resized_w // d) * d)
            final_h = max(d, (resized_h // d) * d)
            left, top = (resized_w - final_w) // 2, (resized_h - final_h) // 2
            result = resized.crop((left, top, left + final_w, top + final_h))

        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=92)

        new_mp = final_w * final_h / 1_000_000
        note = f"{orig_w}x{orig_h} ({orig_mp:.1f}MP) -> {final_w}x{final_h} ({new_mp:.1f}MP)"
        return buf.getvalue(), note


def _image_to_data_url(
    image: Union[str, Path, bytes],
    mime_hint: str = "",
    resize_enabled: bool = False,
    resize_target_mp: float = 1.0,
    snap_enabled: bool = True,
    snap_multiple: int = 64,
) -> tuple[str, Optional[str]]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
    else:
        mime = mime_hint or "image/png"
        data = image

    resize_note = None
    if resize_enabled:
        data, resize_note = _resize_for_target_mp(data, resize_target_mp, snap_enabled, snap_multiple)
        if resize_note:
            mime = "image/jpeg"

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}", resize_note


class LlamaClient:
    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def caption(
        self,
        image: Union[str, Path, bytes],
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 300,
        resize_enabled: bool = False,
        resize_target_mp: float = 1.0,
        snap_enabled: bool = True,
        snap_multiple: int = 64,
    ) -> CaptionResult:
        image_desc = image if isinstance(image, (str, Path)) else f"<{len(image)} bytes>"
        data_url, resize_note = _image_to_data_url(
            image,
            resize_enabled=resize_enabled,
            resize_target_mp=resize_target_mp,
            snap_enabled=snap_enabled,
            snap_multiple=snap_multiple,
        )
        if resize_note:
            log.info("caption(): resized %s: %s", image_desc, resize_note)
        log.debug(
            "caption(): POST %s/v1/chat/completions image=%s prompt_len=%d temp=%s top_p=%s max_tokens=%s",
            self.base_url, image_desc, len(prompt), temperature, top_p, max_tokens,
        )
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        start = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            log.error("caption(): request to %s failed after %.1fs: %s", self.base_url, time.monotonic() - start, exc)
            raise ClientError(f"Could not reach llama-server at {self.base_url}: {exc}") from exc
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            log.error("caption(): HTTP %s from %s: %s", resp.status_code, self.base_url, resp.text[:500])
            raise ClientError(
                f"llama-server returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"].strip()
            finish_reason = choice.get("finish_reason", "stop")
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            completion_tokens = usage.get("completion_tokens", 0)

            if finish_reason == "length":
                log.warning(
                    "caption(): truncated at %d tokens (max_tokens=%s) after %.1fs - "
                    "raise Max tokens in Settings if captions keep getting cut off",
                    completion_tokens, max_tokens, elapsed,
                )
            else:
                log.debug(
                    "caption(): finished in %.1fs, %d completion tokens, %s tok/s",
                    elapsed, completion_tokens, timings.get("predicted_per_second"),
                )

            return CaptionResult(
                content=content,
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                prompt_tokens=usage.get("prompt_tokens", 0),
                elapsed_s=elapsed,
                tokens_per_second=timings.get("predicted_per_second"),
                resize_note=resize_note,
            )
        except (KeyError, IndexError, ValueError) as exc:
            log.error("caption(): unexpected response shape from %s: %s", self.base_url, resp.text[:500])
            raise ClientError(f"Unexpected response from llama-server: {resp.text[:500]}") from exc
