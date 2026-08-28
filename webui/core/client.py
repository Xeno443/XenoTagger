"""HTTP client for a running llama-server's OpenAI-compatible chat API."""

from __future__ import annotations

import base64
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import requests

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

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _image_to_data_url(image: Union[str, Path, bytes], mime_hint: str = "") -> str:
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
    else:
        mime = mime_hint or "image/png"
        data = image
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
    ) -> CaptionResult:
        image_desc = image if isinstance(image, (str, Path)) else f"<{len(image)} bytes>"
        data_url = _image_to_data_url(image)
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
            )
        except (KeyError, IndexError, ValueError) as exc:
            log.error("caption(): unexpected response shape from %s: %s", self.base_url, resp.text[:500])
            raise ClientError(f"Unexpected response from llama-server: {resp.text[:500]}") from exc
