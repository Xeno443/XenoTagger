"""Furrence2 worker process - runs the original 2024 Florence-2 code on
transformers 4.46.3, which can't live in the app's own process (see
install.py). Started and spoken to by worker_client.py only; never
imported by the app.

Usage: python -I worker.py --lib <dir> --model-dir <dir> [--device cuda|cpu]

Protocol: one JSON object per line. stdout is reserved for it (anything
a library prints goes to stderr instead, which the client logs to a
file).
  - After loading, the worker sends {"ready": true, "device": ...} or
    {"error": "..."} and exits.
  - Then per request line:
      {"image_path": str} or {"image_b64": str},
      "tags": [[name, category], ...]  (e621 names with underscores,
                                        best first)
      "seed": str          (shuffles the tag order reproducibly)
      "length_min"/"length_max": int  (length = clamp(60 + 2 * prompt
                                        tags, min, max))
      "prompt": str        (optional - use as-is, skipping all of the
                            above except the image; for tests)
    answered with {"caption": ..., "prompt_tokens": ..., "tags_used": ...,
    "length": ..., "seconds": ...} or {"error": "..."}.
  - EOF on stdin = exit.

Prompt format and generation settings are the Thouph/Furrence-2-Large-
Demo Space's, as verified in _reference/furrence2-test/FINDINGS.md.
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Tag categories the prompt keeps (the Space's Pruner); species go on
# their own line.
PROMPT_CATEGORIES = {"general", "species", "meta"}
# Florence-2's encoder holds 577 image tokens + the prompt in 1024
# positions - more crashes with a CUDA index assert.
PROMPT_TOKEN_BUDGET = 1024 - 577
GENERATE = dict(max_new_tokens=512, early_stopping=False, do_sample=False, num_beams=3)


def build_prompt(tags, length, seed):
    tags = [(name, cat) for name, cat in tags if cat in PROMPT_CATEGORIES]
    random.Random(seed).shuffle(tags)
    species = [name for name, cat in tags if cat == "species"]
    rest = [name for name, cat in tags if cat != "species"]
    return f"{' '.join(rest)}\nspecies: {' '.join(species)}\n\nlength: {length}\n\nSTYLE1 FURRY CAPTION:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Protocol channel = the real stdout; everything printed by libraries
    # lands on stderr.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
    sys.stdout = sys.stderr
    sys.stdin.reconfigure(encoding="utf-8")

    def send(obj):
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    # The lib dir first, so its transformers/huggingface_hub win.
    sys.path[:0] = [args.lib, str(HERE / "vendor"), str(HERE / "vendor" / "timm_stub")]

    try:
        import torch
        from PIL import Image
        from transformers import BartTokenizerFast, CLIPImageProcessor

        from florence2.modeling_florence2 import Florence2ForConditionalGeneration

        device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        t0 = time.time()
        model = Florence2ForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()
        impl_dir = str(HERE / "vendor" / "florence2")
        image_processor = CLIPImageProcessor.from_pretrained(impl_dir)
        tokenizer = BartTokenizerFast.from_pretrained(impl_dir)
    except Exception as exc:  # noqa: BLE001 - reported to the client, which shows it
        send({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    print(f"Furrence2 loaded on {device} in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    send({"ready": True, "device": device})

    def n_tokens(prompt):
        return len(tokenizer(prompt)["input_ids"])

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if "image_b64" in req:
                img = Image.open(io.BytesIO(base64.b64decode(req["image_b64"])))
            else:
                img = Image.open(req["image_path"])
            img = img.convert("RGB")

            tags = [tuple(t) for t in req.get("tags", [])]
            n_prompt_tags = sum(1 for _, cat in tags if cat in PROMPT_CATEGORIES)
            length = max(req.get("length_min", 150), min(req.get("length_max", 200), 60 + 2 * n_prompt_tags))
            kept = list(tags)
            if "prompt" in req:
                prompt = req["prompt"]
            else:
                # Tags arrive best first, so trim from the end until it fits.
                prompt = build_prompt(kept, length, req.get("seed", ""))
                while kept and n_tokens(prompt) > PROMPT_TOKEN_BUDGET:
                    kept.pop()
                    prompt = build_prompt(kept, length, req.get("seed", ""))
            prompt_tokens = n_tokens(prompt)

            pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device, dtype)
            enc = tokenizer(text=prompt, return_tensors="pt").to(device)
            t = time.time()
            with torch.inference_mode():
                ids = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=pixel_values, **GENERATE,
                )
            caption = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
            send({
                "caption": caption, "prompt_tokens": prompt_tokens,
                "tags_used": sum(1 for _, cat in kept if cat in PROMPT_CATEGORIES),
                "length": length, "seconds": round(time.time() - t, 2),
            })
        except Exception as exc:  # noqa: BLE001 - one bad image mustn't kill the worker
            print(f"request failed: {exc!r}", file=sys.stderr, flush=True)
            send({"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
