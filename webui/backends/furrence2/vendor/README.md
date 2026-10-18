Vendored for the Furrence2 worker (`../worker.py`), never imported by the
app itself - the worker puts these folders on its own `sys.path`.

- `florence2/` - Microsoft's original 2024 Florence-2 remote code
  (Apache 2.0, see the file headers), as shipped in the
  Thouph/Furrence-2-Large-Demo Space's `florence2_implementation/`.
  Unmodified. Only the files the worker uses: the model/config code, and
  the stock tokenizer + image processor configs (the Furrence2 model repo
  has neither). Runs on transformers 4.46.3 only - see `../install.py`.
- `timm_stub/` - DropPath + trunc_normal_, the only timm pieces
  `modeling_florence2.py` imports.
