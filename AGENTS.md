# TokenDye - Agent Guide

## Project standards

- **Core library**: `src/tokendye/` - write code and comments in **English**.
- **Docs**: `README.md` - write in **English**.
- **Qwen3.5-specific code**: keep it under `example/Qwen3.5/`.
- **Tooling**: manage dependencies and run scripts with `uv`; prefer latest
  dependency versions. `[project].dependencies` lists only torch (the
  package's own dependency); example-only deps (transformers, accelerate,
  bitsandbytes) go into the `example` dependency group.
- **Model path**: pass the Qwen3.5-4B path through the `QWEN_MODEL_PATH`
  environment variable (default: `example/Qwen3.5/Qwen3.5-4B`).
- **No machine-specific paths**: never hardcode absolute paths such as
  `/home/...` in docs, docstrings, or code (the repo is published).
- **Linting**: use `ruff` (dev dependency group); BLE001 (blind `except`) is
  intentionally ignored.
- **Dye methods**: the project supports both post-embed (embedding-level) and
  DeepEmbed-style (per-layer) dyeing; do not present either as the only
  design.

## Design and usage

See [README.md](README.md) for the dye-method designs (including the
`1 + delta` parameterization and fp32 master weights), how to run the
Qwen3.5 example, and the training objective.
