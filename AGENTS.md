# TokenDye - Agent Guide

## Project standards

- **Core library**: `src/tokendye/` - write code and comments in **English**.
- **Docs**: `README.md` - write in **English**.
- **Qwen3.5-specific code**: keep it under `example/Qwen3.5/`; generic
  helpers (model loading, training / GRPO-DAPO machinery) live in
  `src/tokendye/`.
- **Tooling**: manage dependencies and run scripts with `uv`; prefer latest
  dependency versions. `[project].dependencies` keeps only torch; HF deps
  (transformers, accelerate, bitsandbytes) go into the `hf` extra, Qwen3.5
  kernels into the `qwen3_5` extra (which depends on `tokendye[hf]`). Run
  examples with `uv run --extra hf`.
- **Model path**: pass the Qwen3.5-4B path through the `QWEN_MODEL_PATH`
  environment variable (default: `example/Qwen3.5/Qwen3.5-4B`).
- **No machine-specific paths**: never hardcode absolute paths such as
  `/home/...` in docs, docstrings, or code (the repo is published).
- **Linting**: use `ruff` (dev dependency group); BLE001 (blind `except`) is
  intentionally ignored.
- **No logging in the library**: `src/tokendye/` must not contain any
  logging-related code. Dev scripts log directly with `loguru` (real-time
  stderr + file sinks).
- **No domain rewards in the library**: reward/judge logic is task-specific;
  keep it in `example/Qwen3.5/` (e.g. `judge.py`), not in `src/tokendye/`.
- **Dye methods**: the project supports both post-embed (embedding-level) and
  per-layer dyeing; do not present either as the only
  design.
- **RL rollout memory**: generate with `past_key_values` (KV cache); a
  full-sequence recompute per token OOMs at long lengths (Qwen3.5 also
  materializes seq^2 attention in its linear-attention torch fallback).
- **Data validation**: malformed training/attack data must raise an error
  before any training starts (never train silently on empty targets).

## Design and usage

See [README.md](README.md) for the dye-method designs (including the
`1 + delta` parameterization and fp32 master weights), how to run the
Qwen3.5 example, and the training objective.
