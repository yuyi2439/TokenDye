# TokenDye

A lightweight fine-tuning scheme that gives an LLM "identity labels" by
dyeing its hidden states with per-role signals. The dye method is not fixed;
the project supports both approaches:

- **post-embed**: dye token embeddings once at the embedding layer (the
  original approach).
- **DeepEmbed-style**: scale the hidden states of every transformer layer
  element-wise with per-(role, layer) vectors, parameterized as
  `scale = 1 + delta`, i.e.
  `hidden[layer] = hidden[layer] * (1 + delta[label][layer])`. `delta` is
  initialized to zeros (identity), stored as an fp32 master weight and cast
  to the hidden dtype on forward. Parameter cost (4 labels x 32 layers x 4096
  dims): `4 * 32 * 4096 = 524,288` params ~= 1 MB in FP16.

The current example demonstrates the DeepEmbed-style approach; post-embed may
be used as well.

## Example (Qwen3.5-4B)

The package itself only depends on torch. The Qwen3.5 example additionally
needs transformers / accelerate / bitsandbytes, which live in the `example`
dependency group; run it with `--group example` (uv installs them on demand).

```bash
# Run the example; pass the model path through the environment
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --group example python example/Qwen3.5/deep_embed_qwen.py

# Optionally fine-tune the dye parameters for a few steps
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --group example python example/Qwen3.5/deep_embed_qwen.py --train --steps 3
```

Loading strategy: on CUDA, tries bitsandbytes 4-bit (nf4) first, then falls
back to bf16 GPU, then CPU. Without `QWEN_MODEL_PATH`, the example looks for
`Qwen3.5-4B` next to the script (`example/Qwen3.5/Qwen3.5-4B`).
