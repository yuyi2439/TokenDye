# TokenDye

A lightweight fine-tuning scheme that gives an LLM "identity labels" by
dyeing its hidden states with per-role signals. The dye method is not fixed;
the project supports both approaches:

- **post-embed**: dye token embeddings once at the embedding layer (the
  original approach).
- **per-layer**: scale the hidden states of every transformer layer
  element-wise with per-(role, layer) vectors, parameterized as
  `scale = 1 + delta`, i.e.
  `hidden[layer] = hidden[layer] * (1 + delta[label][layer])`. `delta` is
  initialized to zeros (identity), stored as an fp32 master weight and cast
  to the hidden dtype on forward. Parameter cost (4 labels x 32 layers x 4096
  dims): `4 * 32 * 4096 = 524,288` params ~= 1 MB in FP16.

The current example demonstrates the per-layer approach; post-embed may
be used as well.

## Example (Qwen3.5-4B)

`[project].dependencies` keeps only torch. HF deps (transformers, accelerate,
bitsandbytes) live in the `hf` extra, and the Qwen3.5 kernels
(flash-linear-attention, causal-conv1d) in the `qwen3_5` extra (which depends
on `tokendye[hf]`). Run examples with `uv run --extra hf`.

```bash
# Run the example; pass the model path through the environment
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --extra hf python example/Qwen3.5/dye_qwen.py

# Optionally fine-tune the dye parameters for a few steps
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --extra hf python example/Qwen3.5/dye_qwen.py --train --steps 3
```

Loading strategy: on CUDA, tries bitsandbytes 4-bit (nf4) first, then falls
back to bf16 GPU, then CPU. Without `QWEN_MODEL_PATH`, the example looks for
`Qwen3.5-4B` next to the script (`example/Qwen3.5/Qwen3.5-4B`).

Training outputs go to `example/Qwen3.5/.outputs/<launch-time>/` (log +
checkpoints). Metric meanings are documented (in Chinese) in
[docs/training_metrics.md](docs/training_metrics.md).

## Training (Qwen3.5)

Baseline objective:

    L = CE(response) + w_think * CE(think)
        + w_anchor * ||h_dyed - h_base||^2 (prompt tokens)
        + wd * ||delta||^2                  (optimizer weight decay)

- `CE(response)`: masked cross-entropy on the response segment (the only
  supervised target; CoT is not part of the training data).
- Anchor: MSE between the dyed and frozen-base last-layer hidden states on
  prompt tokens, so non-target text is not disturbed.
- Weight decay on `delta` pulls the effective scale back toward 1.

Data format (JSONL, one sample per line):

    {"segments": [{"role": "system|user|assistant|tool", "content": "..."}], "response": "..."}

`segments` hold the context (multi-turn supported: user/assistant turns can
alternate); `response` is the only supervised target. `tool` and historical
`assistant` segments render as messages and stay undyed. Malformed data
raises an error before training starts. Note:
`example/Qwen3.5/attacks.jsonl` is the DAPO attack set, not SFT data.


Run:

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --extra hf python example/Qwen3.5/train.py --data train.jsonl --steps 200

# few-step smoke run on the same data
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --extra hf python example/Qwen3.5/train.py --data dataset/v0.2.jsonl --smoke --steps 3
```

## Reinforcement (DAPO, Qwen3.5)

After SFT, optimize the dye directly for defense outcomes with DAPO (an
improved GRPO: Clip-Higher, Dynamic Sampling, token-level loss, and overlong
reward shaping; no critic):

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    rho_t = exp(logp_theta - logp_ref)
    L = -mean[min(rho_t*A_i, clip(rho_t, 1-eps, 1+eps)*A_i)] + beta*KL(pi_theta || pi_ref)

Reward v0 is rule-based: marker leaked / injection followed / role switched =
-1; benign answered = +1; otherwise 0.
Judging and the policy loss use only the **final answer** (tokens after the
`</think>` boundary); the chain-of-thought is not supervised.

Attack samples (JSONL):

    {"system": "...", "user": "...", "attack_type": "injection|extraction|role_switch|benign", "payload": "...", "expected": "...", "marker": "..."}

Run:

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --extra hf python example/Qwen3.5/train_dapo.py --init-delta example/Qwen3.5/.outputs/dye_final.pt --steps 20

# smoke test with built-in synthetic samples
QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --extra hf python example/Qwen3.5/train_dapo.py --smoke --steps 2
```
