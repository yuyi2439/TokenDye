"""Generic training helpers: data, SFT losses, and GRPO/DAPO optimization."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

__all__ = [
    "collate",
    "find_answer_start",
    "generate_group",
    "grpo_loss",
    "has_learning_signal",
    "masked_cross_entropy",
    "next_token_labels",
    "shape_overlong_reward",
    "teacher_logps",
    "tokenize_chat",
]


def _logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"cannot get logits from model output: {type(output)}")


def tokenize_chat(
    tokenizer,
    messages: list[dict],
    spans: list[tuple[int, str]],
    max_seq_len: int,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Render chat messages and label tokens by character spans.

    Args:
        messages: chat messages, e.g. [{"role": "user", "content": "..."}].
        spans: (label_id, content) pairs; tokens inside a content span get
            that label, everything else gets -1 (undyed).
        add_generation_prompt: append the generation prompt after the messages.

    Returns:
        (input_ids, labels) truncated to max_seq_len.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)

    label_spans = []
    for label_id, content in spans:
        start = prompt.index(content)
        label_spans.append((label_id, start, start + len(content)))

    labels = []
    for s, e in enc["offset_mapping"]:
        label = -1
        for label_id, cs, ce in label_spans:
            if cs <= s and e <= ce:
                label = label_id
                break
        labels.append(label)
    return enc["input_ids"][:max_seq_len], labels[:max_seq_len]


def collate(
    batch: list[tuple[list[int], list[int]]],
    pad_id: int,
    max_seq_len: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Pad a batch of (input_ids, dye_labels) samples."""
    max_len = min(max_seq_len, max(len(ids) for ids, _ in batch))
    input_ids, dye_mask, attention_mask = [], [], []
    for ids, labels in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        dye_mask.append(labels + [-1] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "dye_mask": torch.tensor(dye_mask, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def masked_cross_entropy(logits, input_ids, mask) -> torch.Tensor:
    """Next-token cross-entropy restricted to masked positions."""
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_labels = input_ids[:, 1:].reshape(-1)
    shift_mask = mask[:, 1:].reshape(-1)
    if shift_mask.sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(shift_logits[shift_mask], shift_labels[shift_mask])


def next_token_labels(
    next_ids: torch.Tensor,
    in_think: torch.Tensor,
    think_end_id: int | None,
    think_label_id: int | None,
    response_label_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label newly generated tokens: think before `</think>`, response after.

    The `</think>` token itself keeps the think label; `in_think` flips to
    False afterwards. Without think ids, everything gets the response label.
    """
    if think_end_id is None or think_label_id is None:
        return torch.full_like(next_ids, response_label_id), in_think
    is_end = next_ids.squeeze(-1) == think_end_id
    labels = torch.where(
        in_think,
        torch.full_like(next_ids, think_label_id),
        torch.full_like(next_ids, response_label_id),
    )
    in_think = in_think & ~is_end
    return labels, in_think


def generate_group(
    model,
    controller,
    prompt_ids: torch.Tensor,
    prompt_labels: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    eos_id: int,
    response_label_id: int,
    think_end_id: int | None = None,
    think_label_id: int | None = None,
    progress_every: int = 0,
    on_progress=None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Sample a group of responses with the dye, stopping each row at EOS.

    Uses `past_key_values` (KV cache) so each step only computes the new
    token; without it, long generations OOM on the full-sequence attention.

    Returns:
        (gen_ids, gen_labels, lengths) where lengths are per-row generated
        token counts before the first EOS.
    """
    device = prompt_ids.device
    gen_ids = prompt_ids.expand(group_size, -1).clone()
    gen_labels = prompt_labels.expand(group_size, -1).clone()
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    in_think = torch.ones(group_size, dtype=torch.bool, device=device)
    past_key_values = None
    step_ids = gen_ids
    step_labels = gen_labels

    for _ in range(max_new_tokens):
        controller.set_labels(step_labels)
        with torch.no_grad():
            out = model(input_ids=step_ids, past_key_values=past_key_values, use_cache=True)
            logits = _logits(out)[:, -1, :].float()
            past_key_values = out.past_key_values
        if finished.any():
            logits[finished] = float("-inf")
            logits[finished, eos_id] = 0.0
        dist = torch.distributions.Categorical(logits=logits / temperature)
        next_ids = dist.sample().unsqueeze(-1)
        finished |= next_ids.squeeze(-1) == eos_id
        gen_ids = torch.cat([gen_ids, next_ids], dim=1)
        labels_t, in_think = next_token_labels(
            next_ids,
            in_think,
            think_end_id,
            think_label_id,
            response_label_id,
        )
        gen_labels = torch.cat([gen_labels, labels_t], dim=1)
        step_ids = next_ids
        step_labels = labels_t
        if finished.all():
            break
        if (
            progress_every
            and on_progress is not None
            and (len(gen_ids[0]) - prompt_ids.shape[1]) % progress_every == 0
        ):
            on_progress(len(gen_ids[0]) - prompt_ids.shape[1], gen_ids.shape[1])
    controller.clear_labels()

    prompt_len = prompt_ids.shape[1]
    lengths = []
    for i in range(group_size):
        eos_pos = (gen_ids[i, prompt_len:] == eos_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel():
            first = prompt_len + int(eos_pos[0])
            gen_labels[i, first + 1 :] = -1  # tokens after EOS are padding for the loss
            lengths.append(int(eos_pos[0]))
        else:
            lengths.append(max_new_tokens)
    return gen_ids, gen_labels, lengths


def find_answer_start(
    gen_ids: torch.Tensor,
    prompt_len: int,
    think_end_id: int,
) -> torch.Tensor:
    """Absolute token index where the final answer begins (after `</think>`).

    Rows that never emit `</think>` get -1 (no final answer yet).
    """
    starts = torch.full((gen_ids.shape[0],), -1, dtype=torch.long, device=gen_ids.device)
    for i in range(gen_ids.shape[0]):
        hit = (gen_ids[i, prompt_len:] == think_end_id).nonzero(as_tuple=True)[0]
        if hit.numel():
            starts[i] = prompt_len + int(hit[0]) + 1
    return starts


def teacher_logps(
    model,
    controller,
    input_ids: torch.Tensor,
    dye_mask: torch.Tensor,
    enabled_grad: bool,
    response_label_id: int,
    answer_start: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of the generated sequence (teacher forcing)."""
    controller.set_labels(dye_mask)
    if enabled_grad:
        logits = _logits(model(input_ids=input_ids))
    else:
        with torch.no_grad():
            logits = _logits(model(input_ids=input_ids))
    controller.clear_labels()
    logps = (
        F.log_softmax(logits[:, :-1].float(), dim=-1)
        .gather(2, input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    mask = dye_mask[:, 1:] == response_label_id
    if answer_start is not None:
        positions = torch.arange(1, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        mask = mask & (positions >= answer_start.unsqueeze(1))
    return logps, mask


def grpo_loss(
    logps_theta: torch.Tensor,
    logps_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    clip_eps: float,
    clip_higher: bool = True,
) -> torch.Tensor:
    """Token-level clipped policy loss with a KL penalty.

    `clip_higher` applies DAPO's decoupled clipping: positive advantages are
    not upper-clipped, so high-reward tokens can keep gaining probability.
    """
    ratio = torch.exp(logps_theta - logps_ref)
    adv = advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    if clip_higher:
        pg = torch.where(adv > 0, -ratio * adv, -torch.minimum(ratio * adv, clipped * adv))
    else:
        pg = -torch.minimum(ratio * adv, clipped * adv)
    kl = torch.exp(logps_ref - logps_theta) - (logps_ref - logps_theta) - 1
    loss = (pg + beta * kl) * mask
    return loss.sum() / mask.sum()


def has_learning_signal(rewards: torch.Tensor, min_std: float = 1e-3) -> bool:
    """DAPO dynamic sampling: drop groups whose rewards are all the same."""
    return rewards.std().item() > min_std


def shape_overlong_reward(reward: float, length: int, max_len: int) -> float:
    """DAPO-style cosine shaping: overlong negative rewards get partial credit."""
    if reward >= 0 or length < max_len:
        return reward
    t = min((length - max_len) / (0.2 * max_len), 1.0)
    return -0.5 - 0.5 * (1 - math.cos(math.pi * t)) / 2
