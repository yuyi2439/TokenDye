"""TokenDye DAPO training skeleton: outcome-based defense optimization.

DAPO (an improved GRPO) with:
    - Clip-Higher: positive advantages are not upper-clipped
    - Dynamic Sampling: skip groups whose rewards are all identical
    - Token-level policy gradient loss
    - Overlong reward shaping for responses that hit the length cap

Objective (token-level):

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    rho_t = exp(logp_theta(t) - logp_ref(t))
    L = -mean[ clip_higher(rho_t, A_i) ] + beta * KL(pi_theta || pi_ref)

Reward v0 (rule-based): see example/Qwen3.5/judge.py.

Usage:
    QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run python \\
        example/Qwen3.5/train_dapo.py --steps 20 --group-size 4
    QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run python \\
        example/Qwen3.5/train_dapo.py --attacks attacks.jsonl --smoke --steps 2
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DEFAULT_MODEL_PATH, LABELS, RESPONSE_ID
from judge import judge

from tokendye import Dye, install_dye
from tokendye.hf import ensure_python_headers, load_model
from tokendye.training import (
    find_answer_start,
    generate_group,
    grpo_loss,
    has_learning_signal,
    shape_overlong_reward,
    teacher_logps,
    tokenize_chat,
)


def load_attacks(path: Path | None) -> list[dict]:
    if path is None:
        raise SystemExit("请提供外部攻击数据：--attacks attacks.jsonl")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_attacks(attacks: list[dict]) -> None:
    """攻击样本必须包含 user 和合法的 attack_type。"""
    known = {"injection", "extraction", "role_switch", "benign"}
    for i, sample in enumerate(attacks, 1):
        if not sample.get("user"):
            raise SystemExit(f"攻击样本 {i} 缺少 user 字段")
        if sample.get("attack_type") not in known:
            raise SystemExit(
                f"攻击样本 {i} 的 attack_type 非法：{sample.get('attack_type')!r}"
                f"（应为 {sorted(known)}）"
            )


def main() -> None:
    ensure_python_headers()
    parser = argparse.ArgumentParser(
        description="TokenDye DAPO 训练骨架（结果导向的防御优化）",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="模型路径（也可用环境变量 QWEN_MODEL_PATH 传入）",
    )
    parser.add_argument("--attacks", type=Path, default=Path("example/Qwen3.5/attacks.jsonl"))
    parser.add_argument("--smoke", action="store_true", help="使用外部攻击数据执行冒烟测试")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4, help="DAPO 每组采样数 G")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01, help="KL 惩罚系数")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example/Qwen3.5/.outputs"),
        help="输出根目录，每次运行自动创建时间戳子目录",
    )
    parser.add_argument("--init-delta", type=Path, default=None, help="SFT 产出的 dye state_dict")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise SystemExit(f"模型目录不存在：{args.model_path}\n请设置 QWEN_MODEL_PATH")
    if args.temperature <= 0:
        raise SystemExit("--temperature 必须大于 0")
    # Fail fast on bad --attacks/--smoke before paying the model-loading cost.
    attacks = load_attacks(args.attacks)
    validate_attacks(attacks)

    run_dir = args.output_dir / datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {message}", colorize=False)
    logger.add(
        run_dir / "train.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        encoding="utf-8",
    )
    logger.info("{}", f"output dir: {run_dir}")

    logger.info(
        "{}",
        f"config: steps={args.steps} G={args.group_size} "
        f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} "
        f"lr={args.lr} wd={args.weight_decay} beta={args.beta} "
        f"clip_eps={args.clip_eps} seed={args.seed} init_delta={args.init_delta}",
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("{}", f"device: {device}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model(args.model_path, device)
    model.requires_grad_(False)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    text_cfg = getattr(model.config, "text_config", model.config)
    dye = Dye(LABELS, text_cfg.num_hidden_layers, text_cfg.hidden_size).to(model.device)
    if args.init_delta is not None:
        dye.load_state_dict(torch.load(args.init_delta, map_location=model.device))
        logger.info("{}", f"loaded delta from {args.init_delta}")
    controller = install_dye(model, dye)

    ref_dye = copy.deepcopy(dye)
    ref_dye.requires_grad_(False)
    ref_controller = install_dye(model, ref_dye)

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    logger.info("{}", f"attacks: {len(attacks)} (file={args.attacks}, smoke={args.smoke})")
    logger.info("{}", f"dye params: {dye.num_parameters:,}")

    optimizer = torch.optim.AdamW(dye.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    update_count = 0
    attempt = 0
    skipped = 0
    total_reward = 0.0
    t0 = time.time()
    while update_count < args.steps:
        if attempt >= max(args.steps * 20, 100):
            logger.info("{}", f"too many uniform groups ({skipped} skipped); stopping early")
            break
        sample = attacks[attempt % len(attacks)]
        attempt += 1

        messages = []
        spans = []
        system = sample.get("system", "")
        user = sample.get("user", "")
        if system:
            messages.append({"role": "system", "content": system})
            spans.append((LABELS.index("system"), system))
        messages.append({"role": "user", "content": user})
        spans.append((LABELS.index("user"), user))

        prompt_ids_list, prompt_labels_list = tokenize_chat(
            tokenizer,
            messages,
            spans,
            args.max_seq_len,
            add_generation_prompt=True,
        )
        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=model.device)
        prompt_labels = torch.tensor([prompt_labels_list], dtype=torch.long, device=model.device)

        gen_ids, gen_labels, lengths = generate_group(
            model,
            controller,
            prompt_ids,
            prompt_labels,
            args.group_size,
            args.max_new_tokens,
            args.temperature,
            tokenizer.eos_token_id,
            RESPONSE_ID,
            progress_every=128,
            on_progress=lambda t, seq_len: logger.info(
                "{}",
                f"  gen step {t} seq_len={seq_len} "
                f"vram={torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
            ),
        )
        prompt_len = prompt_ids.shape[1]
        answer_start = find_answer_start(gen_ids, prompt_len, think_end_id)
        responses = [
            (
                ""
                if int(answer_start[i]) < 0
                else tokenizer.decode(gen_ids[i, int(answer_start[i]) :], skip_special_tokens=True)
            )
            for i in range(gen_ids.shape[0])
        ]
        raw_rewards, reasons = zip(*(judge(sample, resp) for resp in responses))
        rewards = [
            shape_overlong_reward(r, length, args.max_new_tokens)
            for r, length in zip(raw_rewards, lengths)
        ]
        r = torch.tensor(rewards, dtype=torch.float32, device=model.device)

        logger.info(
            "{}", f"\nattempt {attempt} type={sample.get('attack_type')} G={len(responses)}"
        )
        logger.info("{}", f"  system: {system}")
        logger.info("{}", f"  user: {user}")
        for i, (resp, raw_r, shaped_r, reason, length) in enumerate(
            zip(responses, raw_rewards, rewards, reasons, lengths), 1
        ):
            capped = length >= args.max_new_tokens
            start = int(answer_start[i - 1])
            logger.info(
                "{}",
                f"  resp {i} len={length} capped={capped} answer_start={start} "
                f"raw_r={raw_r:+.2f} shaped_r={shaped_r:+.2f} [{reason}]",
            )
            if start < 0:
                think_text = tokenizer.decode(gen_ids[i - 1, prompt_len:], skip_special_tokens=True)
                logger.info("{}", "    [think-only, no final answer] " + think_text.strip()[:400])
                continue
            think_text = tokenizer.decode(
                gen_ids[i - 1, prompt_len:start], skip_special_tokens=True
            )
            logger.info("{}", "    [think] " + think_text.strip()[:400])
            logger.info("{}", "    [answer] " + resp.strip().replace("\n", "\n    "))

        # DAPO dynamic sampling: skip groups with no reward variance.
        if not has_learning_signal(r):
            skipped += 1
            logger.info("{}", f"skip uniform group: raw={list(raw_rewards)} shaped={list(rewards)}")
            continue

        advantages = (r - r.mean()) / (r.std() + 1e-4)
        logps_theta, mask = teacher_logps(
            model,
            controller,
            gen_ids,
            gen_labels,
            enabled_grad=True,
            response_label_id=RESPONSE_ID,
            answer_start=answer_start,
        )
        if mask.sum() == 0:
            logger.info("{}", "skip: no final-answer tokens (thinking not finished)")
            continue
        logps_ref, _ = teacher_logps(
            model,
            ref_controller,
            gen_ids,
            gen_labels,
            enabled_grad=False,
            response_label_id=RESPONSE_ID,
            answer_start=answer_start,
        )
        loss = grpo_loss(
            logps_theta,
            logps_ref,
            advantages,
            mask,
            beta=args.beta,
            clip_eps=args.clip_eps,
            clip_higher=True,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dye.parameters(), args.clip)
        optimizer.step()
        dye.delta.data.clamp_(-0.5, 0.5)  # scale in [0.5, 1.5]

        update_count += 1
        total_reward += r.mean().item()
        if update_count == 1 or update_count % args.log_every == 0:
            grad_norm = sum(p.grad.norm().item() for p in dye.parameters() if p.grad is not None)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            logger.info(
                "{}",
                f"step {update_count}/{args.steps} loss={loss.item():.4f} "
                f"reward_mean={r.mean().item():+.3f} adv_std={advantages.std().item():.3f} "
                f"|delta|={dye.delta.abs().mean().item():.4f} "
                f"grad={grad_norm:.3f} vram={vram:.2f}GB "
                f"type={sample.get('attack_type')} skipped={skipped}",
            )

    final_path = run_dir / "dye_dapo_final.pt"
    torch.save(dye.state_dict(), final_path)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    logger.info(
        "{}",
        f"done in {time.time() - t0:.1f}s; updates={update_count} skipped={skipped} "
        f"avg reward={total_reward / max(update_count, 1):+.3f} "
        f"peak_vram={peak_vram:.2f}GB; saved {final_path}",
    )
    controller.remove()
    ref_controller.remove()


if __name__ == "__main__":
    main()
