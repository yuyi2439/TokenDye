"""TokenDye 示例：DeepEmbed 式逐层染色（Qwen3.5-4B）。

post-embed（在 Embedding 层染色）是另一种可选方法；本示例演示的是
DeepEmbed 式逐层缩放。

用法（uv）：
    QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --group example python example/Qwen3.5/deep_embed_qwen.py
    QWEN_MODEL_PATH=/path/to/Qwen3.5-4B uv run --group example python example/Qwen3.5/deep_embed_qwen.py --train --steps 3

默认模型路径：example/Qwen3.5/Qwen3.5-4B（或通过 QWEN_MODEL_PATH 环境变量指定）
加载策略：CUDA 下优先 bitsandbytes 4-bit (nf4)，失败退回 bf16，再退回 CPU。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokendye import DeepEmbedDye, install_deep_embed

LABELS = ["system", "user", "agent", "response"]
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "Qwen3.5-4B")
SYSTEM_CONTENT = "你是一个乐于助人的中文助手。"
USER_CONTENT = "请用一两句话介绍你自己。"


def _ensure_python_headers() -> None:
    """triton 编译 CUDA 模块需要 Python.h；系统缺头文件时借用 uv 管理的 Python。"""
    import sysconfig

    include = Path(sysconfig.get_paths()["include"])
    if (include / "Python.h").exists():
        return
    major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = sorted(Path.home().glob(f".local/share/uv/python/*/include/{major_minor}"))
    if not candidates:
        print("警告：找不到 Python.h，GPU 前向可能失败（可安装 python3.14-dev）")
        return
    os.environ.setdefault("C_INCLUDE_PATH", str(candidates[-1]))


def _logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"无法从模型输出中取 logits: {type(output)}")


def _labeled_prompt(tokenizer):
    """渲染聊天模板，并把每个 token 标上角色标签（-1 = 不染色）。"""
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": USER_CONTENT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 在渲染后的模板里定位每个角色的正文区间（聊天模板会原样保留正文）
    spans = {}
    for role, content in (("system", SYSTEM_CONTENT), ("user", USER_CONTENT)):
        start = prompt.index(content)
        spans[role] = (start, start + len(content))

    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    labels = []
    for s, e in enc["offset_mapping"]:
        label = -1
        for role, (cs, ce) in spans.items():
            if cs <= s and e <= ce:
                label = LABELS.index(role)
                break
        labels.append(label)
    return prompt, enc["input_ids"], labels


def load_model(model_path: str, device: str):
    """优先 bnb 4-bit（CUDA），失败退回 bf16 GPU，再退回 CPU。"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device == "cuda":
        try:
            import bitsandbytes  # noqa: F401

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("加载策略：bitsandbytes 4-bit (nf4) + CUDA")
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as exc:  # bnb 不可用 / 显存不足 / 加载失败
            print(f"4-bit 量化不可用（{type(exc).__name__}: {exc}），尝试 bf16")

    print(f"加载策略：bf16 + {device}")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    try:
        return model.to(device)
    except Exception as exc:
        if device == "cuda":
            print(f"显存不足（{exc}），退回 CPU")
            return model.to("cpu")
        raise


def main() -> None:
    _ensure_python_headers()
    parser = argparse.ArgumentParser(
        description="TokenDye 示例（DeepEmbed 式逐层染色）",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="模型路径（也可用环境变量 QWEN_MODEL_PATH 传入）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--train", action="store_true", help="对染色参数做几步微调")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise SystemExit(
            f"模型目录不存在：{args.model_path}\n"
            "请通过 QWEN_MODEL_PATH 环境变量或 --model-path 指定 Qwen3.5-4B 所在目录"
        )

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model(args.model_path, device)
    model.requires_grad_(False)

    text_cfg = getattr(model.config, "text_config", model.config)
    num_layers = text_cfg.num_hidden_layers
    d_model = text_cfg.hidden_size

    dye = DeepEmbedDye(LABELS, num_layers, d_model).to(model.device)  # delta 用 fp32 master
    controller = install_deep_embed(model, dye)

    print(f"模型：{args.model_path}")
    print(f"染色配置：{len(LABELS)} 标签 × {num_layers} 层 × {d_model} 维")
    print(
        f"染色参数：{dye.num_parameters:,}（bf16 前向约 "
        f"{dye.num_parameters * 2 / 1024 / 1024:.2f} MB，fp32 master 约 "
        f"{dye.num_parameters * 4 / 1024 / 1024:.2f} MB）"
    )

    prompt, input_ids_list, label_ids = _labeled_prompt(tokenizer)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=model.device)
    labels = torch.tensor([label_ids], dtype=torch.long, device=model.device)
    print("\n提示词：\n" + prompt)
    print(
        "标签分布："
        f"system={int((labels == 0).sum())} user={int((labels == 1).sum())} "
        f"未染色={int((labels == -1).sum())}"
    )

    # 1) 初始化检查：scale 全 1 → 染/不染完全一致
    controller.set_labels(labels)
    with torch.no_grad():
        out_with = model(input_ids=input_ids)
    controller.clear_labels()
    with torch.no_grad():
        out_without = model(input_ids=input_ids)
    diff = (_logits(out_with) - _logits(out_without)).abs().max().item()
    print(f"\n[1] 初始 scale=1：染/不染 logits 最大差异 {diff:.3e}（应为 0，零损伤）")

    # 2) 机制演示：改某一层某个标签的缩放，验证深层可控
    probe_layer = num_layers // 2
    probe = dye.delta.data[0, probe_layer].clone()
    dye.delta.data[0, probe_layer] = (
        probe + 0.10
    )  # 逐通道加 0.1（scale≈1.1），避免被 LayerNorm 抵消
    controller.set_labels(labels)
    with torch.no_grad():
        out_probe = model(input_ids=input_ids)
    diff_probe = (_logits(out_probe) - _logits(out_without)).abs().max().item()
    dye.delta.data[0, probe_layer] = probe
    print(
        f"[2] system 第 {probe_layer} 层缩放改为 1.10 后，logits 最大差异 "
        f"{diff_probe:.3e}（应 > 0，深层信号可达输出）"
    )

    # 3) 染色生成：新生成的 token 标为 response
    gen_ids, gen_labels = input_ids.clone(), labels.clone()
    resp_id = LABELS.index("response")
    for _ in range(args.max_new_tokens):
        controller.set_labels(gen_labels)
        with torch.no_grad():
            logits = _logits(model(input_ids=gen_ids))[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_id], dim=1)
        gen_labels = torch.cat([gen_labels, torch.full_like(next_id, resp_id)], dim=1)
    controller.clear_labels()
    response = tokenizer.decode(gen_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(f"\n[3] 生成回复（{args.max_new_tokens} token，未训练时等同原模型）：\n{response}")

    # 4) 可选：微调染色参数（冻结基座，只动染色向量）
    if args.train:
        # delta 是 fp32 master，小更新不会丢失；weight decay 会把 delta 拉回 0（scale 回 1）
        optimizer = torch.optim.AdamW(dye.parameters(), lr=1e-2, weight_decay=1e-3)
        controller.set_labels(labels)
        before = dye.delta.abs().mean().item()
        print(f"\n[4] 微调 {args.steps} 步（仅染色参数可训练）")
        for step in range(1, args.steps + 1):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            optimizer.step()
            dye.delta.data.clamp_(-0.5, 0.5)  # scale 限制在 [0.5, 1.5]
            print(
                f"    step {step}: loss={loss.item():.4f}  "
                f"delta 平均绝对值={dye.delta.abs().mean().item():.6f}"
            )
        after = dye.delta.abs().mean().item()
        print(f"    完成：delta 平均绝对值 {before:.3e} -> {after:.3e}")

    controller.remove()


if __name__ == "__main__":
    main()
