import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

import tokendye
from tokendye import DyeConfig, DyeLayer

if TYPE_CHECKING:
    from transformers.models import qwen2


RESUME_TRAINING = bool(os.getenv("RESUME_TRAINING", False))
SANITY_CHECK = bool(os.getenv("SANITY_CHECK", False))
TOTAL_EPOCHS = int(os.getenv("TOTAL_EPOCHS", 50))

# ====================================================

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("./.outputs") / f"output_{RUN_TS}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    log_path = OUTPUT_DIR / "log_all.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(f"日志文件: {log_path.resolve()}")
    return logger


def init_dataloader(tokenizer, dyeConfig: DyeConfig):
    full_dataset = tokendye.dataset.from_jsonl5(DATA_PATH, tokenizer, dyeConfig.labels)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
    )

    def collate_fn(batch: list) -> dict[str, torch.Tensor]:
        input_ids_list = [torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]
        dye_mask_list = [torch.tensor(s["dye_mask"], dtype=torch.long) for s in batch]
        target_mask_list = [
            torch.tensor(s["target_mask"], dtype=torch.bool) for s in batch
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        dye_mask = torch.nn.utils.rnn.pad_sequence(
            dye_mask_list,
            batch_first=True,
            padding_value=-1,
        )
        target_mask = torch.nn.utils.rnn.pad_sequence(
            target_mask_list,
            batch_first=True,
            padding_value=False,
        )

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100
        labels[~target_mask.bool()] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dye_mask": dye_mask,
            "labels": labels,
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=6,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if SANITY_CHECK:
        logger.info("Sanity check")
        batch = next(iter(train_dataloader))
        print("input_ids:\n", batch["input_ids"][0])
        print("Decoded input_ids:\n", tokenizer.decode(batch["input_ids"][0]))
        print("dye_mask:\n", batch["dye_mask"][0])
        print("labels (non -100 positions):")
        valid_positions = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0]
        print(tokenizer.decode(batch["input_ids"][0][valid_positions]))

        raise RuntimeError("Sanity check complete - stopping here")

    return train_dataloader, val_dataloader


logger = setup_logging()

MODEL_PATH = "./Qwen2.5-7B-Instruct"
DATA_PATH = "./dataset/v0.1a.jsonl5"
CONFIG_PATH = "./DyeConfig_Qwen2.5-7B-Instruct.json"


def main():
    dyeConfig = DyeConfig.load(CONFIG_PATH)

    logger.info("Loading tokenizer...")
    tokenizer: qwen2.Qwen2Tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    logger.info("Loading dataloader...")
    train_dataloader, val_dataloader = init_dataloader(tokenizer, dyeConfig)

    logger.info("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")  # type: ignore
    model.requires_grad_(False)

    # DYE_TYPES = [
    #     "system",
    #     "user",
    #     "tool_callback",
    #     "file_text",
    # ]
    # dyeConfig = DyeConfig(
    #     model_name="Qwen2.5-7B-Instruct",
    #     labels=[DyeLabel(id=i, name=n) for i, n in enumerate(DYE_TYPES)],
    #     rank=8,
    #     d_model=model.config.hidden_size,
    #     dtype=str(model.dtype).split(".")[-1],
    # )
    # dyeConfig.save()

    logger.info("Loading DyeLayer...")
    dye_modules = nn.ModuleDict()
    for dye_label in dyeConfig.labels:
        module = DyeLayer(dyeConfig).to(model.device)
        module.requires_grad_(True)
        dye_modules[dye_label.name] = module

    def dye_hook(module, input, output):
        dye_mask = getattr(module, "_dye_mask", None)
        if dye_mask is None:
            return output

        batch, seq, d_model = output.shape
        flat_out = output.reshape(-1, d_model)
        flat_mask = dye_mask.reshape(-1)

        new_out = flat_out
        for dye_label in dyeConfig.labels:
            pos = (flat_mask == dye_label.id).nonzero(as_tuple=True)[0]
            if pos.numel():
                updated = dye_modules[dye_label.name](flat_out[pos])
                new_out = new_out.index_copy(0, pos, updated)
        return new_out.view(batch, seq, d_model)

    model.model.embed_tokens._dye_mask = None
    model.model.embed_tokens.register_forward_hook(dye_hook)

    optimizer = torch.optim.AdamW(dye_modules.parameters(), lr=2e-4)

    total_steps = TOTAL_EPOCHS * len(train_dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    # ---- checkpoint 保存配置 ----
    DYE_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "dye_modules_best.pt")
    TRAIN_STATE_PATH = os.path.join(OUTPUT_DIR, "train_state_best.pt")

    best_val_loss = float("inf")
    start_epoch = 0
    if RESUME_TRAINING:
        raise Exception("Todo")
        logger.info("尝试加载上次训练状态...")
        state = torch.load(
            "./.checkpoints/train_state_best.pt", map_location=model.device
        )
        for label, mod in dye_modules.items():
            mod.load_state_dict(state["dye_state_dicts"][label])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]
        logger.info(
            f"已加载 checkpoint，恢复到 epoch {start_epoch}，上次 val_loss={state['best_val_loss']:.4f}"
        )

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            dye_mask = batch["dye_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            model.model.embed_tokens._dye_mask = dye_mask

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dye_modules.parameters(), max_norm=1.0)

            # Log gradient norms for each dye module
            for label, mod in dye_modules.items():
                grad_norm = sum(
                    p.grad.norm().item() for p in mod.parameters() if p.grad is not None
                )
                logger.debug(f"{label}: grad_norm = {grad_norm}")

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                dye_mask = batch["dye_mask"].to("cuda")
                labels = batch["labels"].to("cuda")

                model.model.embed_tokens._dye_mask = dye_mask

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_total_loss += outputs.loss.item()

        avg_val_loss = val_total_loss / len(val_dataloader)
        logger.info(
            f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
        )

        # ---- 保存最优checkpoint（仅当val loss刷新最优时）----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # A. 仅dye权重（最小，用于推理/部署）
            torch.save(
                {
                    "dye_state_dicts": {
                        label: mod.state_dict() for label, mod in dye_modules.items()
                    },
                    "dye_types": list(dye_label),
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                },
                DYE_WEIGHTS_PATH,
            )

            # B. 完整训练状态（用于断点续训）
            torch.save(
                {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "dye_state_dicts": {
                        label: mod.state_dict() for label, mod in dye_modules.items()
                    },
                    "best_val_loss": best_val_loss,
                    "train_loss": avg_train_loss,
                },
                TRAIN_STATE_PATH,
            )

            logger.info(
                f"  ↳ 新的最优 checkpoint (val_loss={avg_val_loss:.4f})，已保存到 {OUTPUT_DIR}"
            )


if __name__ == "__main__":
    main()
