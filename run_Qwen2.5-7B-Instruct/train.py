import os
from datetime import datetime

import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup
from utils import BASE, init_dataloader, load_model_and_tokenizer, setup_logging

from tokendye import DyeModule, ModelDyeConfig

RESUME_TRAINING = bool(os.getenv("RESUME_TRAINING", False))
TOTAL_EPOCHS = 50
PATIENCE = 10
bs_train = 6
bs_val = 10
lr = 3e-4

# ====================================================

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
dyeConfig = ModelDyeConfig.load(BASE / "DyeConfig.json")
RANK = 8  # LoRA rank for dye modules

OUTPUT_DIR = BASE / ".outputs" / RUN_TS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR, "train")


def main():
    logger.info("Dye Config: " + dyeConfig.model_dump_json(indent=2))
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")

    model, tokenizer = load_model_and_tokenizer(logger)

    logger.info("Loading dataloader...")
    train_dataloader, val_dataloader = init_dataloader(
        logger,
        tokenizer,
        dyeConfig,
        bs_train=bs_train,
        bs_val=bs_val,
    )

    logger.info("Setting up DyeLayer...")
    dye_modules = nn.ModuleDict()
    for dye_label in dyeConfig.labels:
        module = DyeModule(dyeConfig, RANK).to(model.device)
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

    optimizer = torch.optim.AdamW(dye_modules.parameters(), lr)

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
    best_val_loss_epoch = 1

    start_epoch = 1
    if RESUME_TRAINING:
        raise Exception("Todo")
        logger.info("尝试加载上次训练状态...")
        state = torch.load(
            "./.checkpoints/train_state_best.pt",
            map_location=model.device,
        )
        for label, mod in dye_modules.items():
            mod.load_state_dict(state["dye_state_dicts"][label])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]
        logger.info(
            f"已加载 checkpoint，恢复到 epoch {start_epoch}，上次 val_loss={state['best_val_loss']:.4f}",
        )

    logger.info("Start training")
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        # Patience Check
        if epoch == best_val_loss_epoch + PATIENCE:
            logger.info("Patience exhausted")
            return
        if epoch + PATIENCE == TOTAL_EPOCHS:
            logger.error("Should improve TOTAL_EPOCHS or lr")

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
        total_val_loss = 0.0
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
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        logger.info(
            f"Epoch {epoch}/{TOTAL_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
        )

        # ---- 保存最优checkpoint（仅当val loss刷新最优时）----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_epoch = epoch

            # A. 仅dye权重（最小，用于推理/部署）
            torch.save(
                {
                    "dye_state_dicts": {
                        label: mod.state_dict() for label, mod in dye_modules.items()
                    },
                    "dye_types": list(dye_label),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                },
                DYE_WEIGHTS_PATH,
            )

            # B. 完整训练状态（用于断点续训）
            torch.save(
                {
                    "epoch": epoch,
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
                f"  ↳ 新的最优 checkpoint (val_loss={avg_val_loss:.4f})，已保存",
            )


if __name__ == "__main__":
    main()
