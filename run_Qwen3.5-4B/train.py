import os
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Optional

import torch
from transformers import get_cosine_schedule_with_warmup
from utils import (
    BASE,
    MyLogHandler,
    TrainConfig,
    init_dataloaders,
    load_model_and_tokenizer,
    setup_logger,
)

from tokendye import ModelDyeConfig
from tokendye.module import setup_dye_modules

RESUME_TRAINING = bool(os.getenv("RESUME_TRAINING", False))
DATA_FILE = Path("./dataset/v0.2.jsonl5")

# ====================================================


def run_trains(
    title: str,  # TODO: Too useless
    workspace: "Path",
    tcs: "list[TrainConfig]",
    mdc: "ModelDyeConfig",
    *,
    logger: "Optional[Logger]" = None,
    unify_dye_weight=False,
    bs_train=4,
    bs_val=8,
):
    logger = logger or setup_logger(workspace, title)

    _seed = torch.initial_seed()
    logger.debug(f"Initial Seed: {_seed}")

    logger.info(f"DataSet: {DATA_FILE}")
    logger.info("ModelDyeConfig: " + mdc.model_dump_json(indent=2))

    logger.debug("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer()
    logger.info("Loaded model and tokenizer")

    logger.debug("Loading dataloader...")
    dataloaders = init_dataloaders(
        DATA_FILE,
        logger,
        tokenizer,
        mdc.labels,
        bs_train,
        bs_val,
    )
    logger.info("Loaded dataloader...")

    for tc in tcs:
        case_name = f"rank_{tc.rank}-lr_{tc.lr:.2e}"
        logger.info(f"Start train case: {case_name}")

        sub_workspace = workspace / case_name
        sub_workspace.mkdir(parents=True, exist_ok=True)

        log_handler = MyLogHandler(logger)
        sub_logger = setup_logger(sub_workspace, f"train-{case_name}", log_handler)

        if unify_dye_weight:
            torch.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)

        train_dye(model, dataloaders, sub_logger, sub_workspace, mdc, tc)

        logger.info(f"(≧∀≦)ゞFinish train case successfully: {case_name}\n")


def train_dye(
    model,
    dataloaders: tuple,
    logger: "Logger",
    workspace: "Path",
    mdc: "ModelDyeConfig",
    tc: "TrainConfig",
):
    logger.info(f"WorkSpace: {workspace}")
    logger.info("TrainConfig: " + tc.model_dump_json(indent=2))

    logger.info("Setting up DyeModule...")
    dye_modules = setup_dye_modules(mdc, tc.rank, model.device)
    dye_modules.train()
    dye_modules.requires_grad_(True)

    def dye_hook(module, input, output):
        dye_mask = getattr(module, "_dye_mask", None)
        if dye_mask is None:
            return output

        batch, seq, d_model = output.shape
        flat_out = output.reshape(-1, d_model)
        flat_mask = dye_mask.reshape(-1)

        new_out = flat_out
        for dye_label in mdc.labels:
            pos = (flat_mask == dye_label.id).nonzero(as_tuple=True)[0]
            if pos.numel():
                updated = dye_modules[dye_label.name](flat_out[pos])
                new_out = new_out.index_copy(0, pos, updated)
        return new_out.view(batch, seq, d_model)

    model.model.embed_tokens._dye_mask = None
    model.model.embed_tokens.register_forward_hook(dye_hook)

    optimizer = torch.optim.AdamW(dye_modules.parameters(), tc.lr)

    train_dataloader, val_dataloader = dataloaders

    total_steps = tc.total_epochs * len(train_dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

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

    tc.save(workspace)
    logger.info("Saved TrainConfig")

    logger.info("Start training")
    is_te_not_enough = False
    for epoch in range(start_epoch, tc.total_epochs + 1):
        # Patience Check
        if epoch == best_val_loss_epoch + tc.patience:
            logger.info("Patience exhausted")
            return
        if epoch + tc.patience >= tc.total_epochs:
            is_te_not_enough = True

        # Train
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
            f"Epoch {epoch}/{tc.total_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
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
                    "model_dye_config": mdc.model_dump_json(),
                    "train_config": tc.model_dump_json(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                },
                workspace / "dye_modules_best.pt",
            )

            # # B. 完整训练状态（用于断点续训）
            # torch.save(
            #     {
            #         "epoch": epoch,
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "scheduler_state_dict": scheduler.state_dict(),
            #         "dye_state_dicts": {
            #             label: mod.state_dict() for label, mod in dye_modules.items()
            #         },
            #         "best_val_loss": best_val_loss,
            #         "train_loss": avg_train_loss,
            #     },
            #     workspace / "train_state_best.pt",
            # )

            logger.info(
                f"  ↳ 新的最优 checkpoint (val_loss={avg_val_loss:.4f})，已保存",
            )
    if is_te_not_enough:
        logger.error("Should improve total_epochs or lr")


if __name__ == "__main__":
    _tc = TrainConfig(rank=8, lr=1e-4)
    _mdc = ModelDyeConfig.load(BASE / "DyeConfig.json")

    _RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    _workspace = BASE / ".outputs" / _RUN_TS
    _workspace.mkdir(parents=True, exist_ok=True)

    run_trains("run_train", _workspace, [_tc], _mdc)
