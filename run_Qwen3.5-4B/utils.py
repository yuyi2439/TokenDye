import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tokendye.dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from logging import Logger

    from tokendye import DyeConfig
    from transformers import TokenizersBackend



DATA_PATH = "./dataset/v0.1a.jsonl5"


BASE = Path(__file__).parent
SANITY_CHECK = bool(os.getenv("SANITY_CHECK", False))


def load_model_and_tokenizer(logger: "Logger"):
    model_path = BASE / MODEL_NAME
    logger.info("Loading tokenizer...")
    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(model_path)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")  # type: ignore
    model.requires_grad_(False)

    return model, tokenizer


def init_dataloader(
    logger, tokenizer, dyeConfig: "DyeConfig", *, bs_train: int, bs_val: int,
):
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
        batch_size=bs_train,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=bs_val,
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


def setup_logging(output_dir: "Path", name: str | None) -> "Logger":

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if name:
        log_path = output_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


if __name__ == "__main__":
    from tokendye import DyeLabel

    DYE_TYPES = [
        "system",
        "user",
        "tool_callback",
        "file_text",
    ]

    logger = setup_logging(BASE, None)
    model, _ = load_model_and_tokenizer(logger)
    dyeConfig = DyeConfig(
        model_name=MODEL_NAME,
        labels=[DyeLabel(id=i, name=n) for i, n in enumerate(DYE_TYPES)],
        rank=8,
        d_model=model.config.hidden_size,
        dtype=str(model.dtype).split(".")[-1],
    )
    dyeConfig.save()
