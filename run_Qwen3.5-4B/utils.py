import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import tokendye.dataset

if TYPE_CHECKING:
    from logging import Logger

    from transformers import TokenizersBackend

    from tokendye import DyeLabel


MODEL_NAME = "Qwen3.5-4B"


BASE = Path(__file__).parent
SANITY_CHECK = bool(os.getenv("SANITY_CHECK", 0))


class TrainConfig(BaseModel):
    rank: int
    lr: float
    total_epochs: int = 35
    patience: int = 10

    def save(self, wordspace: "Path", indent: int = 2):
        p = wordspace / "train_config.json"
        p.write_text(self.model_dump_json(indent=indent))

    @classmethod
    def load(cls, wordspace: "Path"):
        p = wordspace / "train_config.json"
        return cls.model_validate_json(p.read_text())


def load_model_and_tokenizer():
    model_path = BASE / MODEL_NAME

    tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(model_path)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

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


def init_dataloaders(
    data_file: "Path",
    logger: "Logger",
    tokenizer,
    labels: list["DyeLabel"],
    bs_train: int,
    bs_val: int,
):
    full_dataset = tokendye.dataset.from_jsonl5(data_file, tokenizer, labels)
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
        batch = next(iter(train_dataloader))
        valid_positions = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0]
        msgs = [
            "Sanity check",
            "input_ids:",
            batch["input_ids"][0],
            "Decoded input_ids:",
            tokenizer.decode(batch["input_ids"][0]),
            "dye_mask:",
            batch["dye_mask"][0],
            "labels (non -100 positions):",
            tokenizer.decode(batch["input_ids"][0][valid_positions]),
        ]

        for msg in msgs:
            logger.info(msg)

        raise RuntimeError("Sanity check complete - stopping here")

    return train_dataloader, val_dataloader


def setup_logger(workspace: "Path", name: str, handler=None) -> "Logger":
    """
    Args:
        `workspace`: Caller should make sure the folder exists
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if name:
        log_path = workspace / f"{name}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    if handler:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    logger.info(f"╰(*°▽°*)╯ Setup logger({name}) successfully!!")

    return logger


# if __name__ == "__main__":
#     from tokendye import DyeLabel

#     DYE_TYPES = [
#         "system",
#         "user",
#         "tool_callback",
#         "file_text",
#     ]
#     model, _ = load_model_and_tokenizer()
#     mdc = ModelDyeConfig(
#         model_name=MODEL_NAME,
#         labels=[DyeLabel(id=i, name=n) for i, n in enumerate(DYE_TYPES)],
#         d_model=model.config.hidden_size,
#         dtype=str(model.dtype).split(".")[-1],
#     )
#     mdc.save()


class MyLogHandler(logging.Handler):
    def __init__(self, parent_logger: "Logger"):
        self.parent_logger = parent_logger
        super().__init__(logging.INFO)

    def emit(self, record: logging.LogRecord):
        self.parent_logger.log(record.levelno, record.message)
