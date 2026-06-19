import logging
import sys
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from transformers.models import qwen2


MODEL_PATH = "./run_Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct"


def load_model_and_tokenizer(logger: "Logger"):
    logger.info("Loading tokenizer...")
    tokenizer: qwen2.Qwen2Tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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
        MODEL_PATH,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")  # type: ignore
    model.requires_grad_(False)

    return model, tokenizer


def setup_logging(output_dir: "Path", name: str) -> "Logger":
    log_path = output_dir / f"{name}.log"

    logger = logging.getLogger(name)
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
