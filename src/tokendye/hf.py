"""Generic HuggingFace helpers (model loading, Python headers)."""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

import torch


def ensure_python_headers() -> None:
    """triton needs Python.h to compile CUDA helpers; fall back to uv-managed Python."""
    include = Path(sysconfig.get_paths()["include"])
    if (include / "Python.h").exists():
        return
    major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = sorted(Path.home().glob(f".local/share/uv/python/*/include/{major_minor}"))
    if not candidates:
        print("warning: Python.h not found; GPU forward may fail (install python3.x-dev)")
        return
    os.environ.setdefault("C_INCLUDE_PATH", str(candidates[-1]))


def load_model(model_path: str, device: str):
    """Load an HF causal LM: bnb 4-bit on CUDA, else bf16 GPU, else CPU."""
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
            print("loading: bitsandbytes 4-bit (nf4) + CUDA")
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as exc:  # bnb unavailable / out of memory / load failure
            print(f"4-bit unavailable ({type(exc).__name__}: {exc}), trying bf16")

    print(f"loading: bf16 + {device}")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    try:
        return model.to(device)
    except Exception as exc:
        if device == "cuda":
            print(f"out of memory ({exc}), falling back to CPU")
            return model.to("cpu")
        raise
