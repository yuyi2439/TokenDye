import json
import jsonl5
from pathlib import Path

from . import DyeDataset


def from_jsonl(
    data_path: Path | str, tokenizer, dye_types: set[str] | None = None,
) -> DyeDataset:
    with open(data_path) as f:
        # raw_data = []
        # for line in f:
        #     raw_data.append(json.loads(line))
        raw_data = [json.loads(line) for line in f]

    dye_types = dye_types or set(
        item["dye"] for sample in raw_data for item in sample["segments"]
    )

    return DyeDataset(raw_data, tokenizer, dye_types)


def from_jsonl5(
    data_path: Path | str, tokenizer, dye_types: set[str] | None = None,
) -> DyeDataset:
    with open(data_path) as f:
        raw_data = jsonl5.load(f)

    dye_types = dye_types or set(
        item["dye"] for sample in raw_data for item in sample["segments"]
    )

    return DyeDataset(raw_data, tokenizer, dye_types)