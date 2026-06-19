import json
from typing import TYPE_CHECKING

import jsonl5

from . import DyeDataset

if TYPE_CHECKING:
    from os import PathLike

    from ..dye_label import DyeLabel


def from_jsonl(data_path: "PathLike | str", tokenizer, labels: list["DyeLabel"]):
    with open(data_path) as f:
        # raw_data = []
        # for line in f:
        #     raw_data.append(json.loads(line))
        raw_data = [json.loads(line) for line in f]

    return DyeDataset(raw_data, tokenizer, labels)


def from_jsonl5(data_path: "PathLike | str", tokenizer, labels: list["DyeLabel"]):
    with open(data_path) as f:
        raw_data = jsonl5.load(f)

    return DyeDataset(raw_data, tokenizer, labels)
