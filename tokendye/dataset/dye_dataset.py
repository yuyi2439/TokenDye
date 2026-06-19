from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ..dye_label import DyeLabel

def _build_sequence(
    data: dict,
    tokenizer,
    labels: list['DyeLabel']
):
    input_ids = []
    dye_mask = []

    messages = []  # TODO: 优化
    for segment in data["segments"]:
        dye = segment["dye"]
        text = segment["text"]
        if dye == "system":
            messages.append({"role": "system", "content": text})
        elif dye == "user":
            messages.append({"role": "user", "content": text})
        elif dye == "file_text":
            messages.append({"role": "user", "content": text})
        elif dye == "tool_callback":
            messages.append({"role": "tool", "content": text})
        else:
            messages.append({"role": "user", "content": text})

    # tokenize=True直接返回ids，不含generation prompt
    full_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )["input_ids"]  # TODO: 留着attention_mask或许有用

    # 2. 对每段content单独encode，在full_ids里定位边界，构建dye_mask
    dye_mask = [-1] * len(full_ids)

    label_map = {label.name:label.id for label in labels}
    for segment in data["segments"]:
        dye = segment["dye"]
        text = segment["text"]
        dye_id = label_map[dye]

        content_ids = tokenizer.encode(text, add_special_tokens=False)
        positions = _find_all_sublist(full_ids, content_ids)

        if len(positions) == 0:
            raise ValueError(f"content not found in full_ids: {text!r}")
        if len(positions) > 1:
            raise ValueError(
                f"ambiguous match ({len(positions)} hits), "
                f"two segments have identical content: {text!r}"
            )

        pos = positions[0]
        for i in range(len(content_ids)):
            dye_mask[pos + i] = dye_id

    # 3. target部分：从apply_chat_template提取generation prompt
    context_len = len(full_ids)

    full_ids_with_gen: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]  # TODO: 留着attention_mask或许有用
    gen_prompt_ids = full_ids_with_gen[context_len:]  # 只取新增部分

    target_ids = tokenizer.encode(data["target"], add_special_tokens=False)
    eos = [tokenizer.eos_token_id]

    input_ids = full_ids + gen_prompt_ids + target_ids + eos
    dye_mask = dye_mask + [-1] * (len(gen_prompt_ids) + len(target_ids) + 1)

    target_mask = (
        [False] * context_len
        + [False] * len(gen_prompt_ids)
        + [True] * len(target_ids)
        + [True]  # eos算loss
    )

    return input_ids, dye_mask, target_mask


class DyeDataset(Dataset):
    def __init__(
        self,
        raw_data: list[dict],
        tokenizer,
        labels: list['DyeLabel']
    ):
        self.dataset: list[dict] = []

        for data in raw_data:
            input_ids, dye_mask, target_mask = _build_sequence(
                data, tokenizer, labels
            )

            self.dataset.append(
                {
                    "input_ids": input_ids,
                    "dye_mask": dye_mask,
                    "target_mask": target_mask,
                }
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def _find_all_sublist(full, sub) -> list[int]:
    """返回sub在full中所有出现位置的起始index"""
    n, m = len(full), len(sub)
    if m == 0:
        return []
    return [i for i in range(n - m + 1) if full[i : i + m] == sub]
