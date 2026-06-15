from torch.utils.data import Dataset

class DyeDataset(Dataset):
    def __init__(
        self,
        data: list[list[dict[str, str]]],   # 每个样本是 segment 列表
        tokenizer,
        dye_types: list[str],                # 例如 ["system", "user", "tool_callback", "file_text"]
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.dye_to_id = {dye: idx for idx, dye in enumerate(dye_types)}
        self.max_length = max_length
        self.samples = []

        for segments in data:
            input_ids = []
            dye_mask = []
            for seg in segments:
                dye = seg["dye"]
                text = seg["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                dye_id = self.dye_to_id.get(dye, -1)
                input_ids.extend(tokens)
                dye_mask.extend([dye_id] * len(tokens))

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                dye_mask = dye_mask[:max_length]

            self.samples.append({
                "input_ids": input_ids,
                "dye_mask": dye_mask,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]
