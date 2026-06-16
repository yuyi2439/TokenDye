from torch.utils.data import Dataset

type DyeDataItem = dict[str, list]


class DyeDataset(Dataset):
    def __init__(
        self,
        data_list: list[dict],
        tokenizer,
        dye_types: set[str],
    ):
        self.tokenizer = tokenizer
        self.dye_to_id = {dye: idx for idx, dye in enumerate(dye_types)}
        self.dataset: list[DyeDataItem] = []

        for data in data_list:
            input_ids = []
            dye_mask = []

            # 1. 拼接所有context片段
            for segment in data["segments"]:
                dye = segment["dye"]
                text = segment["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                dye_id = self.dye_to_id.get(dye, -1)
                input_ids.extend(tokens)
                dye_mask.extend([dye_id] * len(tokens))

            context_len = len(input_ids)

            # 2. 拼接target，并加上eos，让模型学会在何时停止生成
            target_tokens = tokenizer.encode(data["target"], add_special_tokens=False)
            target_tokens = target_tokens + [tokenizer.eos_token_id]

            input_ids.extend(target_tokens)
            dye_mask.extend([-1] * len(target_tokens))  # target部分不染色

            # 3. target_mask：标记哪些位置要计算loss
            target_mask = [False] * context_len + [True] * len(target_tokens)

            self.dataset.append({
                "input_ids": input_ids,
                "dye_mask": dye_mask,
                "target_mask": target_mask,
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
