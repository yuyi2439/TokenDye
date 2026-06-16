from torch.utils.data import Dataset

type DyeDataItem = dict[str, list[int]]


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
            for segment in data["segments"]:
                dye = segment["dye"]
                text = segment["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                dye_id = self.dye_to_id.get(dye, -1)
                input_ids.extend(tokens)
                dye_mask.extend([dye_id] * len(tokens))
            target_ids = tokenizer.encode(data["target"], add_special_tokens=False)

            self.dataset.append(
                {
                    "input_ids": input_ids,
                    "dye_mask": dye_mask,
                    "target_ids": target_ids,
                },
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DyeDataItem:
        return self.dataset[idx]
