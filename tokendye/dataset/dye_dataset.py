from torch.utils.data import Dataset

type DyeDataItem = dict[str, list]


def build_sequence(data, tokenizer, dye_to_id):
    input_ids = []
    dye_mask = []
    
    # 手动构建符合chat_template的结构，同时记录每段的染色标签
    
    for segment in data["segments"]:
        dye = segment["dye"]
        text = segment["text"]
        dye_id = dye_to_id.get(dye, -1)
        
        if dye == "system":
            # 结构token：不染色（dye_id = -1）
            prefix = "<|im_start|>system\n"
            suffix = "<|im_end|>\n"
        elif dye == "user":
            prefix = "<|im_start|>user\n"
            suffix = "<|im_end|>\n"
        # ... 其他类型
        
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        
        input_ids.extend(prefix_ids)
        dye_mask.extend([-1] * len(prefix_ids))  # 结构token不染色
        
        input_ids.extend(content_ids)
        dye_mask.extend([dye_id] * len(content_ids))  # 内容染色
        
        input_ids.extend(suffix_ids)
        dye_mask.extend([-1] * len(suffix_ids))  # 结构token不染色
    
    context_len = len(input_ids)
    
    # target部分
    assistant_prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    target_ids = tokenizer.encode(data["target"], add_special_tokens=False)
    eos = [tokenizer.eos_token_id]
    
    input_ids.extend(assistant_prefix_ids + target_ids + eos)
    dye_mask.extend([-1] * (len(assistant_prefix_ids) + len(target_ids) + 1))
    
    target_mask = (
        [False] * context_len +
        [False] * len(assistant_prefix_ids) +  # assistant\n 这几个token不计入loss
        [True] * len(target_ids) +
        [True]  # eos也要算loss
    )
    
    return input_ids, dye_mask, target_mask


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
            input_ids, dye_mask, target_mask = build_sequence(data, tokenizer, self.dye_to_id)

            self.dataset.append({
                "input_ids": input_ids,
                "dye_mask": dye_mask,
                "target_mask": target_mask,
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
