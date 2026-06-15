from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader

from tokendye import Dye, DyeDataset

MODEL_PATH = "./Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
)
model = model.to("cuda") # pyright: ignore[reportArgumentType]
model.requires_grad_(False)

###############################################################################

_current_dye_mask = None

def set_dye_mask(mask):
    global _current_dye_mask
    _current_dye_mask = mask

def get_dye_mask():
    return _current_dye_mask

dye_types = ["system", "user", "tool_callback", "file_text"]

rank = 8
d_model = model.config.hidden_size

dye_modules = nn.ModuleDict()
for dye_label in dye_types:
    module = Dye(d_model, rank).to(model.device)
    module.requires_grad_(True)
    dye_modules[dye_label] = module

def dye_hook(module, input, output):
    dye_mask = get_dye_mask()
    if dye_mask is None:
        return output
    batch, seq, d = output.shape
    flat_out = output.view(-1, d)
    flat_mask = dye_mask.view(-1)
    for dye_idx, dye_label in enumerate(dye_types):
        pos = (flat_mask == dye_idx).nonzero(as_tuple=True)[0]
        if pos.numel():
            flat_out[pos] = dye_modules[dye_label](flat_out[pos])
    return flat_out.view(batch, seq, d)


model.model.embed_tokens.register_forward_hook(dye_hook)

optimizer = torch.optim.AdamW(dye_modules.parameters(), lr=1e-3)

def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    input_ids_list = [torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]
    dye_mask_list = [torch.tensor(s["dye_mask"], dtype=torch.long) for s in batch]

    # 动态 padding
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id or 0
    )
    dye_mask = torch.nn.utils.rnn.pad_sequence(
        dye_mask_list, batch_first=True, padding_value=-1
    )
    attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()

    labels = input_ids.clone()
    labels[~attention_mask.bool()] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dye_mask": dye_mask,
        "labels": labels,
    }

raw_data = [
    [{"dye": "system", "text": "You are helpful."}, {"dye": "user", "text": "What is AI?"}],
    [{"dye": "user", "text": "Hello"}, {"dye": "tool_callback", "text": "Result: 42"}],
]

dataset = DyeDataset(raw_data, tokenizer, dye_types, max_length=512)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
)

for _ in range(10):
    for batch in dataloader:
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        dye_mask = batch["dye_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        set_dye_mask(dye_mask)   # 设置全局掩码供 hook 读取

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        print(f"Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()