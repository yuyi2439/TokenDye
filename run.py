import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

import tokendye

MODEL_PATH = "./Qwen2.5-7B-Instruct"
DATA_PATH = "./dataset/v0.1a.jsonl5"
DYE_TYPES = {"system", "user", "tool_callback", "file_text"}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("加载 数据集...")
full_dataset = tokendye.dataset.from_jsonl5(DATA_PATH, tokenizer, dye_types=DYE_TYPES)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size],
)


def collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    input_ids_list = [torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]
    dye_mask_list = [torch.tensor(s["dye_mask"], dtype=torch.long) for s in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    dye_mask = torch.nn.utils.rnn.pad_sequence(
        dye_mask_list,
        batch_first=True,
        padding_value=-1,
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    labels = input_ids.clone()
    labels[~attention_mask.bool()] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dye_mask": dye_mask,
        "labels": labels,
    }


train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
)
model = model.to("cuda")  # type: ignore
model.requires_grad_(False)


d_model = model.config.hidden_size
dtype = model.dtype

dye_modules = nn.ModuleDict()
for dye_label in DYE_TYPES:
    module = tokendye.Dye(d_model, 8, dtype=dtype).to(model.device)
    module.requires_grad_(True)
    dye_modules[dye_label] = module

model.model.embed_tokens._dye_mask = None


def dye_hook(module, input, output):
    dye_mask = getattr(module, "_dye_mask", None)
    if dye_mask is None:
        return output
    batch, seq, d = output.shape
    flat_out = output.view(-1, d)
    flat_mask = dye_mask.view(-1)
    for dye_idx, dye_label in enumerate(DYE_TYPES):
        pos = (flat_mask == dye_idx).nonzero(as_tuple=True)[0]
        if pos.numel():
            flat_out[pos] = dye_modules[dye_label](flat_out[pos])
    return flat_out.view(batch, seq, d)


model.model.embed_tokens.register_forward_hook(dye_hook)

optimizer = torch.optim.AdamW(dye_modules.parameters(), lr=2e-4)


num_epochs = 10
total_steps = num_epochs * len(train_dataloader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(0.1 * total_steps)),
    num_training_steps=total_steps,
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        dye_mask = batch["dye_mask"].to("cuda")
        labels = batch["labels"].to("cuda")

        model.model.embed_tokens._dye_mask = dye_mask

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dye_modules.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)

    model.eval()
    val_total_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            dye_mask = batch["dye_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            model.model.embed_tokens._dye_mask = dye_mask

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            val_total_loss += outputs.loss.item()

    avg_val_loss = val_total_loss / len(val_dataloader)
    print(
        f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
    )
