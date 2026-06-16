import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


# 最简单的染色：直接加一个可学习向量
dye_vectors = {
    "system": torch.zeros(model.config.hidden_size),  # 先用零向量
    "user": torch.zeros(model.config.hidden_size),
}


def make_hook(dye_label, token_mask):
    def hook(module, input, output):
        if dye_label in dye_vectors:
            signal = dye_vectors[dye_label]
            output[token_mask] += signal  # 残差注入
        return output

    return hook


# 挂载到Embedding层
handle = model.model.embed_tokens.register_forward_hook(make_hook("user", ...))
