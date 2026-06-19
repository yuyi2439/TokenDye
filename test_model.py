# import torch
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
# model = model.to("cuda") # pyright: ignore[reportArgumentType]
# 测试对话
prompt = "你好，请介绍一下自己"
messages = [{"role": "user", "content": prompt}]

# 应用聊天模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 编码输入
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 生成回复
print("\n生成中...")
outputs = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
    **inputs,
    max_new_tokens=512,
    # temperature=0.7,
    # do_sample=True,
    # top_p=0.9
)

## from 模型仓库
# outputs = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
# ]
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


# 解码并打印
response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print("\n模型回答：")
print(response)
