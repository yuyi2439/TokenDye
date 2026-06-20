from utils import BASE, load_model_and_tokenizer, setup_logging

logger = setup_logging(BASE, "test_model")

model, tokenizer = load_model_and_tokenizer(logger)

# 测试对话
prompt = "你好，请介绍一下自己"
messages = [{"role": "user", "content": prompt}]

# 应用聊天模板
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")  # type: ignore


# 生成回复
logger.info("生成中...")
outputs = model.generate(  # type: ignore
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.0,
)

## from 模型仓库
# outputs = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
# ]
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1] :],
    skip_special_tokens=False,
)
logger.info("模型回答：")
logger.info(response)

input()
