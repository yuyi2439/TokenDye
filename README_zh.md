<div align="center">

[English](./README.md) | 中文

</div>

# Token 染色 🎨

**为大语言模型的 Embedding 注入可训练的轻量信号，让模型感知 Token 来源。**

Token 染色是一种在冻结的 LLM 嵌入层之后挂载低秩“染料”矩阵的机制。每种染料对应一种特定的 Token 来源（例如 `system`、`user`、`file`、`web`、`database`），并以残差形式向 Token 向量中注入结构性信号。模型因此能够**直接在向量空间中感知每个 Token 的来源**，而无需依赖文本线索。

---

## ✨ 核心优势

- **极轻量** – 每个染色矩阵约 50 万参数（< 1 MB）。训练快速，染料可热插拔。
- **无损退化** – 当不施加染色时，模型行为与原基座完全一致。
- **架构无关** – 任何具有 Embedding 层的模型均可使用（已在 Qwen 2.5-7B 和 Qwen 3.5-4B 上验证）。
- **不可伪造** – 染色标签由系统控制，用户无法通过文本内容冒充系统或工具 Token，天然防御提示注入。
- **在线学习潜力** – 极小的参数量支持高频更新和个性化微调。

---

## 🧪 核心发现

| 基座模型        | 染色效果                                        |
|-----------------|-------------------------------------------------|
| Qwen 2.5-7B     | ✅ 增益 – 染色帮助较弱模型进行来源辨别与跨来源优先级推理。 |
| Qwen 3.5-4B     | ⚠️ 干扰 – 初期染色破坏了模型的思维链。**已修复**：在助手回答前显式结束思考（`</think>`）后，干扰消失，染色行为干净可控。 |

**结论：**  
Token 染色在基座模型**不擅长**的任务上（如多来源信息冲突裁决）能提供可测量的增益。在基座模型已经很强的任务上（如安全拒绝），染色不会带来额外好处，但也不会造成任何损害 – 实现了真正的无损退化。

---

## 📁 项目结构

```
.
├── train.py                  # 核心训练脚本（run_train, run_chain）
├── sweep_lr.py               # 学习率搜索脚本
├── ablation_test.py          # 消融对比实验（染色 vs 不染色）
├── log_viewer.py             # 训练日志可视化（Loss 曲线和各染料梯度范数）
├── DyeConfig.json            # 染色标签配置（system, user, tool, file 等）
├── dataset/
│   ├── v0.1a.jsonl5          # 原始安全攻击数据集（90 条）
│   └── v0.2.jsonl5           # 多来源冲突数据集（71+ 条，持续增长）
├── README.md
└── README_zh.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install transformers peft bitsandbytes accelerate json5
```

### 2. 准备染色标签

编辑 `DyeConfig.json` 定义你的标签。示例：

```json5
{
  "labels": [
    {"name": "assistant", "id": 0},
    {"name": "system",    "id": 1},
    {"name": "user",      "id": 2},
    {"name": "tool",      "id": 100},
    {"name": "tool::read_file",      "id": 101},
    {"name": "tool::search_web",     "id": 102},
    {"name": "tool::read_database",  "id": 103}
  ]
}
```

标签可使用 `::` 实现层级化（例如 `tool::read_file` 表示 `tool` 主类别下的 `read_file` 子类别）。

### 3. 准备训练数据

数据格式（JSON5）：

```json5
{
  "segments": [
    {"dye": "system", "text": "你是一个有用的助手。"},
    {"dye": "user",   "text": "法国的首都是哪里？"}
  ],
  "target": "法国的首都是巴黎。"
}
```

对于多来源冲突场景：

```json5
{
  "segments": [
    {"dye": "system", "text": "当来源矛盾时，以数据库为准，不信任网页。"},
    {"dye": "tool::read_database", "text": "产品 A1001 库存：328 件，状态：在售"},
    {"dye": "tool::search_web",   "text": "某购物网站显示 A1001 已售罄。"},
    {"dye": "user", "text": "产品 A1001 还有货吗？"}
  ],
  "target": "根据数据库，A1001 仍有 328 件库存。虽然网页显示售罄，但我们以数据库为准。"
}
```

### 4. 训练

```bash
python train.py
```

脚本将冻结基座模型（默认为 Qwen 3.5-4B），加载染色模块，并仅优化染色矩阵。

### 5. 运行消融测试

```bash
python ablation_test.py
```

该测试比较同一提示下启用染色（A 组）与关闭染色（B 组）的输出。

---

## 📊 监控

使用 `log_viewer.py` 绘制训练曲线和各染料的梯度范数：

```bash
python log_viewer.py --log_file training.log
```

---

## 🔮 未来方向

- **残差染色叠加** – 将基础 `tool` 染料与子染料（`tool::read_file` 等）结合，实现层级化来源表示。
- **在线学习** – 实时更新染料，用于个性化多源信息路由。
- **迁移至其他架构**（如 RWKV） – 方法仅依赖 Embedding 层，可广泛适用。

---

## 📄 许可证

本项目仅用于研究目的。详情请参阅 [LICENSE](LICENSE) 文件。

---

## 📝 引用

若您在研究中使用了 Token 染色技术，请引用本仓库。