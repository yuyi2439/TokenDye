<div align="center">

English | [中文](./README_zh.md)

</div>

# Token Dye 🎨

**Inject light-weight trainable signals into LLM embeddings to make models source-aware.**

Token Dye is a mechanism that attaches low-rank "dye" matrices right after the embedding layer of a frozen LLM. Each dye corresponds to a specific token source (e.g., `system`, `user`, `file`, `web`, `database`) and adds a structured residual signal to the token vectors. The model can then **perceive the provenance of every token directly in the embedding space**, without relying on textual cues alone.

---

## ✨ Why Token Dye?

- **Ultra-light** – each dye matrix is ~0.5 M parameters (< 1 MB). Training is fast, and dyes can be swapped at will.
- **Loss-less degradation** – when no dye is applied, the base model behaves exactly as the original.
- **Architecture agnostic** – works with any model that has an embedding layer (tested on Qwen 2.5-7B and Qwen 3.5-4B).
- **Unforgeable** – dye labels are controlled by the system, not by the user, so users cannot impersonate system or tool tokens through prompt injection.
- **Online learning ready** – tiny parameter footprint enables high-frequency updates and personalization.

---

## 🧪 Core Findings

| Base Model       | Dye Effect                                      |
|------------------|-------------------------------------------------|
| Qwen 2.5-7B      | ✅ Gain – dyes helped weaker models with source discrimination and priority reasoning. |
| Qwen 3.5-4B      | ⚠️ Interference – dyes initially disrupted the model's chain-of-thought. **Root cause fixed** by explicitly closing the think block (`</think>`) before the assistant's answer. |

**Conclusion:**  
Token Dye provides measurable gains on tasks the base model does **not** already solve perfectly (e.g., multi-source conflict resolution). On tasks where the base model is already strong (e.g., safety refusals), dyes add no extra benefit but also cause no harm – a clean baseline of loss-less degradation.

---

## 📁 Repository Structure

```
.
├── train.py                  # Core training script (run_train, run_chain)
├── sweep_lr.py               # Learning rate search
├── ablation_test.py          # Ablation study (dye vs. no-dye)
├── log_viewer.py             # Training log visualization (loss & per-dye gradient norms)
├── DyeConfig.json            # Dye label definitions (system, user, tool, file, etc.)
├── dataset/
│   ├── v0.1a.jsonl5          # Original safety-attack dataset (90 samples)
│   └── v0.2.jsonl5           # Multi-source conflict dataset (71+ samples, growing)
├── README.md
└── README_zh.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install transformers peft bitsandbytes accelerate json5
```

### 2. Prepare dye labels

Edit `DyeConfig.json` to define your labels. Example:

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

Labels can be hierarchical using `::` (e.g., `tool::read_file`).

### 3. Prepare training data

Data format (JSON5):

```json5
{
  "segments": [
    {"dye": "system", "text": "You are a helpful assistant."},
    {"dye": "user",   "text": "What is the capital of France?"}
  ],
  "target": "The capital of France is Paris."
}
```

For multi-source conflict scenarios:

```json5
{
  "segments": [
    {"dye": "system", "text": "When sources conflict, trust the database over the web."},
    {"dye": "tool::read_database", "text": "Product A1001 stock: 328, status: in-stock"},
    {"dye": "tool::search_web",   "text": "Product A1001 is out of stock according to a shopping site."},
    {"dye": "user", "text": "Is product A1001 available?"}
  ],
  "target": "According to the database, product A1001 has 328 items in stock. Although the web says it's sold out, we trust the database."
}
```

### 4. Train

```bash
python train.py
```

The script will freeze the base model (Qwen 3.5-4B by default), load dye modules, and train only the dye matrices.

### 5. Run ablation test

```bash
python ablation_test.py
```

This compares outputs with dye enabled (A) vs. dye disabled (B) on the same prompts.

---

## 📊 Monitoring

Use `log_viewer.py` to plot training curves and per-dye gradient norms:

```bash
python log_viewer.py --log_file training.log
```

---

## 🔮 Future Directions

- **Residual dye stacking** – combine a base `tool` dye with sub-dyes (`tool::read_file`, etc.) for hierarchical source representation.
- **Online learning** – update dyes on the fly for personalized multi-source routing.
- **Transfer to other architectures** (e.g., RWKV) – the method only requires an embedding layer, making it broadly applicable.

---

## 📄 License

This project is provided for research purposes. See [LICENSE](LICENSE) for details.

---

## 📝 Citation

If you use Token Dye in your research, please cite this repository.