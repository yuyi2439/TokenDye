import os
import re
import matplotlib.pyplot as plt

output = os.getenv("OUTPUT")
assert output is not None

# 读取日志文件
with open("./.outputs/" + output + "/log_all.log", "r") as f:
    lines = f.readlines()

epochs = []
train_losses = []
val_losses = []
grad_data = {}          # {stain_type: [ (epoch, grad_norm), ... ]}
current_grads = []       # 临时存储当前 epoch 的染色梯度

for line in lines:
    # 匹配梯度行
    grad_match = re.search(r"DEBUG \| (\w+): grad_norm = ([\d.]+)", line)
    if grad_match:
        stain_type = grad_match.group(1)
        grad_val = float(grad_match.group(2))
        current_grads.append((stain_type, grad_val))
        continue

    # 匹配 Epoch 信息行
    epoch_match = re.search(r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)", line)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        train_loss = float(epoch_match.group(2))
        val_loss = float(epoch_match.group(3))

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 把前面收集的梯度归到当前 epoch
        for stain_type, val in current_grads:
            if stain_type not in grad_data:
                grad_data[stain_type] = []
            grad_data[stain_type].append((epoch, val))
        current_grads = []   # 清空，准备下一个 epoch

# 过滤掉始终为 0 的染色（如 tool_callback, file_text），避免画图冗余
active_stains = {k: v for k, v in grad_data.items() if any(val != 0.0 for _, val in v)}

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ---- 子图1：Loss ----
axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
# 标记最佳 epoch
best_idx = val_losses.index(min(val_losses))
axes[0].axvline(x=epochs[best_idx], color='gray', linestyle='--',
                label=f'Best epoch ({epochs[best_idx]})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ---- 子图2：Gradient Norm ----
for stain_type, data in active_stains.items():
    ep, vals = zip(*data)
    axes[1].plot(ep, vals, marker='.', label=stain_type)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Gradient Norm')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('training_curves_with_grad.png', dpi=150)
plt.show()