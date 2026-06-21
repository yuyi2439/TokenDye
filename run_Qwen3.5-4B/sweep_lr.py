from datetime import datetime

from train import run_trains
from utils import BASE, TrainConfig

from tokendye import ModelDyeConfig

# 要扫描的学习率列表
learning_rates = [5e-5, 1e-4, 2e-4, 5e-4]

# 搜索阶段的 epoch 数
SEARCH_EPOCHS = 15

tcs = [TrainConfig(rank=8, lr=lr, total_epochs=SEARCH_EPOCHS) for lr in learning_rates]
mdc = ModelDyeConfig.load(BASE / "DyeConfig.json")

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace = BASE / ".outputs" / f"{RUN_TS}-sweep_lr"
workspace.mkdir(parents=True, exist_ok=True)

run_trains("sweep_lr", workspace, tcs, mdc)

print("所有学习率扫描完成。")
