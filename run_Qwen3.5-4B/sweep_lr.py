from datetime import datetime

from train import run_trains
from utils import BASE, TrainConfig, setup_logger

from tokendye import ModelDyeConfig

# 要扫描的学习率列表
lrs = [
    8e-5,
    9e-5,
    1e-4,
    1.2e-4,
    1.6e-4,
]  #!! should: 7e-5 < lr < 2e-4, the best is 1e-4 roughly

# 搜索阶段的 epoch 数
SEARCH_EPOCHS = 15
RANK = 8

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
workspace = BASE / ".outputs" / f"{RUN_TS}-sweep_lr"
workspace.mkdir(parents=True, exist_ok=True)
logger = setup_logger(workspace, "sweep_lr")

mdc = ModelDyeConfig.load(BASE / "DyeConfig.json")
tcs = [TrainConfig(rank=RANK, lr=lr, total_epochs=SEARCH_EPOCHS) for lr in lrs]
logger.info(f"len(tcs): {len(tcs)}")

# import torch
# torch.manual_seed(6124655262693369293)


run_trains("sweep_lr", workspace, tcs, mdc, logger=logger, unify_dye_weight=True)

workspace.rename(workspace.with_name(workspace.name + "-finish"))

print("所有学习率扫描完成。")
