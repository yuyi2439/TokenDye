from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from .config import ModelDyeConfig


class DyeModule(nn.Module):
    def __init__(
        self,
        mdc: "ModelDyeConfig",
        rank: int,
        alpha: float | None = None,
        init_weight=True,
    ):
        super().__init__()
        d_model = mdc.d_model
        dtype = getattr(torch, mdc.dtype)

        self.A = nn.Linear(d_model, rank, bias=False, dtype=dtype)
        self.B = nn.Linear(rank, d_model, bias=False, dtype=dtype)
        self.scaling = (alpha / rank) if alpha is not None else 1.0

        if init_weight:
            self._init_weight()

    def forward(self, x: "Tensor") -> "Tensor":
        delta = self.scaling * self.B(self.A(x))
        return x + delta

    def _init_weight(self, generator: Optional[torch.Generator] = None):
        nn.init.normal_(self.A.weight, std=0.01, generator=generator)
        nn.init.zeros_(self.B.weight)


def setup_dye_modules(mdc: "ModelDyeConfig", rank: int, m_device, sub_module=True):
    labels = mdc.labels
    # TODO 处理子模块

    dye_modules = nn.ModuleDict()
    for _dye_label in labels:
        module = DyeModule(mdc, rank).to(m_device)
        dye_modules[_dye_label.name] = module

    return dye_modules
