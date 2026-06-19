from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from .config import DyeConfig


class DyeModule(nn.Module):
    def __init__(
        self,
        config: "DyeConfig",
        alpha: float | None = None,
    ):
        super().__init__()
        rank = config.rank
        d_model = config.d_model
        dtype = getattr(torch, config.dtype)

        self.A = nn.Linear(d_model, rank, bias=False, dtype=dtype)
        self.B = nn.Linear(rank, d_model, bias=False, dtype=dtype)
        self.scaling = (alpha / rank) if alpha is not None else 1.0

        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: "Tensor") -> "Tensor":
        delta = self.scaling * self.B(self.A(x))
        return x + delta
