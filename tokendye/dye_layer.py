import torch
from torch import nn


class Dye(nn.Module):
    def __init__(
        self,
        d_model: int,
        rank: int,
        alpha: float | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.A = nn.Linear(d_model, rank, bias=False, dtype=dtype)
        self.B = nn.Linear(rank, d_model, bias=False, dtype=dtype)
        self.scaling = (alpha / rank) if alpha is not None else 1.0

        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * self.B(self.A(x))
        return x + delta
