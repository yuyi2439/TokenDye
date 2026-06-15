from torch import nn
import torch

class Dye(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)

        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)
        
        self.A.to(torch.bfloat16)
        self.B.to(torch.bfloat16)
    
    def forward(self, x):
        delta = self.B(self.A(x))
        return x + delta
    
