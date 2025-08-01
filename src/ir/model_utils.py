import torch.nn as nn
from torch import Tensor
from typing import Optional


class CondSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *cond):
        for layer in self.layers:
            if len(cond) > 0:
                x = layer(x, *cond)
            else:
                x = layer(x)
        return x
    
    
class ScriptableCondSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond1: Optional[Tensor], cond2: Optional[Tensor]):
        for layer in self.layers:
            x = layer(x, cond1, cond2)
        return x


if __name__ == "__main__":
    class Dummy(nn.Module):
        def forward(self, x, c1=None, c2=None, c3=None):
            print(f"{x=}, {c1=}, {c2=}, {c3=}")
            return x

    model = CondSequential(Dummy())
    model(42, 1, 2, 3)
    