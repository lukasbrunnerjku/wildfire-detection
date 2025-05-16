import torch.nn as nn


class CondSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond=None):
        for layer in self.layers:
            if cond is not None:
                x = layer(x, cond)
            else:
                x = layer(x)
        return x
