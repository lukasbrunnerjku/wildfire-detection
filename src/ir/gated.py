import torch
from torch import Tensor
from torch import nn


class GatedOpWithActivation(nn.Module):
    def __init__(self, op, act, act_mask):
        super().__init__()
        self.op = op
        self.act = act
        self.act_mask = act_mask

    def forward(self, input):
        x, mask = torch.chunk(self.op(input), 2, dim=1)
        return self.act(x) * self.act_mask(mask)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs) -> None:
        super().__init__()

        self.gated = GatedOpWithActivation(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size, *args, **kwargs),
            nn.Identity(),
            nn.Hardsigmoid(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gated(x)


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs) -> None:
        super().__init__()

        self.gated = GatedOpWithActivation(
            nn.ConvTranspose2d(
                in_channels, out_channels * 2, kernel_size, *args, **kwargs
            ),
            nn.Identity(),
            nn.Hardsigmoid(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gated(x)