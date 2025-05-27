import torch.nn as nn
from torch import Tensor


class MambaOut(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        expand: int = 2,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
    def forward(self, x: Tensor):
        
if __name__ == "__main__":
    