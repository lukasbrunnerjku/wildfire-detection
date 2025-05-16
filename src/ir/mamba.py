"""
UVM-Net Code: https://github.com/zzr-idam/UVM-Net

VmambaIR: https://arxiv.org/pdf/2403.11423 (no code yet)

Educational Code Examples: https://github.com/alxndrTL/mamba.py

Theory: https://huggingface.co/blog/lbourdois/get-on-the-ssm-train
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# Efficient Selective SSM Code: https://github.com/state-spaces/mamba (UVM-Net uses this code)
# ---> REQUIRES LINUX <---
# Mamba or Mamba2 ?
from mamba_ssm import Mamba  # pip install mamba-ssm[causal-conv1d]

from .scan import to_line_scanable_sequence


class DConv(nn.Conv2d):

    def __init__(self, in_channels, K, kernel_size, stride, bias=True):
        super().__init__(
            in_channels,
            K * in_channels,
            kernel_size,
            stride,
            padding="same",
            groups=in_channels,
            bias=bias,
        )


class EFFN(nn.Module):

    def __init__(self, in_channels: int, ffn_expansion_factor: float = 2.0, bias: bool = True):
        """https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py"""
        super().__init__()
        hidden_features = int(in_channels * ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_channels, 2 * hidden_features, 1, 1, 0, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_channels, 1, 1, 0, bias=bias)
        self.dwconv = DConv(2 * hidden_features, 1, 3, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class OSS(nn.Module):

    def __init__(self, in_channels: int):
        """Omni Selective Scan (OSS)"""
        super().__init__()
        self.mamba1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=4 * in_channels, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.mamba2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=2, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(self, fo1, fo2):
        B, C, H, W = fo1.shape  # BxCxHxW

        h_forward = to_line_scanable_sequence(fo1, False, "h_forward")  # BxLxC
        h_backward = to_line_scanable_sequence(fo1, False, "h_backward")  # BxLxC
        w_forward = to_line_scanable_sequence(fo1, False, "w_forward")  # BxLxC
        w_backward = to_line_scanable_sequence(fo1, False, "w_backward")  # BxLxC
        
        fop = torch.concat([h_forward, h_backward, w_forward, w_backward], 2)  # BxLx4C
        fop = self.mamba1(fop)  # BxLx4C
        fop = fop.reshape(B, H * W, 4, C).sum(2)  # BxLxC
        fop = fop.permute(0, 2, 1).reshape(B, C, H, W)

        residual = fop * fo2  # BxCxHxW

        foc = F.adaptive_avg_pool2d(residual, (1, 1))  # BxCx1x1
        foc = foc.reshape(B, C, 1)
        # Here, the sequence length "L" is the channel dimension "C"
        c_forward = foc  # BxCx1
        c_backward = foc.clone().flip(1)  # BxCx1
        foc = torch.concat([c_forward, c_backward], 2)  # BxCx2
        foc = self.mamba2(foc)  # BxCx2
        foc = foc.sum(2, keepdim=True).unsqueeze(-1)  # BxCx1x1

        foss = foc * residual + residual
        return foss
    

class OSSModule(nn.Module):

    def __init__(self, in_channels: int, bias: bool = True):
        super().__init__()
        self.in_projection = nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=bias)
        self.dwconv = DConv(in_channels, 1, 3, 1, bias=bias)
        self.oss = OSS(in_channels)
        self.out_projection = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        """
        The input of the OSS module is initially processed by a convolutional layer,
        generating two information flows. One flow undergoes refinement through
        depthwise convolution and a Silu activation, capturing intricate patterns.
        Simultaneously, the other flow is processed with a Silu activation.
        The two flows enter the core OSS mechanism, which models information across
        all feature dimensions. Subsequently, the two flows are fused within the OSS,
        merging the refined features with complementary information. After passing through
        a 1x1 convolution, the output of OSS generates the final output of the OSS block...
        """
        fo1, fo2 = self.in_projection(x).chunk(2, dim=1)
        fo1 = F.silu(self.dwconv(fo1))
        fo2 = F.silu(fo2)
        foss = self.oss(fo1, fo2)  # BxCxHxW
        x = self.out_projection(foss)
        return x
    

class OSSBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class VmambaIR(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

if __name__ == "__main__":
    pass
