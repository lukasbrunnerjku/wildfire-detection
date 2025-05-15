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


def to_line_scanable_sequence(x: Tensor, no_jump: bool, direction: str):
    B, C, H, W = x.shape

    """
    Imagine H, W as y, x axes here.

    0, 1, 2
    3, 4, 5
    6, 7, 8
    """
    if direction in ("h_forward", "h_backward"):
        # h_forward: "...scanning from the top left to the bottom right"
        if no_jump:
            """
            Along the sequence dimension L we would get
            => 0, 1, 2, 5, 4, 3, 6, 7, 8
            """
            x = torch.stack([x[:, :, 0::2, :], x[:, :, 1::2, ::-1]], dim=3)  # BxCx(H/2)x2xW
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC
        else:
            """
            Along the sequence dimension L we would get
            => 0, 1, 2, 3, 4, 5, 6, 7, 8
            """
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC
        
        if "h_backward":
            """
            Along the sequence dimension L we would get either
            => 8, 7, 6, 3, 4, 5, 2, 1, 0 (every_2nd_row_in_reverse)
            or
            => 8, 7, 6, 5, 4, 3, 2, 1, 0
            """
            x = x[:, ::-1, :]  # BxLxC
    
    elif direction in ("w_forward", "w_backward"):
        # w_forward: "...scanning from the bottom left to the top right"
        if no_jump:
            """
            Starting from
            0, 1, 2
            3, 4, 5
            6, 7, 8
            and after permute
            0, 3, 6
            1, 4, 7
            2, 5, 8

            Along the sequence dimension L we would get
            => 6, 3, 0, 1, 4, 7, 8, 5, 2
            """
            x = x.permute(0, 1, 3, 2)  # BxCxWxH
            x = torch.stack([x[:, :, 0::2, ::-1], x[:, :, 1::2, :]], dim=3)  # BxCx(W/2)x2xH
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC

        else:
            """
            Starting from
            0, 1, 2
            3, 4, 5
            6, 7, 8
            and after permute
            0, 3, 6
            1, 4, 7
            2, 5, 8
            and after reverse in H dimension
            6, 3, 0
            7, 4, 1
            8, 5, 2

            Along the sequence dimension L we would get
            => 6, 3, 0, 7, 4, 1, 8, 5, 2
            """
            x = x.permute(0, 1, 3, 2)  # BxCxWxH
            x = x[:, :, :, ::-1]  # Reverse in H dimension
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC

        if "w_backward":
            x = x[:, ::-1, :]  # BxLxC

    else:
        raise ValueError(f"Unknown {direction=}?")


    return x
    

class OSS(nn.Module):

    def __init__(self, in_channels: int):
        """Omni Selective Scan (OSS)"""
        super().__init__()
        self.h_forward_scan = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=in_channels, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.h_backward_scan = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=in_channels, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.w_forward_scan = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=in_channels, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.w_backward_scan = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=in_channels, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(self, x1, x2):
        """
        
        """
        B, C, H, W = x1.shape  # BxCxHxW

        x1.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC
        return x
    

class OSSModule(nn.Module):

    def __init__(self, in_channels: int, bias: bool = True):
        super().__init__()
        self.in_projection = nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=bias)
        self.dwconv = DConv(in_channels, 1, 3, 1, bias=bias)
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
        x1, x2 = self.in_projection(x).chunk(2, dim=1)
        x1 = F.silu(self.dwconv(x1))
        x2 = F.silu(x2)

        x = self.out_projection(x)
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
