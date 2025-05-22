"""
UVM-Net Code: https://github.com/zzr-idam/UVM-Net

Efficient Selective SSM Code: https://github.com/state-spaces/mamba (UVM-Net uses this code)

VmambaIR: https://arxiv.org/pdf/2403.11423 (no code yet)

Educational Code Examples: https://github.com/alxndrTL/mamba.py

Theory: https://huggingface.co/blog/lbourdois/get-on-the-ssm-train
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mamba_ssm import Mamba  # TODO Mamba or Mamba2 ?

from .scan import to_line_scanable_sequence
from .norms import AdaLayerNorm
from .model_utils import CondSequential


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

    def forward(self, fo1: Tensor, fo2: Tensor):
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


class Norm(nn.Module):

    def __init__(self, norm_type: str, in_channels: int, img_sz: tuple[int], embed_dim: Optional[int] = None):
        super().__init__()
        assert norm_type in ("layer", "group", "ada_group")

        if norm_type == "layer":
            # The ONLY reason we need to know spatial resolution!
            self.norm = nn.LayerNorm((in_channels, *img_sz))  # affine per channel AND per pixel
        elif norm_type == "group":
            self.norm = nn.GroupNorm(get_groups(in_channels), in_channels)  # affine per channel group
        elif norm_type == "ada_group":
            assert embed_dim is not None
            # TODO read more about layer vs group norm
            # !! https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
            self.norm = AdaGroupNorm(embed_dim, in_channels, get_groups(in_channels))  # affine per channel group

    def forward(self, x: Tensor, emb: Optional[Tensor] = None):
        if emb is None:
            return self.norm(x)
        else:
            return self.norm(x, emb)


class OSSBlock(nn.Module):

    def __init__(
        self,
        norm_type: str,
        in_channels: int,
        img_sz: tuple[int],
        embed_dim: Optional[int] = None,
        ffn_expansion_factor: int = 2,
    ):
        super().__init__()
        self.norm1 = Norm(norm_type, in_channels, img_sz, embed_dim)
        self.oss_module = OSSModule(in_channels)
        self.norm2 = Norm(norm_type, in_channels, img_sz, embed_dim)
        self.effn = EFFN(in_channels, ffn_expansion_factor)

    def forward(self, x: Tensor, emb: Optional[Tensor] = None):
        x = self.oss_module(self.norm1(x, emb)) + x
        x = self.effn(self.norm2(x, emb)) + x
        return x
    

class VmambaIR(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_sz: tuple[int],
        block_out_channels: tuple[int, ...] = (48, 96, 192, 384),
        oss_blocks_per_scale: tuple[int, ...] = (4, 4, 6, 8),
        num_class_embeds: Optional[int] = None,
        oss_refine_blocks: int = 2, 
        predict_image: bool = True,
    ):
        """
        L1 objective
        AdamW
        betas = (0.9, 0.999)
        lr = 3e-4
        weight_decay = 1e-4
        """
        super().__init__()
        if len(block_out_channels) != len(oss_blocks_per_scale):
            raise ValueError("Must provide same number of `block_out_channels` and `oss_blocks_per_scale`")

        self.predict_image = predict_image  # Residual or restored image?

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)

        if num_class_embeds is not None:
            norm_type = "ada_group"
            embed_dim = block_out_channels[0] * 4
            self.embedding = nn.Embedding(num_class_embeds, embed_dim)
        else:
            norm_type = "group"
            embed_dim = None
            self.embedding = None

        self.down_blocks = nn.ModuleList([])
        self.downsample_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        self.skip_projections = nn.ModuleList([])
        self.refine_block = None
        self.conv_out = nn.Conv2d(block_out_channels[0], in_channels, 3, 1, 1)

        h, w = img_sz

        # down, total stride of 2^(len(block_out_channels) - 1)
        for i in range(len(block_out_channels) - 1):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i + 1]

            blocks = []
            for _ in range(oss_blocks_per_scale[i]):
                blocks.append(OSSBlock(norm_type, in_channels, (h, w), embed_dim))
            self.down_blocks.append(CondSequential(*blocks))

            self.downsample_blocks.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            h, w = h // 2, w // 2

        # mid, use last block_out_channels and oss_blocks_per_scale
        blocks = []
        for _ in range(oss_blocks_per_scale[-1]):
            blocks.append(OSSBlock(norm_type, block_out_channels[-1], (h, w), embed_dim))
        self.mid_block = CondSequential(*blocks)

        # up, till input resolution
        for i in reversed(range(1, len(block_out_channels))):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i - 1]

            self.upsample_blocks.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            h, w = 2 * h, 2 * w

            self.skip_projections.append(nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))

            blocks = []
            for _ in range(oss_blocks_per_scale[i - 1]):
                blocks.append(OSSBlock(norm_type, out_channels, (h, w), embed_dim))
            self.up_blocks.append(CondSequential(*blocks))

        if oss_refine_blocks > 0:
            blocks = []
            for _ in range(oss_refine_blocks):
                blocks.append(OSSBlock(norm_type, block_out_channels[0], (h, w), embed_dim))
            self.refine_block = CondSequential(*blocks)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VmambaIR with {1e-6 * n_params:.3f} M parameters.")
            
    def forward(self, x: Tensor, emb: Optional[Tensor] = None):
        """Returns the IR residual, not restored image directly."""
        x_input = x

        x = self.conv_in(x)

        if emb is not None:
            emb = self.embedding(emb)

        residuals = ()
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x, emb)
            residuals += (x,)
            x = self.downsample_blocks[i](x)

        x = self.mid_block(x, emb)

        residuals = tuple(reversed(residuals))
        for i in range(len(self.up_blocks)):
            x = self.upsample_blocks[i](x)
            x = torch.concat((x, residuals[i]), 1)
            x = self.skip_projections[i](x)
            x = self.up_blocks[i](x, emb)

        if self.refine_block is not None:
            x = self.refine_block(x, emb)

        x = self.conv_out(x)

        if self.predict_image:
            x = x + x_input

        return x


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128).cuda()
    emb = torch.randint(0, 31, (2,)).cuda()

    model = VmambaIR(1, (128, 128), (32, 64, 96), (2, 2, 2), None).cuda()
    y = model(x, None)
    print(y.shape)

    model = VmambaIR(1, (128, 128), (32, 64, 96), (2, 2, 2), 31).cuda()
    y = model(x, emb)
    print(y.shape)
