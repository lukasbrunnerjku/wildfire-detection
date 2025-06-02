import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm.layers import trunc_normal_, DropPath

from .norms import AdaLayerNorm
from .model_utils import CondSequential


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(
        self,
        dim,
        embed_dim: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        expansion_ratio=8/3,
        kernel_size=7,
        conv_ratio=1.0,
        drop_path=0.,
    ):  # https://github.com/yuweihao/MambaOut
        super().__init__()
        self.norm = AdaLayerNorm(dim, embed_dim, num_embeddings)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self,
        x: Tensor,
        emb: Optional[Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        shortcut = x # [B, H, W, C]
        x, gate = self.norm(x, emb, class_labels)  # We do not use this gate!
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut # [B, H, W, C]

            
class MambaOutIR(nn.Module):

    def __init__(
        self,
        in_channels: int,
        block_out_channels: tuple[int, ...] = (32, 64, 96, 128),
        oss_blocks_per_scale: tuple[int, ...] = (2, 2, 3, 4),
        num_class_embeds: Optional[int] = None,  # Enable conditioning
        oss_refine_blocks: int = 2, 
        predict_image: bool = True,
        local_embeds: bool = False,  # Each Norm Layer learns its own embedding table
        drop_path: float = 0.,
    ):
        super().__init__()
        if len(block_out_channels) != len(oss_blocks_per_scale):
            raise ValueError("Must provide same number of `block_out_channels` and `oss_blocks_per_scale`")

        self.predict_image = predict_image  # Residual or restored image?

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, 1, 1)

        self.local_embeds = local_embeds
        embed_dim = None
        self.embedding = None
        num_embeds = None
        if num_class_embeds is not None:
            embed_dim = block_out_channels[0] * 4
            if local_embeds is False:
                self.embedding = nn.Embedding(num_class_embeds, embed_dim)
            else:
                num_embeds = num_class_embeds

        self.down_blocks = nn.ModuleList([])
        self.downsample_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        self.skip_projections = nn.ModuleList([])
        self.refine_block = None
        self.conv_out = nn.Conv2d(block_out_channels[0], in_channels, 3, 1, 1)

        # down, total stride of 2^(len(block_out_channels) - 1)
        for i in range(len(block_out_channels) - 1):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i + 1]

            blocks = []
            for _ in range(oss_blocks_per_scale[i]):
                blocks.append(GatedCNNBlock(
                    in_channels,
                    embed_dim,
                    num_embeds,
                    drop_path=drop_path,
                ))
            self.down_blocks.append(CondSequential(*blocks))

            self.downsample_blocks.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))

        # mid, use last block_out_channels and oss_blocks_per_scale
        blocks = []
        for _ in range(oss_blocks_per_scale[-1]):
            blocks.append(GatedCNNBlock(
                out_channels,
                embed_dim,
                num_embeds,
                drop_path=drop_path,
            ))
        self.mid_block = CondSequential(*blocks)

        # up, till input resolution
        for i in reversed(range(1, len(block_out_channels))):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i - 1]

            self.upsample_blocks.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))

            self.skip_projections.append(nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0))

            blocks = []
            for _ in range(oss_blocks_per_scale[i - 1]):
                blocks.append(GatedCNNBlock(
                    out_channels,
                    embed_dim,
                    num_embeds,
                    drop_path=drop_path,
                ))
            self.up_blocks.append(CondSequential(*blocks))

        if oss_refine_blocks > 0:
            blocks = []
            for _ in range(oss_refine_blocks):
                blocks.append(GatedCNNBlock(
                    out_channels,
                    embed_dim,
                    num_embeds,
                    drop_path=drop_path,
                ))
            self.refine_block = CondSequential(*blocks)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MambaOutIR with {1e-6 * n_params:.3f} M parameters.")
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def forward(
        self,
        x: Tensor,
        emb: Optional[Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        x_input = x  # BxCxHxW

        x = self.conv_in(x)  # BxCxHxW

        if emb is not None:
            emb = self.embedding(emb)

        residuals = ()
        for i in range(len(self.down_blocks)):
            x = x.permute(0, 2, 3, 1)  # BxHxWxC
            x = self.down_blocks[i](x, emb, class_labels)  
            x = x.permute(0, 3, 1, 2)  # BxCxHxW
            residuals += (x,)  
            x = self.downsample_blocks[i](x)

        x = x.permute(0, 2, 3, 1)  # BxHxWxC
        x = self.mid_block(x, emb, class_labels)
        x = x.permute(0, 3, 1, 2)  # BxCxHxW

        residuals = tuple(reversed(residuals))
        for i in range(len(self.up_blocks)):
            x = self.upsample_blocks[i](x)
            x = torch.concat((x, residuals[i]), 1)
            x = self.skip_projections[i](x)
            x = x.permute(0, 2, 3, 1)  # BxHxWxC
            x = self.up_blocks[i](x, emb, class_labels)
            x = x.permute(0, 3, 1, 2)  # BxCxHxW

        if self.refine_block is not None:
            x = x.permute(0, 2, 3, 1)  # BxHxWxC
            x = self.refine_block(x, emb, class_labels)
            x = x.permute(0, 3, 1, 2)  # BxCxHxW
        
        x = self.conv_out(x)

        if self.predict_image:
            x = x + x_input

        return x  # BxCxHxW
        
        
if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128).cuda()
    emb = torch.randint(0, 31, (2,)).cuda()
    class_labels = torch.randint(0, 31, (2,)).cuda()

    model = MambaOutIR(1, (32, 64, 96), (2, 2, 3), None).cuda()
    y = model(x, None)
    print(y.shape)

    model = MambaOutIR(1, (32, 64, 96), (2, 2, 3), 31).cuda()
    y = model(x, emb)
    print(y.shape)
    
    model = MambaOutIR(1, (32, 64, 96), (2, 2, 3), 31, local_embeds=True).cuda()
    y = model(x, None, class_labels)
    print(y.shape)
    