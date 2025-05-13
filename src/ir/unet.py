import torch
import torch.nn as nn
from typing import Optional

from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_block_types: tuple[str, ...],
        up_block_types: tuple[str, ...],
        block_out_channels: tuple[int, ...],
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        num_class_embeds: Optional[int] = None,
    ):
        super().__init__()

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        
        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # class embedding
        if num_class_embeds is not None:
            embed_dim = block_out_channels[0] * 4
            self.embedding = nn.Embedding(num_class_embeds, embed_dim)
        else:
            embed_dim = None
            self.embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        if resnet_time_scale_shift != "spatial":
            # Not seen a mid block that support ada_group?
            # Does not support "spatial", but cross attention for sparse
            # conditions.
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=block_out_channels[-1],
                attention_head_dim=attention_head_dim,
                resnet_groups=norm_num_groups,
            )
        else:
            # Can be used for dense conditions with "spatial"
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"UNet with {1e-6 * n_params:.3f} M parameters.")

    def forward(self, sample, class_labels=None):
        
        # 1. get embedding
        emb = None
        if self.embedding is not None:
            emb = self.embedding(class_labels).to(dtype=sample.dtype)
        elif self.embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return sample
    

if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128)
    cond = torch.randint(0, 31, (2,))

    # NOTE: stride := 2^(len(block_out_channels) - 1)
    model = UNet(
        in_channels=1,
        out_channels=1,
        down_block_types=["KDownBlock2D", "KDownBlock2D", "KDownBlock2D"],
        up_block_types=["KUpBlock2D", "KUpBlock2D", "KUpBlock2D"],
        block_out_channels=[32, 64, 128],
        num_class_embeds=31,
        resnet_time_scale_shift="ada_group",  # or "spatial" if dense
    )
    with torch.no_grad():
        y = model(x, cond)
    print(y.shape)

    # model = UNet(
    #     in_channels=1,
    #     out_channels=1,
    #     down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D"],
    #     up_block_types=["UpBlock2D", "UpBlock2D", "UpBlock2D"],
    #     block_out_channels=[32, 64, 128],
    # )
    # with torch.no_grad():
    #     y = model(x)
    # print(y.shape)
