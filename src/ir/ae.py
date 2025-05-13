import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from .norms import AdaGroupNorm, get_activation
    

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, conv_cls=None):
        super().__init__()

        if conv_cls is None:
            conv_cls = nn.Conv2d

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(in_channel),
            conv_cls(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel),
            conv_cls(channel, in_channel, 1),
        )

    def forward(self, input):
        return input + self.net(input)
    

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
    

class CondResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        emb_channels,
        conv_cls=None,
        groups: int = 32,  # typical 32 if channels > 32, aim for 4-16 channels per group
        # >> to many groups make normalization unstable! Is independent of batch size.
        # groups = 1 => layernorm and groups = channels => instance norm
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        dropout: float = 0.0,
        non_linearity: str = "swish",
    ):  # see ResnetBlockCondNorm2D from HuggingFace
        super().__init__()

        if conv_cls is None:
            conv_cls = nn.Conv2d

        if groups_out is None:
            groups_out = groups

        if in_channel < 32:
            groups = 1  # layernorm
        
        if channel < 32:
            groups_out = 1  # layernorm

        self.norm1 = AdaGroupNorm(emb_channels, in_channel, groups, eps=eps)
        self.norm2 = AdaGroupNorm(emb_channels, channel, groups_out, eps=eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.nonlinearity = get_activation(non_linearity)
        self.conv1 = conv_cls(in_channel, channel, 3, padding=1)
        self.conv2 = conv_cls(channel, in_channel, 1)

    def forward(self, input, emb):  # 2nd input is condition
        hidden_states = input
        hidden_states = self.norm1(hidden_states, emb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states, emb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return input + hidden_states


class RepeatedResBlock(nn.Module):
    def __init__(self, channel, res_channel, res_block, conv_cls=None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[ResBlock(channel, res_channel, conv_cls) for _ in range(res_block)]
        )

    def forward(self, x):
        return self.net(x)
    

class CondRepeatedResBlock(nn.Module):
    def __init__(
        self,
        channel,
        res_channel,
        res_block,
        emb_channel,
        conv_cls=None,
    ) -> None:
        super().__init__()
        self.net = CondSequential(
            *[CondResBlock(channel, res_channel, emb_channel, conv_cls) for _ in range(res_block)]
        )

    def forward(self, x, emb):
        return self.net(x, emb)


class Down(nn.Module):
    def __init__(self, channel, channel_out, conv_cls=None) -> None:
        super().__init__()
        self.channel = channel

        if conv_cls is None:
            conv_cls = nn.Conv2d

        self.conv_strided = conv_cls(channel, channel_out, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_strided(x)
    

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, res_block, res_channel, conv_cls=None):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            Down(in_channel, channel // 4, conv_cls),
            RepeatedResBlock(channel // 4, res_channel // 4, res_block, conv_cls),
            Down(channel // 4, channel, conv_cls),
            RepeatedResBlock(channel, res_channel, res_block, conv_cls),
        )

    def forward(self, input):
        return self.blocks(input)


class CondEncoder(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        res_block,
        res_channel,
        emb_channel,
        conv_cls=None,
    ):
        super().__init__()
        
        self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
        self.down1 = Down(in_channel, channel // 4, conv_cls)
        self.block1 = CondRepeatedResBlock(
            channel // 4, res_channel // 4, res_block, emb_channel, conv_cls
        )
        self.down2 = Down(channel // 4, channel, conv_cls)
        self.block2 = CondRepeatedResBlock(
            channel, res_channel, res_block, emb_channel, conv_cls
        )

    def forward(self, input, emb):
        x = self.pad(input)
        x = self.down1(x)
        x = self.block1(x, emb)
        x = self.down2(x)
        x = self.block2(x, emb)
        return x
    

class Up(nn.Module):
    def __init__(self, channel, channel_out, conv_transpose_cls=None) -> None:
        super().__init__()
        self.channel = channel

        if conv_transpose_cls is None:
            conv_transpose_cls = nn.ConvTranspose2d

        self.convt_strided = conv_transpose_cls(
            channel, channel_out, 4, stride=2, padding=1
        )

    def forward(self, x):
        return self.convt_strided(x)
    

class Decoder(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        channel,
        res_block,
        res_channel,
        conv_cls=None,
        conv_transpose_cls=None,
    ):
        super().__init__()

        if conv_cls is None:
            conv_cls = nn.Conv2d

        if conv_transpose_cls is None:
            conv_transpose_cls = nn.ConvTranspose2d

        blocks = [
            conv_cls(in_channel, channel, 3, padding=1),
            RepeatedResBlock(channel, res_channel, res_block, conv_cls),
            Up(channel, channel // 2, conv_transpose_cls),
            RepeatedResBlock(channel // 2, res_channel // 2, 1, conv_cls),
            Up(channel // 2, channel // 4, conv_transpose_cls),
            RepeatedResBlock(channel // 4, res_channel // 4, 1, conv_cls),
            conv_cls(channel // 4, out_channel, 1),
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    

class CondDecoder(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        channel,
        res_block,
        res_channel,
        emb_channel,
        conv_cls=None,
        conv_transpose_cls=None,
    ):
        super().__init__()

        if conv_cls is None:
            conv_cls = nn.Conv2d

        if conv_transpose_cls is None:
            conv_transpose_cls = nn.ConvTranspose2d

        self.conv_in = conv_cls(in_channel, channel, 3, padding=1)
        self.conv1 = CondRepeatedResBlock(channel, res_channel, res_block, emb_channel, conv_cls)
        self.up1 = Up(channel, channel // 2, conv_transpose_cls)
        self.conv2 = CondRepeatedResBlock(channel // 2, res_channel // 2, 1, emb_channel, conv_cls)
        self.up2 = Up(channel // 2, channel // 4, conv_transpose_cls)
        self.conv3 = CondRepeatedResBlock(channel // 4, res_channel // 4, 1, emb_channel, conv_cls)
        self.conv_out = conv_cls(channel // 4, out_channel, 1)

    def forward(self, input, emb):
        x = self.conv_in(input)
        x = self.conv1(x, emb)
        x = self.up1(x)
        x = self.conv2(x, emb)
        x = self.up2(x)
        x = self.conv3(x, emb)
        x = self.conv_out(x)
        return x
    

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channels=128,
        residual_blocks=2,
        residual_channels=32,
        embed_dim=64,
        num_cond_embed: int = 0,
        cond_embed_dim: int = 64,
        conv_cls=None,
        conv_transpose_cls=None,
    ):
        """
        Conditions must be be indices to index the embedding table.
        In our case env. temp. start at 0 and end at 31 degree so we
        do not need a remaping from env. temp. to indices.
        """
        super().__init__()

        if conv_cls is None:
            conv_cls = nn.Conv2d

        self.enc_proj = conv_cls(channels, embed_dim, 1)

        if num_cond_embed > 0:
            self.embedding = nn.Embedding(num_cond_embed, cond_embed_dim)
            self.encoder = CondEncoder(
                in_channel,
                channels,
                residual_blocks,
                residual_channels,
                cond_embed_dim,
                conv_cls
            )
            self.decoder = CondDecoder(
                embed_dim,
                in_channel,
                channels,
                residual_blocks,
                residual_channels,
                cond_embed_dim,
                conv_cls,
                conv_transpose_cls,
            )
        else:
            self.embedding = None
            self.encoder = Encoder(
                in_channel,
                channels,
                residual_blocks,
                residual_channels,
                conv_cls
            )
            self.decoder = Decoder(
                embed_dim,
                in_channel,
                channels,
                residual_blocks,
                residual_channels,
                conv_cls,
                conv_transpose_cls,
            )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Autoencoder with {1e-6 * n_params:.3f} M parameters.")

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cond=None):
        # x ... BxCxHxW >> input image (aos)
        # cond ... B, >> indices of embedding table
        if self.embedding:
            emb = self.embedding(cond)
            h = self.enc_proj(self.encoder(x, emb))
            y = self.decoder(h, emb)
        else:
            h = self.enc_proj(self.encoder(x))
            y = self.decoder(h)
        return y
    

if __name__ == "__main__":
    num_cond_embed = 24
    in_channel = 1
    bsz = 2
    x = torch.randn(bsz, in_channel, 128, 128)
    model = Autoencoder(in_channel=in_channel, num_cond_embed=num_cond_embed)
    if num_cond_embed > 0:
        cond = torch.randint(0, num_cond_embed, (bsz,))
        y = model(x, cond)
    else:
        y = model(x)
    print(y.shape)
