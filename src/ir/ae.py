from torch import nn
from torch.nn import functional as F


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


class RepeatedResBlock(nn.Module):
    def __init__(self, channel, res_channel, res_block, conv_cls=None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[ResBlock(channel, res_channel, conv_cls) for _ in range(res_block)]
        )

    def forward(self, x):
        return self.net(x)


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
            Down(in_channel + 1, channel // 4, conv_cls),
            RepeatedResBlock(channel // 4, res_channel // 4, res_block, conv_cls),
            Down(channel // 4, channel, conv_cls),
            RepeatedResBlock(channel, res_channel, res_block, conv_cls),
        )

    def forward(self, input):
        return self.blocks(F.pad(input, (0, 0, 0, 0, 0, 1), mode="constant", value=1.0))


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
    

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channels=128,
        residual_blocks=2,
        residual_channels=32,
        embed_dim=64,
        conv_cls=None,
        conv_transpose_cls=None,
    ):
        super().__init__()

        if conv_cls is None:
            conv_cls = nn.Conv2d

        self.encoder = nn.Sequential(
            Encoder(in_channel, channels, residual_blocks, residual_channels, conv_cls),
            conv_cls(channels, embed_dim, 1),
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

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y
    

if __name__ == "__main__":
    model = Autoencoder(in_channel=1)
    