import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class MultiUp(nn.Module):
    def __init__(self, channels, out_channel):
        super().__init__()
        ups = []
        for ch_in, ch_out in zip(channels, channels[1:]):
            ups.append(Up(ch_in, ch_out))
        ups.append(nn.Conv2d(channels[-1], out_channel, kernel_size=1))
        self.convs = nn.Sequential(*ups)

    def forward(self, x):
        return self.convs(x)


if __name__ == "__main__":
    convs = MultiUp([384, 384, 192, 192, 96, 48, 24], 1)

    z = torch.zeros(4, 384, 1, 1)
    x = convs(z)
    breakpoint()
