"""A two-level U-Net. Forward returns logits for numerically stable BCE."""
import torch
from torch import nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
    )


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = double_conv(1, 16)
        self.enc2 = double_conv(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        z = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(z), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
