import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureEncoding, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(out_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv0(x)
        out = self.conv1(residual)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + residual


class FeatureDecoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureDecoding, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(out_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv0(x)
        out = self.conv1(residual)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + residual


class DDNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDNet, self).__init__()
        self.enc1 = FeatureEncoding(in_channels, 128)
        self.enc2 = FeatureEncoding(128, 128)
        self.enc3 = FeatureEncoding(128, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec2 = FeatureDecoding(256, 128)
        self.dec1 = FeatureDecoding(256, out_channels)

        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)


    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Decoder
        dec2 = self.dec2(torch.cat((enc2, self.upconv2(enc3)), dim=1))
        dec1 = self.dec1(torch.cat((enc1, self.upconv1(dec2)), dim=1))

        return dec1 + x


if __name__ == "__main__":
    # Example usage:
    model = UNet(in_channels=3, out_channels=3)
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
