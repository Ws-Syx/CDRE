import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .entropy_models import BitEstimator, LowerBound

class DistortionTransform(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims)
    
    def forward(self, x):
        x = self.fc(x)
        return x


class DistortionExtractor(nn.Module):
    def __init__(self, in_dims, bottleneck_dims, token_dims):
        super().__init__()

        # encoder
        kernel_size = 3
        mid_channels = 16
        
        self.encoder_sensitive = ExtractorSensitive(in_dims, mid_channels)
        self.encoder = Encoder(in_dims, mid_channels, bottleneck_dims)
        
        # bottleneck
        self.sigmoid = nn.Sigmoid()
        self.bits_estimator = BitEstimator(bottleneck_dims)
        
        # decoder
        tmp = (bottleneck_dims + token_dims) // 2
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_dims, tmp, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.Conv2d(tmp, tmp, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.Conv2d(tmp, token_dims, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        )
        
    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        # return CompressionModel.probs_to_bits(probs)
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits
    
    def forward(self, x_hat, x_raw, original_size=None):
        # create raw and lossy prior
        f1_raw, f2_raw, f3_raw = self.encoder_sensitive(x_raw)
        f1_hat, f2_hat, f3_hat = self.encoder_sensitive(x_hat)
        
        # compress
        y = self.encoder(x_raw, x_hat, f1_hat, f2_hat, f3_hat, f1_raw, f2_raw, f3_raw)

        # quantize and transmition
        y_hat = y + (torch.round(y) - y).detach()
        
        # reconstruct
        distortion_hat = self.decoder(y_hat)
        # flatten to token-like
        distortion_hat = distortion_hat.flatten(2).transpose(1, 2)
        
        # calculate bpp
        bits_z = self.get_z_bits(y_hat, self.bits_estimator)
        bits_z = torch.sum(bits_z)
        bpp = bits_z / (x_raw.shape[0] * x_raw.shape[2] * x_raw.shape[3])
        
        return distortion_hat, {'bpp': bpp, 'f1_raw': f1_raw, 'f2_raw': f2_raw, 'f3_raw': f3_raw, 'f1_hat': f1_hat, 'f2_hat': f2_hat, 'f3_hat': f3_hat}


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Linear transformations for query, key, and value
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        
        # Final linear layer for output
        self.linear_out = nn.Linear(dim, dim)
        
    def forward(self, x, distortion):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, dim).
            distortion (torch.Tensor): Distortion tensor of shape (B, M, dim).
        
        Returns:
            torch.Tensor: Output tensor after cross-attention of shape (B, N, dim).
        """
        batch_size, seq_len_x, _ = x.size()
        _, seq_len_d, _ = distortion.size()
        
        # Linear transformations for query, key, and value
        q = self.linear_q(x)  # (B, N, dim)
        k = self.linear_k(distortion)  # (B, M, dim)
        v = self.linear_v(distortion)  # (B, M, dim)
        
        # Reshape to include multiple heads
        q = q.view(batch_size, seq_len_x, self.num_heads, -1).transpose(1, 2)  # (B, num_heads, N, dim_per_head)
        k = k.view(batch_size, seq_len_d, self.num_heads, -1).transpose(1, 2)  # (B, num_heads, M, dim_per_head)
        v = v.view(batch_size, seq_len_d, self.num_heads, -1).transpose(1, 2)  # (B, num_heads, M, dim_per_head)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  # (B, num_heads, N, M)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, N, M)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, N, dim_per_head)
        
        # Reshape and transpose to get the final output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_x, -1)  # (B, N, dim)
        
        # Apply final linear layer for output
        output = self.linear_out(attn_output)  # (B, N, dim)
        
        return output


class ExtractorSensitive(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtractorSensitive, self).__init__()
        
        # mid_channels = (in_channels + out_channels) // 2
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(attention)
        attention = self.sigmoid(attention)
        return x * attention


class ModulationModule(nn.Module):
    def __init__(self, in_channels, embed_channels):
        super(ModulationModule, self).__init__()
        self.scale_conv = nn.Sequential(
            nn.Conv2d(embed_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.shift_conv = nn.Sequential(
            nn.Conv2d(embed_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x, f_hat, f_raw):
        combined_features = torch.cat((f_hat, f_raw), dim=1)
        scale = self.scale_conv(combined_features)
        shift = self.shift_conv(combined_features)
        # print("x: ", x.shape)
        # print("scale: ", scale.shape)
        # print("shift: ", shift.shape)
        x = x + x * scale + shift
        x = x + self.refine(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, embed_channels, out_channels):
        super(Encoder, self).__init__()

        mid_channels = (in_channels + out_channels) // 2
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(3 * in_channels, mid_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(mid_channels), 
            nn.LeakyReLU(inplace=True),
            SAM(mid_channels)
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(mid_channels + 2 * embed_channels, mid_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(mid_channels), 
            nn.LeakyReLU(inplace=True),
            SAM(mid_channels)
        )
        self.stage2_mod = ModulationModule(mid_channels, embed_channels)
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(mid_channels + 2 * embed_channels, mid_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(mid_channels), 
            nn.LeakyReLU(inplace=True),
            SAM(mid_channels)
        )
        self.stage3_mod = ModulationModule(mid_channels, embed_channels)
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(mid_channels + 2 * embed_channels, mid_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            SAM(mid_channels)
        )
        self.stage4_mod = ModulationModule(mid_channels, embed_channels)
        
        self.stage5 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),  
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(inplace=True),
            SAM(out_channels)
        )
        
    def forward(self, x, x_hat, f1_hat, f2_hat, f3_hat, f1_raw, f2_raw, f3_raw):
        x = self.stage1(torch.cat((x, x_hat, 5.0 * torch.abs(x - x_hat)), dim=1))
        
        x = self.stage2_mod(x, f1_hat, f1_raw)
        x = self.stage2(torch.cat((x, f1_hat, f1_raw), dim=1))
        
        x = self.stage3_mod(x, f2_hat, f2_raw)
        x = self.stage3(torch.cat((x, f2_hat, f2_raw), dim=1))
        
        x = self.stage4_mod(x, f3_hat, f3_raw)
        x = self.stage4(torch.cat((x, f3_hat, f3_raw), dim=1))
        
        x = self.stage5(x)
        
        return x

if __name__ == "__main__":
    x = torch.zeros((4, 9, 384, 640))

    model = DistortionExtractor(9, 8, 96)
    y = model(x)
    print(y.shape)


