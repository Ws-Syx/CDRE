import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F

# from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

# class DistortionTransform(nn.Module):
#     def __init__(self, in_dims, out_dims):
#         super().__init__()
#         self.fc = nn.Linear(in_dims, out_dims)
    
#     def forward(self, x):
#         x = self.fc(x)
#         return x


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
        
        # decoder
        self.decoder = Decoder(bottleneck_dims, token_dims)
    
    def forward(self, x_hat, x_raw):
        # create raw and lossy prior
        f1_raw, f2_raw, f3_raw = self.encoder_sensitive(x_raw)
        f1_hat, f2_hat, f3_hat = self.encoder_sensitive(x_hat)
        
        # compress
        y = self.encoder(x_raw, x_hat, f1_hat, f2_hat, f3_hat, f1_raw, f2_raw, f3_raw)

        # quantize
        y = self.sigmoid(y)
        y_hat = y + (torch.round(y) - y).detach()
        # reconstruct
        distortion_hat = self.decoder(y_hat)
        
        return distortion_hat, {'f1_raw': f1_raw, 'f2_raw': f2_raw, 'f3_raw': f3_raw, 'f1_hat': f1_hat, 'f2_hat': f2_hat, 'f3_hat': f3_hat}


class EmbeddingModule(nn.Module):
    def __init__(self, in_channels, distortion_channels, reduction=16):
        super(EmbeddingModule, self).__init__()

        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + distortion_channels, (in_channels + distortion_channels) // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d((in_channels + distortion_channels) // reduction, in_channels + distortion_channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        # Convolutional layer to combine features
        self.combine_conv = nn.Conv2d(in_channels + distortion_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, distortion):
        # Concatenate the input features and distortion
        combined = torch.cat((x, distortion), dim=1)
        
        # Apply spatial attention
        avg_out = torch.mean(combined, dim=1, keepdim=True)
        max_out, _ = torch.max(combined, dim=1, keepdim=True)
        spatial_attention_map = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        # print(f"combined: {combined.shape}, spatial_attention_map: {spatial_attention_map.shape}")
        combined = combined * spatial_attention_map
        
        # Apply channel attention
        channel_attention_map = self.channel_attention(combined)
        # print(f"combined: {combined.shape}, channel_attention_map: {channel_attention_map.shape}")
        combined = combined * channel_attention_map
        
        # Combine features and reduce to original input channels
        fused_features = self.combine_conv(combined)
        
        return fused_features


class EmbeddingModule_v2(nn.Module):
    def __init__(self, in_channels, distortion_channels):
        super(EmbeddingModule_v2, self).__init__()

        middle_channel = 16
        self.enc = nn.Conv2d(in_channels + distortion_channels, middle_channel, kernel_size=3, stride=1, padding=1)
        self.dec = nn.Conv2d(middle_channel, in_channels, kernel_size=3, stride=1, padding=1)

        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(middle_channel, middle_channel // 2, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(middle_channel // 2, middle_channel, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, distortion):
        # Concatenate the input features and distortion
        combined = self.enc(torch.cat((x, distortion), dim=1))
        
        # Apply spatial attention
        avg_out = torch.mean(combined, dim=1, keepdim=True)
        max_out, _ = torch.max(combined, dim=1, keepdim=True)
        spatial_attention_map = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        # print(f"combined: {combined.shape}, spatial_attention_map: {spatial_attention_map.shape}")
        combined = combined * spatial_attention_map
        
        # Apply channel attention
        channel_attention_map = self.channel_attention(combined)
        combined = combined * channel_attention_map
        
        # Combine features and reduce to original input channels
        fused_features = self.dec(combined)
        
        return fused_features
        

class DistortionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DistortionLayer, self).__init__()

        mid_channels = (in_channels + out_channels) // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


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
        x = self.stage1(torch.cat((x, x_hat, x - x_hat), dim=1))
        
        x = self.stage2_mod(x, f1_hat, f1_raw)
        x = self.stage2(torch.cat((x, f1_hat, f1_raw), dim=1))
        
        x = self.stage3_mod(x, f2_hat, f2_raw)
        x = self.stage3(torch.cat((x, f2_hat, f2_raw), dim=1))
        
        x = self.stage4_mod(x, f3_hat, f3_raw)
        x = self.stage4(torch.cat((x, f3_hat, f3_raw), dim=1))
        
        x = self.stage5(x)
        
        return x


class Decoder(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(Decoder, self).__init__()  
        mid_channels = (in_channels + out_channels) // 2
        
        self.stage1 = nn.Sequential(  
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),  
        )
        self.stage2 = nn.Sequential(  
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        )
        self.stage3 = nn.Sequential(  
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        )
        self.stage4 = nn.Sequential(  
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.stage5 = nn.Sequential(  
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
        )
  
    def forward(self, x):  
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x  


if __name__ == "__main__":
    x = torch.zeros((4, 9, 384, 640))

    model = DistortionExtractor(9, 8, 96)
    y = model(x)
    print(y.shape)


