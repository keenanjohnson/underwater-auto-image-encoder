"""
Attention-enhanced U-Net for improved underwater image enhancement
Incorporates spatial and channel attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionDoubleConv(nn.Module):
    """Double convolution with attention"""
    
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = CBAM(out_channels) if use_attention else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.attention:
            x = self.attention(x)
        return x


class AttentionUNet(nn.Module):
    """U-Net with attention mechanisms for underwater image enhancement"""
    
    def __init__(self, n_channels=3, n_classes=3, base_features=32):
        super(AttentionUNet, self).__init__()
        
        self.encoder1 = AttentionDoubleConv(n_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = AttentionDoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = AttentionDoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = AttentionDoubleConv(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = AttentionDoubleConv(base_features * 8, base_features * 16)
        
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.decoder4 = AttentionDoubleConv(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.decoder3 = AttentionDoubleConv(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.decoder2 = AttentionDoubleConv(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.decoder1 = AttentionDoubleConv(base_features * 2, base_features)
        
        self.final_conv = nn.Conv2d(base_features, n_classes, kernel_size=1)
        
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_features * 16, base_features * 16 // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 16 // 16, base_features * 16, 1),
            nn.Sigmoid()
        )
        
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        global_context = self.global_attention(bottleneck)
        bottleneck = bottleneck * global_context
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final_conv(dec1)
        return self.output_activation(out)


class WaterNet(nn.Module):
    """Specialized network for underwater image enhancement with water-specific modules"""
    
    def __init__(self, n_channels=3, n_classes=3, base_features=32):
        super(WaterNet, self).__init__()
        
        self.color_correction_module = nn.Sequential(
            nn.Conv2d(n_channels, base_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, n_channels, 1),
            nn.Tanh()
        )
        
        self.main_network = AttentionUNet(n_channels * 2, n_classes, base_features)
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        color_corrected = self.color_correction_module(x)
        color_corrected = x + color_corrected * 0.5
        
        combined_input = torch.cat([x, color_corrected], dim=1)
        
        enhanced = self.main_network(combined_input)
        
        output = enhanced + x * self.residual_weight
        
        return torch.clamp(output, 0, 1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AttentionUNet(n_channels=3, n_classes=3).to(device)
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)
    
    print(f"AttentionUNet output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    water_model = WaterNet().to(device)
    water_output = water_model(dummy_input)
    print(f"\nWaterNet output shape: {water_output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in water_model.parameters()):,}")