"""
Custom loss functions for underwater image enhancement
Includes perceptual, color, and quality-specific losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, feature_layers: List[int] = [3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.ModuleList()
        
        for layer_idx in feature_layers:
            self.feature_extractor.append(
                nn.Sequential(*[vgg[i] for i in range(layer_idx + 1)])
            )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, output, target):
        loss = 0
        
        for extractor in self.feature_extractor:
            output_features = extractor(output)
            target_features = extractor(target)
            loss += F.mse_loss(output_features, target_features)
        
        return loss / len(self.feature_extractor)


class ColorLoss(nn.Module):
    """Color consistency loss for underwater images"""
    
    def __init__(self, weight_lab: float = 0.5):
        super(ColorLoss, self).__init__()
        self.weight_lab = weight_lab
    
    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space (simplified)"""
        rgb = rgb.permute(0, 2, 3, 1)
        
        rgb_linear = torch.where(
            rgb > 0.04045,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92
        )
        
        xyz_transform = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb.device, dtype=rgb.dtype)
        
        xyz = torch.matmul(rgb_linear, xyz_transform.T)
        
        xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device)
        
        def f(t):
            delta = 6.0 / 29.0
            return torch.where(
                t > delta ** 3,
                torch.pow(t, 1.0 / 3.0),
                t / (3 * delta ** 2) + 4.0 / 29.0
            )
        
        xyz_f = f(xyz)
        
        L = 116.0 * xyz_f[..., 1] - 16.0
        a = 500.0 * (xyz_f[..., 0] - xyz_f[..., 1])
        b = 200.0 * (xyz_f[..., 1] - xyz_f[..., 2])
        
        lab = torch.stack([L, a, b], dim=-1)
        return lab.permute(0, 3, 1, 2)
    
    def forward(self, output, target):
        output_lab = self.rgb_to_lab(output)
        target_lab = self.rgb_to_lab(target)
        
        color_loss = F.mse_loss(output_lab[:, 1:], target_lab[:, 1:])
        
        luminance_loss = F.l1_loss(output_lab[:, 0], target_lab[:, 0])
        
        return self.weight_lab * color_loss + (1 - self.weight_lab) * luminance_loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)
    
    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.exp(torch.tensor([
                -(x - window_size // 2) ** 2 / (2.0 * sigma ** 2)
                for x in range(window_size)
            ]))
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        channel = img1.size(1)
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class UnderwaterEnhancementLoss(nn.Module):
    """Combined loss for underwater image enhancement"""
    
    def __init__(self,
                 lambda_pixel: float = 1.0,
                 lambda_perceptual: float = 0.1,
                 lambda_color: float = 0.5,
                 lambda_ssim: float = 0.5):
        super(UnderwaterEnhancementLoss, self).__init__()
        
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_color = lambda_color
        self.lambda_ssim = lambda_ssim
        
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss() if lambda_perceptual > 0 else None
        self.color_loss = ColorLoss() if lambda_color > 0 else None
        self.ssim_loss = SSIMLoss() if lambda_ssim > 0 else None
    
    def forward(self, output, target):
        total_loss = 0
        loss_dict = {}
        
        if self.lambda_pixel > 0:
            pixel_loss = self.pixel_loss(output, target)
            total_loss += self.lambda_pixel * pixel_loss
            loss_dict['pixel'] = pixel_loss.item()
        
        if self.lambda_perceptual > 0 and self.perceptual_loss:
            perceptual_loss = self.perceptual_loss(output, target)
            total_loss += self.lambda_perceptual * perceptual_loss
            loss_dict['perceptual'] = perceptual_loss.item()
        
        if self.lambda_color > 0 and self.color_loss:
            color_loss = self.color_loss(output, target)
            total_loss += self.lambda_color * color_loss
            loss_dict['color'] = color_loss.item()
        
        if self.lambda_ssim > 0 and self.ssim_loss:
            ssim_loss = self.ssim_loss(output, target)
            total_loss += self.lambda_ssim * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_fn = UnderwaterEnhancementLoss().to(device)
    
    dummy_output = torch.randn(2, 3, 256, 256).to(device)
    dummy_target = torch.randn(2, 3, 256, 256).to(device)
    
    total_loss, loss_dict = loss_fn(dummy_output, dummy_target)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")