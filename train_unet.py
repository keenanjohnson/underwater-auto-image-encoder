#!/usr/bin/env python3
"""
Standard U-Net Training Script for Underwater Image Enhancement
Uses only the standard U-Net architecture with no model selection options.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np


class DoubleConv(nn.Module):
    """Standard U-Net double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StandardUNet(nn.Module):
    """
    Standard U-Net Architecture
    Feature progression: 64 → 128 → 256 → 512 → 1024
    ~31M parameters
    """
    def __init__(self, n_channels=3, n_classes=3):
        super(StandardUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder - Standard U-Net channel progression
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return torch.sigmoid(logits)


class CombinedLoss(nn.Module):
    """Combined L1 and MSE loss as specified in config"""
    def __init__(self, l1_weight=0.8, mse_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.l1_weight * l1 + self.mse_weight * mse


def load_config(config_path="config.yaml"):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config_path="config.yaml"):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model - NO MODEL SELECTION, always Standard U-Net
    model = StandardUNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Standard U-Net")
    print(f"Total parameters: {total_params:,}")
    
    # Initialize loss and optimizer
    criterion = CombinedLoss(
        l1_weight=config['training']['loss']['l1_weight'],
        mse_weight=config['training']['loss']['mse_weight']
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Batch Size: {config['data']['batch_size']}")
    print(f"  Image Size: {config['data']['image_size']}")
    print(f"  Loss: {config['training']['loss']['l1_weight']} L1 + {config['training']['loss']['mse_weight']} MSE")
    
    # Placeholder for training loop
    print("\nReady to train Standard U-Net!")
    print("Note: Implement DataLoader and training loop based on your dataset structure")
    
    return model


def inference(model, image_path, config, device):
    """
    Run inference on a single image
    Always uses Standard U-Net
    """
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor()
    ])
    
    # Process image
    with torch.no_grad():
        # Load image (implement based on your image format)
        # input_tensor = transform(image).unsqueeze(0).to(device)
        # output = model(input_tensor)
        pass
    
    print(f"Inference complete using Standard U-Net")


if __name__ == "__main__":
    # Always trains Standard U-Net - no model selection needed
    model = train_model("config.yaml")
    print("\nStandard U-Net model ready for underwater image enhancement!")