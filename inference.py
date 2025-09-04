#!/usr/bin/env python3
"""
Inference script for underwater image enhancement
Run trained model on new images
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import logging
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from src.models.unet_autoencoder import UNetAutoencoder, LightweightUNet
from src.models.attention_unet import AttentionUNet, WaterNet

# Colab-compatible model blocks (matching exact architecture from notebook)
class ColabDoubleConv(torch.nn.Module):
    """Double Convolution Block matching Colab notebook (with bias)"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ColabDown(torch.nn.Module):
    """Downscaling with maxpool then double conv - Colab version"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            ColabDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ColabUp(torch.nn.Module):
    """Upscaling then double conv - Colab version"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ColabDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ColabUNetAutoencoder(torch.nn.Module):
    """Colab-compatible U-Net matching the exact architecture from the notebook"""
    def __init__(self, n_channels=3, n_classes=3):
        super(ColabUNetAutoencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder - matching Colab notebook exactly
        self.inc = ColabDoubleConv(n_channels, 64)
        self.down1 = ColabDown(64, 128)
        self.down2 = ColabDown(128, 256)
        self.down3 = ColabDown(256, 512)
        self.down4 = ColabDown(512, 1024)
        
        # Decoder - matching Colab notebook exactly
        self.up1 = ColabUp(1024, 512)
        self.up2 = ColabUp(512, 256)
        self.up3 = ColabUp(256, 128)
        self.up4 = ColabUp(128, 64)
        self.outc = torch.nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return torch.sigmoid(logits)

# Import blocks from local model for non-Colab models
from src.models.unet_autoencoder import DoubleConv, Down, Up

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Inferencer:
    """Inference class for underwater image enhancement"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """Initialize inferencer with trained model"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Handle Colab checkpoint format - create default config
            self.config = self._create_default_config(checkpoint)
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        self.setup_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        
        self.setup_transforms()
    
    def _create_default_config(self, checkpoint):
        """Create default config for Colab-trained models"""
        # Check if checkpoint has model_config (from final save in Colab)
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            return {
                'model': {
                    'type': 'UNetAutoencoder',  # Default to UNetAutoencoder as used in Colab
                    'n_channels': model_config.get('n_channels', 3),
                    'n_classes': model_config.get('n_classes', 3),
                    'base_features': 64,  # Default U-Net base features
                    'bilinear': False
                },
                'data': {
                    'image_size': [model_config.get('image_size', 256), model_config.get('image_size', 256)]
                },
                'inference': {
                    'resize_inference': True  # Default: resize to training size for faster inference
                }
            }
        else:
            # Fallback config for basic Colab checkpoints
            return {
                'model': {
                    'type': 'UNetAutoencoder',
                    'n_channels': 3,
                    'n_classes': 3,
                    'base_features': 64,
                    'bilinear': False
                },
                'data': {
                    'image_size': [256, 256]
                },
                'inference': {
                    'resize_inference': True
                }
            }
    
    def setup_model(self):
        """Initialize model architecture"""
        model_type = self.config['model']['type']
        model_params = {
            'n_channels': self.config['model']['n_channels'],
            'n_classes': self.config['model']['n_classes']
        }
        
        if model_type == 'LightweightUNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = LightweightUNet(**model_params)
        elif model_type == 'UNetAutoencoder':
            # Check if this is a Colab checkpoint (without base_features parameter)
            if 'base_features' not in self.config['model'] or self.config['model']['base_features'] == 64:
                # Use Colab-compatible model for exact architecture match
                self.model = ColabUNetAutoencoder(**model_params)
            else:
                model_params['base_features'] = self.config['model']['base_features']
                model_params['bilinear'] = self.config['model'].get('bilinear', False)
                self.model = UNetAutoencoder(**model_params)
        elif model_type == 'AttentionUNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = AttentionUNet(**model_params)
        elif model_type == 'WaterNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = WaterNet(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
    
    def setup_transforms(self):
        """Setup image transformations"""
        if self.config.get('inference', {}).get('resize_inference', False):
            # Use inference_size if available, otherwise fall back to data image_size
            inference_config = self.config.get('inference', {})
            if 'inference_size' in inference_config:
                image_size = tuple(inference_config['inference_size'])
            else:
                image_size = tuple(self.config['data']['image_size'])
            
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
            self.inverse_transform = transforms.Compose([
                transforms.ToPILImage()
            ])
        else:
            self.transform = transforms.ToTensor()
            self.inverse_transform = transforms.ToPILImage()
    
    def pad_to_multiple(self, tensor, multiple=32):
        """Pad tensor to be divisible by multiple"""
        _, _, h, w = tensor.shape
        
        # Calculate padding needed
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        # Pad to make divisible by multiple
        if pad_h > 0 or pad_w > 0:
            # Use reflection padding to avoid artifacts
            padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
            tensor = torch.nn.functional.pad(tensor, padding, mode='reflect')
        
        return tensor, (pad_h, pad_w)
    
    def process_image_tiled(self, image_path: Path, tile_size=1024, overlap=128, progress_callback=None):
        """Process large image using tiling to avoid memory issues
        
        Args:
            image_path: Path to input image
            tile_size: Size of each tile
            overlap: Overlap between tiles
            progress_callback: Optional callback(message) for progress updates
        """
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        width, height = original_size
        
        logger.info(f"Processing {width}x{height} image with {tile_size}x{tile_size} tiles")
        if progress_callback:
            progress_callback(f"Processing {width}x{height} image with {tile_size}x{tile_size} tiles")
        
        # Create output image
        output_img = Image.new('RGB', original_size)
        
        # Calculate total tiles for progress tracking
        tiles_x = (width + tile_size - overlap - 1) // (tile_size - overlap)
        tiles_y = (height + tile_size - overlap - 1) // (tile_size - overlap)
        total_tiles = tiles_x * tiles_y
        current_tile = 0
        
        # Process tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                current_tile += 1
                # Define tile boundaries
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                
                # Extract tile
                tile = img.crop((x, y, x_end, y_end))
                
                # Process tile
                input_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                padded_tensor, (pad_h, pad_w) = self.pad_to_multiple(input_tensor, multiple=32)
                
                with torch.no_grad():
                    output_tensor = self.model(padded_tensor)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    _, _, h, w = input_tensor.shape
                    output_tensor = output_tensor[:, :, :h, :w]
                
                output_tensor = output_tensor.squeeze(0).cpu()
                enhanced_tile = self.inverse_transform(output_tensor)
                
                # Handle overlap blending for smoother results
                if overlap > 0 and (x > 0 or y > 0):
                    # Simple paste for now - could add feathering
                    paste_x = x + overlap // 2 if x > 0 else x
                    paste_y = y + overlap // 2 if y > 0 else y
                    paste_x_end = x_end - overlap // 2 if x_end < width else x_end
                    paste_y_end = y_end - overlap // 2 if y_end < height else y_end
                    
                    crop_x = overlap // 2 if x > 0 else 0
                    crop_y = overlap // 2 if y > 0 else 0
                    crop_x_end = enhanced_tile.size[0] - (overlap // 2 if x_end < width else 0)
                    crop_y_end = enhanced_tile.size[1] - (overlap // 2 if y_end < height else 0)
                    
                    enhanced_tile_cropped = enhanced_tile.crop((crop_x, crop_y, crop_x_end, crop_y_end))
                    output_img.paste(enhanced_tile_cropped, (paste_x, paste_y))
                else:
                    output_img.paste(enhanced_tile, (x, y))
                
                logger.info(f"Processed tile ({x}, {y}) to ({x_end}, {y_end})")
                if progress_callback:
                    progress_callback(f"Processed tile {current_tile}/{total_tiles} ({x}, {y}) to ({x_end}, {y_end})")
        
        return output_img

    def process_image(self, image_path: Path, output_path: Path = None, progress_callback=None):
        """Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Optional output path for saving
            progress_callback: Optional callback(message) for progress updates
        """
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        width, height = original_size
        
        # Check if we should resize during inference
        resize_inference = self.config.get('inference', {}).get('resize_inference', False)
        
        # Use tiling for large images to avoid memory issues
        if not resize_inference and (width > 2048 or height > 2048):
            logger.info(f"Large image detected ({width}x{height}), using tiled processing")
            output_img = self.process_image_tiled(image_path, progress_callback=progress_callback)
        elif resize_inference:
            # Use training resolution
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            output_tensor = output_tensor.squeeze(0).cpu()
            output_img = self.inverse_transform(output_tensor)
            
            # Resize back to original size
            output_img = output_img.resize(original_size, Image.LANCZOS)
        else:
            # Process at original resolution with padding
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Pad to make dimensions compatible with U-Net
            padded_tensor, (pad_h, pad_w) = self.pad_to_multiple(input_tensor, multiple=32)
            
            with torch.no_grad():
                output_tensor = self.model(padded_tensor)
            
            # Remove padding from output
            if pad_h > 0 or pad_w > 0:
                _, _, h, w = input_tensor.shape
                output_tensor = output_tensor[:, :, :h, :w]
            
            output_tensor = output_tensor.squeeze(0).cpu()
            output_img = self.inverse_transform(output_tensor)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(output_path, quality=95)
            logger.info(f"Saved enhanced image to {output_path}")
        
        return output_img
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         extensions: list = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
        """Process all images in a directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for img_path in tqdm(image_files, desc="Processing images"):
            output_path = output_dir / f"{img_path.stem}_enhanced{img_path.suffix}"
            self.process_image(img_path, output_path)
    
    def compare_images(self, image_path: Path, save_comparison: bool = True):
        """Create side-by-side comparison of original and enhanced"""
        original = Image.open(image_path).convert('RGB')
        enhanced = self.process_image(image_path)
        
        width, height = original.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(original, (0, 0))
        comparison.paste(enhanced, (width, 0))
        
        if save_comparison:
            output_path = image_path.parent / f"{image_path.stem}_comparison.jpg"
            comparison.save(output_path, quality=95)
            logger.info(f"Saved comparison to {output_path}")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description="Run inference on underwater images")
    parser.add_argument('input', type=str, help='Input image or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (optional for Colab checkpoints)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory or file')
    parser.add_argument('--compare', action='store_true',
                        help='Create side-by-side comparison')
    parser.add_argument('--full-size', action='store_true',
                        help='Process at original resolution (4606x4030) instead of resizing')
    
    args = parser.parse_args()
    
    # Override config for full-size processing if requested
    config_override = None
    if args.full_size:
        config_override = {'inference': {'resize_inference': False}}
    
    inferencer = Inferencer(args.checkpoint, args.config)
    
    # Apply full-size override if requested
    if config_override:
        inferencer.config.update(config_override)
        inferencer.setup_transforms()  # Refresh transforms
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        if args.compare:
            inferencer.compare_images(input_path)
        else:
            if output_path.suffix:
                inferencer.process_image(input_path, output_path)
            else:
                output_file = output_path / f"{input_path.stem}_enhanced{input_path.suffix}"
                inferencer.process_image(input_path, output_file)
    elif input_path.is_dir():
        inferencer.process_directory(input_path, output_path)
    else:
        logger.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()