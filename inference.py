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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Inferencer:
    """Inference class for underwater image enhancement"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """Initialize inferencer with trained model"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
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
    
    def process_image(self, image_path: Path, output_path: Path = None):
        """Process a single image"""
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Check if we should resize during inference
        resize_inference = self.config.get('inference', {}).get('resize_inference', False)
        
        if resize_inference:
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
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory or file')
    parser.add_argument('--compare', action='store_true',
                        help='Create side-by-side comparison')
    
    args = parser.parse_args()
    
    inferencer = Inferencer(args.checkpoint, args.config)
    
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