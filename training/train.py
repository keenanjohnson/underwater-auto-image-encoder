#!/usr/bin/env python3
"""
Underwater Image Enhancement Training Script
Standalone version matching the Jupyter notebook workflow exactly
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from PIL import Image

# Optional 8-bit optimizer support
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ushape_transformer import UShapeTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnderwaterDataset(Dataset):
    """Dataset that extracts patches from full-resolution images.

    Supports two modes:
    - Training: Random patches with augmentation
    - Validation: Grid-based patches covering the entire image
    """

    def __init__(self, input_dir, target_dir, file_indices, image_size=512,
                 patches_per_image=1, augment=False, use_grid=False,
                 grid_stride=None, image_width=4606, image_height=4030):
        """
        Args:
            input_dir: Directory with input images
            target_dir: Directory with target images
            file_indices: List of file indices to use
            image_size: Patch size to extract (e.g., 512 -> 512×512 patches)
            patches_per_image: Number of random patches per image (only for random mode)
            augment: Whether to apply augmentation (flips)
            use_grid: If True, extract all patches in a grid pattern (for validation)
            grid_stride: Stride for grid extraction (default: image_size for non-overlapping)
            image_width: Expected image width (for grid calculation)
            image_height: Expected image height (for grid calculation)
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.use_grid = use_grid
        self.grid_stride = grid_stride if grid_stride is not None else image_size
        self.image_width = image_width
        self.image_height = image_height

        # Get all files with various image extensions
        input_extensions = ('.tiff', '.tif', '.TIFF', '.TIF')
        target_extensions = ('.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')

        all_input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(input_extensions)])
        all_target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(target_extensions)])

        # Filter indices to valid range
        valid_indices = [i for i in file_indices if i < min(len(all_input_files), len(all_target_files))]

        # Select files based on indices
        self.input_files = [all_input_files[i] for i in valid_indices]
        self.target_files = [all_target_files[i] for i in valid_indices]
        self.num_images = len(self.input_files)

        # Generate patch positions
        self.patch_positions = []
        if self.use_grid:
            self._generate_grid_positions()
        else:
            self._generate_random_positions()

        # Simple transforms (no resize - we extract patches directly)
        self.to_tensor = transforms.ToTensor()

    def _generate_grid_positions(self):
        """Generate grid patch positions covering each image."""
        self.patch_positions = []
        patch_size = self.image_size
        stride = self.grid_stride

        # Calculate grid positions
        for img_idx in range(self.num_images):
            y = 0
            while y + patch_size <= self.image_height:
                x = 0
                while x + patch_size <= self.image_width:
                    self.patch_positions.append((img_idx, x, y))
                    x += stride
                y += stride

        # Calculate patches per image for logging
        patches_x = (self.image_width - patch_size) // stride + 1
        patches_y = (self.image_height - patch_size) // stride + 1
        self.patches_per_image_grid = patches_x * patches_y
        logger.info(f"Grid validation: {patches_x}x{patches_y} = {self.patches_per_image_grid} patches per image")

    def _generate_random_positions(self):
        """Generate random patch positions for all images."""
        self.patch_positions = []
        for img_idx in range(self.num_images):
            for _ in range(self.patches_per_image):
                # Store just the image index; random x,y computed in __getitem__
                self.patch_positions.append((img_idx, None, None))

    def reshuffle_patches(self):
        """Reshuffle patch positions for a new epoch.
        Only applies to random mode; grid mode positions are fixed."""
        if not self.use_grid:
            self._generate_random_positions()
            logger.debug(f"Reshuffled patches: {len(self.patch_positions)} patches from {self.num_images} images")

    def __len__(self):
        return len(self.patch_positions)

    def __getitem__(self, idx):
        # Get the image index and position for this patch
        img_idx, grid_x, grid_y = self.patch_positions[idx]

        # Load images
        input_path = self.input_dir / self.input_files[img_idx]
        target_path = self.target_dir / self.target_files[img_idx]

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Get image dimensions
        width, height = input_img.size

        # Calculate crop position
        if self.image_size is not None:
            patch_size = self.image_size

            if self.use_grid:
                # Use pre-computed grid position
                x, y = grid_x, grid_y
            else:
                # Random crop position
                max_x = max(0, width - patch_size)
                max_y = max(0, height - patch_size)
                x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
                y = np.random.randint(0, max_y + 1) if max_y > 0 else 0

            # Extract the same patch from both images
            input_img = input_img.crop((x, y, x + patch_size, y + patch_size))
            target_img = target_img.crop((x, y, x + patch_size, y + patch_size))

        # Apply augmentation (only for training)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Random vertical flip
            if np.random.random() > 0.5:
                input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)

        # Convert to tensors
        input_tensor = self.to_tensor(input_img)
        target_tensor = self.to_tensor(target_img)

        return input_tensor, target_tensor


class DoubleConv(nn.Module):
    """Double Convolution Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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


class UNetAutoencoder(nn.Module):
    """U-Net based autoencoder for image enhancement"""
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetAutoencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
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

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return torch.sigmoid(logits)


class CombinedLoss(nn.Module):
    """Combined loss function for image enhancement"""
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.alpha * l1 + self.beta * mse


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_split(split_file, num_files):
    """Load train/validation split from file"""
    if split_file.exists():
        try:
            # Try to load as JSON first
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            train_indices = split_data['train']
            val_indices = split_data['validation']
        except json.JSONDecodeError:
            # Parse as plain text with comma-separated indices
            with open(split_file, 'r') as f:
                lines = f.readlines()

            train_indices = []
            val_indices = []

            for line in lines:
                line = line.strip()
                if 'Training' in line or 'training' in line:
                    continue
                elif 'Validation' in line or 'validation' in line:
                    continue
                elif line and not line.startswith('#'):
                    indices = [int(x.strip()) for x in line.split(',') if x.strip().isdigit()]
                    if not train_indices:
                        train_indices.extend(indices)
                    else:
                        val_indices.extend(indices)

        logger.info(f"Loaded split: {len(train_indices)} train, {len(val_indices)} validation")
        return train_indices, val_indices
    else:
        # Create 80/20 split
        logger.info("No split file found, creating 80/20 train/validation split")
        indices = np.random.permutation(num_files)
        split_point = int(0.8 * num_files)
        train_indices = indices[:split_point].tolist()
        val_indices = indices[split_point:].tolist()
        logger.info(f"Created split: {len(train_indices)} train, {len(val_indices)} validation")
        return train_indices, val_indices


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = len(dataloader)

    with tqdm(dataloader, desc="Training") as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                # Mixed precision forward pass
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            # Calculate PSNR in FP32 for accuracy
            with torch.no_grad():
                batch_psnr = calculate_psnr(outputs.float(), targets.float()).item()

            total_loss += batch_loss
            total_psnr += batch_psnr

            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'PSNR': f'{batch_psnr:.2f} dB'
            })

    return total_loss / num_batches, total_psnr / num_batches


def validate_epoch(model, dataloader, criterion, device, use_amp=False):
    """Validate for one epoch with optional mixed precision"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                if use_amp:
                    with autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                batch_loss = loss.item()
                # Calculate PSNR in FP32 for accuracy
                batch_psnr = calculate_psnr(outputs.float(), targets.float()).item()

                total_loss += batch_loss
                total_psnr += batch_psnr

                pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'PSNR': f'{batch_psnr:.2f} dB'
                })

    return total_loss / num_batches, total_psnr / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Underwater Image Enhancement Model')

    # Data arguments
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--target-dir', type=str, required=True, help='Directory with target images')
    parser.add_argument('--split-file', type=str, default=None, help='Train/val split file (optional)')

    # Training arguments
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Training patch size (use 0 for full-size, default: 1024)')
    parser.add_argument('--patches-per-image', type=int, default=1,
                       help='Number of random patches to extract per image per epoch (default: 1)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for models')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Loss configuration
    parser.add_argument('--l1-weight', type=float, default=0.8, help='L1 loss weight (default: 0.8)')
    parser.add_argument('--mse-weight', type=float, default=0.2, help='MSE loss weight (default: 0.2)')

    # Model selection
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'ushape_transformer'],
                       help='Model architecture (default: unet)')

    # Early stopping
    parser.add_argument('--early-stopping', type=int, default=15,
                       help='Early stopping patience - stop if val loss does not improve for N epochs (default: 15, 0 to disable)')

    # Mixed precision training
    parser.add_argument('--amp', action='store_true',
                       help='Enable automatic mixed precision (FP16) training for reduced memory usage')

    # Gradient checkpointing
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Enable gradient checkpointing to reduce memory usage (trades compute for memory)')

    # 8-bit optimizer
    parser.add_argument('--optimizer-8bit', action='store_true',
                       help='Use 8-bit Adam optimizer (requires bitsandbytes) for ~1.5-2GB memory savings')

    # torch.compile
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for potential speedups and memory optimization (PyTorch 2.0+)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using device: {device}")
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"Using device: {device} (Apple Silicon)")
        # MPS doesn't provide direct memory query, but M1/M2/M3 typically have unified memory
        logger.info("Note: MPS uses unified memory shared with system RAM")
    else:
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        logger.warning("No GPU acceleration available. Training will be slow.")

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)

    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif', '.TIFF', '.TIF'))])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG', '.jpg', '.JPG'))])

    logger.info(f"Found {len(input_files)} input images")
    logger.info(f"Found {len(target_files)} target images")

    num_files = min(len(input_files), len(target_files))

    # Load split
    split_file = Path(args.split_file) if args.split_file else input_dir.parent / 'split.txt'
    train_indices, val_indices = load_split(split_file, num_files)

    # Create datasets
    image_size = None if args.image_size == 0 else args.image_size
    patches_per_image = args.patches_per_image
    resolution_str = "Full-size" if image_size is None else f"{image_size}×{image_size}"

    logger.info(f"Training patch size: {resolution_str}")
    logger.info(f"Patches per image: {patches_per_image}")
    logger.info(f"Batch size: {args.batch_size}")

    train_dataset = UnderwaterDataset(input_dir, target_dir, train_indices,
                                     image_size=image_size,
                                     patches_per_image=patches_per_image,
                                     augment=True)
    val_dataset = UnderwaterDataset(input_dir, target_dir, val_indices,
                                   image_size=image_size,
                                   use_grid=True,  # Validate on all patches
                                   augment=False)

    logger.info(f"Training images: {train_dataset.num_images}")
    logger.info(f"Training patches per epoch: {len(train_dataset)}")
    logger.info(f"Validation images: {val_dataset.num_images}")
    logger.info(f"Validation patches per epoch: {len(val_dataset)}")

    # Create dataloaders
    # Disable pin_memory for MPS (not supported yet)
    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )

    # Test memory with one batch
    logger.info("Testing memory with sample batch...")
    try:
        sample_batch = next(iter(train_loader))
        sample_inputs = sample_batch[0].to(device)
        logger.info(f"Batch shape: {sample_inputs.shape}")

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Memory for data batch: {memory_used:.2f} GB")
    except Exception as e:
        logger.error(f"Memory test failed: {e}")
        logger.error("Batch size may be too large for available GPU memory")
        return

    # Create model based on selection
    use_gradient_checkpointing = args.gradient_checkpointing

    if args.model == 'ushape_transformer':
        # U-shape Transformer requires specific image size (must be divisible by 16)
        model_img_size = image_size if image_size is not None else 256
        if model_img_size % 16 != 0:
            logger.warning(f"Image size {model_img_size} not divisible by 16, adjusting to {(model_img_size // 16) * 16}")
            model_img_size = (model_img_size // 16) * 16

        model = UShapeTransformer(
            img_dim=model_img_size,
            in_ch=3,
            out_ch=3,
            return_single=True,  # Return single output for non-GAN training
            gradient_checkpointing=use_gradient_checkpointing
        ).to(device)
        logger.info(f"Using U-shape Transformer model (img_dim={model_img_size})")
        if use_gradient_checkpointing:
            logger.info("Gradient checkpointing: ENABLED (memory-efficient mode)")
    else:
        model = UNetAutoencoder(n_channels=3, n_classes=3).to(device)
        logger.info("Using U-Net Autoencoder model")
        if use_gradient_checkpointing:
            logger.warning("Gradient checkpointing requested but not supported for U-Net model")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")

    # Loss and optimizer
    criterion = CombinedLoss(alpha=args.l1_weight, beta=args.mse_weight)

    # Select optimizer
    use_8bit_optimizer = args.optimizer_8bit
    if use_8bit_optimizer:
        if HAS_BITSANDBYTES:
            optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
            logger.info("Using 8-bit Adam optimizer (bitsandbytes)")
        else:
            logger.warning("8-bit optimizer requested but bitsandbytes not installed. Falling back to standard Adam.")
            logger.warning("Install with: pip install bitsandbytes")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
            use_8bit_optimizer = False
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # torch.compile for PyTorch 2.0+ (deferred until after checkpoint loading)
    use_compile = args.compile
    compile_mode = None
    if use_compile:
        if hasattr(torch, 'compile'):
            # Use 'default' mode when gradient checkpointing is enabled
            # 'reduce-overhead' uses CUDA graphs which conflict with checkpointing's control flow
            if use_gradient_checkpointing:
                compile_mode = 'default'
                logger.info("Using torch.compile with 'default' mode (gradient checkpointing enabled)")
            else:
                compile_mode = 'reduce-overhead'
                logger.info("Using torch.compile with 'reduce-overhead' mode")
        else:
            logger.warning("torch.compile requested but not available (requires PyTorch 2.0+)")
            use_compile = False

    # Mixed precision setup (supports CUDA and MPS)
    use_amp = args.amp and device.type in ('cuda', 'mps')
    scaler = GradScaler(device.type) if use_amp else None

    if args.amp and device.type not in ('cuda', 'mps'):
        logger.warning("AMP requested but not using CUDA or MPS - disabling mixed precision")
    if use_amp:
        logger.info("Mixed precision (AMP) training: ENABLED")
        logger.info(f"Model size with FP16: ~{total_params * 2 / 1e6:.2f} MB")
    else:
        logger.info("Mixed precision (AMP) training: DISABLED")

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []
    patience_counter = 0

    # Resume from checkpoint
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_psnrs = checkpoint.get('train_psnrs', [])
            val_psnrs = checkpoint.get('val_psnrs', [])
            patience_counter = checkpoint.get('patience_counter', 0)
            # Restore scaler state if available and using AMP
            if use_amp and scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("Restored AMP scaler state from checkpoint")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    # Apply torch.compile after checkpoint loading to avoid state_dict key mismatches
    if use_compile and compile_mode is not None:
        logger.info("Compiling model (this may take a moment on first forward pass)...")
        model = torch.compile(model, mode=compile_mode)
        logger.info("Model compiled successfully")

    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 80)

        # Reshuffle random patches for this epoch (new random crops each epoch)
        train_dataset.reshuffle_patches()

        # Train
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device,
                                              scaler=scaler, use_amp=use_amp)
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)

        # Validate
        val_loss, val_psnr = validate_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        # Update learning rate
        scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")
        logger.info(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_psnrs': train_psnrs,
            'val_psnrs': val_psnrs,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'training_config': {
                'resolution': resolution_str,
                'batch_size': args.batch_size,
                'image_size': image_size if image_size is not None else 4606,
                'patches_per_image': patches_per_image,
                'model': args.model,
                'amp': use_amp,
                'gradient_checkpointing': use_gradient_checkpointing,
                'optimizer_8bit': use_8bit_optimizer,
                'compile': use_compile,
            }
        }
        # Save scaler state for AMP
        if use_amp and scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save latest checkpoint
        latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_checkpoint)

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = output_dir / 'best_model.pth'

            # Add model config for inference
            checkpoint['model_config'] = {
                'n_channels': 3,
                'n_classes': 3,
                'image_size': image_size if image_size is not None else 4606,
                'model': args.model,
            }

            torch.save(checkpoint, best_model_path)
            logger.info(f"✓ Saved best model: {best_model_path}")
        else:
            patience_counter += 1
            if args.early_stopping > 0:
                logger.info(f"No improvement in val loss ({patience_counter}/{args.early_stopping})")
                if patience_counter >= args.early_stopping:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_checkpoint = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_checkpoint)
            logger.info(f"✓ Saved periodic checkpoint: {periodic_checkpoint}")

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': 3,
            'n_classes': 3,
            'image_size': image_size if image_size is not None else 4606,
            'model': args.model,
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_psnrs': train_psnrs,
            'val_psnrs': val_psnrs,
        },
        'training_config': {
            'resolution': resolution_str,
            'batch_size': args.batch_size,
            'patches_per_image': patches_per_image,
            'epochs': len(train_losses),
            'model': args.model,
            'amp': use_amp,
            'gradient_checkpointing': use_gradient_checkpointing,
            'optimizer_8bit': use_8bit_optimizer,
            'compile': use_compile,
        }
    }
    torch.save(final_checkpoint, final_model_path)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info(f"✓ Best model saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"✓ Final model saved to: {final_model_path}")
    logger.info(f"✓ Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
