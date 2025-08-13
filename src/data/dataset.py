"""
Dataset classes for underwater image enhancement
Handles paired RAW/JPEG images for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class UnderwaterImageDataset(Dataset):
    """Dataset for paired underwater images (raw input, enhanced target)"""
    
    def __init__(self, 
                 input_dir: str,
                 target_dir: str,
                 transform=None,
                 input_ext: str = '.tiff',
                 target_ext: str = '.jpg'):
        """
        Initialize dataset
        
        Args:
            input_dir: Directory with input RAW/TIFF images
            target_dir: Directory with target enhanced JPEG images
            transform: Optional transforms to apply
            input_ext: Input file extension
            target_ext: Target file extension
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        
        self.input_files = sorted(list(self.input_dir.glob(f'*{input_ext}')))
        self.target_files = sorted(list(self.target_dir.glob(f'*{target_ext}')))
        
        self._validate_pairs()
        
        logger.info(f"Dataset initialized with {len(self)} image pairs")
    
    def _validate_pairs(self):
        """Validate that input and target files match"""
        input_stems = {f.stem.replace('_cropped', '') for f in self.input_files}
        target_stems = {f.stem for f in self.target_files}
        
        common_stems = input_stems & target_stems
        
        self.input_files = [f for f in self.input_files 
                           if f.stem.replace('_cropped', '') in common_stems]
        self.target_files = [f for f in self.target_files 
                           if f.stem in common_stems]
        
        if len(self.input_files) != len(self.target_files):
            logger.warning(f"Mismatch in paired files. Using {len(self.input_files)} pairs")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]
        
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        else:
            input_img = np.array(input_img).astype(np.float32) / 255.0
            target_img = np.array(target_img).astype(np.float32) / 255.0
            
            input_img = torch.from_numpy(input_img).permute(2, 0, 1)
            target_img = torch.from_numpy(target_img).permute(2, 0, 1)
        
        return {
            'input': input_img,
            'target': target_img,
            'input_path': str(input_path),
            'target_path': str(target_path)
        }


class UnpairedUnderwaterDataset(Dataset):
    """Dataset for unpaired underwater images (for unsupervised learning)"""
    
    def __init__(self,
                 raw_dir: str,
                 enhanced_dir: Optional[str] = None,
                 transform=None,
                 file_ext: str = '.tiff'):
        """
        Initialize unpaired dataset
        
        Args:
            raw_dir: Directory with raw underwater images
            enhanced_dir: Optional directory with enhanced images (for CycleGAN-style training)
            transform: Optional transforms
            file_ext: File extension to look for
        """
        self.raw_dir = Path(raw_dir)
        self.enhanced_dir = Path(enhanced_dir) if enhanced_dir else None
        self.transform = transform
        
        self.raw_files = sorted(list(self.raw_dir.glob(f'*{file_ext}')))
        self.enhanced_files = []
        
        if self.enhanced_dir:
            self.enhanced_files = sorted(list(self.enhanced_dir.glob(f'*{file_ext}')))
        
        logger.info(f"Unpaired dataset: {len(self.raw_files)} raw images")
        if self.enhanced_files:
            logger.info(f"                  {len(self.enhanced_files)} enhanced images")
    
    def __len__(self):
        return len(self.raw_files)
    
    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        raw_img = Image.open(raw_path).convert('RGB')
        
        if self.transform:
            raw_img = self.transform(raw_img)
        else:
            raw_img = np.array(raw_img).astype(np.float32) / 255.0
            raw_img = torch.from_numpy(raw_img).permute(2, 0, 1)
        
        result = {
            'raw': raw_img,
            'raw_path': str(raw_path)
        }
        
        if self.enhanced_files:
            enhanced_idx = np.random.randint(0, len(self.enhanced_files))
            enhanced_path = self.enhanced_files[enhanced_idx]
            enhanced_img = Image.open(enhanced_path).convert('RGB')
            
            if self.transform:
                enhanced_img = self.transform(enhanced_img)
            else:
                enhanced_img = np.array(enhanced_img).astype(np.float32) / 255.0
                enhanced_img = torch.from_numpy(enhanced_img).permute(2, 0, 1)
            
            result['enhanced'] = enhanced_img
            result['enhanced_path'] = str(enhanced_path)
        
        return result


def create_data_loaders(input_dir: str,
                       target_dir: str,
                       batch_size: int = 4,
                       num_workers: int = 4,
                       train_split: float = 0.8,
                       transform=None,
                       pin_memory: bool = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        input_dir: Directory with input images
        target_dir: Directory with target images
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        train_split: Fraction of data for training
        transform: Optional transforms
        pin_memory: Whether to pin memory (auto-detected if None)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = UnderwaterImageDataset(input_dir, target_dir, transform=transform)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Auto-detect pin_memory based on CUDA availability
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_dataset = UnderwaterImageDataset(
        input_dir="processed/cropped",
        target_dir="processed/enhanced",
        input_ext='.tiff',
        target_ext='.jpg'
    )
    
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        print(f"Sample input shape: {sample['input'].shape}")
        print(f"Sample target shape: {sample['target'].shape}")
        print(f"Input path: {sample['input_path']}")
        print(f"Target path: {sample['target_path']}")