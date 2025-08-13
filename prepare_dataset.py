#!/usr/bin/env python3
"""
Dataset preparation script for training
Helps organize and verify paired images for training
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare and verify dataset for training"""
    
    def __init__(self, raw_dir: str, enhanced_dir: str, output_dir: str = "dataset"):
        """
        Initialize dataset preparer
        
        Args:
            raw_dir: Directory with raw/input images
            enhanced_dir: Directory with manually enhanced target images
            output_dir: Output directory for organized dataset
        """
        self.raw_dir = Path(raw_dir)
        self.enhanced_dir = Path(enhanced_dir)
        self.output_dir = Path(output_dir)
        
        self.input_dir = self.output_dir / "input"
        self.target_dir = self.output_dir / "target"
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
    
    def find_image_pairs(self):
        """Find matching pairs of raw and enhanced images"""
        raw_files = {}
        enhanced_files = {}
        
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.gpr', '.dng']:
            for f in self.raw_dir.glob(f'*{ext}'):
                base_name = f.stem.replace('_raw', '').replace('_cropped', '')
                raw_files[base_name] = f
            
            for f in self.enhanced_dir.glob(f'*{ext}'):
                base_name = f.stem.replace('_enhanced', '').replace('_edited', '')
                enhanced_files[base_name] = f
        
        common_names = set(raw_files.keys()) & set(enhanced_files.keys())
        
        pairs = []
        for name in common_names:
            pairs.append((raw_files[name], enhanced_files[name]))
        
        logger.info(f"Found {len(pairs)} matching image pairs")
        logger.info(f"Raw images without pairs: {len(raw_files) - len(pairs)}")
        logger.info(f"Enhanced images without pairs: {len(enhanced_files) - len(pairs)}")
        
        return pairs
    
    def verify_dimensions(self, pairs):
        """Verify that paired images have compatible dimensions"""
        valid_pairs = []
        mismatched = []
        
        for raw_path, enhanced_path in tqdm(pairs, desc="Verifying dimensions"):
            try:
                raw_img = Image.open(raw_path)
                enhanced_img = Image.open(enhanced_path)
                
                if raw_img.size == enhanced_img.size:
                    valid_pairs.append((raw_path, enhanced_path))
                else:
                    mismatched.append({
                        'raw': raw_path.name,
                        'raw_size': raw_img.size,
                        'enhanced': enhanced_path.name,
                        'enhanced_size': enhanced_img.size
                    })
            except Exception as e:
                logger.warning(f"Error reading pair {raw_path.name}, {enhanced_path.name}: {e}")
        
        if mismatched:
            logger.warning(f"Found {len(mismatched)} pairs with mismatched dimensions:")
            for m in mismatched[:5]:
                logger.warning(f"  {m['raw']} ({m['raw_size']}) != {m['enhanced']} ({m['enhanced_size']})")
        
        return valid_pairs
    
    def copy_pairs(self, pairs, symlink=False):
        """Copy or symlink paired images to dataset directory"""
        for idx, (raw_path, enhanced_path) in enumerate(tqdm(pairs, desc="Organizing dataset")):
            base_name = f"img_{idx:05d}"
            
            input_ext = raw_path.suffix
            target_ext = enhanced_path.suffix
            
            input_dest = self.input_dir / f"{base_name}{input_ext}"
            target_dest = self.target_dir / f"{base_name}{target_ext}"
            
            if symlink:
                if input_dest.exists():
                    input_dest.unlink()
                if target_dest.exists():
                    target_dest.unlink()
                
                input_dest.symlink_to(raw_path.absolute())
                target_dest.symlink_to(enhanced_path.absolute())
            else:
                shutil.copy2(raw_path, input_dest)
                shutil.copy2(enhanced_path, target_dest)
    
    def analyze_dataset(self, pairs):
        """Analyze dataset statistics"""
        logger.info("\nDataset Statistics:")
        logger.info(f"Total pairs: {len(pairs)}")
        
        if pairs:
            widths = []
            heights = []
            aspects = []
            
            for raw_path, _ in pairs[:100]:
                try:
                    img = Image.open(raw_path)
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspects.append(w / h)
                except:
                    pass
            
            if widths:
                logger.info(f"Image dimensions (sample of {len(widths)}):")
                logger.info(f"  Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
                logger.info(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
                logger.info(f"  Aspect ratio: min={min(aspects):.2f}, max={max(aspects):.2f}, mean={np.mean(aspects):.2f}")
        
        train_size = int(0.8 * len(pairs))
        val_size = len(pairs) - train_size
        logger.info(f"\nSuggested split (80/20):")
        logger.info(f"  Training: {train_size} pairs")
        logger.info(f"  Validation: {val_size} pairs")
    
    def create_split_file(self, pairs, split_ratio=0.8):
        """Create train/val split file"""
        n_train = int(len(pairs) * split_ratio)
        
        indices = list(range(len(pairs)))
        np.random.shuffle(indices)
        
        train_indices = sorted(indices[:n_train])
        val_indices = sorted(indices[n_train:])
        
        split_file = self.output_dir / 'split.txt'
        with open(split_file, 'w') as f:
            f.write(f"# Training indices (n={len(train_indices)})\n")
            f.write(','.join(map(str, train_indices)) + '\n')
            f.write(f"# Validation indices (n={len(val_indices)})\n")
            f.write(','.join(map(str, val_indices)) + '\n')
        
        logger.info(f"Created split file: {split_file}")
    
    def prepare(self, symlink=False, verify_dims=True):
        """Main preparation pipeline"""
        logger.info("Starting dataset preparation...")
        
        pairs = self.find_image_pairs()
        
        if not pairs:
            logger.error("No matching pairs found!")
            return
        
        if verify_dims:
            pairs = self.verify_dimensions(pairs)
        
        self.copy_pairs(pairs, symlink=symlink)
        self.analyze_dataset(pairs)
        self.create_split_file(pairs)
        
        logger.info(f"\nDataset prepared successfully!")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Target directory: {self.target_dir}")
        logger.info(f"\nTo train, run:")
        logger.info(f"  python train.py --config config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument('raw_dir', type=str,
                        help='Directory with raw/input images')
    parser.add_argument('enhanced_dir', type=str,
                        help='Directory with enhanced/target images')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying files')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip dimension verification')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.raw_dir, args.enhanced_dir, args.output)
    preparer.prepare(symlink=args.symlink, verify_dims=not args.no_verify)


if __name__ == "__main__":
    main()