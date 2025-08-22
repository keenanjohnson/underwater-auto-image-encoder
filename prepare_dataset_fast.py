#!/usr/bin/env python3
"""
Optimized dataset preparation script for training
Helps organize and verify paired images for training with parallel processing
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastDatasetPreparer:
    """Prepare and verify dataset for training with parallel processing"""
    
    def __init__(self, raw_dir: str, enhanced_dir: str, output_dir: str = "dataset", n_workers: int = None):
        """
        Initialize dataset preparer
        
        Args:
            raw_dir: Directory with raw/input images
            enhanced_dir: Directory with manually enhanced target images
            output_dir: Output directory for organized dataset
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.raw_dir = Path(raw_dir)
        self.enhanced_dir = Path(enhanced_dir)
        self.output_dir = Path(output_dir)
        
        self.input_dir = self.output_dir / "input"
        self.target_dir = self.output_dir / "target"
        
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_workers = n_workers or min(multiprocessing.cpu_count(), 8)
    
    def find_image_pairs(self):
        """Find matching pairs of raw and enhanced images (parallelized)"""
        raw_files = {}
        enhanced_files = {}
        
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.gpr', '.dng']
        
        # Parallel file scanning
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Scan raw directory
            raw_futures = []
            for ext in extensions:
                future = executor.submit(self._scan_directory, self.raw_dir, ext, ['_raw', '_cropped'])
                raw_futures.append(future)
            
            # Scan enhanced directory
            enhanced_futures = []
            for ext in extensions:
                future = executor.submit(self._scan_directory, self.enhanced_dir, ext, ['_enhanced', '_edited'])
                enhanced_futures.append(future)
            
            # Collect results
            for future in as_completed(raw_futures):
                raw_files.update(future.result())
            
            for future in as_completed(enhanced_futures):
                enhanced_files.update(future.result())
        
        common_names = set(raw_files.keys()) & set(enhanced_files.keys())
        
        pairs = [(raw_files[name], enhanced_files[name]) for name in common_names]
        
        logger.info(f"Found {len(pairs)} matching image pairs")
        logger.info(f"Raw images without pairs: {len(raw_files) - len(pairs)}")
        logger.info(f"Enhanced images without pairs: {len(enhanced_files) - len(pairs)}")
        
        return pairs
    
    def _scan_directory(self, directory: Path, extension: str, remove_suffixes: list):
        """Scan directory for files with given extension"""
        files = {}
        for f in directory.glob(f'*{extension}'):
            base_name = f.stem
            for suffix in remove_suffixes:
                base_name = base_name.replace(suffix, '')
            files[base_name] = f
        return files
    
    def verify_dimensions(self, pairs):
        """Verify that paired images have compatible dimensions (parallelized)"""
        valid_pairs = []
        mismatched = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._check_dimensions, pair): pair 
                      for pair in pairs}
            
            for future in tqdm(as_completed(futures), total=len(pairs), desc="Verifying dimensions"):
                pair = futures[future]
                try:
                    result = future.result()
                    if result['match']:
                        valid_pairs.append(pair)
                    else:
                        mismatched.append(result)
                except Exception as e:
                    logger.warning(f"Error checking pair {pair[0].name}, {pair[1].name}: {e}")
        
        if mismatched:
            logger.warning(f"Found {len(mismatched)} pairs with mismatched dimensions:")
            for m in mismatched[:5]:
                logger.warning(f"  {m['raw']} ({m['raw_size']}) != {m['enhanced']} ({m['enhanced_size']})")
        
        return valid_pairs
    
    def _check_dimensions(self, pair):
        """Check if a pair of images has matching dimensions"""
        raw_path, enhanced_path = pair
        try:
            raw_img = Image.open(raw_path)
            enhanced_img = Image.open(enhanced_path)
            
            if raw_img.size == enhanced_img.size:
                return {'match': True}
            else:
                return {
                    'match': False,
                    'raw': raw_path.name,
                    'raw_size': raw_img.size,
                    'enhanced': enhanced_path.name,
                    'enhanced_size': enhanced_img.size
                }
        except Exception as e:
            raise e
    
    def copy_pairs(self, pairs, symlink=False, use_hardlink=False):
        """Copy or link paired images to dataset directory (parallelized)"""
        
        if symlink:
            copy_func = self._create_symlink
        elif use_hardlink:
            copy_func = self._create_hardlink
        else:
            copy_func = self._copy_file
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for idx, pair in enumerate(pairs):
                future = executor.submit(copy_func, idx, pair)
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(pairs), desc="Organizing dataset"):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Error organizing file: {e}")
    
    def _copy_file(self, idx, pair):
        """Copy a single file pair"""
        raw_path, enhanced_path = pair
        base_name = f"img_{idx:05d}"
        
        input_dest = self.input_dir / f"{base_name}{raw_path.suffix}"
        target_dest = self.target_dir / f"{base_name}{enhanced_path.suffix}"
        
        shutil.copy(raw_path, input_dest)  # Using copy instead of copy2 (faster)
        shutil.copy(enhanced_path, target_dest)
    
    def _create_symlink(self, idx, pair):
        """Create symlinks for a file pair"""
        raw_path, enhanced_path = pair
        base_name = f"img_{idx:05d}"
        
        input_dest = self.input_dir / f"{base_name}{raw_path.suffix}"
        target_dest = self.target_dir / f"{base_name}{enhanced_path.suffix}"
        
        if input_dest.exists():
            input_dest.unlink()
        if target_dest.exists():
            target_dest.unlink()
        
        input_dest.symlink_to(raw_path.absolute())
        target_dest.symlink_to(enhanced_path.absolute())
    
    def _create_hardlink(self, idx, pair):
        """Create hard links for a file pair (same filesystem only)"""
        raw_path, enhanced_path = pair
        base_name = f"img_{idx:05d}"
        
        input_dest = self.input_dir / f"{base_name}{raw_path.suffix}"
        target_dest = self.target_dir / f"{base_name}{enhanced_path.suffix}"
        
        if input_dest.exists():
            input_dest.unlink()
        if target_dest.exists():
            target_dest.unlink()
        
        os.link(raw_path, input_dest)
        os.link(enhanced_path, target_dest)
    
    def analyze_dataset(self, pairs):
        """Analyze dataset statistics (parallelized sampling)"""
        logger.info("\nDataset Statistics:")
        logger.info(f"Total pairs: {len(pairs)}")
        
        if pairs:
            sample_size = min(100, len(pairs))
            sample_pairs = pairs[:sample_size]
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(self._get_image_stats, pair[0]): pair 
                          for pair in sample_pairs}
                
                stats = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            stats.append(result)
                    except:
                        pass
            
            if stats:
                widths = [s['width'] for s in stats]
                heights = [s['height'] for s in stats]
                aspects = [s['aspect'] for s in stats]
                
                logger.info(f"Image dimensions (sample of {len(stats)}:")
                logger.info(f"  Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
                logger.info(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
                logger.info(f"  Aspect ratio: min={min(aspects):.2f}, max={max(aspects):.2f}, mean={np.mean(aspects):.2f}")
        
        train_size = int(0.8 * len(pairs))
        val_size = len(pairs) - train_size
        logger.info(f"\nSuggested split (80/20):")
        logger.info(f"  Training: {train_size} pairs")
        logger.info(f"  Validation: {val_size} pairs")
    
    def _get_image_stats(self, image_path):
        """Get statistics for a single image"""
        try:
            img = Image.open(image_path)
            w, h = img.size
            return {'width': w, 'height': h, 'aspect': w / h}
        except:
            return None
    
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
    
    def prepare(self, symlink=False, use_hardlink=False, verify_dims=True):
        """Main preparation pipeline"""
        logger.info(f"Starting dataset preparation with {self.n_workers} workers...")
        
        pairs = self.find_image_pairs()
        
        if not pairs:
            logger.error("No matching pairs found!")
            return
        
        if verify_dims:
            pairs = self.verify_dimensions(pairs)
        
        self.copy_pairs(pairs, symlink=symlink, use_hardlink=use_hardlink)
        self.analyze_dataset(pairs)
        self.create_split_file(pairs)
        
        logger.info(f"\nDataset prepared successfully!")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Target directory: {self.target_dir}")
        logger.info(f"\nTo train, run:")
        logger.info(f"  python train.py --config config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Fast dataset preparation for training")
    parser.add_argument('raw_dir', type=str,
                        help='Directory with raw/input images')
    parser.add_argument('enhanced_dir', type=str,
                        help='Directory with enhanced/target images')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying files')
    parser.add_argument('--hardlink', action='store_true',
                        help='Create hard links instead of copying (faster, same filesystem only)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip dimension verification')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count, max 8)')
    
    args = parser.parse_args()
    
    if args.symlink and args.hardlink:
        logger.error("Cannot use both --symlink and --hardlink")
        sys.exit(1)
    
    preparer = FastDatasetPreparer(args.raw_dir, args.enhanced_dir, args.output, args.workers)
    preparer.prepare(symlink=args.symlink, use_hardlink=args.hardlink, verify_dims=not args.no_verify)


if __name__ == "__main__":
    main()