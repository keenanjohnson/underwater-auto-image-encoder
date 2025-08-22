#!/usr/bin/env python3
"""
Optimized Image Preprocessing Pipeline for Underwater GPR Images
Handles GPR to RAW/DNG conversion and cropping with parallel processing
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import rawpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm
import tempfile
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastGPRPreprocessor:
    """Handles GPR file preprocessing with parallel processing"""
    
    def __init__(self, output_dir: str = "processed", crop_size: Tuple[int, int] = (4606, 4030), 
                 n_workers: int = None, batch_size: int = 4):
        """
        Initialize the preprocessor
        
        Args:
            output_dir: Directory to save processed images
            crop_size: Target crop size (width, height)
            n_workers: Number of parallel workers (default: CPU count)
            batch_size: Number of files to process in parallel batches
        """
        self.output_dir = Path(output_dir)
        self.crop_width, self.crop_height = crop_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_dir = self.output_dir / "raw"
        self.cropped_dir = self.output_dir / "cropped"
        self.raw_dir.mkdir(exist_ok=True)
        self.cropped_dir.mkdir(exist_ok=True)
        
        self.n_workers = n_workers or min(multiprocessing.cpu_count(), 8)
        self.batch_size = batch_size
        
    def check_gpr_tools(self) -> bool:
        """Check if gpr_tools is available"""
        try:
            result = subprocess.run(['which', 'gpr_tools'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Found gpr_tools at: {result.stdout.strip()}")
                return True
        except Exception as e:
            logger.error(f"Error checking for gpr_tools: {e}")
        return False
    
    def convert_gpr_batch(self, gpr_paths: List[Path]) -> List[Optional[Path]]:
        """
        Convert multiple GPR files to DNG in parallel
        
        Args:
            gpr_paths: List of GPR file paths
            
        Returns:
            List of converted DNG paths (None for failed conversions)
        """
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._convert_single_gpr, gpr_path): gpr_path 
                      for gpr_path in gpr_paths}
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting GPR to DNG"):
                try:
                    dng_path = future.result()
                    results.append(dng_path)
                except Exception as e:
                    logger.error(f"Conversion failed: {e}")
                    results.append(None)
        
        return results
    
    def _convert_single_gpr(self, gpr_path: Path) -> Optional[Path]:
        """Convert a single GPR file to DNG"""
        dng_path = self.raw_dir / f"{gpr_path.stem}.dng"
        
        try:
            cmd = [
                'gpr_tools',
                '-i', str(gpr_path),
                '-o', str(dng_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and dng_path.exists():
                return dng_path
            else:
                logger.error(f"Conversion failed for {gpr_path.name}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Conversion timeout for {gpr_path.name}")
            return None
        except Exception as e:
            logger.error(f"Error converting {gpr_path}: {e}")
            return None
    
    def crop_raw_batch(self, raw_paths: List[Path]) -> List[Optional[Path]]:
        """
        Crop multiple RAW/DNG images in parallel
        
        Args:
            raw_paths: List of RAW/DNG file paths
            
        Returns:
            List of cropped image paths (None for failed crops)
        """
        # Filter out None values
        valid_paths = [p for p in raw_paths if p is not None]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._crop_single_raw, raw_path): raw_path 
                      for raw_path in valid_paths}
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping images"):
                try:
                    cropped_path = future.result()
                    results.append(cropped_path)
                except Exception as e:
                    logger.error(f"Cropping failed: {e}")
                    results.append(None)
        
        return results
    
    def _crop_single_raw(self, raw_path: Path) -> Optional[Path]:
        """Crop a single RAW/DNG image"""
        try:
            with rawpy.imread(str(raw_path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            
            height, width = rgb.shape[:2]
            
            if width < self.crop_width or height < self.crop_height:
                logger.warning(f"Image {raw_path.name} too small to crop to {self.crop_width}x{self.crop_height}")
                return None
            
            center_x = width // 2
            center_y = height // 2
            left = center_x - self.crop_width // 2
            top = center_y - self.crop_height // 2
            right = left + self.crop_width
            bottom = top + self.crop_height
            
            cropped = rgb[top:bottom, left:right]
            
            output_path = self.cropped_dir / f"{raw_path.stem}_cropped.tiff"
            
            img = Image.fromarray(cropped)
            img.save(output_path, 'TIFF', compression='none')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error cropping {raw_path}: {e}")
            return None
    
    def process_gpr_files_parallel(self, gpr_paths: List[Path]) -> List[Tuple[Optional[Path], Optional[Path]]]:
        """
        Process multiple GPR files through the entire pipeline in parallel
        
        Args:
            gpr_paths: List of GPR file paths
            
        Returns:
            List of (DNG path, cropped image path) tuples
        """
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(gpr_paths), self.batch_size):
            batch = gpr_paths[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(gpr_paths) + self.batch_size - 1)//self.batch_size}")
            
            # Convert GPR to DNG in parallel
            dng_paths = self.convert_gpr_batch(batch)
            
            # Crop DNG images in parallel
            cropped_paths = self.crop_raw_batch(dng_paths)
            
            # Combine results
            for dng_path, cropped_path in zip(dng_paths, cropped_paths):
                results.append((dng_path, cropped_path))
        
        return results
    
    def process_directory(self, input_dir: Path, pattern: str = "*.gpr") -> List[Tuple[Path, Path]]:
        """
        Process all GPR files in a directory with parallel processing
        
        Args:
            input_dir: Directory containing GPR files
            pattern: File pattern to match
            
        Returns:
            List of (DNG path, cropped path) tuples
        """
        # Search for both lowercase and uppercase GPR files
        gpr_files = list(input_dir.glob("*.gpr")) + list(input_dir.glob("*.GPR"))
        logger.info(f"Found {len(gpr_files)} GPR files")
        
        if not gpr_files:
            logger.warning("No GPR files found")
            return []
        
        # Process all files in parallel
        all_results = self.process_gpr_files_parallel(gpr_files)
        
        # Filter successful results
        successful_results = [(dng, crop) for dng, crop in all_results if dng and crop]
        
        logger.info(f"Successfully processed {len(successful_results)}/{len(gpr_files)} files")
        return successful_results
    
    def process_directory_streaming(self, input_dir: Path) -> None:
        """
        Process GPR files with streaming approach for very large directories
        Processes and saves files as they complete rather than waiting for all
        """
        gpr_files = list(input_dir.glob("*.gpr")) + list(input_dir.glob("*.GPR"))
        logger.info(f"Found {len(gpr_files)} GPR files")
        
        if not gpr_files:
            logger.warning("No GPR files found")
            return
        
        success_count = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all conversions
            conversion_futures = {executor.submit(self._convert_single_gpr, gpr_path): gpr_path 
                                 for gpr_path in gpr_files}
            
            # As conversions complete, immediately submit cropping tasks
            crop_futures = {}
            
            for future in tqdm(as_completed(conversion_futures), total=len(gpr_files), 
                             desc="Processing GPR files"):
                try:
                    dng_path = future.result()
                    if dng_path:
                        # Immediately submit cropping task
                        crop_future = executor.submit(self._crop_single_raw, dng_path)
                        crop_futures[crop_future] = dng_path
                except Exception as e:
                    logger.error(f"Error in conversion: {e}")
            
            # Process cropping results
            for future in tqdm(as_completed(crop_futures), total=len(crop_futures), 
                             desc="Cropping images"):
                try:
                    cropped_path = future.result()
                    if cropped_path:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error in cropping: {e}")
        
        logger.info(f"Successfully processed {success_count}/{len(gpr_files)} files")

def main():
    parser = argparse.ArgumentParser(description="Fast preprocessing of GPR images for ML training")
    parser.add_argument('input', type=str, help='Input GPR file or directory')
    parser.add_argument('--output-dir', '-o', type=str, default='processed',
                        help='Output directory for processed files')
    parser.add_argument('--crop-width', type=int, default=4606,
                        help='Crop width (default: 4606)')
    parser.add_argument('--crop-height', type=int, default=4030,
                        help='Crop height (default: 4030)')
    parser.add_argument('--skip-gpr-check', action='store_true',
                        help='Skip GPR tools availability check')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count, max 8)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for parallel processing (default: 4)')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode for very large directories')
    
    args = parser.parse_args()
    
    preprocessor = FastGPRPreprocessor(
        output_dir=args.output_dir,
        crop_size=(args.crop_width, args.crop_height),
        n_workers=args.workers,
        batch_size=args.batch_size
    )
    
    if not args.skip_gpr_check and not preprocessor.check_gpr_tools():
        logger.error("gpr_tools not found! Please install it first:")
        logger.error("Run: bash install_gpr_tools.sh")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.gpr':
        # Single file processing
        dng_path = preprocessor._convert_single_gpr(input_path)
        if dng_path:
            cropped_path = preprocessor._crop_single_raw(dng_path)
            if cropped_path:
                logger.info("Processing complete!")
                logger.info(f"  DNG: {dng_path}")
                logger.info(f"  Cropped: {cropped_path}")
    elif input_path.is_dir():
        if args.streaming:
            # Streaming mode for very large directories
            preprocessor.process_directory_streaming(input_path)
        else:
            # Batch mode
            results = preprocessor.process_directory(input_path)
            logger.info(f"\nProcessing Summary:")
            logger.info(f"  Total processed: {len(results)}")
            logger.info(f"  Output directory: {preprocessor.output_dir}")
    else:
        logger.error(f"Invalid input: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()