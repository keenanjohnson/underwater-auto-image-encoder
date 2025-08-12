#!/usr/bin/env python3
"""
Image Preprocessing Pipeline for Underwater GPR Images
Handles GPR to RAW/DNG conversion and cropping to 4606x4030
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import rawpy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPRPreprocessor:
    """Handles GPR file preprocessing including conversion and cropping"""
    
    def __init__(self, output_dir: str = "processed", crop_size: Tuple[int, int] = (4606, 4030)):
        """
        Initialize the preprocessor
        
        Args:
            output_dir: Directory to save processed images
            crop_size: Target crop size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.crop_width, self.crop_height = crop_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_dir = self.output_dir / "raw"
        self.cropped_dir = self.output_dir / "cropped"
        self.raw_dir.mkdir(exist_ok=True)
        self.cropped_dir.mkdir(exist_ok=True)
        
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
    
    def convert_gpr_to_dng(self, gpr_path: Path) -> Optional[Path]:
        """
        Convert GPR file to DNG using gpr_tools
        
        Args:
            gpr_path: Path to GPR file
            
        Returns:
            Path to converted DNG file or None if conversion failed
        """
        dng_path = self.raw_dir / f"{gpr_path.stem}.dng"
        
        try:
            cmd = [
                'gpr_tools',
                '-i', str(gpr_path),
                '-o', str(dng_path)
            ]
            
            logger.info(f"Converting {gpr_path.name} to DNG...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and dng_path.exists():
                logger.info(f"Successfully converted to {dng_path.name}")
                return dng_path
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting {gpr_path}: {e}")
            return None
    
    def crop_raw_image(self, raw_path: Path) -> Optional[Path]:
        """
        Crop RAW/DNG image to specified dimensions
        
        Args:
            raw_path: Path to RAW/DNG file
            
        Returns:
            Path to cropped image or None if cropping failed
        """
        try:
            with rawpy.imread(str(raw_path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
            
            height, width = rgb.shape[:2]
            logger.info(f"Original dimensions: {width}x{height}")
            
            if width < self.crop_width or height < self.crop_height:
                logger.warning(f"Image too small to crop to {self.crop_width}x{self.crop_height}")
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
            
            logger.info(f"Saved cropped image: {output_path.name} ({self.crop_width}x{self.crop_height})")
            return output_path
            
        except Exception as e:
            logger.error(f"Error cropping {raw_path}: {e}")
            return None
    
    def process_gpr_file(self, gpr_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Process a single GPR file through the entire pipeline
        
        Args:
            gpr_path: Path to GPR file
            
        Returns:
            Tuple of (DNG path, cropped image path)
        """
        logger.info(f"Processing: {gpr_path}")
        
        dng_path = self.convert_gpr_to_dng(gpr_path)
        if not dng_path:
            return None, None
        
        cropped_path = self.crop_raw_image(dng_path)
        
        return dng_path, cropped_path
    
    def process_directory(self, input_dir: Path, pattern: str = "*.gpr") -> List[Tuple[Path, Path]]:
        """
        Process all GPR files in a directory
        
        Args:
            input_dir: Directory containing GPR files
            pattern: File pattern to match
            
        Returns:
            List of (DNG path, cropped path) tuples
        """
        gpr_files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(gpr_files)} GPR files")
        
        results = []
        for gpr_file in gpr_files:
            dng_path, cropped_path = self.process_gpr_file(gpr_file)
            if dng_path and cropped_path:
                results.append((dng_path, cropped_path))
        
        logger.info(f"Successfully processed {len(results)} files")
        return results

def main():
    parser = argparse.ArgumentParser(description="Preprocess GPR images for ML training")
    parser.add_argument('input', type=str, help='Input GPR file or directory')
    parser.add_argument('--output-dir', '-o', type=str, default='processed',
                        help='Output directory for processed files')
    parser.add_argument('--crop-width', type=int, default=4606,
                        help='Crop width (default: 4606)')
    parser.add_argument('--crop-height', type=int, default=4030,
                        help='Crop height (default: 4030)')
    parser.add_argument('--skip-gpr-check', action='store_true',
                        help='Skip GPR tools availability check')
    
    args = parser.parse_args()
    
    preprocessor = GPRPreprocessor(
        output_dir=args.output_dir,
        crop_size=(args.crop_width, args.crop_height)
    )
    
    if not args.skip_gpr_check and not preprocessor.check_gpr_tools():
        logger.error("gpr_tools not found! Please install it first:")
        logger.error("Run: bash install_gpr_tools.sh")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.gpr':
        dng_path, cropped_path = preprocessor.process_gpr_file(input_path)
        if dng_path and cropped_path:
            logger.info("Processing complete!")
            logger.info(f"  DNG: {dng_path}")
            logger.info(f"  Cropped: {cropped_path}")
    elif input_path.is_dir():
        results = preprocessor.process_directory(input_path)
        logger.info(f"\nProcessing Summary:")
        logger.info(f"  Total processed: {len(results)}")
        logger.info(f"  Output directory: {preprocessor.output_dir}")
    else:
        logger.error(f"Invalid input: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()