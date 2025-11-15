#!/usr/bin/env python3
"""
Compare different denoising methods to find optimal settings
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse
from PIL import Image
import numpy as np

def run_denoise(input_file, output_file, method, params=""):
    """Run denoise_tiff.py with specified method and parameters"""
    cmd = f"python denoise_tiff.py {input_file} --output {output_file} --method {method} {params}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Compare denoising methods")
    parser.add_argument('input', type=str, help='Input TIFF file')
    parser.add_argument('--output-dir', type=str, default='denoise_comparison',
                        help='Output directory for comparison')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Test configurations - optimized for underwater imagery
    tests = [
        # Very light denoising
        ("bilateral_light", "bilateral", "--bilateral-d 5 --bilateral-sigma-color 25 --bilateral-sigma-space 25"),
        ("bilateral_medium", "bilateral", "--bilateral-d 7 --bilateral-sigma-color 50 --bilateral-sigma-space 50"),
        ("bilateral_standard", "bilateral", "--bilateral-d 9 --bilateral-sigma-color 75 --bilateral-sigma-space 75"),
        
        # NL-Means with different strengths
        ("nlmeans_verylight", "nlmeans", "--nlmeans-h 0.03 --nlmeans-patch-size 5 --nlmeans-patch-distance 7"),
        ("nlmeans_light", "nlmeans", "--nlmeans-h 0.05 --nlmeans-patch-size 5 --nlmeans-patch-distance 7"),
        ("nlmeans_medium", "nlmeans", "--nlmeans-h 0.08 --nlmeans-patch-size 7 --nlmeans-patch-distance 11"),
        
        # Wavelet denoising
        ("wavelet_light", "wavelet", "--wavelet-sigma 0.05"),
        ("wavelet_medium", "wavelet", "--wavelet-sigma 0.1"),
        
        # Total Variation
        ("tv_light", "tv_chambolle", "--tv-weight 0.05"),
        ("tv_medium", "tv_chambolle", "--tv-weight 0.1"),
        
        # Median filter for comparison
        ("median_3", "median", "--median-kernel 3"),
        
        # Light Gaussian
        ("gaussian_light", "gaussian", "--gaussian-sigma 0.5"),
    ]
    
    print(f"Testing {len(tests)} denoising configurations on {input_path.name}")
    print("=" * 60)
    
    results = []
    for name, method, params in tests:
        output_file = output_dir / f"{input_path.stem}_{name}.tiff"
        success = run_denoise(input_path, output_file, method, params)
        if success:
            print(f"✓ {name}: Saved to {output_file.name}")
            results.append(output_file.name)
        else:
            print(f"✗ {name}: Failed")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print(f"Comparison complete! Results saved to {output_dir}/")
    print("\nRecommended next steps:")
    print("1. Open the output directory and visually compare the results")
    print("2. Look for a balance between noise reduction and detail preservation")
    print("3. For underwater images, bilateral_light or nlmeans_verylight often work best")
    print("\nGenerated files:")
    for result in results:
        print(f"  - {result}")

if __name__ == "__main__":
    main()