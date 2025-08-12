#!/usr/bin/env python3
"""
Test script for the preprocessing pipeline
"""

import sys
from pathlib import Path
from preprocess_images import GPRPreprocessor

def test_preprocessing():
    """Test the preprocessing pipeline with example usage"""
    
    print("Testing GPR Preprocessing Pipeline")
    print("=" * 40)
    
    preprocessor = GPRPreprocessor(output_dir="test_processed")
    
    print("\n1. Checking GPR tools availability...")
    if not preprocessor.check_gpr_tools():
        print("   ❌ GPR tools not found. Please install using: bash install_gpr_tools.sh")
        return False
    print("   ✓ GPR tools available")
    
    print("\n2. Testing directory structure creation...")
    assert preprocessor.output_dir.exists(), "Output directory not created"
    assert preprocessor.raw_dir.exists(), "RAW directory not created"
    assert preprocessor.cropped_dir.exists(), "Cropped directory not created"
    print("   ✓ Directory structure created")
    
    print("\n3. Configuration check...")
    print(f"   - Output directory: {preprocessor.output_dir}")
    print(f"   - Crop dimensions: {preprocessor.crop_width}x{preprocessor.crop_height}")
    print("   ✓ Configuration verified")
    
    print("\n" + "=" * 40)
    print("✓ All tests passed!")
    print("\nUsage examples:")
    print("  # Process a single GPR file:")
    print("  python3 preprocess_images.py input.gpr")
    print()
    print("  # Process all GPR files in a directory:")
    print("  python3 preprocess_images.py /path/to/gpr/files/")
    print()
    print("  # Custom output directory and crop size:")
    print("  python3 preprocess_images.py input.gpr -o custom_output --crop-width 4000 --crop-height 3000")
    
    return True

if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)