#!/usr/bin/env python3
"""
Test script to verify if rawpy can read GPR files directly
"""

import sys
import rawpy
import numpy as np
from pathlib import Path
from PIL import Image

def test_gpr_reading(gpr_path):
    """Test if rawpy can read a GPR file"""
    
    print(f"\nTesting GPR file: {gpr_path}")
    print("-" * 50)
    
    try:
        # Attempt to open GPR file with rawpy
        print("Attempting to open with rawpy...")
        with rawpy.imread(str(gpr_path)) as raw:
            print(f"✓ Successfully opened GPR file!")
            print(f"  Image size: {raw.sizes.width} x {raw.sizes.height}")
            print(f"  Raw type: {raw.raw_type}")
            print(f"  Color description: {raw.color_desc.decode('utf-8')}")
            
            # Try to process the image
            print("\nAttempting to demosaic and process...")
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8
            )
            
            print(f"✓ Successfully processed!")
            print(f"  Output shape: {rgb.shape}")
            print(f"  Output dtype: {rgb.dtype}")
            print(f"  Min/Max values: {rgb.min()}/{rgb.max()}")
            
            # Save as test output
            output_path = Path(gpr_path).parent / f"{Path(gpr_path).stem}_rawpy_test.jpg"
            Image.fromarray(rgb).save(output_path)
            print(f"\n✓ Saved test output to: {output_path}")
            
            return True
            
    except Exception as e:
        print(f"✗ Failed to read GPR file with rawpy")
        print(f"  Error: {type(e).__name__}: {e}")
        return False


def find_gpr_files(directory="."):
    """Find GPR files in the directory"""
    path = Path(directory)
    gpr_files = list(path.glob("**/*.gpr")) + list(path.glob("**/*.GPR"))
    return gpr_files


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific file
        gpr_file = Path(sys.argv[1])
        if gpr_file.exists():
            success = test_gpr_reading(gpr_file)
            sys.exit(0 if success else 1)
        else:
            print(f"Error: File not found: {gpr_file}")
            sys.exit(1)
    else:
        # Find and test any GPR file in the project
        print("Looking for GPR files in the project...")
        gpr_files = find_gpr_files()
        
        if gpr_files:
            print(f"Found {len(gpr_files)} GPR file(s)")
            # Test the first one
            success = test_gpr_reading(gpr_files[0])
            
            if not success:
                print("\n" + "="*50)
                print("CONCLUSION: rawpy cannot directly read GPR files")
                print("The standard rawpy installation doesn't include GPR support.")
                print("\nAlternatives:")
                print("1. Continue using gpr_tools for GPR → TIFF conversion")
                print("2. Use Adobe DNG Converter to convert GPR → DNG first")
                print("3. Build rawpy/LibRaw from source with GPR SDK support")
                print("4. Accept TIFF/JPEG input only for the GUI application")
        else:
            print("No GPR files found in the project.")
            print("\nTo test GPR support, run:")
            print("  python test_gpr_rawpy.py /path/to/file.gpr")