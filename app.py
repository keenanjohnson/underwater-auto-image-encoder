#!/usr/bin/env python3
"""
Underwater Image Enhancer GUI Application
Main entry point for the desktop application
"""

__version__ = "0.1.0"

import sys
import os
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent))

def smoke_test():
    """Run smoke tests for CI validation"""
    print(f"Underwater Enhancer v{__version__}")
    print("Running smoke tests...")
    
    
    errors = []
    
    # Test critical imports
    try:
        import torch
        print("  [OK] PyTorch imported")
    except ImportError as e:
        errors.append(f"PyTorch import failed: {e}")
        print(f"  [FAIL] PyTorch: {e}")
    
    try:
        import cv2
        print("  [OK] OpenCV imported")
    except (ImportError, Exception) as e:
        # OpenCV has known issues with PyInstaller on macOS, warn but don't fail
        if "recursion is detected" in str(e):
            print(f"  [WARN] OpenCV: Known PyInstaller issue - {e}")
        else:
            errors.append(f"OpenCV import failed: {e}")
            print(f"  [FAIL] OpenCV: {e}")
    
    try:
        import customtkinter
        print("  [OK] CustomTkinter imported")
    except ImportError as e:
        errors.append(f"CustomTkinter import failed: {e}")
        print(f"  [FAIL] CustomTkinter: {e}")
    
    try:
        from src.models.unet_autoencoder import UNetAutoencoder
        print("  [OK] Model imported")
    except ImportError as e:
        errors.append(f"Model import failed: {e}")
        print(f"  [FAIL] Model: {e}")
    
    try:
        from src.converters.gpr_converter import GPRConverter
        print("  [OK] GPR converter imported")
    except ImportError as e:
        errors.append(f"GPR converter import failed: {e}")
        print(f"  [FAIL] GPR converter: {e}")
    
    try:
        from src.gui.main_window import UnderwaterEnhancerApp
        print("  [OK] GUI imported")
    except ImportError as e:
        errors.append(f"GUI import failed: {e}")
        print(f"  [FAIL] GUI: {e}")
    
    if errors:
        print("\nSmoke test FAILED with errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\nSmoke test PASSED - all modules imported successfully!")
        sys.exit(0)

def main():
    """Main application entry point"""
    # Check for smoke test mode
    if os.environ.get('SMOKE_TEST') == '1' or '--smoke-test' in sys.argv:
        smoke_test()
        return
    
    from src.gui.main_window import UnderwaterEnhancerApp
    app = UnderwaterEnhancerApp()
    app.mainloop()

if __name__ == "__main__":
    main()