#!/usr/bin/env python3
"""Test if GPR converter works in bundled app"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from converters.gpr_converter import GPRConverter

try:
    # Check if GPR conversion is available
    if GPRConverter.is_available():
        print("✓ GPR converter is available")
        gpr_path = GPRConverter.get_gpr_tools_path()
        print(f"✓ GPR tools found at: {gpr_path}")
    else:
        print("✗ GPR converter is not available")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)