"""
Runtime hook to ensure src modules can be imported
"""
import sys
import os
from pathlib import Path

# Get the directory where the executable is located
if hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))

# For macOS app bundles, also check Resources directory
if sys.platform == 'darwin' and hasattr(sys, '_MEIPASS'):
    # Check both Frameworks and Resources for src
    possible_paths = [
        os.path.join(base_path, 'src'),
        os.path.join(base_path, '..', 'Resources', 'src'),
        os.path.join(base_path, '..', 'Frameworks', 'src'),
    ]
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            # Also add parent so "src.models" can be imported
            parent = os.path.dirname(abs_path)
            if parent not in sys.path:
                sys.path.insert(0, parent)
else:
    # Add src directory to Python path
    src_path = os.path.join(base_path, 'src')
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

# Also add the base path
if base_path not in sys.path:
    sys.path.insert(0, base_path)