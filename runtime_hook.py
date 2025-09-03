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

# Add src directory to Python path
src_path = os.path.join(base_path, 'src')
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

# Also add the base path
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Debug output disabled for production
# print(f"Runtime hook: Added paths to sys.path")
# print(f"  Base: {base_path}")
# print(f"  Src: {src_path}")