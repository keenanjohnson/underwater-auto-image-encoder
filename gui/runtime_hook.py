"""
Runtime hook to ensure src modules can be imported and PyTorch works with PyInstaller
"""
import sys
import os
from pathlib import Path
import inspect

# Disable PyTorch JIT before importing torch
# This prevents torch._sources.parse_def from being called
os.environ['PYTORCH_JIT'] = '0'
os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '0'

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

# PyTorch compatibility fixes for PyInstaller
# PyTorch's source code introspection fails in frozen apps, so we patch it
if hasattr(sys, '_MEIPASS'):
    # Patch inspect module first
    _original_getsource = inspect.getsource
    _original_getsourcelines = inspect.getsourcelines
    _original_findsource = inspect.findsource

    def _patched_getsource(object):
        """Patched getsource that returns empty string instead of failing"""
        try:
            return _original_getsource(object)
        except (OSError, TypeError):
            # Return empty source code when running in PyInstaller
            return ""

    def _patched_getsourcelines(object):
        """Patched getsourcelines that returns empty list instead of failing"""
        try:
            return _original_getsourcelines(object)
        except (OSError, TypeError):
            # Return empty source lines when running in PyInstaller
            return ([], 0)

    def _patched_findsource(object):
        """Patched findsource that returns empty source instead of failing"""
        try:
            return _original_findsource(object)
        except (OSError, TypeError):
            # Return empty source when running in PyInstaller
            return ([], 0)

    inspect.getsource = _patched_getsource
    inspect.getsourcelines = _patched_getsourcelines
    inspect.findsource = _patched_findsource