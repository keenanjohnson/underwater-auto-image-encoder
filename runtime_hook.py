"""
Runtime hook to ensure src modules can be imported and torch paths are set correctly
"""
import sys
import os

# Apply torch fixes for bundled applications
if hasattr(sys, '_MEIPASS'):
    # We're in a PyInstaller bundle, apply torch patches
    sys.path.insert(0, sys._MEIPASS)
    try:
        import torch_bundle_fix
    except ImportError:
        # Fallback: just set environment variables
        os.environ['TORCH_DISABLE_SOURCE_INSPECTION'] = '1'
        os.environ['PYTORCH_DISABLE_LIBRARY_VALIDATION'] = '1'

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

# Fix torch paths for macOS app bundles
if sys.platform == 'darwin' and hasattr(sys, '_MEIPASS'):
    # Set environment variable for torch_shm_manager
    torch_bin = os.path.join(base_path, 'torch', 'bin')
    if os.path.exists(torch_bin):
        # Add torch/bin to PATH so torch can find torch_shm_manager
        current_path = os.environ.get('PATH', '')
        if torch_bin not in current_path:
            os.environ['PATH'] = f"{torch_bin}:{current_path}"

        # Also set TORCH_SHIM_MANAGER explicitly if needed
        shm_manager = os.path.join(torch_bin, 'torch_shm_manager')
        if os.path.exists(shm_manager):
            os.environ['TORCH_SHM_MANAGER'] = shm_manager