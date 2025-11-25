"""
PyInstaller hook for torchvision - collects all necessary components
Includes C++ extensions and essential modules for inference

Size optimization: Excludes datasets, video modules, and Python source files
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import os
import glob

# Collect torchvision dynamic libraries
binaries = collect_dynamic_libs('torchvision')

# Collect only essential data files (not Python source files for smaller size)
datas = collect_data_files('torchvision', include_py_files=False)

# Explicitly collect torchvision C++ extension libraries
# These contain operators like torchvision::nms
try:
    import torchvision
    tv_path = os.path.dirname(torchvision.__file__)

    # Look for .so/.dylib/.pyd files in torchvision directory
    for ext in ['*.so', '*.dylib', '*.pyd', '*.dll']:
        for lib_file in glob.glob(os.path.join(tv_path, '**', ext), recursive=True):
            # Add to binaries with correct destination path
            rel_path = os.path.relpath(os.path.dirname(lib_file), tv_path)
            dest_dir = os.path.join('torchvision', rel_path) if rel_path != '.' else 'torchvision'
            binaries.append((lib_file, dest_dir))
except ImportError:
    pass

# Collect all torchvision submodules
# Note: Excluding modules can cause import errors, so include all
hiddenimports = collect_submodules('torchvision')

# Keep excludedimports empty for stability
excludedimports = []
