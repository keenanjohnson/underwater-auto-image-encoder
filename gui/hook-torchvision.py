"""
PyInstaller hook for torchvision - collects all necessary components
Includes C++ extensions and essential modules for inference

Size optimization: Excludes datasets and video modules not needed for inference
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import os
import glob

# Collect torchvision dynamic libraries and source files
datas = collect_data_files('torchvision', include_py_files=True)
binaries = collect_dynamic_libs('torchvision')

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

# Collect torchvision submodules, filtering out unused ones
all_submodules = collect_submodules('torchvision')

# Modules to exclude (not needed for inference)
exclude_patterns = [
    'torchvision.datasets',      # Dataset loaders (CIFAR, ImageNet, etc.)
    'torchvision.io.video',      # Video I/O (large ffmpeg bindings)
    'torchvision.prototype',     # Experimental features
]

hiddenimports = [
    mod for mod in all_submodules
    if not any(mod.startswith(excl) for excl in exclude_patterns)
]

# Exclude heavy modules not needed for inference
excludedimports = [
    'torchvision.datasets',
    'torchvision.io.video',
    'torchvision.prototype',
]
