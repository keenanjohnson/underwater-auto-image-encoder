"""
PyInstaller hook for PyTorch - excludes unnecessary components to reduce size
KEEPS CUDA SUPPORT - only removes tests and benchmarks
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import os

# Collect all PyTorch dynamic libraries (including CUDA)
datas = collect_data_files('torch', include_py_files=False)
binaries = collect_dynamic_libs('torch')

# Exclude test files and documentation (NOT CUDA libraries)
excludedimports = [
    'torch.testing',
    'torch.utils.tensorboard',
    'torch.utils.benchmark',
    'torch.testing._internal',
]
