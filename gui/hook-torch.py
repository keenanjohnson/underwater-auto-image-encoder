"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - minimal exclusions to ensure stability
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all PyTorch dynamic libraries (including CUDA)
datas = collect_data_files('torch', include_py_files=False)
binaries = collect_dynamic_libs('torch')

# Only exclude modules that are truly safe to exclude
# Most PyTorch modules are interdependent, so we minimize exclusions
excludedimports = [
    'torch.utils.tensorboard',  # Safe to exclude - only for TensorBoard logging
]
