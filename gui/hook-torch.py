"""
PyInstaller hook for PyTorch - excludes unnecessary components to reduce size
KEEPS CUDA SUPPORT - only removes tests and benchmarks
"""
from PyInstaller.utils.hooks import collect_submodules

# Exclude test files and documentation (NOT CUDA libraries)
excludedimports = [
    'torch.testing',
    'torch.utils.tensorboard',
    'torch.utils.benchmark',
    'torch.testing._internal',
]
