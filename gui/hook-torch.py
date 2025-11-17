"""
PyInstaller hook for PyTorch - excludes unnecessary components to reduce size
KEEPS CUDA SUPPORT - only removes safe-to-exclude modules
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all PyTorch dynamic libraries (including CUDA)
datas = collect_data_files('torch', include_py_files=False)
binaries = collect_dynamic_libs('torch')

# Only exclude modules that are truly optional and cause issues
# Note: Can't exclude torch.testing as it's referenced internally
# Note: Excluding torch.distributed.rpc to avoid duplicate type registration
excludedimports = [
    'torch.utils.tensorboard',  # Safe to exclude - only for TensorBoard logging
    'torch.distributed.rpc',     # Causes duplicate type registration errors
    'torch.distributed.pipeline',  # Not needed for inference
]
