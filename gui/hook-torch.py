"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - NO exclusions to ensure maximum compatibility
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all PyTorch dynamic libraries (including CUDA)
datas = collect_data_files('torch', include_py_files=False)
binaries = collect_dynamic_libs('torch')

# Don't exclude anything from PyTorch - it's too complex and interdependent
# The size savings from exclusions aren't worth the compatibility issues
excludedimports = []
