"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - includes source files for JIT functionality
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all PyTorch dynamic libraries (including CUDA)
# IMPORTANT: include_py_files=True to include .py source files
# PyTorch's JIT system needs access to source code via torch._sources
datas = collect_data_files('torch', include_py_files=True)
binaries = collect_dynamic_libs('torch')

# Don't exclude anything from PyTorch - it's too complex and interdependent
# The size savings from exclusions aren't worth the compatibility issues
excludedimports = []
