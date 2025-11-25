"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT

Size optimization: Only include essential data files, not Python source files
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect PyTorch dynamic libraries (including CUDA)
binaries = collect_dynamic_libs('torch')

# Collect only essential data files (not Python source files)
# This saves significant space as PyTorch has many .py files
# Note: If JIT compilation fails, set include_py_files=True
datas = collect_data_files('torch', include_py_files=False)

# Note: PyTorch modules are highly interdependent
# Excluding modules can cause circular import errors
# Keep excludedimports minimal or empty for stability
excludedimports = []
