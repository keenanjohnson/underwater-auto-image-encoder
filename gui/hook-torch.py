"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - includes source files required for JIT
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all PyTorch dynamic libraries (including CUDA)
binaries = collect_dynamic_libs('torch')

# IMPORTANT: include_py_files=True is required!
# PyTorch's JIT system uses torch._sources to parse Python source files
# Without them, you get: "RuntimeError: Expected a single top-level function"
datas = collect_data_files('torch', include_py_files=True)

# Don't exclude any PyTorch modules - they are highly interdependent
excludedimports = []
