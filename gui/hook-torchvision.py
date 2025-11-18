"""
PyInstaller hook for torchvision - collects all necessary components
Includes C++ extensions and all modules for compatibility
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect torchvision dynamic libraries and source files
# include_py_files=True to include source files that may be needed
datas = collect_data_files('torchvision', include_py_files=True)
binaries = collect_dynamic_libs('torchvision')

# Don't exclude any torchvision modules
# Excluding modules causes issues with operator registration
# The size savings aren't worth the compatibility problems
excludedimports = []
