"""
PyInstaller hook for customtkinter - ensures distutils compatibility
CustomTkinter uses distutils.version which needs explicit collection
"""
from PyInstaller.utils.hooks import collect_submodules

# Collect all customtkinter submodules
hiddenimports = collect_submodules('customtkinter')

# CustomTkinter imports distutils.version for version comparison
# Need to explicitly include distutils and its submodules
hiddenimports.extend([
    'distutils',
    'distutils.version',
    'distutils.errors',
    'distutils.spawn',
])
