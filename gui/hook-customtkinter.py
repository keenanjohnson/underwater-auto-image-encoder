"""
PyInstaller hook for customtkinter - collects data files and ensures distutils compatibility
CustomTkinter needs theme JSON files and uses distutils.version
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all customtkinter submodules
hiddenimports = collect_submodules('customtkinter')

# Collect customtkinter data files (themes, assets, etc.)
datas = collect_data_files('customtkinter')

# CustomTkinter imports distutils.version for version comparison
# Need to explicitly include distutils and its submodules
hiddenimports.extend([
    'distutils',
    'distutils.version',
    'distutils.errors',
    'distutils.spawn',
])
