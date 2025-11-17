"""
PyInstaller hook for torchvision - excludes unnecessary model types
Only keeps transforms which is what we use
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect torchvision dynamic libraries
datas = collect_data_files('torchvision', include_py_files=False)
binaries = collect_dynamic_libs('torchvision')

# Exclude model types we don't use (keeps transforms)
excludedimports = [
    'torchvision.models.detection',
    'torchvision.models.segmentation',
    'torchvision.models.video',
    'torchvision.models.optical_flow',
    'torchvision.datasets',  # Don't need built-in datasets
]
