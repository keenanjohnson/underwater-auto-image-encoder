"""
PyInstaller hook for torchvision - excludes unnecessary model types
Only keeps transforms which is what we use
"""

# Exclude model types we don't use (keeps transforms)
excludedimports = [
    'torchvision.models.detection',
    'torchvision.models.segmentation',
    'torchvision.models.video',
    'torchvision.models.optical_flow',
    'torchvision.datasets',  # Don't need built-in datasets
]
