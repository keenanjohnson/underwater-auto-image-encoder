"""
PyInstaller hook for scikit-image - excludes unused modules
"""

# Exclude modules we don't use
excludedimports = [
    'skimage.feature',
    'skimage.segmentation',
    'skimage.morphology',
    'skimage.measure',
    'skimage.restoration',
    'skimage.filters',
    'skimage.data',  # Don't need sample images
]
