"""
Converters package for image format conversion
"""

# Conditionally import GPRConverter based on feature flag
try:
    from gui.features import GPR_SUPPORT_ENABLED
except ImportError:
    GPR_SUPPORT_ENABLED = False

if GPR_SUPPORT_ENABLED:
    from .gpr_converter import GPRConverter
    __all__ = ['GPRConverter']
else:
    __all__ = []
