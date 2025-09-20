"""
PyInstaller hook for PyTorch to ensure CUDA DLLs are bundled on Windows
Place this file in the same directory as pyinstaller.spec
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import platform

# Collect PyTorch dynamic libraries
binaries = collect_dynamic_libs('torch')

# On Windows, explicitly collect CUDA DLLs
if platform.system() == 'Windows':
    # Additional patterns for CUDA DLLs that might be missed
    datas = collect_data_files('torch', include_py_files=False)

    # Ensure torch._C and CUDA modules are included
    hiddenimports = [
        'torch._C',
        'torch._C._cuda',
        'torch.cuda',
        'torch.cuda.amp',
        'torch.cuda.memory',
        'torch.backends.cuda',
        'torch.backends.cudnn',
        'torch.utils.cuda',
    ]
else:
    datas = []
    hiddenimports = []