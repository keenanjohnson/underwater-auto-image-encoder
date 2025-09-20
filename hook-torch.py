"""
PyInstaller hook for PyTorch to ensure required binaries are bundled
Place this file in the same directory as pyinstaller.spec
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import platform
import os
from pathlib import Path

# Collect PyTorch dynamic libraries
binaries = collect_dynamic_libs('torch')

# Platform-specific binary collection
if platform.system() == 'Darwin':
    # macOS: Ensure torch_shm_manager is included
    try:
        import torch
        torch_path = Path(torch.__file__).parent
        torch_bin = torch_path / 'bin' / 'torch_shm_manager'
        if torch_bin.exists():
            binaries.append((str(torch_bin), 'torch/bin'))
    except:
        pass

    # Collect all torch submodules
    hiddenimports = collect_submodules('torch')
    datas = collect_data_files('torch', include_py_files=False)

elif platform.system() == 'Windows':
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
    # Linux
    hiddenimports = collect_submodules('torch')
    datas = collect_data_files('torch', include_py_files=False)