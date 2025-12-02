"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - includes source files for JIT functionality
Excludes server-class GPU architectures to reduce size
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import re

# Collect all PyTorch dynamic libraries (including CUDA)
# IMPORTANT: include_py_files=True to include .py source files
# PyTorch's JIT system needs access to source code via torch._sources
datas = collect_data_files('torch', include_py_files=True)
binaries = collect_dynamic_libs('torch')

# Exclude server-class and obsolete CUDA architectures to reduce binary size
# Keep: sm_60/61 (Pascal GTX 1000), sm_75 (Turing RTX 2000), sm_86 (RTX 3000), sm_89 (RTX 4000)
# Exclude: sm_50/52 (Maxwell GTX 900 - 2014), sm_70 (Volta V100), sm_80 (A100), sm_90 (H100)
EXCLUDED_ARCHITECTURES = ['sm_50', 'sm_52', 'sm_70', 'sm_80', 'sm_90']

def should_exclude_binary(filepath):
    """Check if binary should be excluded based on CUDA architecture"""
    filename = filepath.lower()
    for arch in EXCLUDED_ARCHITECTURES:
        if arch in filename:
            return True
    return False

# Filter out server architecture binaries
binaries = [(src, dst) for src, dst in binaries if not should_exclude_binary(src)]

# Exclude training/debugging modules not needed for inference
# These are safer to exclude as they're clearly separate subsystems
excludedimports = [
    'torch.distributed',         # Multi-GPU/multi-node training
    'torch.testing',             # Test utilities
    'torch.onnx',                # ONNX export
    'torch.quantization',        # Model quantization
    'torch.profiler',            # Performance profiling
    'torch.utils.tensorboard',   # TensorBoard integration
    'torch.utils.benchmark',     # Benchmarking utilities
]
