"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - includes source files for JIT functionality

Size optimization: Excludes unused modules while keeping core inference functionality
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import os
import glob

# Collect all PyTorch dynamic libraries (including CUDA)
# IMPORTANT: include_py_files=True to include .py source files
# PyTorch's JIT system needs access to source code via torch._sources
datas = collect_data_files('torch', include_py_files=True)
binaries = collect_dynamic_libs('torch')

# Filter out unused CUDA architectures to reduce size
# Keep only commonly used architectures (sm_70=Volta, sm_75=Turing, sm_80/86=Ampere, sm_89=Ada, sm_90=Hopper)
# This can significantly reduce binary size on CUDA builds
def filter_cuda_binaries(binaries_list):
    """Filter CUDA binaries to keep only essential architectures."""
    filtered = []
    # Architectures to keep (covers most modern GPUs: GTX 1000+, RTX series, datacenter)
    keep_archs = {'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90', 'compute_70', 'compute_80', 'compute_90'}
    # Older architectures to exclude (Pascal and older: GTX 900 series and below)
    exclude_archs = {'sm_50', 'sm_52', 'sm_53', 'sm_60', 'sm_61', 'sm_62',
                     'compute_50', 'compute_52', 'compute_60', 'compute_61'}

    for binary_path, dest in binaries_list:
        filename = os.path.basename(binary_path).lower()
        # Check if this is an architecture-specific CUDA binary
        should_exclude = False
        for arch in exclude_archs:
            if arch in filename:
                should_exclude = True
                break
        if not should_exclude:
            filtered.append((binary_path, dest))
    return filtered

binaries = filter_cuda_binaries(binaries)

# Exclude heavy/unused PyTorch modules for inference-only use
# These modules are not needed for running a trained model
excludedimports = [
    # Training-only modules
    'torch.distributed',           # Multi-GPU distributed training
    'torch.distributed.elastic',
    'torch.distributed.fsdp',
    'torch.distributed.pipeline',
    'torch.distributed.rpc',
    'torch.distributed.optim',
    'torch.optim',                 # Optimizers (not needed for inference)
    'torch.optim.lr_scheduler',

    # Quantization (not used)
    'torch.ao',
    'torch.ao.quantization',
    'torch.quantization',

    # Profiling/debugging (not needed in production)
    'torch.profiler',
    'torch.autograd.profiler',
    'torch.utils.tensorboard',
    'torch.utils.benchmark',
    'torch.utils.bottleneck',
    'torch.utils.collect_env',

    # Mobile/export backends (not used)
    'torch.backends.xnnpack',
    'torch.onnx',
    'torch.package',

    # Testing modules
    'torch.testing',
    'torch.utils.hipify',
]
