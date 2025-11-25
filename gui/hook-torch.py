"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT - includes source files for JIT functionality

Note: CUDA libraries are highly interdependent and cannot be selectively excluded
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all PyTorch dynamic libraries (including CUDA)
# IMPORTANT: include_py_files=True to include .py source files
# PyTorch's JIT system needs access to source code via torch._sources
datas = collect_data_files('torch', include_py_files=True)
binaries = collect_dynamic_libs('torch')

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
