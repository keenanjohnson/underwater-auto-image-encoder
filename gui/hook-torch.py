"""
PyInstaller hook for PyTorch - collects all necessary components
KEEPS CUDA SUPPORT

Size optimization: Only include essential data files, not Python source files
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect PyTorch dynamic libraries (including CUDA)
binaries = collect_dynamic_libs('torch')

# Collect only essential data files (not Python source files)
# This saves significant space as PyTorch has many .py files
# Note: If JIT compilation fails, set include_py_files=True
datas = collect_data_files('torch', include_py_files=False)

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
