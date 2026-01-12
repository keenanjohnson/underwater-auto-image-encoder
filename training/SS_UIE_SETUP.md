# SS-UIE Model Setup Guide

The SS-UIE (State-Space Underwater Image Enhancement) model requires special setup due to its use of [Mamba](https://github.com/state-spaces/mamba) state-space models with custom CUDA kernels.

## Requirements

- **CUDA 12.8+** toolkit (for sm_120 / Blackwell architecture support)
- **PyTorch 2.9+** with CUDA 12.8
- **GCC 11+** for compiling CUDA extensions

## RTX 50-Series (Blackwell) GPU Support

If you have an RTX 5090, 5080, or other Blackwell architecture GPU (compute capability 12.0 / sm_120), you'll encounter this error with pre-built mamba-ssm wheels:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

This happens because the pre-built `mamba-ssm` and `causal-conv1d` packages don't include sm_120 kernels. You need to build from source.

## Step 1: Install CUDA 12.8 Toolkit

The system CUDA toolkit must be 12.8+ to compile for sm_120.

### Ubuntu 24.04

```bash
# Download and install the CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt update

# Install CUDA 12.8 toolkit (toolkit only, won't change drivers)
sudo apt install cuda-toolkit-12-8
```

### Ubuntu 22.04

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8
```

### Set Environment Variables

Add to your `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then reload: `source ~/.bashrc`

Verify installation:

```bash
/usr/local/cuda-12.8/bin/nvcc --version
# Should show: Cuda compilation tools, release 12.8
```

## Step 2: Uninstall Pre-built Packages

```bash
source env/bin/activate
pip uninstall mamba-ssm causal-conv1d -y
```

## Step 3: Build causal-conv1d from Source

```bash
# Set build environment
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"

# Clone and install
git clone https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d
cd /tmp/causal-conv1d
pip install .
```

## Step 4: Build mamba-ssm from Source

```bash
# Ensure environment is set
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"

# Clone repository
git clone https://github.com/state-spaces/mamba.git /tmp/mamba
cd /tmp/mamba

# Edit setup.py to add sm_120 support (around line 193)
# Find the line with cc_flag and add:
#   cc_flag.append("arch=compute_120,code=sm_120")

# Install (this takes several minutes)
pip install .
```

### Automated setup.py Patch

You can use sed to patch the file:

```bash
cd /tmp/mamba
# Add sm_120 architecture flag
sed -i '/cc_flag = \[\]/a\    cc_flag.append("-gencode", "arch=compute_120,code=sm_120")' setup.py
pip install .
```

## Step 5: Verify Installation

```bash
python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
python -c "import causal_conv1d; print('causal-conv1d OK')"
```

## Step 6: Test SS-UIE Model

```bash
python -c "
import torch
from src.models.ss_uie import SSUIEModel, is_ss_uie_available
print(f'SS-UIE available: {is_ss_uie_available()}')
model = SSUIEModel(H=256, W=256).cuda()
x = torch.randn(1, 3, 256, 256).cuda()
y = model(x)
print(f'Output shape: {y.shape}')
print('SS-UIE model works!')
"
```

## Troubleshooting

### "Invalid handle. Cannot load symbol cublasLtCreate"

This can occur if there's a mismatch between CUDA versions. Ensure:
- PyTorch CUDA version matches the toolkit version
- `LD_LIBRARY_PATH` includes the correct CUDA lib64 path

### Compilation errors with GCC

If you get GCC-related errors, try using GCC 11:

```bash
sudo apt install gcc-11 g++-11
export CC=gcc-11
export CXX=g++-11
```

### Out of memory during compilation

Limit parallel jobs:

```bash
MAX_JOBS=4 pip install .
```

## High-Resolution Inference

SS-UIE has **resolution-dependent components** that require special handling for images larger than the training resolution:

### Why Tiling is Required

1. **SWSA (Spectral-Wise Self-Attention)**: Uses learnable FFT filters (K) with dimensions matching the training resolution. These filters cannot be arbitrarily resized.

2. **MCSS (Multi-scale Cycle Selective Scan)**: Mamba state-space models process sequences of fixed length (H×W).

### How Inference Works

For high-resolution images (e.g., 4606×4030), the inference script automatically:

1. Splits the image into 512×512 tiles (matching training resolution)
2. Processes each tile through the model
3. Uses weighted blending at overlapping regions
4. Edge tiles are resized to 512×512, processed, then resized back

### Inference Command

```bash
# Full-size processing with automatic tiling
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth --full-size

# Or resize to training resolution (faster, lower quality)
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth
```

### Tiling Parameters

| Parameter | Default for SS-UIE | Notes |
|-----------|-------------------|-------|
| Tile size | 512×512 | Must match training H/W |
| Overlap | ~170px (tile_size/3) | Larger than U-Net due to global receptive field |
| Edge handling | Resize + process + resize | Not padding (would break FFT filters) |

### Quality Considerations

- Larger overlap reduces tile boundary artifacts but increases processing time
- The global receptive field of Mamba means each tile learns context independently
- Some subtle discontinuities at boundaries are unavoidable with any tiling approach

## References

- [Mamba GitHub Issue #745: RTX 5090 support](https://github.com/state-spaces/mamba/issues/745)
- [SS-UIE Paper (AAAI 2025)](https://github.com/LintaoPeng/SS-UIE)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive)
- [PyTorch sm_120 Support Issue](https://github.com/pytorch/pytorch/issues/159207)
