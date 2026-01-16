"""
SS-UIE (State-Space Underwater Image Enhancement) Model Wrapper

Based on: "Adaptive Dual-domain Learning for Underwater Image Enhancement" (AAAI 2025)
GitHub: https://github.com/LintaoPeng/SS-UIE

This model uses Mamba blocks (state-space models) combined with FFT-based
frequency domain attention for underwater image enhancement.

Note: This model requires CUDA and the mamba-ssm package.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Check for mamba-ssm availability before importing SS-UIE
try:
    import mamba_ssm  # noqa: F401
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# Check for other required dependencies
try:
    import timm  # noqa: F401
    import einops  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def _get_ss_uie_model():
    """Lazily import SS_UIE_model to avoid import errors when dependencies missing."""
    import platform

    if not HAS_MAMBA:
        if platform.system() == 'Darwin':
            raise ImportError(
                "SS-UIE model is not available on macOS.\n\n"
                "The SS-UIE model requires NVIDIA CUDA, which is not supported on macOS.\n"
                "Please use a different model type (U-Net or U-Shape Transformer) on macOS,\n"
                "or run SS-UIE inference on a Windows/Linux system with an NVIDIA GPU."
            )
        else:
            raise ImportError(
                "SS-UIE model requires mamba-ssm package (CUDA only).\n\n"
                "This model requires an NVIDIA GPU with CUDA support.\n"
                "Install with: pip install mamba-ssm causal-conv1d\n\n"
                "If you don't have an NVIDIA GPU, please use a different model type\n"
                "(U-Net or U-Shape Transformer)."
            )
    if not HAS_DEPS:
        raise ImportError(
            "SS-UIE model requires timm and einops packages. "
            "Install with: pip install timm einops"
        )

    # Add the SS-UIE library path for imports
    ss_uie_path = Path(__file__).parent.parent.parent / "lib" / "SS-UIE"
    if str(ss_uie_path) not in sys.path:
        sys.path.insert(0, str(ss_uie_path))

    # Import the model - this will use the net.blocks imports internally
    from net.model import SS_UIE_model
    return SS_UIE_model


class SSUIEModel(nn.Module):
    """
    SS-UIE Model wrapper for underwater image enhancement.

    This model uses State-Space Models (Mamba) combined with FFT-based
    frequency domain attention for underwater image enhancement.

    Important: This model is resolution-aware. The H and W parameters
    must match the training resolution. For different resolutions at
    inference time, use tiled processing.

    Args:
        in_channels (int): Number of input channels (default: 3)
        channels (int): Base channel count (default: 16)
        num_memblock (int): Number of memory blocks (default: 6)
        num_resblock (int): Number of residual blocks per memory block (default: 6)
        drop_rate (float): Dropout rate (default: 0.0)
        H (int): Expected input height (default: 256)
        W (int): Expected input width (default: 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 16,
        num_memblock: int = 6,
        num_resblock: int = 6,
        drop_rate: float = 0.0,
        H: int = 256,
        W: int = 256,
    ):
        super(SSUIEModel, self).__init__()

        # Get the SS_UIE_model class (handles import and dependency checks)
        SS_UIE_model = _get_ss_uie_model()

        self.H = H
        self.W = W
        self.in_channels = in_channels
        self.channels = channels
        self.num_memblock = num_memblock
        self.num_resblock = num_resblock

        # Create the underlying SS-UIE model
        self.model = SS_UIE_model(
            in_channels=in_channels,
            channels=channels,
            num_memblock=num_memblock,
            num_resblock=num_resblock,
            drop_rate=drop_rate,
            H=H,
            W=W
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W), values in [0, 1]

        Returns:
            Enhanced image tensor of shape (B, C, H, W), values in [0, 1]
        """
        # SS-UIE model uses residual learning: output = input + learned_residual
        # Following the paper exactly: return raw output without sigmoid or clamp
        # The paper only clamps for PSNR metric calculation, not the actual output
        # Image conversion (to uint8) will naturally clip out-of-range values
        out = self.model(x)
        return out


def is_ss_uie_available() -> bool:
    """Check if SS-UIE model dependencies are available."""
    return HAS_MAMBA and HAS_DEPS


def get_ss_uie_unavailable_reason() -> str:
    """Get a user-friendly reason why SS-UIE is not available."""
    import platform

    if platform.system() == 'Darwin':
        return (
            "SS-UIE is not available on macOS because it requires NVIDIA CUDA, "
            "which is not supported on Apple hardware."
        )
    if not HAS_MAMBA:
        return (
            "SS-UIE requires the mamba-ssm package, which needs an NVIDIA GPU with CUDA. "
            "Install with: pip install mamba-ssm causal-conv1d"
        )
    if not HAS_DEPS:
        return "SS-UIE requires timm and einops packages. Install with: pip install timm einops"
    return ""
