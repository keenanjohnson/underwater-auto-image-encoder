#!/usr/bin/env python3
"""
GPU Detection and Verification Utility
Checks if GPU/CUDA is available and provides diagnostic information
"""

import sys
import platform
import os

def check_gpu():
    """Check GPU/CUDA availability and provide diagnostic info"""

    print("="*60)
    print("GPU/CUDA Detection Tool")
    print("="*60)

    # System info
    print(f"\nSystem Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")

        # CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA Status:")
        print(f"  CUDA available: {cuda_available}")

        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")

            # GPU details
            gpu_count = torch.cuda.device_count()
            print(f"\n  GPU count: {gpu_count}")

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Total memory: {props.total_memory / (1024**3):.1f} GB")

                # Current memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    print(f"    Memory allocated: {allocated:.1f} GB")
                    print(f"    Memory reserved: {reserved:.1f} GB")

            # Test tensor operation
            print(f"\n  Testing GPU computation...")
            try:
                device = torch.device('cuda:0')
                test_tensor = torch.randn(1000, 1000).to(device)
                result = torch.mm(test_tensor, test_tensor)
                print(f"  ✓ GPU computation test successful")
            except Exception as e:
                print(f"  ✗ GPU computation test failed: {e}")

        else:
            print(f"  PyTorch CUDA support: {torch.cuda.is_available()}")
            print(f"\n  Possible reasons GPU is not detected:")

            if platform.system() == 'Windows':
                print(f"  • NVIDIA drivers may not be installed")
                print(f"  • CUDA runtime libraries may be missing")
                print(f"  • PyTorch may not have CUDA support bundled")
                print(f"  • GPU may not be CUDA-capable")

                # Check for NVIDIA driver
                nvidia_smi = os.environ.get('CUDA_PATH')
                if nvidia_smi:
                    print(f"\n  CUDA_PATH found: {nvidia_smi}")
                else:
                    print(f"\n  CUDA_PATH not found in environment variables")

            elif platform.system() == 'Darwin':
                print(f"  • macOS does not support CUDA (Apple Silicon uses MPS)")
                print(f"  • MPS (Metal Performance Shaders) support:")
                print(f"    MPS available: {torch.backends.mps.is_available()}")
                print(f"    MPS built: {torch.backends.mps.is_built()}")

            elif platform.system() == 'Linux':
                print(f"  • NVIDIA drivers may not be installed")
                print(f"  • CUDA toolkit may not be installed")
                print(f"  • PyTorch may not have CUDA support")

        # Check MPS for Apple Silicon
        if platform.system() == 'Darwin':
            print(f"\nApple Silicon (MPS) Status:")
            print(f"  MPS available: {torch.backends.mps.is_available()}")
            print(f"  MPS built: {torch.backends.mps.is_built()}")

            if torch.backends.mps.is_available():
                print(f"\n  Testing MPS computation...")
                try:
                    device = torch.device('mps')
                    test_tensor = torch.randn(1000, 1000).to(device)
                    result = torch.mm(test_tensor, test_tensor)
                    print(f"  ✓ MPS computation test successful")
                except Exception as e:
                    print(f"  ✗ MPS computation test failed: {e}")

    except ImportError as e:
        print(f"\n✗ PyTorch not installed or import error: {e}")

    # Environment variables
    print(f"\nRelevant Environment Variables:")
    env_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDNN_PATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            if var == 'PATH':
                # Only show CUDA-related paths
                cuda_paths = [p for p in value.split(os.pathsep) if 'cuda' in p.lower() or 'nvidia' in p.lower()]
                if cuda_paths:
                    print(f"  {var} (CUDA-related):")
                    for path in cuda_paths[:3]:  # Limit to first 3
                        print(f"    - {path}")
            else:
                print(f"  {var}: {value}")

    print("\n" + "="*60)

    # Return status for programmatic use
    if 'torch' in sys.modules:
        return torch.cuda.is_available()
    return False

if __name__ == "__main__":
    # Run check and exit with appropriate code
    gpu_available = check_gpu()

    if gpu_available:
        print("\n✓ GPU is available and ready for use!")
        sys.exit(0)
    else:
        print("\n✗ GPU is not available. Running on CPU.")
        sys.exit(1)