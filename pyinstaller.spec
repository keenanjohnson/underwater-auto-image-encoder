# -*- mode: python ; coding: utf-8 -*-

import platform
import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Ensure src is importable
spec_dir = os.path.dirname(os.path.abspath(SPECPATH))
sys.path.insert(0, spec_dir)

block_cipher = None

# Determine platform-specific binaries
system = platform.system().lower()
if system == 'windows':
    gpr_binary = ('binaries/win32/gpr_tools.exe', 'binaries/win32')
elif system == 'darwin':
    gpr_binary = ('binaries/darwin/gpr_tools', 'binaries/darwin')
else:
    gpr_binary = ('binaries/linux/gpr_tools', 'binaries/linux')

# Check if binary exists - REQUIRED for packaging
binary_path = Path(gpr_binary[0])
if not binary_path.exists():
    print(f"\n" + "="*60)
    print(f"FATAL ERROR: GPR tools binary not found!")
    print(f"Expected location: {binary_path}")
    print(f"\nThe gpr_tools binary MUST be compiled before packaging.")
    print(f"Please run the appropriate build script:")
    if system == 'windows':
        print(f"  build_scripts\\compile_gpr_tools.bat")
    else:
        print(f"  ./build_scripts/compile_gpr_tools.sh")
    print("="*60 + "\n")
    raise FileNotFoundError(f"Required gpr_tools binary not found at {binary_path}")

# Verify it's a valid executable (not a placeholder)
file_size = binary_path.stat().st_size
if file_size < 10000:  # Less than 10KB is definitely wrong
    print(f"\n" + "="*60)
    print(f"FATAL ERROR: GPR tools binary appears invalid!")
    print(f"Binary path: {binary_path}")
    print(f"File size: {file_size} bytes (expected > 10KB)")
    print(f"The file may be corrupted or a placeholder.")
    print("="*60 + "\n")
    raise ValueError(f"Invalid gpr_tools binary at {binary_path} (size: {file_size} bytes)")

print(f"✓ Found valid GPR tools binary: {binary_path} (size: {file_size:,} bytes)")
binaries = [gpr_binary]

# Manually list all src modules since collect_submodules doesn't work in CI
src_modules = [
    'src',
    'src.models',
    'src.models.unet_autoencoder',
    'src.models.attention_unet',
    'src.utils',
    'src.converters',
    'src.converters.gpr_converter',
    'src.gui',
    'src.gui.main_window',
    'src.gui.image_processor',
]

# Try to collect submodules, fallback to manual list
try:
    src_hiddenimports = collect_submodules('src')
    if not src_hiddenimports:
        src_hiddenimports = src_modules
except:
    src_hiddenimports = src_modules

# Ensure all src Python files are included
src_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            src_files.append((os.path.join(root, file), root))

# Silently collect files (debug output removed)

# Exclude unnecessary modules while keeping GPU support
excludes = [
    # Exclude torch components we don't need for inference
    'torch.distributions',
    # 'torch.testing',  # Required by torch internals
    'torch.utils.tensorboard',
    'torch.utils.bottleneck',
    'torch.utils.benchmark',
    'torch.utils.cpp_extension',
    'torch.utils.mobile_optimizer',
    'torch.profiler',
    'torch.ao',  # quantization
    'torch.jit.mobile',
    'torch.onnx',
    'torch.quantization',
    'torch.fx',  # graph mode
    'torch.package',

    # Exclude torchvision extras we don't use
    'torchvision.datasets',
    'torchvision.io.video',
    'torchvision.models.detection',
    'torchvision.models.segmentation',
    'torchvision.models.video',
    'torchvision.models.quantization',
    'torchvision.ops',

    # Test frameworks
    'pytest',
    'unittest',
    'test',
    'tests',

    # Development/debug tools
    'IPython',
    'jupyter',
    'notebook',
    'ipykernel',
    'ipywidgets',
    'tensorboard',
    'setuptools',
    'pip',

    # Visualization libraries we don't use
    'matplotlib',
    'seaborn',
    'plotly',
    'bokeh',

    # Other ML frameworks
    'tensorflow',
    'keras',
    'jax',
    'transformers',

    # Web frameworks
    'flask',
    'django',
    'fastapi',
    'uvicorn',

    # Unused data science libraries
    'pandas',
    'scipy.stats',  # Keep core scipy for skimage
    'sympy',
    'statsmodels',
]

a = Analysis(
    ['app.py'],
    pathex=[str(Path.cwd())],  # Add absolute path to current directory
    binaries=binaries,
    datas=[
        ('config.yaml', '.') if Path('config.yaml').exists() else ('config.yaml', '.'),
    ] + src_files,  # Include all Python files from src
    hiddenimports=[
        'customtkinter',
        'torch',
        'torchvision',
        'PIL',
        'PIL.Image',
        'numpy',
        'cv2',
        'tqdm',
        'yaml',
        'skimage',
        'tkinter',
        'darkdetect',
    ] + src_hiddenimports,  # Add all src submodules
    hookspath=['.'],  # Use local hooks directory
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if system == 'darwin':
    # macOS: Use onedir mode for .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='UnderwaterEnhancer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,  # Strip symbols to reduce size
        upx=True,
        console=False,  # No console for macOS app
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=True,  # Strip symbols to reduce size
        upx=True,
        upx_exclude=['*.pyd', '*.so', '*.dylib'],  # Don't compress shared libs
        name='UnderwaterEnhancer',
    )
else:
    # Windows/Linux: Use onefile mode with optimizations

    # Platform-specific UPX exclusions
    if system == 'windows':
        # On Windows, only exclude critical DLLs that break when compressed
        upx_excludes = [
            '*.pyd',  # Python extensions
            'nvToolsExt*.dll',  # NVIDIA tools (known to fail)
            'cudart*.dll',  # CUDA runtime
            'cublas*.dll',  # CUDA BLAS
            'cufft*.dll',  # CUDA FFT
            'curand*.dll',  # CUDA random
            'cusolver*.dll',  # CUDA solver
            'cusparse*.dll',  # CUDA sparse
            'cudnn*.dll',  # cuDNN
            'nvrtc*.dll',  # NVIDIA runtime compilation
            'api-ms-win*.dll',  # Windows API
            'vcruntime*.dll',  # Visual C++ runtime
            'msvcp*.dll',  # Microsoft C++ runtime
        ]
    else:
        # Linux: exclude .so files
        upx_excludes = ['*.so']

    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='UnderwaterEnhancer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True if system == 'linux' else False,  # Strip only on Linux
        upx=True,
        upx_exclude=upx_excludes,
        runtime_tmpdir=None,
        console=False,  # Set to True for debugging
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/icon.ico' if system == 'windows' and Path('assets/icon.ico').exists() else None,
    )

# macOS specific
if system == 'darwin':
    app = BUNDLE(
        coll,
        name='UnderwaterEnhancer.app',
        icon='assets/icon.icns' if Path('assets/icon.icns').exists() else None,
        bundle_identifier='com.seattleaquarium.underwaterenhancer',
        info_plist={
            'CFBundleShortVersionString': '0.2.0',
            'CFBundleVersion': '0.2.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.13.0',
            'CFBundleExecutable': 'UnderwaterEnhancer',
            'CFBundleName': 'Underwater Enhancer',
            'CFBundleDisplayName': 'Underwater Enhancer',
            'CFBundleIdentifier': 'com.seattleaquarium.underwaterenhancer',
            'NSRequiresAquaSystemAppearance': False,
            'NSAppleEventsUsageDescription': 'This app needs to control other apps.',
            'NSCameraUsageDescription': 'This app needs camera access to process images.',
            'NSPhotoLibraryUsageDescription': 'This app needs photo library access to process images.',
        },
    )