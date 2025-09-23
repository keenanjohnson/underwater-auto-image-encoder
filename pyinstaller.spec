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
    # Safe exclusions - these are definitely not needed for inference
    'torch.utils.tensorboard',  # TensorBoard logging
    'torch.utils.bottleneck',  # Profiling tool
    'torch.utils.benchmark',  # Benchmarking tool
    'torch.utils.cpp_extension',  # For building C++ extensions
    'torch.utils.mobile_optimizer',  # Mobile optimization

    # Test frameworks
    'pytest',
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

# Remove unnecessary binaries to reduce size
print("Removing unnecessary binaries...")
binaries_to_remove = []
for (dest, source, kind) in a.binaries:
    dest_lower = dest.lower()

    # Remove test and benchmark binaries
    if any(x in dest_lower for x in ['_test', 'test_', 'benchmark', '_bench']):
        binaries_to_remove.append((dest, source, kind))
        print(f"  Removing test/benchmark: {dest}")
        continue

    # Remove duplicate/unnecessary MKL libraries (keep core ones)
    if 'mkl' in dest_lower:
        # Remove large MKL libraries we don't need
        if any(x in dest_lower for x in ['mkl_avx2', 'mkl_avx512', 'mkl_mc', 'mkl_mc3', 'mkl_def', 'mkl_vml_']):
            binaries_to_remove.append((dest, source, kind))
            print(f"  Removing MKL library: {dest}")
            continue

    # Remove debug libraries
    if dest_lower.endswith('_d.dll') or dest_lower.endswith('_d.so') or '_debug' in dest_lower:
        binaries_to_remove.append((dest, source, kind))
        print(f"  Removing debug library: {dest}")
        continue

    # Remove CUDA libraries for architectures we don't need (keep only common ones)
    if 'cuda' in dest_lower and any(x in dest_lower for x in ['sm_35', 'sm_37', 'sm_50', 'sm_52']):
        binaries_to_remove.append((dest, source, kind))
        print(f"  Removing old CUDA arch: {dest}")
        continue

    # Remove large torch libraries we don't need
    if 'torch' in dest_lower:
        # Remove torch testing libraries
        if any(x in dest_lower for x in ['torch_test', 'test_torch', '_test.', 'testing']):
            binaries_to_remove.append((dest, source, kind))
            print(f"  Removing torch test library: {dest}")
            continue

        # Remove torch profiler libraries if present
        if 'profiler' in dest_lower:
            binaries_to_remove.append((dest, source, kind))
            print(f"  Removing torch profiler: {dest}")
            continue

print(f"Removing {len(binaries_to_remove)} unnecessary binaries")
for item in binaries_to_remove:
    if item in a.binaries:
        a.binaries.remove(item)

# Remove unnecessary data files
print("Removing unnecessary data files...")
datas_to_remove = []
for (dest, source, kind) in a.datas:
    dest_lower = dest.lower()

    # Remove documentation and examples
    if any(x in dest_lower for x in ['/doc/', '/docs/', '/documentation/', '/examples/', '/example/',
                                      '/test/', '/tests/', '/testing/', '/sample/', '/samples/',
                                      '.md', '.rst', '.txt', 'readme', 'license', 'changelog']):
        datas_to_remove.append((dest, source, kind))
        continue

    # Remove source maps and debug files
    if any(dest.endswith(x) for x in ['.map', '.pdb', '.exp', '.lib', '.a']):
        datas_to_remove.append((dest, source, kind))
        continue

    # Remove TypeScript files and other source files we don't need at runtime
    if any(dest.endswith(x) for x in ['.ts', '.tsx', '.jsx', '.coffee', '.scss', '.sass', '.less']):
        datas_to_remove.append((dest, source, kind))
        continue

    # Remove translation/locale files except English
    if ('/locale/' in dest or '/locales/' in dest or '/translations/' in dest) and '/en' not in dest_lower:
        datas_to_remove.append((dest, source, kind))
        continue

print(f"Removing {len(datas_to_remove)} unnecessary data files")
for item in datas_to_remove:
    if item in a.datas:
        a.datas.remove(item)

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