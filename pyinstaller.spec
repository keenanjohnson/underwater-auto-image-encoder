# -*- mode: python ; coding: utf-8 -*-

import platform
import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

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

# Collect PyTorch binaries for each platform
torch_binaries = []
if system == 'windows':
    try:
        # Collect PyTorch CUDA DLLs
        import torch
        torch_path = Path(torch.__file__).parent

        # Find CUDA DLLs in torch/lib directory
        cuda_dll_patterns = [
            'cudnn*.dll',
            'cublas*.dll',
            'cufft*.dll',
            'curand*.dll',
            'cusparse*.dll',
            'cudart*.dll',
            'nvrtc*.dll',
            'c10_cuda.dll',
            'torch_cuda*.dll',
            'caffe2_nvrtc.dll'
        ]

        lib_path = torch_path / 'lib'
        if lib_path.exists():
            for pattern in cuda_dll_patterns:
                dlls = list(lib_path.glob(pattern))
                for dll in dlls:
                    torch_binaries.append((str(dll), 'torch/lib'))
                    print(f"✓ Found CUDA DLL: {dll.name}")

        # Also check torch/bin for additional DLLs
        bin_path = torch_path / 'bin'
        if bin_path.exists():
            for pattern in cuda_dll_patterns:
                dlls = list(bin_path.glob(pattern))
                for dll in dlls:
                    torch_binaries.append((str(dll), 'torch/bin'))
                    print(f"✓ Found CUDA DLL: {dll.name}")

        if torch_binaries:
            print(f"✓ Found {len(torch_binaries)} CUDA DLLs for bundling")
        else:
            print("⚠ No CUDA DLLs found - GPU support may not work")

    except ImportError:
        print("⚠ PyTorch not found - cannot bundle CUDA DLLs")
    except Exception as e:
        print(f"⚠ Error collecting CUDA DLLs: {e}")

elif system == 'darwin':
    # macOS: Collect torch_shm_manager and other torch binaries
    try:
        import torch
        torch_path = Path(torch.__file__).parent

        # Critical binaries needed by PyTorch on macOS
        torch_bin_path = torch_path / 'bin'
        if torch_bin_path.exists():
            # torch_shm_manager is required for multiprocessing
            shm_manager = torch_bin_path / 'torch_shm_manager'
            if shm_manager.exists():
                torch_binaries.append((str(shm_manager), 'torch/bin'))
                print(f"✓ Found torch_shm_manager for macOS")
            else:
                print("⚠ torch_shm_manager not found - this may cause issues")

            # Include any other binaries in torch/bin
            for binary in torch_bin_path.iterdir():
                if binary.is_file() and binary.stat().st_mode & 0o111:  # Is executable
                    torch_binaries.append((str(binary), 'torch/bin'))
                    print(f"✓ Found torch binary: {binary.name}")

        # Also collect .dylib files from torch/lib
        torch_lib_path = torch_path / 'lib'
        if torch_lib_path.exists():
            for dylib in torch_lib_path.glob('*.dylib'):
                torch_binaries.append((str(dylib), 'torch/lib'))
                print(f"✓ Found torch library: {dylib.name}")

    except ImportError:
        print("⚠ PyTorch not found - cannot bundle torch binaries for macOS")
    except Exception as e:
        print(f"⚠ Error collecting torch binaries: {e}")

# Combine with GPR binary and torch binaries
all_binaries = binaries + torch_binaries

# Silently collect files (debug output removed)

a = Analysis(
    ['app.py'],
    pathex=[str(Path.cwd())],  # Add absolute path to current directory
    binaries=all_binaries,
    datas=[
        ('config.yaml', '.') if Path('config.yaml').exists() else ('config.yaml', '.'),
        ('check_gpu.py', '.') if Path('check_gpu.py').exists() else ('check_gpu.py', '.'),
    ] + src_files,  # Include all Python files from src
    hiddenimports=[
        'customtkinter',
        'torch',
        'torch.cuda',
        'torch._C',
        'torch._C._cuda',
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
    excludes=[],
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
        strip=False,
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
        strip=False,
        upx=True,
        upx_exclude=[],
        name='UnderwaterEnhancer',
    )
else:
    # Windows/Linux: Use onefile mode
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
        strip=False,
        upx=True,
        upx_exclude=[],
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
            'CFBundleShortVersionString': '0.1.0',
            'CFBundleVersion': '0.1.0',
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