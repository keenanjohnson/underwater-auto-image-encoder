# -*- mode: python ; coding: utf-8 -*-

import platform
import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Ensure src is importable
# When running "pyinstaller gui/pyinstaller.spec" from project root:
# - SPECPATH is set by PyInstaller to the DIRECTORY containing the spec file
# - We need to get to the project root directory (one level up from SPECPATH)
#
# IMPORTANT: SPECPATH is the directory path, not the file path!
print(f"\nDEBUG: SPECPATH = {SPECPATH}")
print(f"DEBUG: os.getcwd() = {os.getcwd()}")

# SPECPATH is already the directory containing this spec file (.../gui)
spec_file_dir = SPECPATH
print(f"DEBUG: spec_file_dir = {spec_file_dir}")

# Go up one level to get project root
spec_dir = os.path.dirname(spec_file_dir)
print(f"DEBUG: spec_dir (project root) = {spec_dir}")

sys.path.insert(0, spec_dir)

block_cipher = None

# Determine platform-specific binaries (relative to project root)
system = platform.system().lower()
if system == 'windows':
    gpr_binary = (os.path.join(spec_dir, 'binaries/win32/gpr_tools.exe'), 'binaries/win32')
elif system == 'darwin':
    gpr_binary = (os.path.join(spec_dir, 'binaries/darwin/gpr_tools'), 'binaries/darwin')
else:
    gpr_binary = (os.path.join(spec_dir, 'binaries/linux/gpr_tools'), 'binaries/linux')

# Debug: print what we're looking for
print(f"\nProject root: {spec_dir}")
print(f"Looking for binary at: {gpr_binary[0]}")
print(f"Binary directory exists: {os.path.exists(os.path.dirname(gpr_binary[0]))}")
if os.path.exists(os.path.dirname(gpr_binary[0])):
    print(f"Contents of binary directory: {os.listdir(os.path.dirname(gpr_binary[0]))}")

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

# Ensure all src Python files are included (from project root)
src_files = []
src_path = os.path.join(spec_dir, 'src')
for root, dirs, files in os.walk(src_path):
    for file in files:
        if file.endswith('.py'):
            rel_root = os.path.relpath(root, spec_dir)
            src_files.append((os.path.join(root, file), rel_root))

# Silently collect files (debug output removed)

a = Analysis(
    [os.path.join(spec_dir, 'gui', 'app.py')],
    pathex=[spec_dir],  # Add absolute path to project root
    binaries=binaries,
    datas=src_files,  # Include all Python files from src
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
    hookspath=[os.path.join(spec_dir, 'gui')],  # Use gui hooks directory
    hooksconfig={},
    runtime_hooks=[os.path.join(spec_dir, 'gui', 'runtime_hook.py')],
    excludes=[
        # Exclude unused libraries to reduce size (keeps CUDA support)
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'setuptools',
        'wheel',
        'pip',
        # Exclude test modules
        'torch.testing',
        'torch.utils.tensorboard',
        'torch.utils.benchmark',
        # Exclude unused torchvision modules
        'torchvision.datasets',
        'torchvision.models.detection',
        'torchvision.models.segmentation',
        'torchvision.models.video',
        # Exclude unused image processing
        'skimage.data',
    ],
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
        icon=os.path.join(spec_dir, 'assets', 'icon.ico') if system == 'windows' and Path(os.path.join(spec_dir, 'assets', 'icon.ico')).exists() else None,
    )

# macOS specific
if system == 'darwin':
    app = BUNDLE(
        coll,
        name='UnderwaterEnhancer.app',
        icon=os.path.join(spec_dir, 'assets', 'icon.icns') if Path(os.path.join(spec_dir, 'assets', 'icon.icns')).exists() else None,
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