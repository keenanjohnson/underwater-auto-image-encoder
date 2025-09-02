# -*- mode: python ; coding: utf-8 -*-

import platform
import sys
from pathlib import Path

block_cipher = None

# Determine platform-specific binaries
system = platform.system().lower()
if system == 'windows':
    gpr_binary = ('binaries/win32/gpr_tools.exe', 'binaries/win32')
elif system == 'darwin':
    gpr_binary = ('binaries/darwin/gpr_tools', 'binaries/darwin')
else:
    gpr_binary = ('binaries/linux/gpr_tools', 'binaries/linux')

# Check if binary exists, if not, create placeholder
binary_path = Path(gpr_binary[0])
if not binary_path.exists():
    print(f"Warning: {binary_path} not found. Creating placeholder.")
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    # For now, we'll include the binary only if it exists
    binaries = []
else:
    binaries = [gpr_binary]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('src', 'src'),
        ('config.yaml', '.') if Path('config.yaml').exists() else ('config.yaml', '.'),
    ],
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
        exe,
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