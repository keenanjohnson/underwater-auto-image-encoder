# GUI Application with Bundled GPR Support - Implementation Plan

## Overview
Create a single executable Python GUI application that bundles the gpr_tools binary internally, allowing users to process GPR files directly without external dependencies.

## Architecture Decision
- **Approach**: Bundle platform-specific gpr_tools binaries within PyInstaller executable
- **Rationale**: Provides seamless GPR support without requiring users to install additional tools
- **Trade-off**: Larger executable size (~5-10MB per platform binary) for better user experience

## Directory Structure
```
auto-image-encoder/
├── app.py                          # Main GUI application
├── src/
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py         # CustomTkinter interface
│   │   └── image_processor.py     # Processing logic
│   ├── converters/
│   │   ├── __init__.py
│   │   └── gpr_converter.py       # GPR to TIFF conversion wrapper
│   └── models/                    # Existing ML models
├── binaries/
│   ├── darwin/                    # macOS
│   │   └── gpr_tools
│   ├── linux/                     # Linux
│   │   └── gpr_tools
│   └── win32/                     # Windows
│       └── gpr_tools.exe
├── build_scripts/
│   ├── compile_gpr_tools.sh       # Build gpr_tools for current platform
│   ├── build_app.py               # PyInstaller build automation
│   └── sign_app.sh                # Code signing for macOS/Windows
├── pyinstaller.spec                # PyInstaller configuration
├── requirements_gui.txt           # GUI-specific dependencies
└── assets/
    ├── icon.ico                   # Windows icon
    ├── icon.icns                  # macOS icon
    └── icon.png                   # Linux/source icon
```

## Implementation Steps

### Phase 1: GPR Converter Module

Create `src/converters/gpr_converter.py`:
```python
"""
GPR to TIFF converter using bundled gpr_tools binary
"""
import sys
import platform
import subprocess
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GPRConverter:
    """Handles GPR to TIFF conversion using bundled gpr_tools"""
    
    @staticmethod
    def get_gpr_tools_path():
        """Get path to bundled gpr_tools binary for current platform"""
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            base_path = Path(sys._MEIPASS) / 'binaries'
        else:
            # Running in development mode
            base_path = Path(__file__).parent.parent.parent / 'binaries'
        
        system = platform.system().lower()
        if system == 'windows':
            binary_name = 'gpr_tools.exe'
            binary_path = base_path / 'win32' / binary_name
        elif system == 'darwin':
            binary_path = base_path / 'darwin' / 'gpr_tools'
        else:  # linux
            binary_path = base_path / 'linux' / 'gpr_tools'
        
        if not binary_path.exists():
            raise FileNotFoundError(f"gpr_tools binary not found at {binary_path}")
        
        # Ensure binary is executable on Unix systems
        if system != 'windows':
            import stat
            binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)
        
        return binary_path
    
    @classmethod
    def convert(cls, gpr_path: Path, output_path: Path = None, 
                keep_temp: bool = False) -> Path:
        """
        Convert GPR file to TIFF
        
        Args:
            gpr_path: Path to input GPR file
            output_path: Optional output path (auto-generated if None)
            keep_temp: Keep temporary files for debugging
        
        Returns:
            Path to converted TIFF file
        """
        gpr_tools = cls.get_gpr_tools_path()
        
        if output_path is None:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
                output_path = Path(tmp.name)
        
        # Build command
        cmd = [str(gpr_tools), str(gpr_path), '-o', str(output_path)]
        
        try:
            # Run conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30  # 30 second timeout
            )
            
            logger.info(f"Successfully converted {gpr_path.name} to TIFF")
            
            if not output_path.exists():
                raise RuntimeError(f"Conversion succeeded but output file not found: {output_path}")
            
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"GPR conversion timed out for {gpr_path}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"GPR conversion failed: {e.stderr}")
            raise
        finally:
            # Cleanup temp files if needed
            if not keep_temp and output_path.exists() and output_path.parent == Path(tempfile.gettempdir()):
                try:
                    output_path.unlink()
                except:
                    pass
    
    @classmethod
    def is_gpr_file(cls, file_path: Path) -> bool:
        """Check if file is a GPR file"""
        return file_path.suffix.lower() in ['.gpr']
    
    @classmethod
    def batch_convert(cls, gpr_files: list[Path], output_dir: Path = None,
                     progress_callback=None) -> list[Path]:
        """
        Convert multiple GPR files to TIFF
        
        Args:
            gpr_files: List of GPR file paths
            output_dir: Directory for output files
            progress_callback: Optional callback(current, total, filename)
        
        Returns:
            List of converted TIFF paths
        """
        converted_files = []
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, gpr_path in enumerate(gpr_files):
            if progress_callback:
                progress_callback(i, len(gpr_files), gpr_path.name)
            
            if output_dir:
                output_path = output_dir / f"{gpr_path.stem}.tiff"
            else:
                output_path = None
            
            try:
                tiff_path = cls.convert(gpr_path, output_path)
                converted_files.append(tiff_path)
            except Exception as e:
                logger.error(f"Failed to convert {gpr_path}: {e}")
                # Continue with other files
        
        return converted_files
```

### Phase 2: Modified Image Processor

Update the image processing pipeline to handle GPR files:
```python
"""
Image processor that handles GPR, TIFF, and JPEG inputs
"""
from pathlib import Path
from src.converters.gpr_converter import GPRConverter
import tempfile

class ImageProcessor:
    def __init__(self, model_path: str):
        self.inferencer = Inferencer(model_path)
        self.gpr_converter = GPRConverter()
    
    def process_image(self, input_path: Path, output_path: Path):
        """Process image, handling GPR conversion if needed"""
        
        # Check if input is GPR
        if self.gpr_converter.is_gpr_file(input_path):
            # Convert to TIFF first
            with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
                tiff_path = Path(tmp.name)
            
            tiff_path = self.gpr_converter.convert(input_path, tiff_path)
            
            try:
                # Process the TIFF
                result = self.inferencer.process_image(tiff_path, output_path)
            finally:
                # Clean up temp TIFF
                if tiff_path.exists():
                    tiff_path.unlink()
            
            return result
        else:
            # Direct processing for TIFF/JPEG
            return self.inferencer.process_image(input_path, output_path)
```

### Phase 3: PyInstaller Configuration

`pyinstaller.spec`:
```python
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

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[gpr_binary],  # Include platform-specific binary
    datas=[
        ('src', 'src'),
        ('config.yaml', '.'),
    ],
    hiddenimports=[
        'customtkinter',
        'torch',
        'torchvision',
        'PIL',
        'numpy',
        'cv2',
        'tqdm',
        'yaml',
        'skimage',
        'tkinter',
        'darkdetect',  # For system theme detection
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
    icon='assets/icon.ico' if system == 'windows' else None,
)

# macOS specific
if system == 'darwin':
    app = BUNDLE(
        exe,
        name='UnderwaterEnhancer.app',
        icon='assets/icon.icns',
        bundle_identifier='com.seattleaquarium.underwaterenhancer',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHighResolutionCapable': True,
        },
    )
```

### Phase 4: Build Automation Script

`build_scripts/build_app.py`:
```python
#!/usr/bin/env python3
"""
Build script for creating platform-specific executables
"""
import sys
import platform
import subprocess
import shutil
from pathlib import Path

def compile_gpr_tools():
    """Compile gpr_tools for current platform"""
    print("Compiling gpr_tools...")
    
    system = platform.system().lower()
    
    # Run the compile script
    if system == 'windows':
        # Windows compilation (requires Visual Studio)
        subprocess.run(['build_scripts/compile_gpr_tools.bat'], check=True)
    else:
        # Unix compilation
        subprocess.run(['bash', 'build_scripts/compile_gpr_tools.sh'], check=True)
    
    print("✓ gpr_tools compiled successfully")

def verify_binary():
    """Verify the gpr_tools binary exists"""
    system = platform.system().lower()
    
    if system == 'windows':
        binary_path = Path('binaries/win32/gpr_tools.exe')
    elif system == 'darwin':
        binary_path = Path('binaries/darwin/gpr_tools')
    else:
        binary_path = Path('binaries/linux/gpr_tools')
    
    if not binary_path.exists():
        print(f"✗ Binary not found at {binary_path}")
        print("Please compile gpr_tools first:")
        print("  ./build_scripts/compile_gpr_tools.sh")
        return False
    
    print(f"✓ Found binary at {binary_path}")
    return True

def build_executable():
    """Build the executable using PyInstaller"""
    print("\nBuilding executable with PyInstaller...")
    
    # Clean previous builds
    for dir in ['build', 'dist']:
        if Path(dir).exists():
            shutil.rmtree(dir)
    
    # Run PyInstaller
    subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        'pyinstaller.spec',
        '--clean',
        '--noconfirm'
    ], check=True)
    
    print("✓ Executable built successfully")
    
    # Show output location
    system = platform.system().lower()
    if system == 'darwin':
        print(f"\nApplication bundle created: dist/UnderwaterEnhancer.app")
    elif system == 'windows':
        print(f"\nExecutable created: dist/UnderwaterEnhancer.exe")
    else:
        print(f"\nExecutable created: dist/UnderwaterEnhancer")

def main():
    print("="*50)
    print("Underwater Enhancer Build Script")
    print("="*50)
    
    # Step 1: Verify or compile gpr_tools
    if not verify_binary():
        compile_gpr_tools()
        if not verify_binary():
            print("Failed to compile gpr_tools")
            sys.exit(1)
    
    # Step 2: Install dependencies
    print("\nInstalling Python dependencies...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements_gui.txt'
    ], check=True)
    
    # Step 3: Build executable
    build_executable()
    
    print("\n" + "="*50)
    print("Build completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
```

### Phase 5: gpr_tools Compilation Script

`build_scripts/compile_gpr_tools.sh`:
```bash
#!/bin/bash
# Compile gpr_tools for current platform

set -e

echo "Compiling gpr_tools for $(uname -s)..."

# Clone gpr repository if not exists
if [ ! -d "temp/gpr" ]; then
    mkdir -p temp
    cd temp
    git clone https://github.com/gopro/gpr.git
    cd ..
fi

cd temp/gpr

# Build based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    mkdir -p build
    cd build
    cmake ..
    make
    
    # Copy binary
    mkdir -p ../../../binaries/darwin
    cp gpr_tools ../../../binaries/darwin/
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    mkdir -p build
    cd build
    cmake ..
    make
    
    # Copy binary
    mkdir -p ../../../binaries/linux
    cp gpr_tools ../../../binaries/linux/
fi

echo "✓ gpr_tools compiled successfully"
```

## Testing Strategy

### 1. Unit Tests for GPR Converter
```python
# tests/test_gpr_converter.py
import pytest
from pathlib import Path
from src.converters.gpr_converter import GPRConverter

def test_gpr_tools_binary_exists():
    """Test that gpr_tools binary can be found"""
    binary_path = GPRConverter.get_gpr_tools_path()
    assert binary_path.exists()

def test_gpr_conversion(sample_gpr_file):
    """Test GPR to TIFF conversion"""
    output = GPRConverter.convert(sample_gpr_file)
    assert output.exists()
    assert output.suffix == '.tiff'
```

### 2. Integration Tests
- Test full pipeline: GPR → TIFF → ML Enhancement → Output
- Test batch processing with mixed file types
- Test error handling for corrupted GPR files

## Deployment Checklist

### Pre-Build
- [ ] Compile gpr_tools for target platform
- [ ] Verify binary permissions and execution
- [ ] Update version numbers in spec file
- [ ] Run all tests

### Build Process
- [ ] Clean previous builds
- [ ] Run PyInstaller with spec file
- [ ] Test executable on clean system
- [ ] Verify GPR conversion works in bundled app

### Platform-Specific
- **macOS**: 
  - [ ] Code sign the app bundle
  - [ ] Notarize for distribution
  - [ ] Test on Intel and Apple Silicon
  
- **Windows**:
  - [ ] Code sign the executable
  - [ ] Test on Windows 10/11
  - [ ] Check antivirus compatibility
  
- **Linux**:
  - [ ] Test on Ubuntu/Debian
  - [ ] Create AppImage for distribution
  - [ ] Test on different distros

## File Size Estimates

- Base PyInstaller Python: ~40MB (smaller with CustomTkinter vs NiceGUI)
- CustomTkinter + dependencies: ~10MB
- PyTorch models: ~100MB
- ML model checkpoint: ~120MB
- gpr_tools binary: ~5MB per platform
- **Total executable**: ~275MB

## Error Handling

The application should gracefully handle:
1. Missing gpr_tools binary → Show clear error message
2. GPR conversion failure → Fall back to manual file selection
3. Corrupted GPR files → Skip and continue with batch
4. Permission errors → Request admin rights or show instructions

## Requirements File

`requirements_gui.txt`:
```
# GUI Framework
customtkinter==5.2.0
darkdetect==0.8.0  # For system theme detection

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-image>=0.21.0

# ML Dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# Utilities
PyYAML>=6.0
tqdm>=4.65.0

# Packaging
pyinstaller>=6.0.0
```

## Future Enhancements

1. **Caching**: Cache TIFF conversions to speed up re-processing
2. **Parallel Processing**: Convert multiple GPR files in parallel
3. **Format Detection**: Auto-detect file format without extension
4. **Direct GPR Support**: Eventually implement native Python GPR decoder