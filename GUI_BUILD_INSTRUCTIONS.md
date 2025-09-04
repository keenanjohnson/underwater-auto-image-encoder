# Build Instructions for Underwater Image Enhancer

## Prerequisites

### All Platforms
- Python 3.8 or higher
- Git
- 8GB RAM minimum (16GB recommended)

### Platform-Specific
- **Windows**: Visual Studio Build Tools (for some Python packages)
- **macOS**: Xcode Command Line Tools
- **Linux**: build-essential package

## Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/Seattle-Aquarium/auto-image-encoder.git
cd auto-image-encoder
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# For GUI application only
pip install -r requirements-gui.txt

# For full development (includes training, etc.)
pip install -r requirements.txt
```

### 4. Install GPR Tools
```bash
# Automatic installation
bash install_gpr_tools.sh

# Or manually place gpr_tools binary in:
# - Windows: binaries/win32/gpr_tools.exe
# - macOS: binaries/darwin/gpr_tools
# - Linux: binaries/linux/gpr_tools
```

### 5. Run Development Version
```bash
python app.py
```

## Building Standalone Application

### Install PyInstaller
```bash
pip install pyinstaller
```

### Build Application

#### Windows
```bash
pyinstaller pyinstaller.spec
# Output: dist/UnderwaterEnhancer.exe
```

#### macOS
```bash
pyinstaller pyinstaller.spec
# Output: dist/UnderwaterEnhancer.app

# Sign the app (optional, prevents security warnings)
codesign --deep --force --sign - dist/UnderwaterEnhancer.app
```

#### Linux
```bash
pyinstaller pyinstaller.spec
# Output: dist/UnderwaterEnhancer
chmod +x dist/UnderwaterEnhancer
```

## Testing Build

### Smoke Test
```bash
# Test that all modules load correctly
./dist/UnderwaterEnhancer --smoke-test

# Or set environment variable
SMOKE_TEST=1 ./dist/UnderwaterEnhancer
```

### Manual Test
1. Run the built application
2. Load a model checkpoint file
3. Process a test image
4. Verify output quality

## Packaging for Distribution

### Windows
1. Build with PyInstaller
2. (Optional) Create installer with NSIS or Inno Setup
3. Compress to ZIP for distribution

### macOS
1. Build with PyInstaller
2. Sign the application (requires Apple Developer account for distribution)
3. Create DMG:
```bash
# Install create-dmg
brew install create-dmg

# Create DMG
create-dmg \
  --volname "Underwater Enhancer" \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "UnderwaterEnhancer.app" 175 120 \
  --app-drop-link 425 120 \
  UnderwaterEnhancer.dmg \
  dist/
```

### Linux
1. Build with PyInstaller
2. Create AppImage or .deb package (optional)
3. Compress to tar.gz for distribution

## Troubleshooting

### Common Issues

**Import Errors in Built App**
- Ensure all dependencies are installed in the virtual environment
- Check pyinstaller.spec hiddenimports section
- Run with --debug flag for detailed output

**GPR Files Not Working**
- Verify gpr_tools binary is included in binaries/ folder
- Check binary has execute permissions (macOS/Linux)
- Ensure rawpy is installed for DNG processing

**Large File Size**
- Use UPX to compress executables (optional):
```bash
# Install UPX
# Windows: Download from https://upx.github.io/
# macOS: brew install upx
# Linux: apt-get install upx

# Rebuild with compression
pyinstaller pyinstaller.spec --upx-dir=/path/to/upx
```

**macOS Security Warnings**
- Sign the app: `codesign --deep --force --sign - dist/UnderwaterEnhancer.app`
- Clear quarantine: `xattr -cr dist/UnderwaterEnhancer.app`
- Users may need to allow in System Preferences > Security & Privacy

### Debug Mode
```bash
# Build with debug output
pyinstaller pyinstaller.spec --debug all

# Run with console window (Windows)
pyinstaller pyinstaller.spec --console
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build Application

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements-gui.txt
        pip install pyinstaller
    
    - name: Build application
      run: pyinstaller pyinstaller.spec
    
    - name: Upload artifact
      uses: actions/upload-artifact@v2
      with:
        name: UnderwaterEnhancer-${{ matrix.os }}
        path: dist/
```

## Version Management

1. Update version in `app.py`:
```python
__version__ = "0.1.0-beta"
```

2. Tag release in git:
```bash
git tag v0.1.0-beta
git push origin v0.1.0-beta
```

3. Create GitHub release with built artifacts

## Support

For build issues, please open an issue at:
https://github.com/Seattle-Aquarium/auto-image-encoder/issues