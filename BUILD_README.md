# Building the Underwater Enhancer GUI Application

## Prerequisites

### All Platforms
- Python 3.8 or higher
- Git
- 8GB RAM minimum (16GB recommended)
- **CMake** (required for building gpr_tools)

### Platform-Specific
- **Windows**: Visual Studio 2019+ with C++ support (or Build Tools for Visual Studio)
- **macOS**: Xcode Command Line Tools
- **Linux**: build-essential package

## Quick Build

### Automated Build (Recommended)
```bash
python build_scripts/build_app.py
```

This will:
1. **Compile gpr_tools from source** (REQUIRED - build will fail without it)
2. Install Python dependencies
3. Build the standalone executable using PyInstaller with bundled gpr_tools
4. Output to `dist/` directory

### Platform-Specific Outputs
- **macOS**: `dist/UnderwaterEnhancer.app`
- **Windows**: `dist/UnderwaterEnhancer.exe`
- **Linux**: `dist/UnderwaterEnhancer`

## Manual Build Steps

### 1. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Compile GPR Tools (REQUIRED)
```bash
# Unix/macOS
chmod +x build_scripts/compile_gpr_tools.sh
./build_scripts/compile_gpr_tools.sh

# Windows
build_scripts\compile_gpr_tools.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements_gui.txt
```

### 4. Build with PyInstaller
```bash
pyinstaller pyinstaller.spec --clean --noconfirm
```

## GitHub Actions Build

The application is automatically built for all platforms when:
- Pushing to `main` or `basic-gui` branches
- Creating a release tag (v*)
- Manually triggering the workflow

### CI/CD Features
- ✅ Cross-platform builds (Windows, macOS, Linux)
- ✅ **Mandatory GPR tools compilation (CI fails if compilation fails)**
- ✅ CPU-only PyTorch for smaller artifacts
- ✅ Build artifacts available for 7 days
- ✅ Automatic release creation on version tags
- ✅ Cached dependencies for faster builds

## Build Requirements

### All Platforms
- Python 3.8+
- Git
- **CMake** (required for building gpr_tools)

### Platform-Specific
- **macOS**: Xcode Command Line Tools
- **Windows**: Visual Studio 2019+ with C++ support (or Build Tools for Visual Studio)
- **Linux**: build-essential, cmake, libgl1-mesa-glx

## GPR Support (MANDATORY)

**The gpr_tools binary is REQUIRED for building the application.** The build will fail if gpr_tools cannot be compiled.

The build process:
1. Checks for existing binary in `binaries/<platform>/`
2. Compiles from source if missing
3. **FAILS the build if compilation fails**
4. Validates the binary is valid (>10KB)
5. Bundles it into the executable

**Note**: The application uses only the bundled binary - there is no fallback to system PATH. This ensures consistent behavior across all installations.

## Development Setup (Running from Source)

### 1. Clone Repository
```bash
git clone https://github.com/Seattle-Aquarium/auto-image-encoder.git
cd auto-image-encoder
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
# For GUI application only
pip install -r requirements_gui.txt

# For full development (includes training, etc.)
pip install -r requirements.txt
```

### 4. Install GPR Tools
```bash
# Automatic compilation
chmod +x build_scripts/compile_gpr_tools.sh
./build_scripts/compile_gpr_tools.sh

# Or manually place gpr_tools binary in:
# - Windows: binaries/win32/gpr_tools.exe
# - macOS: binaries/darwin/gpr_tools
# - Linux: binaries/linux/gpr_tools
```

### 5. Run Development Version
```bash
python app.py
```

## Testing the Build

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

## Troubleshooting

### Common Issues

**Large Executable Size**
- The executable is ~166MB due to bundled PyTorch and dependencies. This is normal.

**Build Failures Due to GPR Tools**
If the build fails with "GPR tools binary not found":
1. Ensure CMake is installed and in PATH
2. For Windows: Ensure Visual Studio C++ build tools are installed
3. Check compilation logs for specific errors
4. Manually run the compile script to see detailed errors:
   - Windows: `build_scripts\compile_gpr_tools.bat`
   - Unix/macOS: `./build_scripts/compile_gpr_tools.sh`

**Import Errors in Built App**
- Ensure all dependencies are installed in the virtual environment
- Check pyinstaller.spec hiddenimports section
- Run with --debug flag for detailed output

**GPR Files Not Working**
- Verify gpr_tools binary is included in binaries/ folder
- Check binary has execute permissions (macOS/Linux)
- Ensure rawpy is installed for DNG processing

**Build Failures**
1. Check Python version (3.8+ required)
2. Ensure all dependencies installed: `pip install -r requirements_gui.txt`
3. Clean build directory: `rm -rf build dist`
4. Try again with console mode: Edit `pyinstaller.spec`, set `console=True`

**macOS Security Warnings**
- Sign the app: `codesign --deep --force --sign - dist/UnderwaterEnhancer.app`
- Clear quarantine: `xattr -cr dist/UnderwaterEnhancer.app`
- Users may need to allow in System Preferences > Security & Privacy
- See [MACOS_APP_INSTALLATION.md](MACOS_APP_INSTALLATION.md) for user instructions

### Debug Mode
```bash
# Build with debug output
pyinstaller pyinstaller.spec --debug all

# Run with console window (Windows)
pyinstaller pyinstaller.spec --console
```

### PyTorch Installation
For CPU-only PyTorch (smaller download):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

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

## CI/CD with GitHub Actions

The `.github/workflows/build-gui.yml` workflow automatically:
1. Builds for Windows, macOS, and Linux
2. **Compiles gpr_tools from source (mandatory - CI fails if compilation fails)**
3. Validates the gpr_tools binary (size check)
4. Bundles gpr_tools into the executable
5. Creates platform-specific archives
6. Uploads artifacts to GitHub (available for 7 days)
7. Creates releases when tagged

### CI/CD Features
- ✅ Cross-platform builds (Windows, macOS, Linux)
- ✅ Mandatory GPR tools compilation
- ✅ CPU-only PyTorch for smaller artifacts
- ✅ Build artifacts available for 7 days
- ✅ Automatic release creation on version tags
- ✅ Cached dependencies for faster builds

### Triggering a Release
```bash
git tag v1.0.0
git push origin v1.0.0
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