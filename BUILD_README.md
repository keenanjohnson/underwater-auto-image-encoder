# Building the Underwater Enhancer GUI Application

## Quick Build

### Local Build
```bash
python build_scripts/build_app.py
```

This will:
1. Compile gpr_tools for your platform (if not already present)
2. Install Python dependencies
3. Build the standalone executable using PyInstaller
4. Output to `dist/` directory

### Platform-Specific Outputs
- **macOS**: `dist/UnderwaterEnhancer.app`
- **Windows**: `dist/UnderwaterEnhancer.exe`
- **Linux**: `dist/UnderwaterEnhancer`

## Manual Build Steps

### 1. Compile GPR Tools (Optional - for GPR file support)
```bash
# Unix/macOS
chmod +x build_scripts/compile_gpr_tools.sh
./build_scripts/compile_gpr_tools.sh

# Windows
build_scripts\compile_gpr_tools.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements_gui.txt
```

### 3. Build with PyInstaller
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
- ✅ Automatic GPR tools compilation
- ✅ CPU-only PyTorch for smaller artifacts
- ✅ Build artifacts available for 7 days
- ✅ Automatic release creation on version tags
- ✅ Cached dependencies for faster builds

## Build Requirements

### All Platforms
- Python 3.8+
- Git

### Platform-Specific
- **macOS**: Xcode Command Line Tools, CMake
- **Windows**: Visual Studio 2019+ or MinGW, CMake
- **Linux**: build-essential, cmake, libgl1-mesa-glx

## GPR Support

GPR file support requires the gpr_tools binary. The build script will:
1. Check for existing binary in `binaries/<platform>/`
2. Attempt to compile from source if missing
3. Continue without GPR support if compilation fails

**Important**: GPR support requires the bundled binary - there is no fallback to system PATH. This ensures consistent behavior across all installations.

## Troubleshooting

### Large Executable Size
The executable is ~166MB due to bundled PyTorch and dependencies. This is normal.

### Missing GPR Support
If GPR files aren't working:
1. Ensure gpr_tools compiled successfully
2. Check `binaries/<platform>/gpr_tools` exists
3. Rebuild with `python build_scripts/build_app.py`

### Build Failures
1. Check Python version (3.8+ required)
2. Ensure all dependencies installed: `pip install -r requirements_gui.txt`
3. Clean build directory: `rm -rf build dist`
4. Try again with console mode: Edit `pyinstaller.spec`, set `console=True`

## CI/CD with GitHub Actions

The `.github/workflows/build-gui.yml` workflow automatically:
1. Builds for Windows, macOS, and Linux
2. Compiles gpr_tools for each platform
3. Creates platform-specific archives
4. Uploads artifacts to GitHub
5. Creates releases when tagged

To trigger a release:
```bash
git tag v1.0.0
git push origin v1.0.0
```