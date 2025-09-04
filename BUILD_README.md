# Building the Underwater Enhancer GUI Application

## Quick Build

### Local Build
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

### 1. Compile GPR Tools (REQUIRED)
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

## Troubleshooting

### Large Executable Size
The executable is ~166MB due to bundled PyTorch and dependencies. This is normal.

### Build Failures Due to GPR Tools
If the build fails with "GPR tools binary not found":
1. Ensure CMake is installed and in PATH
2. For Windows: Ensure Visual Studio C++ build tools are installed
3. Check compilation logs for specific errors
4. Manually run the compile script to see detailed errors:
   - Windows: `build_scripts\compile_gpr_tools.bat`
   - Unix/macOS: `./build_scripts/compile_gpr_tools.sh`

### Build Failures
1. Check Python version (3.8+ required)
2. Ensure all dependencies installed: `pip install -r requirements_gui.txt`
3. Clean build directory: `rm -rf build dist`
4. Try again with console mode: Edit `pyinstaller.spec`, set `console=True`

## CI/CD with GitHub Actions

The `.github/workflows/build-gui.yml` workflow automatically:
1. Builds for Windows, macOS, and Linux
2. **Compiles gpr_tools from source (mandatory - CI fails if compilation fails)**
3. Validates the gpr_tools binary (size check)
4. Bundles gpr_tools into the executable
5. Creates platform-specific archives
6. Uploads artifacts to GitHub
7. Creates releases when tagged

To trigger a release:
```bash
git tag v1.0.0
git push origin v1.0.0
```