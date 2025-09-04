# Windows CI Build Fix

## Issue
```
CMake Error: generator : MinGW Makefiles
Does not match the generator used previously: Visual Studio 16 2019
Either remove the CMakeCache.txt file and CMakeFiles directory or choose a different binary directory.
```

## Root Causes
1. CMakeCache.txt from previous build attempts wasn't cleaned
2. Visual Studio 2019 not available on GitHub Actions runners
3. Need to use appropriate build tools for CI environment

## Solutions Applied

### 1. Clean Build Directory
- Always remove `build` directory before configuring
- Ensures no cache conflicts between different generators

### 2. Setup MSVC for Windows CI
- Added `ilammy/msvc-dev-cmd@v1` action to setup MSVC environment
- Provides necessary compiler tools for Windows builds

### 3. Install Ninja Build System
- Ninja is faster and more reliable for CI builds
- Works well with MSVC compiler tools
- Simpler than Visual Studio generators

### 4. Detect CI Environment
- Use Ninja by default in CI (via `CI` environment variable)
- Fall back to other generators for local builds

### 5. Flexible Binary Location Detection
- Check multiple possible output locations:
  - `source\app\gpr_tools\Release\gpr_tools.exe` (Visual Studio)
  - `source\app\gpr_tools\gpr_tools.exe` (Ninja in subdirectory)
  - `gpr_tools.exe` (Ninja in root build directory)

## Updated Files

### `.github/workflows/build-gui.yml`
- Added MSVC setup action
- Install Ninja for Windows builds
- Check for existing CMake before installing

### `build_scripts/compile_gpr_tools.bat`
- Clean build directory before configuration
- Detect CI environment and use appropriate generator
- Try multiple binary locations when copying

## Benefits
- ✅ No more CMake cache conflicts
- ✅ Reliable Windows CI builds
- ✅ Faster builds with Ninja
- ✅ Works on both CI and local environments