# Underwater Image Enhancer GUI

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_gui.txt
```

### 2. Run the GUI Application
```bash
python app.py
```

### 3. Using the Application

1. **Select Model**: Click "Browse..." to select your trained model file (`.pth`)
2. **Select Input Folder**: Choose the folder containing your GPR/TIFF/JPEG images
3. **Select Output Folder**: Choose where to save enhanced images
4. **Choose Output Format**: Select TIFF or JPEG output format
5. **Start Processing**: Click "▶ Start Processing" to begin

The application will:
- Automatically detect and convert GPR files to 4606×4030 resolution
- Process all images at full resolution (no downscaling)
- Save enhanced images in your chosen format
- Display progress and logs in real-time

## Building Standalone Executable

### Automatic Build
```bash
python build_scripts/build_app.py
```

This will:
1. Check for/compile gpr_tools (optional - for GPR file support)
2. Install all dependencies
3. Create application icons
4. Build the executable using PyInstaller
5. Output to `dist/` directory (~166MB with PyTorch)

### Manual Build with GPR Support

1. **Compile gpr_tools** (for GPR file support):
```bash
chmod +x build_scripts/compile_gpr_tools.sh
./build_scripts/compile_gpr_tools.sh
```

2. **Build the executable**:
```bash
pyinstaller pyinstaller.spec --clean --noconfirm
```

### Output Locations
- **macOS**: `dist/UnderwaterEnhancer.app` (see [macOS Installation Guide](MACOS_APP_INSTALLATION.md))
- **Windows**: `dist/UnderwaterEnhancer.exe`
- **Linux**: `dist/UnderwaterEnhancer`

### macOS Security Note
Downloaded macOS apps require security approval. See [MACOS_APP_INSTALLATION.md](MACOS_APP_INSTALLATION.md) for instructions.

## Features

- ✅ Native GPR file support (requires bundled gpr_tools binary)
- ✅ Full resolution processing (4606×4030, no downscaling)
- ✅ Batch processing with progress tracking
- ✅ Dark/Light mode toggle
- ✅ TIFF and JPEG output formats
- ✅ Real-time processing logs
- ✅ Time estimates for batch processing
- ✅ Cancel processing at any time
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Automated CI/CD builds via GitHub Actions

## Troubleshooting

### Missing gpr_tools Binary
If GPR files cannot be processed:
1. The app will show "GPR support not available" in logs
2. The app will still work with TIFF/JPEG files
3. To add GPR support, compile gpr_tools using the provided script
4. Note: GPR support requires the bundled binary - no system PATH fallback

### PyTorch Installation
For CPU-only PyTorch (smaller download):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### GUI Doesn't Launch
Ensure all dependencies are installed:
```bash
pip install customtkinter darkdetect PyYAML Pillow
```

## File Structure
```
app.py                  # Main entry point
src/
├── gui/
│   ├── main_window.py  # GUI interface
│   └── image_processor.py  # Processing logic
├── converters/
│   └── gpr_converter.py    # GPR conversion
binaries/               # Platform-specific gpr_tools
build_scripts/          # Build automation
```