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
- Automatically detect and convert GPR files
- Process images through the ML model
- Save enhanced images in your chosen format
- Display progress and logs in real-time

## Building Standalone Executable

### Automatic Build
```bash
python build_scripts/build_app.py
```

This will:
1. Install all dependencies
2. Create application icons
3. Build the executable using PyInstaller
4. Output to `dist/` directory

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
- **macOS**: `dist/UnderwaterEnhancer.app`
- **Windows**: `dist/UnderwaterEnhancer.exe`
- **Linux**: `dist/UnderwaterEnhancer`

## Features

- ✅ Native GPR file support (with bundled gpr_tools)
- ✅ Batch processing with progress tracking
- ✅ Dark/Light mode toggle
- ✅ TIFF and JPEG output formats
- ✅ Real-time processing logs
- ✅ Time estimates for batch processing
- ✅ Cancel processing at any time
- ✅ Cross-platform (Windows, macOS, Linux)

## Troubleshooting

### Missing gpr_tools Binary
If you see warnings about missing gpr_tools:
1. The app will still work with TIFF/JPEG files
2. To add GPR support, compile gpr_tools using the provided script

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