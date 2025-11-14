# GUI Application

This directory contains the graphical user interface (GUI) application for the Underwater Image Enhancer.

## Contents

- **[app.py](app.py)** - Main GUI application entry point
- **[pyinstaller.spec](pyinstaller.spec)** - PyInstaller configuration for building standalone executables
- **[runtime_hook.py](runtime_hook.py)** - PyInstaller runtime hook for module imports
- **[create_icon.py](create_icon.py)** - Utility to generate application icons
- **[test_gui_size.py](test_gui_size.py)** - GUI size/layout testing script
- **[requirements-gui.txt](requirements-gui.txt)** - GUI-specific Python dependencies
- **[gui_screenshot.png](gui_screenshot.png)** - Application screenshot for documentation
- **[docs/](docs/)** - GUI-specific documentation
  - [GUI_README.md](docs/GUI_README.md) - Detailed GUI user guide
  - [MACOS_APP_INSTALLATION.md](docs/MACOS_APP_INSTALLATION.md) - macOS installation instructions
- **[release_notes/](release_notes/)** - Version release notes

## Quick Start

### Development Mode
```bash
# From project root
python gui/app.py
```

### Building Standalone Executable
```bash
# From project root
pyinstaller gui/pyinstaller.spec
```

The built application will be in the `dist/` directory:
- **macOS**: `dist/UnderwaterEnhancer.app`
- **Windows**: `dist/UnderwaterEnhancer.exe`
- **Linux**: `dist/UnderwaterEnhancer`

## Documentation

For detailed information, see:
- [GUI User Guide](docs/GUI_README.md) - End-user documentation
- [BUILD_README.md](../BUILD_README.md) - Build and packaging instructions
- [CLAUDE.md](../CLAUDE.md) - Developer guidance

## Architecture

The GUI uses:
- **CustomTkinter** - Modern-looking GUI framework
- **PyInstaller** - For creating standalone executables
- **src/gui/** - Core GUI components (in parent src/ directory)
  - `main_window.py` - Main application window
  - `image_processor.py` - Image processing logic

## Platform-Specific Notes

### macOS
- Application includes proper .app bundle structure
- Icon in .icns format
- May require security approval on first run

### Windows
- Single .exe file with all dependencies
- Icon in .ico format
- Windows Defender may require approval

### Linux
- Single executable binary
- May require execute permissions: `chmod +x dist/UnderwaterEnhancer`
