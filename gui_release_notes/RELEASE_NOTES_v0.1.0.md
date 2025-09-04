# Release Notes - v0.1.0-beta

## 🌊 Underwater Image Enhancer GUI - First Beta Release

### Overview
This is the first beta release of the Underwater Image Enhancer GUI application, designed for the Seattle Aquarium to automate the enhancement of underwater ROV survey images using ML models.

### ✨ Features

#### Core Functionality
- **ML-based Image Enhancement**: Process underwater images using trained U-Net models
- **GPR File Support**: Native support for GoPro RAW (GPR) files with automatic conversion
- **Batch Processing**: Process entire folders of images with progress tracking
- **Multiple Output Formats**: Export as TIFF (lossless) or JPEG (compressed)

#### Image Processing Pipeline
- **Automatic Center Cropping**: Images cropped to consistent 4606×4030 resolution matching training data
- **Tiled Processing**: Large images processed in tiles to avoid memory issues
- **Full Resolution Support**: Process images at original resolution without downscaling

#### User Interface
- **Native Desktop Application**: Built with CustomTkinter for modern, native look
- **Real-time Progress Tracking**: See processing status including tile-by-tile updates
- **Activity Log**: Detailed processing log with timestamps
- **Dark/Light Theme**: Toggle between dark and light modes
- **Drag & Drop Support**: Easy file and folder selection

### 📋 System Requirements
- **OS**: Windows 10+, macOS 11+, Linux (Ubuntu 20.04+)
- **Memory**: 8GB RAM minimum (16GB recommended for large images)
- **GPU**: CUDA-compatible GPU recommended but not required
- **Python**: 3.8+ (for development)

### 🚀 Installation

#### For Users (Standalone Application)
1. Download the appropriate release for your platform:
   - Windows: `UnderwaterEnhancer.exe`
   - macOS: `UnderwaterEnhancer.app`
   - Linux: `UnderwaterEnhancer`
2. Place your trained model file (`best_model.pth`) in an accessible location
3. Run the application

#### For Developers
```bash
# Clone repository
git clone https://github.com/Seattle-Aquarium/auto-image-encoder.git
cd auto-image-encoder

# Install dependencies
pip install -r requirements-gui.txt

# Run application
python app.py
```

### 🎯 Usage
1. Launch the application
2. Select your trained model file (.pth)
3. Choose input folder containing images
4. Select output folder for processed images
5. Choose output format (TIFF or JPEG)
6. Click "Process Images"

### 🔧 Technical Details
- **Model Architecture**: U-Net autoencoder with skip connections
- **Input Formats**: GPR, TIFF, JPEG, PNG
- **Processing Resolution**: 4606×4030 (center cropped from GoPro originals)
- **Tile Size**: 1024×1024 with 128px overlap for large images

### ⚠️ Known Limitations (Beta)
- GPR support requires bundled gpr_tools binary
- Large batch processing may require significant memory
- First image in batch may take longer (model loading)
- macOS may show security warnings for unsigned app

### 🐛 Bug Reports
Please report issues at: https://github.com/Seattle-Aquarium/auto-image-encoder/issues

### 📝 Change Log
- Initial beta release
- Core image enhancement functionality
- GPR file support with automatic conversion
- Batch processing with progress tracking
- CustomTkinter GUI with modern interface
- Automatic center cropping to training dimensions
- Tiled processing for memory efficiency
- Real-time progress updates including tile information

### 🙏 Acknowledgments
- Seattle Aquarium ROV team for requirements and testing
- GoPro for gpr_tools library
- PyTorch team for ML framework

---
**Note**: This is a beta release. Please backup your data before processing and report any issues encountered.