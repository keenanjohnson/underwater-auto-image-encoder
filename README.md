# Underwater Image Enhancement with ML

Automated machine learning pipeline that replaces manual image editing for underwater GoPro images captured during ROV surveys. Converts RAW GPR files to enhanced images matching manual Adobe Lightroom editing quality.

### For End Users: GUI Application

**No programming required** - Desktop application available for Windows, macOS, and Linux.

👉 **[See GUI Documentation](gui/README.md)**

**Quick steps:**
1. Download the application for your platform
2. Download a trained model (.pth file)
3. Drag & drop images to enhance

### For Developers: Training & Command Line

Train custom models or integrate into automated workflows.

👉 **[See Training Documentation](training/README.md)**

**Quick start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (downloads dataset automatically)
python training/setup_and_train.py

# Run inference on images
python inference/inference.py input.jpg --checkpoint output/best_model.pth
```

## Common Workflows

### Process Images with GUI
```bash
python gui/app.py
```
See [gui/README.md](gui/README.md) for details.

### Train a Model
```bash
# Quick: All-in-one script
python training/setup_and_train.py

# Manual: Step-by-step
python dataset_prep/download_dataset.py
python training/train.py --input-dir dataset/input --target-dir dataset/target
```
See [training/README.md](training/README.md) for details.

### Run Inference (Command Line)
```bash
# Single image
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth

# Batch process directory
python inference/inference.py /path/to/images --checkpoint checkpoints/best_model.pth --output enhanced/

# With comparison view
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth --compare
```

### Preprocess GPR Files
```bash
# Convert GPR to TIFF
python preprocessing/preprocess_images.py /path/to/gpr/files --output-dir processed
```

## Documentation

### User Documentation
- **[gui/README.md](gui/README.md)** - GUI application guide
- **[gui/docs/GUI_README.md](gui/docs/GUI_README.md)** - Detailed GUI user guide
- **[gui/docs/MACOS_APP_INSTALLATION.md](gui/docs/MACOS_APP_INSTALLATION.md)** - macOS installation

### Developer Documentation
- **[training/README.md](training/README.md)** - Training quick start
- **[TRAINING.md](TRAINING.md)** - Complete training guide
- **[SETUP_CONFIG.md](SETUP_CONFIG.md)** - Configuration reference
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines
- **[BUILD_README.md](BUILD_README.md)** - Build instructions
- **[CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)** - Cleanup utilities

## Pre-trained Models

Example trained models are available for download here:
https://huggingface.co/Seattle-Aquarium

## References

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Project Discussion](https://github.com/Seattle-Aquarium/CCR_development/issues/29)
- [Sample Data](https://github.com/Seattle-Aquarium/CCR_development/tree/rmt_edits/files/ML_image_processing)

## Contributing

This project is developed to support the Seattle Aquarium's ROV survey enhancement pipeline. For questions or contributions, refer to the main [CCR development repository](https://github.com/Seattle-Aquarium/CCR_development).

---

**Quick Links:**
- **GUI Users**: Start with [gui/README.md](gui/README.md)
- **Training Models**: Start with [training/README.md](training/README.md)
