# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Git Commit Policy
**NEVER automatically add or commit changes.** Only stage and commit files when explicitly requested by the user. This applies to all situations - no exceptions.

## Project Overview

Underwater image enhancement ML pipeline that automates manual GoPro RAW editing for Seattle Aquarium ROV surveys. Replaces time-intensive manual Adobe Lightroom editing with trained U-Net autoencoder.

## Common Development Commands

### Preprocessing GPR Files
```bash
# Fast preprocessing (recommended) - uses parallel processing
python preprocess_images_fast.py /path/to/gpr/files --output-dir processed

# Standard preprocessing
python preprocess_images.py /path/to/gpr/files --output-dir processed
```

### Dataset Preparation

#### From Hugging Face (Set-Based Structure)

**Quick Start - All-in-One Script (Recommended for VMs):**
```bash
# One command: download, prepare, crop, and train
huggingface-cli login  # One-time setup
python setup_and_train.py

# Custom parameters
python setup_and_train.py --batch-size 4 --epochs 100 --skip-output-crop

# Start fresh (cleanup and re-train)
python cleanup_training.py --force && python setup_and_train.py --skip-download
```

**Manual Step-by-Step:**
```bash
# Authenticate
huggingface-cli login

# Download
python download_dataset.py --output dataset_raw

# Prepare
python prepare_huggingface_dataset.py dataset_raw --output training_dataset

# Crop (REQUIRED - images are different sizes)
python crop_tiff.py training_dataset/input --output-dir training_dataset/input_cropped --width 4606 --height 4030

# Train
python train.py --input-dir training_dataset/input_cropped --target-dir training_dataset/target
```

#### From Local Files
```bash
# Fast dataset preparation (recommended)
python prepare_dataset_fast.py ./processed/cropped/ ./training_data/human_output_jpeg/ --output dataset

# Standard dataset preparation
python prepare_dataset.py ./processed/cropped/ ./training_data/human_output_jpeg/ --output dataset

# Create subset for testing
python create_subset.py --input-dir dataset --output-dir dataset_subset --num-samples 100
```

### Training
```bash
# Local training with U-Net (auto-detects GPU/CPU)
python train.py

# Resume training from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pth

# Monitor training progress
tensorboard --logdir logs
```

### Inference
```bash
# Process single image
python inference.py input.jpg --checkpoint checkpoints/best_model.pth

# Process directory with comparison
python inference.py /path/to/images --checkpoint checkpoints/best_model.pth --compare

# Full resolution processing (uses tiled processing for >2048px)
python inference.py input.jpg --checkpoint checkpoints/best_model.pth --full-size
```

### Post-Processing
```bash
# Apply denoising to enhanced images
python denoise_tiff.py enhanced_images/ --output final_images/ --method bilateral

# Compare denoising methods
python compare_denoise_methods.py input.tiff --output comparison.png
```

### Testing
```bash
# Test preprocessing pipeline
python test_preprocessing.py

# No formal test suite yet - evaluate visually using inference with --compare flag
```

### Cleanup & Utilities
```bash
# Preview what will be removed (safe)
python cleanup_training.py --dry-run

# Clean up all intermediate files (keeps raw dataset)
python cleanup_training.py

# Complete cleanup including raw dataset
python cleanup_training.py --remove-raw-dataset --force

# Keep specific artifacts
python cleanup_training.py --keep-checkpoints --keep-outputs

# See full cleanup guide
cat CLEANUP_GUIDE.md
```

## High-Level Architecture

### Model Architecture (Standard U-Net)
Located in `src/models/unet_autoencoder.py`:
- **Encoder**: 5 levels (64→128→256→512→1024 channels)
- **Decoder**: Symmetric upsampling with skip connections
- **~31M parameters** optimized for underwater enhancement
- Combined L1+MSE loss (80%/20% split) for detail preservation and color consistency

### Data Pipeline
1. **GPR Processing** (`preprocess_images.py`):
   - Uses gpr_tools for GPR→DNG conversion
   - Center crops to 4606×4030
   - Saves as TIFF for lossless processing

2. **Dataset Organization** (`prepare_dataset.py`):
   - Pairs raw/enhanced images
   - Creates train/val splits (80/20)
   - Handles file naming consistency

3. **Training Loop** (`train.py`):
   - Loads config from `config.yaml`
   - Implements early stopping, learning rate scheduling
   - Saves best model based on validation loss
   - Generates validation comparisons every 5 epochs

4. **Inference** (`inference.py`):
   - Handles arbitrary input sizes via tiled processing
   - Supports batch processing of directories
   - Creates side-by-side comparisons

### Key Design Decisions
- **PyTorch over TensorFlow**: Better community support for vision tasks
- **U-Net over plain autoencoder**: Skip connections preserve fine details
- **Tiled processing**: Enables full-resolution inference without memory limits
- **Combined loss**: Balances sharpness (L1) with smooth gradients (MSE)

## Dataset Structure

### Hugging Face Dataset Structure (New Format)
```
dataset_raw/           # Downloaded from HuggingFace
├── set01/
│   ├── input/        # Raw images for set 1
│   └── output/       # Enhanced targets for set 1
├── set02/
│   ├── input/
│   └── output/
└── ...

training_dataset/     # Prepared for training (after running prepare_hf_set_dataset.py)
├── input/           # Combined inputs from all sets
├── target/          # Combined targets from all sets
└── split.txt        # Train/val split indices
```

### Legacy Local Dataset Structure
```
dataset/
├── input_GPR/          # Raw GPR input images (1000 samples)
└── human_output_JPEG/  # Manually edited targets

photos/                 # Full dataset (3414 pairs)
├── input_GPR/
└── human_output_JPEG/
```

## Configuration (config.yaml)
Key settings:
- `batch_size`: 16 (reduce for memory issues)
- `image_size`: [256, 256] (training patches)
- `learning_rate`: 0.0001
- `epochs`: 50
- Loss weights: L1=0.8, MSE=0.2

## External Dependencies
- **GPR Tools**: Install via `./build_scripts/compile_gpr_tools.sh` or build from https://github.com/keenanjohnson/gpr_tools (fork with MSVC fixes)
- **Python 3.8+** with PyTorch, OpenCV, scikit-image
- **CUDA GPU** recommended (auto-fallback to CPU)

## Current Status (from TODO.md)
- ✅ Dev environment setup
- ✅ GPR preprocessing automation  
- ✅ ML framework selection (PyTorch)
- ✅ Model architecture (U-Net)
- ✅ Model implementation
- ✅ Initial training
- ⏳ Performance evaluation
- ⏳ Pipeline integration
- 🆕 GUI Application development (in progress)

## GUI Application Development

**Design Document**: See `DESIGN_GUI.md` for complete GUI application design and implementation plan.

### GUI Overview
- **Framework**: NiceGUI with native desktop mode
- **Packaging**: PyInstaller for standalone executable distribution
- **Target Users**: Marine biologists at Seattle Aquarium
- **Key Features**: Single/batch processing, native file dialogs, real-time preview

### GUI Development Status
- ✅ Design document completed
- ✅ Technology selection (NiceGUI + PyInstaller)
- ⏳ Phase 1: Core GUI implementation
- ⏳ Phase 2: Enhanced features
- ⏳ Phase 3: GPR support integration
- ⏳ Phase 4: Polish & packaging

### Running GUI Application (once implemented)
```bash
# Development mode
python app.py

# Build standalone executable
pyinstaller pyinstaller.spec

# Run packaged application
./dist/UnderwaterEnhancer  # Linux/macOS
./dist/UnderwaterEnhancer.exe  # Windows
```

## Resources
- Discussion: https://github.com/Seattle-Aquarium/CCR_development/issues/29
- Sample data: https://github.com/Seattle-Aquarium/CCR_development/tree/rmt_edits/files/ML_image_processing
- Google Colab notebook: `train_underwater_enhancer_colab.ipynb`
- GUI Design Document: `DESIGN_GUI.md`