# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Git Commit Policy
**NEVER automatically add or commit changes.** Only stage and commit files when explicitly requested by the user. This applies to all situations - no exceptions.

## Project Overview

Underwater image enhancement ML pipeline that automates manual GoPro RAW editing for Seattle Aquarium ROV surveys. Replaces time-intensive manual Adobe Lightroom editing with trained deep learning models.

## Environment Setup

```bash
python3.10 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt  # Full dev (training + inference)
pip install -r requirements_gui.txt  # GUI-only
```

## Common Commands

### Training (All-in-One)
```bash
huggingface-cli login  # One-time setup
python training/setup_and_train.py  # Downloads dataset, prepares, trains

# Resume from checkpoint
python training/train.py --resume checkpoints/checkpoint_epoch_10.pth

# Monitor progress
tensorboard --logdir logs
```

### Inference
```bash
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth
python inference/inference.py /path/to/images --checkpoint checkpoints/best_model.pth --compare
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth --full-size
```

### GPR Preprocessing
```bash
python preprocessing/preprocess_images_fast.py /path/to/gpr/files --output-dir processed
```

### GUI Development
```bash
python gui/app.py  # Development mode
pyinstaller gui/pyinstaller.spec --clean --noconfirm  # Build executable
./dist/UnderwaterEnhancer --smoke-test  # Test build
```

### Cleanup
```bash
python training/cleanup_training.py --dry-run  # Preview
python training/cleanup_training.py --force  # Execute
```

## High-Level Architecture

### Model Architectures

Two model options configured via `model:` in [setup_and_train_config.yaml](setup_and_train_config.yaml):

| Model | File | Params | Use Case |
|-------|------|--------|----------|
| U-Net | [src/models/unet_autoencoder.py](src/models/unet_autoencoder.py) | ~31M | Faster training, good baseline |
| U-shape Transformer | [src/models/ushape_transformer.py](src/models/ushape_transformer.py) | ~50M | Better quality, slower training |

**U-Net**: 5-level encoder (64→128→256→512→1024 channels), symmetric decoder with skip connections. Combined L1+MSE loss (80%/20%).

**U-shape Transformer**: CMSFFT (Cross-scale Multi-scale Fusion FFT) + SGFMT (Spatial-Gated Feed-forward Modulation Transformer) attention mechanisms.

### Data Pipeline

```
GPR Files → gpr_tools → DNG → TIFF (4606×4030) → Training patches (512×512)
```

1. **GPR Processing** ([preprocessing/](preprocessing/)): gpr_tools converts GPR→DNG, center crops to 4606×4030, saves as TIFF
2. **Dataset Prep** ([dataset_prep/](dataset_prep/)): Pairs raw/enhanced images, creates 80/20 train/val splits
3. **Training** ([training/train.py](training/train.py)): Early stopping, LR scheduling, saves best model by validation loss
4. **Inference** ([inference/inference.py](inference/inference.py)): Tiled processing for arbitrary sizes, batch support

### GUI Application

- **Framework**: NiceGUI with native desktop mode
- **Packaging**: PyInstaller (outputs ~166MB with bundled PyTorch)
- **Entry point**: [gui/app.py](gui/app.py)
- **Core logic**: [src/gui/image_processor.py](src/gui/image_processor.py), [src/gui/main_window.py](src/gui/main_window.py)
- **GPR support**: [src/converters/gpr_converter.py](src/converters/gpr_converter.py) + bundled `binaries/<platform>/gpr_tools`

Build docs: [BUILD_README.md](BUILD_README.md) | macOS security: [gui/docs/MACOS_APP_INSTALLATION.md](gui/docs/MACOS_APP_INSTALLATION.md)

### Key Configuration

Training config in [setup_and_train_config.yaml](setup_and_train_config.yaml):
- `model`: "unet" or "ushape_transformer"
- `batch_size`: 12-16 (reduce for memory issues)
- `image_size`: 512 (training patch size)
- `epochs`: 150

## External Dependencies

- **GPR Tools**: `./build_scripts/compile_gpr_tools.sh` compiles from https://github.com/keenanjohnson/gpr_tools
- **Hardware**: CUDA GPU recommended for training (~50x faster), CPU fine for inference
- **Memory**: 12GB+ GPU RAM for batch_size=12-16 with image_size=512

## Resources

- Pre-trained models: https://huggingface.co/Seattle-Aquarium
- Dataset: https://huggingface.co/datasets/Seattle-Aquarium/Seattle_Aquarium_benthic_imagery
- Parent project: https://github.com/Seattle-Aquarium/CCR_development