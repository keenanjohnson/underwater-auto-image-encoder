# Training the Underwater Image Enhancement Model

Train a U-Net model to enhance underwater images using PyTorch.

## Quick Start

### All-in-One Training Script (Recommended)

The easiest way to train - downloads dataset, prepares data, and trains the model:

```bash
# One-time setup: Authenticate with Hugging Face
huggingface-cli login

# Download, prepare, and train in one command
python training/setup_and_train.py
```

**What it does:**
1. Downloads dataset from Hugging Face
2. Prepares training/validation splits
3. Crops images to standard size
4. Trains the model with optimal settings

### Configuration

Customize training by creating a config file:

```bash
# Copy example config
cp setup_and_train_config_example.yaml setup_and_train_config.yaml

# Edit settings (batch size, epochs, image size, etc.)
nano setup_and_train_config.yaml

# Run with custom config
python training/setup_and_train.py
```

See [SETUP_CONFIG.md](../SETUP_CONFIG.md) for all configuration options.

## Manual Training Workflow

### Step 1: Prepare Dataset

**Option A: Download from Hugging Face**
```bash
# Authenticate
huggingface-cli login

# Download dataset
python dataset_prep/download_dataset.py --output dataset_raw

# Prepare for training
python dataset_prep/prepare_huggingface_dataset.py dataset_raw --output training_dataset

# Crop to standard size (REQUIRED)
python dataset_prep/crop_tiff.py training_dataset/input --output-dir training_dataset/input_cropped --width 4606 --height 4030
```

**Option B: From Local GPR Files**
```bash
# Preprocess GPR files
python preprocessing/preprocess_images.py /path/to/gpr/files --output-dir processed

# Prepare dataset
python dataset_prep/prepare_dataset.py ./processed/cropped/ ./training_data/human_output_jpeg/ --output dataset
```

### Step 2: Train Model

```bash
# Basic training
python training/train.py --input-dir dataset/input --target-dir dataset/target

# With custom settings
python training/train.py \
  --input-dir dataset/input \
  --target-dir dataset/target \
  --image-size 512 \
  --batch-size 8 \
  --epochs 50

# Resume from checkpoint
python training/train.py \
  --input-dir dataset/input \
  --target-dir dataset/target \
  --resume checkpoints/checkpoint_epoch_10.pth
```

### Step 3: Monitor Progress

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir logs

# Open browser to http://localhost:6006
```

## Training Options

### Key Parameters

- `--input-dir` (required): Directory with input images (.tiff/.tif)
- `--target-dir` (required): Directory with target/enhanced images
- `--image-size` (default: 512): Training patch size in pixels
  - `512`: Recommended for most GPUs (8GB+ VRAM)
  - `1024`: For high-end GPUs (24GB+ VRAM)
  - `0`: Full-size training (requires 32GB+ VRAM)
- `--batch-size` (default: 16): Images per batch
  - Reduce if running out of memory
  - Increase for faster training on large GPUs
- `--epochs` (default: 50): Number of training epochs
- `--lr` (default: 0.0001): Learning rate

### Output Locations

- `checkpoints/` - Model checkpoints saved during training
- `output/` - Best and final models
- `logs/` - TensorBoard training logs
- `validation_comparisons/` - Visual comparisons during training

## Hardware Requirements

### Minimum
- **GPU**: 8GB VRAM (e.g., RTX 3060)
- **RAM**: 16GB
- **Storage**: 20GB for dataset + models
- **Training time**: ~2-3 hours for 50 epochs

### Recommended
- **GPU**: 24GB VRAM (e.g., RTX 4090, A5000)
- **RAM**: 32GB
- **Storage**: 50GB
- **Training time**: ~45 minutes for 50 epochs

### Cloud/Free Options
- **Google Colab**: Free GPU training available
  - See [train_underwater_enhancer_colab.ipynb](../train_underwater_enhancer_colab.ipynb)
  - T4 GPU (16GB) included in free tier

## Cleanup

```bash
# Preview what will be removed
python training/cleanup_training.py --dry-run

# Clean up intermediate files (keeps raw dataset)
python training/cleanup_training.py

# Complete cleanup including dataset
python training/cleanup_training.py --remove-raw-dataset --force

# Keep specific artifacts
python training/cleanup_training.py --keep-checkpoints --keep-outputs
```

See [CLEANUP_GUIDE.md](../CLEANUP_GUIDE.md) for details.

## Detailed Documentation

- **[TRAINING.md](../TRAINING.md)** - Complete training guide with all command-line options
- **[SETUP_CONFIG.md](../SETUP_CONFIG.md)** - Configuration file reference
- **[CLAUDE.md](../CLAUDE.md)** - Developer guidance and architecture details

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size or image size
python training/train.py --input-dir dataset/input --target-dir dataset/target --batch-size 4 --image-size 256
```

### Dataset Not Found
```bash
# Verify dataset structure
ls dataset/input
ls dataset/target

# Should contain matching .tif/.tiff files
```

### CUDA Not Available
- Training will automatically fall back to CPU
- CPU training is ~10-20x slower
- Consider using Google Colab for free GPU access

## After Training

Once training completes, you'll have:

1. **Best model**: `output/best_model.pth` (lowest validation loss)
2. **Final model**: `output/final_model.pth` (last epoch)
3. **Checkpoints**: `checkpoints/checkpoint_epoch_*.pth`

### Use Your Model

```bash
# Run inference with your trained model
python inference/inference.py input_image.jpg --checkpoint output/best_model.pth

# Or load in the GUI application
python gui/app.py
# Then select your .pth file in the GUI
```

## Support

For questions or issues:
- Check [TRAINING.md](../TRAINING.md) for detailed documentation
- Review [SETUP_CONFIG.md](../SETUP_CONFIG.md) for configuration options
- See example workflow in `setup_and_train_config_example.yaml`
