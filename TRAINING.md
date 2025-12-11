# Training Script Documentation

This document explains how to use the standalone `training/train.py` script, which replaces the Jupyter notebook workflow.

## Overview

The `training/train.py` script provides a command-line interface for training the underwater image enhancement model with the same functionality as the Jupyter notebook, but with better flexibility and integration into automated workflows.

## Quick Start

### Basic Usage (1024×1024 patches)

```bash
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --image-size 1024 \
  --batch-size 16 \
  --epochs 50
```

### Full-Size Training (requires 32GB+ GPU)

```bash
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --image-size 0 \
  --batch-size 1 \
  --epochs 50
```

## Command-Line Arguments

### Data Arguments

- `--input-dir` (required): Directory containing input images (.tiff/.tif files)
- `--target-dir` (required): Directory containing target/ground truth images
- `--split-file` (optional): Path to train/validation split file. If not provided, looks for `split.txt` in parent of input-dir, or creates an 80/20 split

### Training Arguments

- `--image-size` (default: 1024): Training image size in pixels
  - Use `1024` for 1024×1024 patches (recommended)
  - Use `512` for 512×512 patches
  - Use `256` for 256×256 patches
  - Use `0` for full-size/no resizing (requires significant GPU memory)

- `--batch-size` (default: 16): Number of images per batch
  - 1024×1024: batch=16-24 on 24GB+ GPU
  - Full-size: batch=1 on 32GB GPU

- `--epochs` (default: 50): Number of training epochs

- `--lr` (default: 1e-4): Learning rate

- `--num-workers` (default: 4): Number of data loading worker processes

### Output Arguments

- `--output-dir` (default: 'output'): Directory for saving trained models
- `--checkpoint-dir` (default: 'checkpoints'): Directory for saving training checkpoints
- `--resume` (optional): Path to checkpoint to resume training from

### Loss Configuration

- `--l1-weight` (default: 0.8): Weight for L1 loss component
- `--mse-weight` (default: 0.2): Weight for MSE loss component

### Training Control

- `--early-stopping` (default: 15): Stop training if validation loss doesn't improve for N epochs (0 to disable)

## Example Workflows

### 1. Training on 1024×1024 Patches (Recommended)

```bash
# Create output directories
mkdir -p output checkpoints

# Train model
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --image-size 1024 \
  --batch-size 16 \
  --epochs 50 \
  --output-dir output \
  --checkpoint-dir checkpoints
```

**Expected Results:**
- Training time: ~10-15 min/epoch on RTX 4090
- Memory usage: ~19GB on 24GB GPU
- Output: `output/best_model.pth` and `output/final_model.pth`

### 2. Training with Custom Split

```bash
# If you have a custom train/val split file
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --split-file dataset/my_split.txt \
  --image-size 1024 \
  --batch-size 24
```

### 3. Resuming from Checkpoint

```bash
# Resume training from a previous checkpoint
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --resume checkpoints/latest_checkpoint.pth \
  --epochs 100
```

### 4. Full-Size Training (80GB GPU)

```bash
# Train on full-resolution images (4606×4030)
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --image-size 0 \
  --batch-size 1 \
  --epochs 50 \
  --output-dir output_fullsize
```

**Warning:** Requires 24GB+ GPU memory for batch size 1, or will run very slowly

### 5. Small Dataset / Quick Test

```bash
# Fast training for testing (256×256 patches)
python training/train.py \
  --input-dir dataset/input_GPR \
  --target-dir dataset/human_output_JPEG \
  --image-size 256 \
  --batch-size 64 \
  --epochs 10
```

## Hardware Acceleration

The training script automatically detects and uses the best available hardware acceleration:

1. **CUDA (NVIDIA GPUs)** - Preferred for maximum performance
2. **MPS (Apple Silicon)** - For M1/M2/M3 Macs with unified memory
3. **CPU** - Fallback (very slow, not recommended)

### Example Device Detection:

```
# NVIDIA GPU
Using device: cuda
GPU: NVIDIA RTX 4090
GPU Memory: 24.00 GB

# Apple Silicon
Using device: mps (Apple Silicon)
Note: MPS uses unified memory shared with system RAM

# CPU fallback
Using device: cpu
WARNING: No GPU acceleration available. Training will be slow.
```

## GPU Memory Requirements

| Image Size | Batch Size | Memory Required | Recommended GPU |
|------------|------------|-----------------|-----------------|
| 256×256 | 64 | ~4-5 GB | Any modern GPU, M1/M2/M3 |
| 512×512 | 32 | ~8-10 GB | 12GB+, M1 Pro/Max |
| 1024×1024 | 16 | ~19 GB | 24GB (RTX 4090/5090), M2 Ultra |
| 1024×1024 | 24 | ~29 GB | 32GB (RTX 5090) |
| Full (4606×4030) | 1 | ~21 GB | 24GB+, M2 Ultra (64GB+) |
| Full (4606×4030) | 2 | ~42 GB | 48GB+ |
| Full (4606×4030) | 4 | ~84 GB | 96GB (H100) |

**Note for Apple Silicon:** MPS uses unified memory, so memory requirements compete with system RAM. Ensure sufficient total memory (e.g., 64GB+ for larger training jobs).

## Output Files

The training script creates the following files:

### In `output/` directory:
- `best_model.pth`: Best model based on validation loss
  - Contains model weights and configuration
  - Ready to use with `inference.py` or GUI app
- `final_model.pth`: Model from the final epoch
  - Contains training history

### In `checkpoints/` directory:
- `latest_checkpoint.pth`: Most recent checkpoint (updated every epoch)
  - Can be used to resume training
- `checkpoint_epoch_5.pth`, `checkpoint_epoch_10.pth`, etc.: Periodic checkpoints every 5 epochs

## Model Configuration

The saved models include configuration that the inference script uses automatically:

```python
{
    'model_state_dict': ...,  # Trained weights
    'model_config': {
        'n_channels': 3,
        'n_classes': 3,
        'image_size': 1024,  # Training resolution
    },
    'training_history': {
        'train_losses': [...],
        'val_losses': [...],
        'train_psnrs': [...],
        'val_psnrs': [...],
    }
}
```

## Using Trained Models

### With Inference Script

```bash
# Process single image
python inference/inference.py input.jpg \
  --checkpoint output/best_model.pth \
  --output enhanced.tiff

# Process directory
python inference/inference.py input_dir/ \
  --checkpoint output/best_model.pth \
  --output output_dir/
```

### With GUI Application

1. Launch GUI: `python app.py`
2. Click "Select Model"
3. Choose `output/best_model.pth`
4. Load and process images

## Monitoring Training

The script provides real-time progress information:

```
2025-01-15 10:30:00 - INFO - Using device: cuda
2025-01-15 10:30:00 - INFO - GPU: NVIDIA RTX 5090
2025-01-15 10:30:00 - INFO - GPU Memory: 32.00 GB
2025-01-15 10:30:01 - INFO - Training resolution: 1024×1024
2025-01-15 10:30:01 - INFO - Batch size: 16
2025-01-15 10:30:02 - INFO - Training samples: 800
2025-01-15 10:30:02 - INFO - Validation samples: 200

Epoch 1/50
--------------------------------------------------------------------------------
Training: 100%|████████| 50/50 [01:23<00:00, Loss: 0.0234, PSNR: 28.45 dB]
Validation: 100%|██████| 13/13 [00:15<00:00, Loss: 0.0198, PSNR: 30.12 dB]
Train Loss: 0.0234, Train PSNR: 28.45 dB
Val Loss: 0.0198, Val PSNR: 30.12 dB
✓ Saved best model: output/best_model.pth
```

## Troubleshooting

### Out of Memory Errors

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size or image size:
```bash
# Reduce batch size
python training/train.py --batch-size 8 ...

# Or use smaller images
python training/train.py --image-size 512 ...
```

### Split File Format

The script accepts two formats:

**JSON format:**
```json
{
  "train": [0, 1, 2, 3, ...],
  "validation": [800, 801, 802, ...]
}
```

**Plain text format:**
```
Training indices:
0, 1, 2, 3, 4, ...

Validation indices:
800, 801, 802, ...
```

### No GPU Available

If training on CPU:
```
2025-01-15 10:30:00 - INFO - Using device: cpu
```

Training will be **very slow** on CPU. Consider:
- Using Google Colab with free T4 GPU
- Reducing image size to 256×256
- Using a smaller dataset for testing

## Comparison to Jupyter Notebook

| Feature | Jupyter Notebook | train.py Script |
|---------|------------------|-----------------|
| Ease of use | Interactive, visual | Command-line |
| Automation | Manual execution | Easy to automate |
| Reproducibility | Requires careful cell execution | Fully reproducible |
| Remote training | Difficult | Easy (SSH/cloud) |
| Hyperparameter tuning | Manual cell edits | Command-line args |
| Integration | Limited | CI/CD ready |
| Progress tracking | Cell outputs | Log files |

## Next Steps

After training:

1. **Evaluate the model:**
   ```bash
   python inference/inference.py test_image.tiff \
     --checkpoint output/best_model.pth \
     --compare
   ```

2. **Analyze noise:**
   - Compare enhanced vs original images
   - Measure high-frequency noise reduction
   - Decide if you need to train at higher resolution

3. **Deploy:**
   - Use model with GUI application
   - Batch process your entire dataset
   - Share model checkpoint with team

## Support

For issues or questions:
- Check `CLAUDE.md` for project overview
- Review training logs in console output
- Check GPU memory with `nvidia-smi`
- Verify dataset structure matches expected format
