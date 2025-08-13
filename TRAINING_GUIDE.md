# Training Guide for Underwater Image Enhancement

## Quick Start

### 1. Prepare Your Dataset
```bash
# Organize your raw and enhanced image pairs
python prepare_dataset.py /path/to/raw/images /path/to/enhanced/images --output dataset

# This will create:
# dataset/input/    - Raw images (renamed consistently)
# dataset/target/   - Enhanced images (matching names)
# dataset/split.txt - Train/validation split
```

### 2. Start Training
```bash
# Train with default configuration
python train.py --config config.yaml

# Resume training from checkpoint
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. Monitor Training
```bash
# View tensorboard logs
tensorboard --logdir logs

# Check training progress
tail -f logs/training.log
```

### 4. Run Inference
```bash
# Process single image
python inference.py input.jpg --checkpoint checkpoints/best_model.pth

# Process directory
python inference.py /path/to/images --checkpoint checkpoints/best_model.pth --output enhanced_images

# Create comparison images
python inference.py input.jpg --checkpoint checkpoints/best_model.pth --compare
```

## Configuration Options

### Model Selection
Edit `config.yaml` to choose model architecture:
```yaml
model:
  type: "LightweightUNet"  # Options: LightweightUNet, UNetAutoencoder, AttentionUNet, WaterNet
  base_features: 32        # More features = larger model, better quality
```

### Training Parameters
```yaml
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 4           # Reduce if GPU memory issues
  optimizer: "adam"       # Options: adam, adamw, sgd
```

### Data Settings
```yaml
data:
  image_size: [512, 512]  # Training resolution (full res: [4606, 4030])
  train_split: 0.8        # 80% training, 20% validation
```

## Memory Management

### For High-Resolution Training (4606x4030):
- Set `batch_size: 1`
- Enable `mixed_precision: true`
- Use `LightweightUNet` model
- Consider patch-based training for very large images

### For GPU Memory Issues:
- Reduce `batch_size`
- Reduce `base_features`
- Reduce `image_size`
- Enable mixed precision

## Model Architectures

### LightweightUNet (Recommended for testing)
- **Parameters**: ~2M
- **Memory**: Low
- **Training speed**: Fast
- **Quality**: Good for initial experiments

### UNetAutoencoder (Production)
- **Parameters**: ~30M
- **Memory**: High
- **Training speed**: Slower
- **Quality**: Higher quality results

### AttentionUNet (Advanced)
- **Parameters**: ~10M
- **Memory**: Medium
- **Training speed**: Medium
- **Quality**: Better feature extraction with attention

### WaterNet (Specialized)
- **Parameters**: ~15M
- **Memory**: Medium
- **Training speed**: Medium
- **Quality**: Optimized for underwater characteristics

## Loss Function Components

The combined loss includes:
- **Pixel Loss**: Direct pixel comparison (L1)
- **Perceptual Loss**: VGG feature matching
- **Color Loss**: LAB color space consistency
- **SSIM Loss**: Structural similarity

Adjust weights in config.yaml:
```yaml
training:
  loss:
    lambda_pixel: 1.0      # Pixel accuracy
    lambda_perceptual: 0.1 # Natural appearance
    lambda_color: 0.5      # Color correction (important for underwater)
    lambda_ssim: 0.5       # Detail preservation
```

## Training Tips

### Progressive Training Strategy:
1. **Start small**: Train on 512x512 images first
2. **Fine-tune**: Use best checkpoint for higher resolution
3. **Full resolution**: Final training on 4606x4030 if memory allows

### Data Augmentation:
- Random horizontal/vertical flips are enabled
- NO color augmentation (interferes with color learning)
- Consider random crops for memory efficiency

### Monitoring:
- Validation loss should decrease steadily
- Check tensorboard for image comparisons every 5 epochs
- Early stopping prevents overfitting (patience=20 epochs)

## Troubleshooting

### CUDA Out of Memory:
```bash
# Reduce batch size
batch_size: 1

# Enable mixed precision
mixed_precision: true

# Use smaller model
model:
  type: "LightweightUNet"
  base_features: 16
```

### Poor Color Correction:
```bash
# Increase color loss weight
lambda_color: 1.0

# Use WaterNet model
model:
  type: "WaterNet"
```

### Training Too Slow:
```bash
# Use LightweightUNet
model:
  type: "LightweightUNet"

# Reduce image size
image_size: [256, 256]

# Increase batch size (if memory allows)
batch_size: 8
```

## Example Workflow

```bash
# 1. Preprocess GPR files
python preprocess_images.py /path/to/gpr/files

# 2. Prepare dataset (assuming you have enhanced pairs)
python prepare_dataset.py processed/cropped /path/to/enhanced/images

# 3. Start training
python train.py

# 4. Monitor progress
tensorboard --logdir logs

# 5. Test best model
python inference.py test_image.jpg --checkpoint checkpoints/best_model.pth --compare

# 6. Process full dataset
python inference.py /path/to/test/images --checkpoint checkpoints/best_model.pth
```