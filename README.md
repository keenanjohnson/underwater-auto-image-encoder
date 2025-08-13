# Underwater Image Enhancement with ML

An automated machine learning pipeline that replaces manual image editing for underwater GoPro images captured during ROV surveys. The system converts RAW GPR files to enhanced JPEGs that match manual Adobe Lightroom editing quality.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- GoPro GPR tools (automatically installed in dev container)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd auto-image-encoder

# Install dependencies
pip install -r requirements.txt

# Verify GPR tools installation
python -c "from preprocess_images import GPRPreprocessor; print('✓ Setup complete')"
```

### Usage Pipeline

#### 1. Preprocess GPR Files
```bash
# Convert and crop GPR files to training format
python preprocess_images.py /path/to/gpr/files --output-dir processed

# This creates:
# processed/raw/      - DNG files from GPR conversion
# processed/cropped/  - Center-cropped 4606×4030 TIFF images
```

#### 2. Prepare Training Dataset
```bash
# Organize paired raw/enhanced images for training
python prepare_dataset.py processed/cropped /path/to/manually/edited/images --output dataset

# Creates organized dataset structure:
# dataset/input/   - Raw images (renamed consistently)
# dataset/target/  - Enhanced images (matching names)
# dataset/split.txt - Train/validation split
```

#### 3. Train the Model
```bash
# Start training with default configuration
python train.py --config config.yaml

# Monitor training progress
tensorboard --logdir logs

# Resume from checkpoint (if needed)
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

#### 4. Run Inference
```bash
# Process single image
python inference.py input.jpg --checkpoint checkpoints/best_model.pth

# Process entire directory
python inference.py /path/to/images --checkpoint checkpoints/best_model.pth --output enhanced_images

# Create side-by-side comparisons
python inference.py input.jpg --checkpoint checkpoints/best_model.pth --compare
```

## 🏗️ Architecture

### Model Options
- **LightweightUNet**: Fast training, ~2M parameters (recommended for testing)
- **UNetAutoencoder**: Full-featured, ~30M parameters (production quality)
- **AttentionUNet**: Enhanced with attention mechanisms (~10M parameters)
- **WaterNet**: Specialized for underwater characteristics (~15M parameters)

### Key Features
- **Skip connections** for detail preservation
- **Multi-scale processing** for global and local enhancement
- **Underwater-specific loss functions** including color correction
- **Attention mechanisms** for better feature extraction
- **Progressive training** from low to high resolution

## 📊 Training Details

### Loss Function Components
- **Pixel Loss (L1)**: Direct pixel-wise accuracy
- **Perceptual Loss**: VGG feature matching for natural appearance
- **Color Loss**: LAB color space consistency for underwater color correction
- **SSIM Loss**: Structural similarity for detail preservation

### Training Strategy
1. **Phase 1**: Train on 512×512 downsampled images for fast iteration
2. **Phase 2**: Fine-tune on patches from full resolution
3. **Phase 3**: Full resolution training (4606×4030) if memory permits

### Memory Optimization
- Mixed precision training (FP16)
- Gradient checkpointing
- Configurable batch sizes
- Progressive image scaling

## 📁 Project Structure

```
auto-image-encoder/
├── src/
│   ├── models/
│   │   ├── unet_autoencoder.py     # U-Net architectures
│   │   └── attention_unet.py       # Attention-enhanced models
│   ├── data/
│   │   └── dataset.py              # Dataset loaders
│   └── utils/
│       └── losses.py               # Custom loss functions
├── preprocess_images.py            # GPR → RAW conversion + cropping
├── prepare_dataset.py              # Dataset organization
├── train.py                        # Training script
├── inference.py                    # Inference script
├── config.yaml                     # Training configuration
├── TRAINING_GUIDE.md              # Detailed training guide
└── requirements.txt               # Python dependencies
```

## ⚙️ Configuration

Edit `config.yaml` to customize training:

```yaml
# Model selection
model:
  type: "LightweightUNet"  # LightweightUNet, UNetAutoencoder, AttentionUNet, WaterNet
  base_features: 32

# Training parameters
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 4
  
# Data settings
data:
  image_size: [512, 512]  # Training resolution
  train_split: 0.8

# Loss weights (adjust for underwater characteristics)
training:
  loss:
    lambda_pixel: 1.0      # Pixel accuracy
    lambda_perceptual: 0.1 # Natural appearance
    lambda_color: 0.5      # Color correction (important!)
    lambda_ssim: 0.5       # Detail preservation
```

## 🔧 Troubleshooting

### GPU Memory Issues
```yaml
# Reduce memory usage
data:
  batch_size: 1
  image_size: [256, 256]

hardware:
  mixed_precision: true

model:
  type: "LightweightUNet"
  base_features: 16
```

### Poor Color Correction
```yaml
# Emphasize color learning
training:
  loss:
    lambda_color: 1.0

model:
  type: "WaterNet"  # Specialized for underwater
```

## 📈 Monitoring Training

- **TensorBoard**: `tensorboard --logdir logs`
- **Validation images**: Updated every 5 epochs
- **Automatic checkpointing**: Best model saved based on validation loss
- **Early stopping**: Prevents overfitting (20 epoch patience)

## 🎯 Expected Results

The trained model should reproduce manual Lightroom adjustments:
- **Denoise**: Noise reduction (equivalent to setting 55)
- **White balance**: Automatic underwater color correction
- **Tone adjustments**: Exposure, highlights, shadows, whites, blacks
- **Contrast enhancement**: Improved clarity and detail

## 📚 References

- [GoPro GPR Tools](https://github.com/gopro/gpr)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Underwater Enhancement Research](https://github.com/Seattle-Aquarium/CCR_development/issues/29)

## 🤝 Contributing

This project is part of the Seattle Aquarium's ROV survey enhancement pipeline. For questions or contributions, refer to the main [CCR development repository](https://github.com/Seattle-Aquarium/CCR_development).

## 📄 License

[Add appropriate license information]