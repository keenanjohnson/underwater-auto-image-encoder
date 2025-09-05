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
python ./prepare_dataset.py ./processed/cropped/ ./training_data/human_output_jpeg/ --output dataset

# Creates organized dataset structure:
# dataset/input/   - Raw images (renamed consistently)
# dataset/target/  - Enhanced images (matching names)
# dataset/split.txt - Train/validation split
```

#### 3. Train the Model

**Option A: Local Training (CPU/GPU)**
```bash
# Train with standard U-Net using config
python train_unet.py

# The script will:
# - Load config.yaml settings
# - Initialize Standard U-Net (~31M parameters)
# - Use Combined L1+MSE loss (80%/20%)
# - Auto-detect GPU/CPU
```

**Option B: Google Colab Training (Recommended)**
1. Upload `train_underwater_enhancer_colab.ipynb` to Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → GPU)
3. Run all cells - dataset loads from Google Drive after first run
4. Models save automatically to Google Drive

#### 4. Run Inference
```bash
# Process single image
python inference.py input.jpg --checkpoint checkpoints/best_model.pth

# Process entire directory
python inference.py /path/to/images --checkpoint checkpoints/best_model.pth --output enhanced_images

# Create side-by-side comparisons
python inference.py input.jpg --checkpoint checkpoints/best_model.pth --compare

# Process at full resolution (uses tiled processing for large images)
python inference.py input.jpg --checkpoint checkpoints/best_model.pth --full-size
```

**Note**: For large images (>2048px), the inference script automatically uses tiled processing to avoid memory issues. This processes the image in overlapping tiles and blends them seamlessly.

#### 5. Post-Process with Denoising (Optional)
```bash
# Apply denoising to TIFF outputs from inference
python denoise_tiff.py enhanced_images/ --output final_images/

# Use specific denoising algorithm (options: bilateral, nlmeans, gaussian, median, tv_chambolle, wavelet, bm3d_approximation)
python denoise_tiff.py enhanced_images/ --method bilateral --output final_images/

# Process single TIFF file with Non-Local Means (default)
python denoise_tiff.py input.tiff --output output.tiff --nlmeans-h 0.1

# Use bilateral filter (recommended for underwater images)
python denoise_tiff.py input.tiff --method bilateral --bilateral-sigma-color 75

# Preserve original value range
python denoise_tiff.py input.tiff --preserve-range
```

## 🖥️ GUI Application

A standalone desktop application is available for easy batch processing without command-line usage.

### Features
- Process GPR, TIFF, and JPEG images
- Batch processing with progress tracking
- Dark/Light mode interface
- Cross-platform (Windows, macOS, Linux)

### Quick Start
```bash
# Run the GUI
python app.py

# Or build standalone executable
python build_scripts/build_app.py
```

See [GUI_README.md](GUI_README.md) for detailed GUI documentation.

## 🏗️ Architecture

### Standard U-Net Model
- **Architecture**: Standard U-Net with skip connections
- **Parameters**: ~31M parameters for optimal quality
- **Feature progression**: 64 → 128 → 256 → 512 → 1024 channels
- **Purpose**: Designed specifically for underwater image enhancement

### Key Features
- **Skip connections** for detail preservation
- **Multi-scale processing** for global and local enhancement
- **Combined L1+MSE loss** for sharp details and color consistency
- **Optimized for underwater characteristics** (color correction, contrast enhancement)
- **Tiled processing** for high-resolution images (>2048px) to avoid memory issues
- **Post-processing denoising** with 7 different algorithms for final quality enhancement

## 📊 Training Details

### Loss Function
- **Combined L1+MSE Loss**: 
  - 80% L1 Loss for sharp detail preservation
  - 20% MSE Loss for color and brightness consistency
- **Optimized for underwater enhancement**: Balances detail and color correction

### Training Options

**Option 1: Google Colab (Recommended)**
- Use `train_underwater_enhancer_colab.ipynb` 
- Free GPU access, dataset caching to Google Drive
- Automatic model saving and resuming

**Option 2: Local Training**
- Use `train_unet.py` with your local GPU/CPU
- Requires manual dataset setup
- Good for development and testing

### Memory Optimization
- Mixed precision training (FP16)
- Gradient checkpointing
- Configurable batch sizes
- Progressive image scaling

## 📁 Project Structure

```
auto-image-encoder/
├── dataset/                        # Training dataset (1000 image pairs)
│   ├── input_GPR/                  # Raw GPR input images
│   └── human_output_JPEG/          # Manually edited target images
├── photos/                         # Full dataset (3414 image pairs)
│   ├── input_GPR/                  # All GPR input images
│   └── human_output_JPEG/          # All manually edited images
├── train_unet.py                   # Local training script (Standard U-Net)
├── train_underwater_enhancer_colab.ipynb  # Google Colab notebook
├── inference.py                    # Inference script with tiled processing
├── denoise_tiff.py                # Post-processing denoising script
├── create_subset.py                # Dataset subset creation script
├── config.yaml                     # Training configuration
└── requirements.txt                # Python dependencies
```

## ⚙️ Configuration

Edit `config.yaml` to customize training:

```yaml
# Model Configuration (Standard U-Net)
model:
  n_channels: 3  # Input channels (RGB)
  n_classes: 3   # Output channels (RGB)
  # Fixed architecture: 64 → 128 → 256 → 512 → 1024

# Training parameters
training:
  epochs: 50
  learning_rate: 0.0001
  lr_scheduler: "reduce_on_plateau"
  optimizer: "adam"
  
  # Loss configuration
  loss:
    l1_weight: 0.8    # L1 loss for sharp details
    mse_weight: 0.2   # MSE loss for color consistency
  
# Data settings
data:
  batch_size: 16
  image_size: [256, 256]  # Balanced for quality and GPU memory
  train_split: 0.8
  input_dir: "dataset/input_GPR"
  target_dir: "dataset/human_output_JPEG"
```

## 🔧 Troubleshooting

### GPU Memory Issues
```yaml
# Reduce memory usage
data:
  batch_size: 8      # Reduce from 16
  image_size: [128, 128]  # Reduce from 256

hardware:
  mixed_precision: false  # Keep false for stability
```

### Training Not Converging
```yaml
# Adjust learning rate and loss weights
training:
  learning_rate: 0.001    # Increase learning rate
  loss:
    l1_weight: 0.9        # Emphasize sharp details
    mse_weight: 0.1       # Reduce color smoothing
```

### For CPU Training
- Use Google Colab with GPU instead (much faster)
- If you must use CPU, reduce batch_size to 1 and image_size to [128, 128]

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

- [GoPro GPR Tools](https://github.com/keenanjohnson/gpr_tools) (fork with MSVC fixes)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Underwater Enhancement Research](https://github.com/Seattle-Aquarium/CCR_development/issues/29)

## 🤝 Contributing

This project is part of the Seattle Aquarium's ROV survey enhancement pipeline. For questions or contributions, refer to the main [CCR development repository](https://github.com/Seattle-Aquarium/CCR_development).

