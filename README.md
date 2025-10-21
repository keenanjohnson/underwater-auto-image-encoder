# Underwater Image Enhancement with ML

An automated machine learning pipeline that replaces manual image editing for underwater GoPro images captured during ROV surveys. The system converts RAW GPR files to enhanced JPEGs that match manual Adobe Lightroom editing quality.

![Underwater Image Enhancer GUI](gui_screenshot.png)

## 🖥️ Desktop Application (GUI)

**For most users: A user-friendly desktop application is available that requires no programming knowledge.**

### Download and Run
```bash
# Windows
Download UnderwaterEnhancer.exe from releases

# macOS  
Download UnderwaterEnhancer.app from releases

# Linux
Download UnderwaterEnhancer from releases
```

### GUI Models

You can use models trained from this repo to process images.

For example, two different models are provided in this google drive folder:
https://drive.google.com/drive/u/0/folders/1Vdctr52LTxoS6eecFiGS5LROZYSqJ3vl

Models are in the .pth format and can be loaded into the GUI.

#### Windows Security Warning
When running the Windows executable for the first time, you may see a security warning from Windows Defender SmartScreen. This occurs because the application is not digitally signed. To run the application:

1. Click **"More info"** on the Windows Defender SmartScreen warning
2. Click **"Run anyway"** at the bottom of the dialog

The application is safe to use. This warning appears for all unsigned executables and will decrease over time as more users run the application.

### GUI Features
- **Drag & Drop Interface** - Simply drag images into the application
- **Batch Processing** - Process hundreds of images at once
- **Real-time Preview** - See enhanced results instantly
- **Multiple Format Support** - GPR, JPEG, PNG, TIFF input formats
- **Progress Tracking** - Visual progress bars for batch operations
- **Dark/Light Themes** - Comfortable viewing in any environment
- **No Installation Required** - Standalone executable, just download and run

### Running from Source
```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Run the GUI application
python app.py
```

### Building the Executable
```bash
# Build standalone executable for your platform
python build_scripts/build_app.py

# Find the executable in:
# Windows: dist/UnderwaterEnhancer.exe
# macOS:   dist/UnderwaterEnhancer.app
# Linux:   dist/UnderwaterEnhancer
```

For detailed GUI documentation, see [GUI_README.md](GUI_README.md).

## 🚀 Quick Start (Command Line)

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

### Download Pre-prepared Dataset from Hugging Face

If you're working on a remote VM or want to use a pre-prepared dataset, you can download it directly from Hugging Face:

```bash
# Install huggingface_hub (included in requirements.txt)
pip install huggingface_hub

# Download the dataset (downloads to ./dataset by default)
python download_dataset.py

# Download to a custom directory
python download_dataset.py --output my_dataset

# Download a different dataset
python download_dataset.py --repo-id username/dataset-name
```

The script downloads the dataset with the correct structure:
```
dataset/
├── input/   # Raw/input images
└── target/  # Enhanced/target images
```

**For private datasets**, authenticate first:
```bash
huggingface-cli login
python download_dataset.py --repo-id username/private-dataset
```

After downloading, you can skip directly to training (step 3 below).

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

```bash
# Train with standard U-Net (auto-detects GPU/CPU)
python train.py

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pth

# Monitor progress
tensorboard --logdir logs
```

For detailed training options and Google Colab setup, see [TRAINING.md](TRAINING.md).

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


## 🏗️ Architecture

### U-Net Model
- Standard U-Net with skip connections (~31M parameters)
- Feature progression: 64 → 128 → 256 → 512 → 1024 channels
- Combined L1+MSE loss (80%/20%) for detail preservation and color consistency
- Tiled processing for high-resolution images (>2048px)
- Optional post-processing denoising with multiple algorithms

For detailed architecture information, see [CLAUDE.md](CLAUDE.md).

## 📊 Training

The model uses a standard U-Net architecture trained on paired raw/enhanced underwater images.

**Quick start:**
```bash
python train.py
```

**Key features:**
- Auto-detects GPU/CPU
- Early stopping and learning rate scheduling
- Validation comparisons every 5 epochs
- Google Colab notebook available for free GPU training

For complete training documentation, command-line options, and hardware requirements, see [TRAINING.md](TRAINING.md).

## 📁 Project Structure

```
auto-image-encoder/
├── dataset/                        # Training dataset (1000 image pairs)
│   ├── input_GPR/                  # Raw GPR input images
│   └── human_output_JPEG/          # Manually edited target images
├── photos/                         # Full dataset (3414 image pairs)
│   ├── input_GPR/                  # All GPR input images
│   └── human_output_JPEG/          # All manually edited images
├── train.py                        # Local training script (Standard U-Net)
├── train_underwater_enhancer_colab.ipynb  # Google Colab notebook
├── inference.py                    # Inference script with tiled processing
├── denoise_tiff.py                # Post-processing denoising script
├── create_subset.py                # Dataset subset creation script
├── config.yaml                     # Training configuration
└── requirements.txt                # Python dependencies
```

## ⚙️ Configuration

Edit `config.yaml` to customize training parameters (batch size, learning rate, loss weights, etc.).

Key settings:
- `batch_size`: 16 (reduce if out of memory)
- `image_size`: [256, 256] (training patch size)
- `learning_rate`: 0.0001
- Loss weights: L1=0.8, MSE=0.2

See [TRAINING.md](TRAINING.md) for complete configuration options and troubleshooting.

## 🎯 Results

The trained model reproduces manual Lightroom adjustments including denoising, white balance correction, tone adjustments, and contrast enhancement for underwater images.

## 📚 References

- [GoPro GPR Tools](https://github.com/keenanjohnson/gpr_tools) (fork with MSVC fixes)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Underwater Enhancement Research](https://github.com/Seattle-Aquarium/CCR_development/issues/29)

## 🤝 Contributing

This project is part of the Seattle Aquarium's ROV survey enhancement pipeline. For questions or contributions, refer to the main [CCR development repository](https://github.com/Seattle-Aquarium/CCR_development).

