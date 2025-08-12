# Autoencoder Architecture Design for Underwater Image Enhancement

## Overview
The architecture is designed specifically for enhancing underwater GoPro images, transforming RAW captures into professionally edited JPEGs that match manual Lightroom adjustments.

## Core Architecture: U-Net Based Autoencoder

### Why U-Net?
- **Skip connections**: Preserve fine details during encoding/decoding
- **Multi-scale feature extraction**: Captures both local details and global context
- **Proven effectiveness**: Widely successful in image-to-image translation tasks
- **Efficient gradient flow**: Skip connections prevent vanishing gradients

## Implemented Architectures

### 1. LightweightUNet (Recommended for Initial Training)
```
Input (3, H, W) → Enhanced Output (3, H, W)

Encoder Path:
├── Conv Block 1: 3 → 32 channels
├── Pool → Conv Block 2: 32 → 64 channels  
├── Pool → Conv Block 3: 64 → 128 channels
└── Pool → Bottleneck: 128 → 256 channels

Decoder Path (with skip connections):
├── UpConv + Skip → Conv Block: 256 → 128 channels
├── UpConv + Skip → Conv Block: 128 → 64 channels
├── UpConv + Skip → Conv Block: 64 → 32 channels
└── Final Conv: 32 → 3 channels + Sigmoid activation
```

**Key Features:**
- Base features: 32 (adjustable)
- Parameters: ~2M (efficient for training)
- Memory efficient for high-resolution images
- Suitable for initial experiments

### 2. Full UNetAutoencoder (For Production)
```
Deeper architecture with 5 encoding/decoding levels
Base features: 64 → 128 → 256 → 512 → 1024
More parameters (~30M) for capturing complex underwater characteristics
```

## Architecture Components

### 1. DoubleConv Blocks
- Two 3×3 convolutions with batch normalization
- ReLU activation between convolutions
- Extracts features while maintaining spatial dimensions

### 2. Encoder (Downsampling Path)
- MaxPooling for spatial reduction
- Progressively increases feature channels
- Captures hierarchical features

### 3. Bottleneck
- Deepest layer with maximum feature extraction
- Global context understanding
- Critical for color correction and dehazing

### 4. Decoder (Upsampling Path)
- Transpose convolutions for upsampling
- Concatenation with encoder features (skip connections)
- Progressive feature refinement

### 5. Output Layer
- 1×1 convolution to 3 channels (RGB)
- Sigmoid activation for [0,1] range output

## Loss Function Design

### Combined Underwater Enhancement Loss
```python
Total Loss = λ₁·L_pixel + λ₂·L_perceptual + λ₃·L_color + λ₄·L_ssim
```

**Components:**
1. **Pixel Loss (L1)**: Direct pixel-wise comparison
2. **Perceptual Loss**: VGG feature matching for natural appearance
3. **Color Loss**: LAB color space consistency for underwater color correction
4. **SSIM Loss**: Structural similarity for detail preservation

**Default Weights:**
- λ₁ = 1.0 (pixel accuracy)
- λ₂ = 0.1 (perceptual quality)
- λ₃ = 0.5 (color correction - important for underwater)
- λ₄ = 0.5 (structural details)

## Underwater-Specific Design Considerations

### 1. Color Correction Focus
- LAB color space loss component
- Addresses blue/green color cast
- Maintains natural color balance

### 2. Detail Enhancement
- Skip connections preserve fine details
- Important for marine life and substrate texture

### 3. Contrast Improvement
- Deep bottleneck for global adjustments
- Handles low contrast from water absorption

### 4. Adaptive to Depth Variations
- Multi-scale features capture varying conditions
- Handles different lighting/depth scenarios

## Input/Output Specifications

**Input:**
- Format: TIFF (from preprocessed GPR files)
- Dimensions: 4606 × 4030 pixels (cropped)
- Channels: 3 (RGB)
- Value range: [0, 1] normalized

**Output:**
- Format: Enhanced RGB image
- Dimensions: Same as input
- Channels: 3 (RGB)
- Value range: [0, 1] (can be scaled to [0, 255])

## Training Strategy

### Progressive Training Approach
1. **Phase 1**: Train on downsampled images (512×512) for faster iteration
2. **Phase 2**: Fine-tune on patches from full resolution
3. **Phase 3**: Full resolution training (if memory permits)

### Data Augmentation (Recommended)
- Random crops for memory efficiency
- Horizontal/vertical flips
- Slight rotation (±5°)
- NO color augmentation (would interfere with color correction learning)

## Memory Optimization

For full resolution (4606×4030) training:
- Use gradient checkpointing
- Mixed precision training (FP16)
- Patch-based training if needed
- Batch size = 1 for full resolution

## Evaluation Metrics

1. **PSNR**: Peak Signal-to-Noise Ratio
2. **SSIM**: Structural Similarity Index
3. **UIQM**: Underwater Image Quality Measure (if available)
4. **Color Difference**: ΔE in LAB space
5. **Perceptual Distance**: LPIPS metric

## Future Enhancements

1. **Attention Mechanisms**: Add spatial/channel attention
2. **Multi-Scale Discriminator**: For adversarial training
3. **Depth-Aware Processing**: Incorporate depth information if available
4. **Temporal Consistency**: For video sequences

## Implementation Status
✅ Core U-Net architectures implemented
✅ Custom loss functions designed
✅ Dataset loaders ready
✅ Configuration system in place
⏳ Ready for training implementation