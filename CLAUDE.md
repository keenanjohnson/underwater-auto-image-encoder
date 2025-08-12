# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an underwater image enhancement project that uses machine learning to automate the manual image editing process for GoPro RAW files (.gpr) captured by the Seattle Aquarium's ROV surveys.

## Key Technical Context

### Input/Output Pipeline
- **Input**: GoPro RAW files (.gpr) from HERO12 cameras (27.3MP, 3-second intervals)
- **Processing**: Convert GPR → standard RAW (4606 x 4030 cropped) → ML autoencoder → enhanced JPEG
- **Output**: Enhanced JPEG images matching manual editing quality

### Current Manual Workflow
The project aims to replace manual Adobe Lightroom Classic editing which includes:
- Denoise (setting: 55)
- Crop to 4606 x 4030
- White balance adjustment
- Tone adjustments (exposure, highlights, shadows, whites, blacks)
- Individual image tweaking for variation

### External Dependencies
- GoPro GPR tools: https://github.com/gopro/gpr (for RAW file conversion)
- Dataset includes manually edited GPR/JPEG pairs from Seattle Aquarium staff

## Development Focus Areas

### Machine Learning Framework Selection
When implementing the autoencoder, consider:
- Image-to-image translation capabilities
- Support for high-resolution images (4606 x 4030)
- Ability to preserve underwater image characteristics

### Data Pipeline Implementation
1. GPR file reading and conversion using gopro/gpr tools
2. Image preprocessing and normalization
3. Model inference
4. Output saving with appropriate metadata

### Training Considerations
- Dataset split: training/validation from existing manually edited pairs
- Loss functions appropriate for underwater image enhancement
- Handling variation in substrate type, algae cover, depth affecting brightness/color/clarity

## Resources
- Active discussion on approaches: https://github.com/Seattle-Aquarium/CCR_development/issues/29
- Example input/output files: https://github.com/Seattle-Aquarium/CCR_development/tree/rmt_edits/files/ML_image_processing

## Project TODO List

The project maintains a TODO.md file with the following tasks:
1. ✓ Set up VS Code dev container for isolated development environment
2. Create GPR to RAW conversion and cropping automation script
3. Select ML framework (TensorFlow or PyTorch)
4. Design autoencoder model architecture
5. Implement autoencoder model
6. Train autoencoder on preprocessed dataset
7. Evaluate model performance and adjust
8. Integrate trained model into processing pipeline

Note: Check TODO.md for the latest status of each task.