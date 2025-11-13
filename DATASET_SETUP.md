# Dataset Setup Guide

This guide explains how to download and prepare the underwater image dataset from Hugging Face for training.

## New Dataset Structure (2024+)

The Seattle Aquarium benthic imagery dataset is now organized into "sets" on Hugging Face. Each set contains matched input/output pairs:

```
dataset/
├── set01/
│   ├── input/    # Raw underwater images
│   └── output/   # Manually enhanced versions
├── set02/
│   ├── input/
│   └── output/
└── ...
```

## Quick Start

### 1. Download Dataset from Hugging Face

```bash
# Download the entire dataset
python download_dataset.py --output dataset_raw

# Or download to a custom location
python download_dataset.py --repo-id Seattle-Aquarium/Seattle_Aquarium_benthic_imagery --output my_dataset
```

The download script will automatically detect the dataset structure and provide next steps.

### 2. Prepare Dataset for Training

After downloading, organize the dataset for training using the `prepare_huggingface_dataset.py` script:

```bash
# Process all sets in the downloaded dataset
python prepare_huggingface_dataset.py dataset_raw --output training_dataset

# This will:
# - Find all set01/, set02/, etc. directories
# - Match input/output pairs by filename
# - Create organized input/ and target/ directories
# - Generate train/validation split (80/20 by default)
```

### 3. Start Training

```bash
python train.py \
  --input-dir training_dataset/input \
  --target-dir training_dataset/target \
  --image-size 1024 \
  --batch-size 8 \
  --epochs 50
```

## Advanced Usage

### Process Specific Sets Only

If you only want to use certain sets for training:

```bash
# Use only set01 and set02
python prepare_huggingface_dataset.py dataset_raw/set01 dataset_raw/set02 --output training_dataset
```

### Save Disk Space with Symlinks

Instead of copying files, you can create symlinks to save disk space:

```bash
python prepare_huggingface_dataset.py dataset_raw --output training_dataset --symlink
```

**Note**: Symlinks won't work if you move or delete the original `dataset_raw` directory.

### Custom Train/Validation Split

Change the split ratio (default is 80/20):

```bash
# 90% training, 10% validation
python prepare_huggingface_dataset.py dataset_raw --output training_dataset --split-ratio 0.9
```

### Download Specific Datasets

To download a different dataset:

```bash
python download_dataset.py \
  --repo-id username/dataset-name \
  --output my_dataset
```

### Private Datasets

For private Hugging Face datasets, login first:

```bash
# Login once
huggingface-cli login

# Then download
python download_dataset.py --repo-id username/private-dataset
```

## Verifying Your Dataset

After preparation, you should see:

```
training_dataset/
├── input/          # img_00000.tif, img_00001.tif, ...
├── target/         # img_00000.jpg, img_00001.jpg, ...
└── split.txt       # Train/validation indices
```

Check the logs to verify:
- Number of pairs found in each set
- Total pairs prepared
- Train/validation split counts

## Troubleshooting

### No pairs found

**Problem**: Script reports "No matching pairs found!"

**Solution**:
- Check that input and output directories exist in each set
- Verify that filenames match between input/ and output/ (same base name, different extensions)
- Try running with a single set first to debug

### Mismatched filenames

**Problem**: Script warns about unmatched files

**Solution**: The script matches files by their stem (filename without extension). Ensure input and output files have the same base name:
- `image_001.tif` (input) matches `image_001.jpg` (output) ✓
- `img001.tif` (input) does NOT match `image001.jpg` (output) ✗

### Memory issues during training

**Problem**: Out of memory errors

**Solution**:
- Reduce `--batch-size` (try 4 or 2)
- Reduce `--image-size` (try 512 instead of 1024)
- Use a subset of the data for testing first

## Legacy Dataset Format

The old dataset format (all files in one directory with different extensions) is no longer supported. Please use the new set-based structure from Hugging Face.

## Dataset Information

- **Source**: Seattle Aquarium benthic ROV surveys
- **Format**: Input (TIFF/TIF), Output (JPEG/JPG)
- **Purpose**: Training underwater image enhancement models
- **Size**: Varies by set, typically 100-1000 pairs per set

## Next Steps

After preparing your dataset:

1. **Verify data quality**: Visually inspect some input/target pairs
2. **Start training**: See [CLAUDE.md](CLAUDE.md) for training commands
3. **Monitor progress**: Use TensorBoard or check validation metrics
4. **Test inference**: Use trained model on new images with `inference.py`

## Additional Resources

- Main documentation: [CLAUDE.md](CLAUDE.md)
- Dataset repository: https://huggingface.co/datasets/Seattle-Aquarium/Seattle_Aquarium_benthic_imagery
- Training script help: `python train.py --help`
