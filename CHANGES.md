# Recent Changes: Configuration File Support

## Summary

Added YAML configuration file support to `setup_and_train.py` for easier version control and experiment management.

## Changes Made

### 1. New Files Created

- **`setup_and_train_config.yaml`** - Active configuration file (gitignored)
  - Contains all default settings for the training pipeline
  - Organized into sections: download, preparation, cropping, training, steps

- **`setup_and_train_config_example.yaml`** - Example configuration file (tracked in git)
  - Template for users to copy and customize
  - Includes detailed comments and common scenario examples

- **`SETUP_CONFIG.md`** - Configuration guide documentation
  - Explains how to use the config system
  - Provides examples for common scenarios
  - Best practices for version control

### 2. Modified Files

- **`setup_and_train.py`**
  - Added `import yaml` for YAML parsing
  - Added `load_config()` function to read YAML files
  - Changed all argparse defaults from hardcoded values to `None`
  - Added config loading logic with CLI override support
  - Refactored to use local variables instead of `args.*` throughout
  - Added `--config` argument to specify custom config file
  - Added `--split-ratio` argument for train/val split control

- **`requirements.txt`**
  - Added `PyYAML>=6.0.0` dependency

- **`.gitignore`**
  - Added `setup_and_train_config.yaml` (active config, not tracked)
  - Added `!setup_and_train_config_example.yaml` (example config, tracked)

## Configuration Hierarchy

Settings are applied in this order (later overrides earlier):

1. **Hardcoded defaults** in the script (fallback values)
2. **Config file values** from `setup_and_train_config.yaml`
3. **Command-line arguments** (highest priority)

## Usage Examples

### Basic usage (uses config file):
```bash
python training/setup_and_train.py
```

### Override specific settings:
```bash
python training/setup_and_train.py --batch-size 4 --epochs 100
```

### Use different config file:
```bash
python training/setup_and_train.py --config experiments/experiment_01.yaml
```

### Save experiment configurations:
```bash
# Create experiment config
cp setup_and_train_config.yaml experiments/high_batch.yaml
# Edit experiments/high_batch.yaml
# Run with that config
python training/setup_and_train.py --config experiments/high_batch.yaml
```

## Benefits

1. **Version Control**: Easy to track what settings produced which results
2. **Diff-friendly**: `git diff` shows exactly what changed between experiments
3. **Reproducibility**: Save exact configuration used for each training run
4. **Team Collaboration**: Share configurations easily with team members
5. **Less Typing**: No need to type long command lines repeatedly
6. **Clear Defaults**: All settings visible in one place

## Configuration Sections

### Download
- `repo_id`: HuggingFace repository to download from
- `dataset_dir`: Where to store raw dataset
- `hf_token`: Authentication token (optional)

### Preparation
- `working_dir`: Where to organize prepared dataset
- `sets`: Specific sets to process (null = all)
- `symlink`: Use symlinks instead of copying
- `split_ratio`: Train/validation split ratio

### Cropping
- `input`: Dimensions for input image cropping
- `output`: Dimensions for output image cropping (disabled by default)

### Training
- `image_size`: Training patch size
- `batch_size`: Batch size
- `epochs`: Number of epochs
- `learning_rate`: Learning rate
- `output_dir`: Where to save models
- `checkpoint_dir`: Where to save checkpoints
- `resume`: Path to checkpoint to resume from

### Steps
- `skip_download`: Skip download step
- `skip_prepare`: Skip preparation step
- `skip_crop`: Skip cropping step
- `prepare_only`: Don't start training

## Migration Guide

### Before (old way):
```bash
python training/setup_and_train.py \
  --dataset-dir dataset_raw \
  --working-dir training_dataset \
  --batch-size 8 \
  --epochs 50 \
  --image-size 1024 \
  --crop-width 4606 \
  --crop-height 4030
```

### After (new way):
1. Set up config once:
```yaml
# setup_and_train_config.yaml
download:
  dataset_dir: "dataset_raw"
preparation:
  working_dir: "training_dataset"
cropping:
  input:
    width: 4606
    height: 4030
training:
  batch_size: 8
  epochs: 50
  image_size: 1024
```

2. Run with simple command:
```bash
python training/setup_and_train.py
```

3. Override when needed:
```bash
python training/setup_and_train.py --batch-size 4
```

## Backward Compatibility

The script remains fully backward compatible:
- All command-line arguments still work
- If no config file exists, defaults are used
- Can run entirely from CLI without config file
- CLI arguments override config file values

## Testing

Tested successfully:
- Config file loading ✓
- Default value fallbacks ✓
- CLI override functionality ✓
- Help text generation ✓
- YAML parsing ✓

## Next Steps

Users should:
1. Copy `setup_and_train_config_example.yaml` to `setup_and_train_config.yaml`
2. Customize the config for their environment
3. Run `python training/setup_and_train.py`
4. Save experiment configs in `experiments/` directory
5. Version control experiment configs in git
