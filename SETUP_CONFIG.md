# Setup and Train Configuration Guide

This guide explains how to use the configuration file system for `setup_and_train.py`.

## Overview

Instead of passing many command-line arguments every time, you can store all your settings in a YAML configuration file. This makes it easy to:
- Version control your training configurations
- Compare different experiment settings with git diff
- Share configurations with team members
- Switch between different training scenarios quickly

## Quick Start

1. **Copy the example config:**
   ```bash
   cp setup_and_train_config_example.yaml setup_and_train_config.yaml
   ```

2. **Edit the config file** with your settings

3. **Run the script:**
   ```bash
   python training/setup_and_train.py
   ```

That's it! The script will load settings from `setup_and_train_config.yaml` automatically.

## Configuration Structure

The config file is organized into sections:

### Download Section
```yaml
download:
  repo_id: "Seattle-Aquarium/Seattle_Aquarium_benthic_imagery"
  dataset_dir: "dataset_raw"
  # Note: hf_token is NOT stored in config for security
  # Use --hf-token CLI argument or HF_TOKEN env variable instead
```

**Security Note**: The `hf_token` is intentionally NOT configurable via the YAML file. Always use:
- CLI argument: `--hf-token your_token`
- Environment variable: `export HF_TOKEN=your_token`
- Or login once: `huggingface-cli login`

### Preparation Section
```yaml
preparation:
  working_dir: "training_dataset"
  sets: null  # null = all sets, or specify: ["set01", "set02"]
  symlink: false  # true = save disk space, false = copy files
  split_ratio: 0.8  # 80% train, 20% validation
```

### Cropping Section
```yaml
cropping:
  input:
    width: 4606
    height: 4030
  output:
    enabled: false  # Output cropping is skipped by default
    width: 4606
    height: 4030
```

### Training Section
```yaml
training:
  image_size: 1024  # Training patch size
  batch_size: 8
  epochs: 50
  learning_rate: 0.0001
  output_dir: "output"
  checkpoint_dir: "checkpoints"
  resume: null  # Path to checkpoint to resume from
```

### Steps Section
```yaml
steps:
  skip_download: false
  skip_prepare: false
  skip_crop: false
  prepare_only: false  # true = don't start training
```

## Command-Line Overrides

You can override any config value with command-line arguments:

```bash
# Use config file but override batch size
python training/setup_and_train.py --batch-size 4

# Use different config file
python training/setup_and_train.py --config my_experiment.yaml

# Override multiple settings
python training/setup_and_train.py --batch-size 4 --epochs 100 --skip-download
```

Command-line arguments always take precedence over config file values.

## Common Scenarios

### 1. Quick Test Run (Small Subset)
```yaml
preparation:
  sets: ["set01"]
training:
  batch_size: 4
  epochs: 10
```

### 2. Memory-Constrained Environment
```yaml
training:
  image_size: 512
  batch_size: 2
```

### 3. High-Performance GPU
```yaml
training:
  image_size: 1024
  batch_size: 16
  epochs: 100
```

### 4. Resume Interrupted Training
```yaml
steps:
  skip_download: true
  skip_prepare: true
  skip_crop: true
training:
  resume: "checkpoints/checkpoint_epoch_25.pth"
```

### 5. Save Disk Space
```yaml
preparation:
  symlink: true  # Use symlinks instead of copying files
```

### 6. Data Preparation Only
```yaml
steps:
  prepare_only: true  # Stop before training
```

## Version Control Best Practices

### Recommended .gitignore entries:
```gitignore
# Ignore the active config (contains local paths)
setup_and_train_config.yaml

# Keep the example
!setup_and_train_config_example.yaml
```

### Versioning experiment configs:
```bash
# Save your current config as an experiment
cp setup_and_train_config.yaml experiments/experiment_01_baseline.yaml

# Later, reuse that config
python training/setup_and_train.py --config experiments/experiment_01_baseline.yaml
```

### Comparing experiments:
```bash
# See what changed between experiments
diff experiments/experiment_01_baseline.yaml experiments/experiment_02_larger_batch.yaml
```

## Multiple Configurations Example

Create different configs for different scenarios:

```bash
configs/
├── dev.yaml          # Small, fast config for development
├── test.yaml         # Single set, quick training
├── production.yaml   # Full dataset, high quality
└── resume.yaml       # Skip prep, resume training
```

Use them with:
```bash
python training/setup_and_train.py --config configs/dev.yaml
python training/setup_and_train.py --config configs/production.yaml
```

## Validation

The script will:
1. Try to load the config file
2. Warn if file doesn't exist (but continue with defaults)
3. Warn if there are YAML syntax errors (but continue with defaults)
4. Apply config values, then override with CLI arguments

## Default Behavior

If no config file exists, the script uses these defaults:
- Download from default Seattle Aquarium repository
- Process all sets
- Crop inputs to 4606×4030
- Skip output cropping
- Train with image size 1024, batch 8, 50 epochs
- Learning rate 1e-4
- 80/20 train/val split

## Security Best Practices

**Never store secrets in config files:**
- The `hf_token` parameter is intentionally NOT available in the config file
- Always use CLI arguments or environment variables for sensitive data
- Config files may be committed to version control, so keep them secret-free

**Recommended authentication methods (in order of preference):**
1. **One-time login** (best for local development):
   ```bash
   huggingface-cli login
   ```

2. **Environment variable** (best for CI/CD):
   ```bash
   export HF_TOKEN=your_token_here
   python training/setup_and_train.py
   ```

3. **CLI argument** (use for one-off runs):
   ```bash
   python training/setup_and_train.py --hf-token your_token_here
   ```

## Tips

1. **Start with the example:** Copy `setup_and_train_config_example.yaml` and modify incrementally
2. **Comment your changes:** YAML supports comments with `#`
3. **Use meaningful names:** When saving configs for experiments, use descriptive names
4. **Keep it simple:** Start with defaults and only change what you need
5. **Version control:** Commit experiment configs to track what settings produced which results
6. **Keep secrets out:** Never add tokens or passwords to config files

## Troubleshooting

### Config file not loading
- Check file exists: `ls setup_and_train_config.yaml`
- Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('setup_and_train_config.yaml'))"`

### Wrong values being used
- Check precedence: CLI args > config file > defaults
- Add `--config` to verify which file is being loaded

### Need to see what settings are active
- Check the log output at the start of the script
- Settings are logged after loading config and applying overrides
