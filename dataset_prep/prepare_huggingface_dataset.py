#!/usr/bin/env python3
"""
Prepare dataset from Hugging Face set-based format.

Handles the HuggingFace dataset structure where images are organized into
set directories (set01/, set02/, etc.), each containing input/ and output/ subdirectories.

This script can:
- Process a single set directory
- Process multiple sets and combine them
- Automatically detect and process all sets in a directory
"""

import argparse
import shutil
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_set_directories(root_dir: Path):
    """Find all set directories in the root directory."""
    set_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith('set')])
    return set_dirs


def count_images_in_set(set_dir: Path):
    """Count input and output images in a set directory."""
    input_dir = set_dir / "input"
    output_dir = set_dir / "output"

    input_count = 0
    output_count = 0

    if input_dir.exists():
        input_count = len([f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith('.')])

    if output_dir.exists():
        output_count = len([f for f in output_dir.iterdir() if f.is_file() and not f.name.startswith('.')])

    return input_count, output_count


def find_image_pairs(set_dir: Path):
    """
    Find matching pairs of input and output images in a set directory.

    Returns:
        List of tuples: [(input_path, output_path), ...]
    """
    input_dir = set_dir / "input"
    output_dir = set_dir / "output"

    if not input_dir.exists() or not output_dir.exists():
        logger.warning(f"Missing input or output directory in {set_dir}")
        return []

    # Get all input files (various extensions)
    input_extensions = ('.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    input_files = {}

    for f in input_dir.iterdir():
        if f.is_file() and f.suffix in input_extensions:
            # Store by stem (filename without extension)
            input_files[f.stem] = f

    # Get all output files
    output_extensions = ('.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    output_files = {}

    for f in output_dir.iterdir():
        if f.is_file() and f.suffix in output_extensions:
            output_files[f.stem] = f

    # Find matching pairs by filename stem
    common_stems = set(input_files.keys()) & set(output_files.keys())

    pairs = []
    for stem in sorted(common_stems):
        pairs.append((input_files[stem], output_files[stem]))

    unmatched_input = len(input_files) - len(pairs)
    unmatched_output = len(output_files) - len(pairs)

    if unmatched_input > 0 or unmatched_output > 0:
        logger.warning(f"{set_dir.name}: {unmatched_input} unmatched input, {unmatched_output} unmatched output files")

    return pairs


def prepare_dataset(
    source_dirs,
    output_dir: str = "dataset",
    symlink: bool = False,
    split_ratio: float = 0.8
):
    """
    Prepare dataset from set-based directories.

    Args:
        source_dirs: List of set directory paths or a single root directory
        output_dir: Output directory for organized dataset
        symlink: Create symlinks instead of copying files
        split_ratio: Train/validation split ratio (default: 0.8)
    """
    output_path = Path(output_dir)
    input_dest_dir = output_path / "input"
    target_dest_dir = output_path / "target"

    input_dest_dir.mkdir(parents=True, exist_ok=True)
    target_dest_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pairs from all sets
    all_pairs = []
    set_info = []

    for source_dir in source_dirs:
        source_path = Path(source_dir)

        if not source_path.exists():
            logger.warning(f"Directory does not exist: {source_dir}")
            continue

        # Check if this is a single set directory or contains multiple sets
        if (source_path / "input").exists() and (source_path / "output").exists():
            # This is a single set directory
            logger.info(f"Processing set: {source_path.name}")
            pairs = find_image_pairs(source_path)
            all_pairs.extend(pairs)
            set_info.append((source_path.name, len(pairs)))
        else:
            # Check for multiple set directories
            set_dirs = find_set_directories(source_path)

            if set_dirs:
                logger.info(f"Found {len(set_dirs)} set directories in {source_path}")

                for set_dir in set_dirs:
                    logger.info(f"  Processing {set_dir.name}...")
                    pairs = find_image_pairs(set_dir)
                    all_pairs.extend(pairs)
                    set_info.append((set_dir.name, len(pairs)))
            else:
                logger.warning(f"No valid set structure found in {source_path}")

    if not all_pairs:
        logger.error("No matching pairs found!")
        return

    logger.info(f"\nTotal pairs found: {len(all_pairs)}")
    logger.info("Set breakdown:")
    for set_name, count in set_info:
        logger.info(f"  {set_name}: {count} pairs")

    # Copy or symlink files with sequential naming
    logger.info(f"\nOrganizing files into {output_path}...")

    for idx, (input_img_path, output_img_path) in enumerate(tqdm(all_pairs, desc="Organizing files")):
        base_name = f"img_{idx:05d}"

        input_ext = input_img_path.suffix
        output_ext = output_img_path.suffix

        input_file_dest = input_dest_dir / f"{base_name}{input_ext}"
        output_file_dest = target_dest_dir / f"{base_name}{output_ext}"

        if symlink:
            # Remove existing symlinks if they exist
            if input_file_dest.exists() or input_file_dest.is_symlink():
                input_file_dest.unlink()
            if output_file_dest.exists() or output_file_dest.is_symlink():
                output_file_dest.unlink()

            input_file_dest.symlink_to(input_img_path.absolute())
            output_file_dest.symlink_to(output_img_path.absolute())
        else:
            # Copy files
            if input_file_dest.exists() or input_file_dest.is_symlink():
                input_file_dest.unlink()
            if output_file_dest.exists() or output_file_dest.is_symlink():
                output_file_dest.unlink()

            shutil.copy2(input_img_path, input_file_dest)
            shutil.copy2(output_img_path, output_file_dest)

    logger.info(f"✓ Organized {len(all_pairs)} pairs")

    # Create train/val split
    logger.info(f"\nCreating train/validation split ({split_ratio:.0%}/{1-split_ratio:.0%})...")

    num_pairs = len(all_pairs)
    np.random.seed(42)
    indices = np.random.permutation(num_pairs)
    split_point = int(split_ratio * num_pairs)

    train_indices = sorted(indices[:split_point].tolist())
    val_indices = sorted(indices[split_point:].tolist())

    split_file = output_path / 'split.txt'
    with open(split_file, 'w') as f:
        f.write(f"# Training indices (n={len(train_indices)})\n")
        f.write(','.join(map(str, train_indices)) + '\n')
        f.write(f"# Validation indices (n={len(val_indices)})\n")
        f.write(','.join(map(str, val_indices)) + '\n')

    logger.info(f"✓ Created split: {len(train_indices)} train, {len(val_indices)} validation")
    logger.info(f"✓ Split file: {split_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Dataset prepared successfully!")
    logger.info("=" * 80)
    logger.info(f"Input directory:  {input_dest_dir}")
    logger.info(f"Target directory: {target_dest_dir}")
    logger.info(f"Total pairs:      {len(all_pairs)}")
    logger.info(f"Training:         {len(train_indices)} pairs")
    logger.info(f"Validation:       {len(val_indices)} pairs")
    logger.info(f"Method:           {'Symlinked' if symlink else 'Copied'}")
    logger.info("=" * 80)
    logger.info("\nTo start training, run:")
    logger.info(f"  python train.py --input-dir {input_dest_dir} --target-dir {target_dest_dir}")
    logger.info("\nOr with custom settings:")
    logger.info(f"  python train.py --input-dir {input_dest_dir} --target-dir {target_dest_dir} \\")
    logger.info(f"    --image-size 512 --batch-size 8 --epochs 50")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset from Hugging Face set-based format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sets in a directory
  python prepare_huggingface_dataset.py dataset_raw --output training_dataset

  # Process a single set
  python prepare_huggingface_dataset.py dataset_raw/set01 --output training_dataset

  # Process specific sets
  python prepare_huggingface_dataset.py dataset_raw/set01 dataset_raw/set02 --output training_dataset

  # Use symlinks instead of copying (saves disk space)
  python prepare_huggingface_dataset.py dataset_raw --output training_dataset --symlink

  # Custom train/val split ratio
  python prepare_huggingface_dataset.py dataset_raw --output training_dataset --split-ratio 0.9
        """
    )

    parser.add_argument('source_dirs', type=str, nargs='+',
                        help='Directory containing set folders, or specific set directories')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for organized dataset (default: dataset)')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying files (saves space)')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Train/validation split ratio (default: 0.8)')

    args = parser.parse_args()

    prepare_dataset(
        source_dirs=args.source_dirs,
        output_dir=args.output,
        symlink=args.symlink,
        split_ratio=args.split_ratio
    )


if __name__ == "__main__":
    main()
