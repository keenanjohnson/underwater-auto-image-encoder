#!/usr/bin/env python3
"""
Prepare dataset from Hugging Face format where input and output files
are in the same directory with matching filenames.

Input: Directory with paired .tif (input) and .jpg (output) files
Output: Organized dataset/input and dataset/target directories
"""

import argparse
import shutil
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_paired_dataset(source_dir: str, output_dir: str = "dataset", symlink: bool = False):
    """
    Organize paired images from a single directory into input/target structure.

    Args:
        source_dir: Directory containing both .tif and .jpg files
        output_dir: Output directory for organized dataset
        symlink: Create symlinks instead of copying files
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return
    if not source_path.is_dir():
        logger.error(f"Source path is not a directory: {source_dir}")
        return
    output_path = Path(output_dir)

    input_dir = output_path / "input"
    target_dir = output_path / "target"

    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all .tif files (inputs)
    tif_files = sorted(list(source_path.glob("*.tif")) + list(source_path.glob("*.TIF")))

    if not tif_files:
        logger.error(f"No .tif files found in {source_dir}")
        return

    logger.info(f"Found {len(tif_files)} .tif files")

    # Find matching pairs
    pairs = []
    missing_targets = []

    for tif_file in tif_files:
        base_name = tif_file.stem
        jpg_file = source_path / f"{base_name}.jpg"

        if jpg_file.exists():
            pairs.append((tif_file, jpg_file))
        else:
            missing_targets.append(tif_file.name)

    logger.info(f"Found {len(pairs)} matching pairs")

    if missing_targets:
        logger.warning(f"Found {len(missing_targets)} .tif files without matching .jpg:")
        for name in missing_targets[:5]:
            logger.warning(f"  - {name}")
        if len(missing_targets) > 5:
            logger.warning(f"  ... and {len(missing_targets) - 5} more")

    if not pairs:
        logger.error("No matching pairs found!")
        return

    # Copy or symlink files
    logger.info(f"Organizing files into {output_path}...")

    for idx, (tif_path, jpg_path) in enumerate(pairs):
        # Use sequential naming for consistency
        base_name = f"img_{idx:05d}"

        input_dest = input_dir / f"{base_name}.tif"
        target_dest = target_dir / f"{base_name}.jpg"

        if symlink:
            # Remove existing symlinks if they exist
            if input_dest.exists() or input_dest.is_symlink():
                input_dest.unlink()
            if target_dest.exists() or target_dest.is_symlink():
                target_dest.unlink()

            input_dest.symlink_to(tif_path.absolute())
            target_dest.symlink_to(jpg_path.absolute())
        else:
            # Remove existing files/symlinks before copying
            if input_dest.exists() or input_dest.is_symlink():
                input_dest.unlink()
            if target_dest.exists() or target_dest.is_symlink():
                target_dest.unlink()

            shutil.copy2(tif_path, input_dest)
            shutil.copy2(jpg_path, target_dest)

    logger.info(f"✓ Organized {len(pairs)} pairs")

    # Create train/val split
    logger.info("Creating train/validation split (80/20)...")

    num_pairs = len(pairs)
    np.random.seed(42)
    indices = np.random.permutation(num_pairs)
    split_point = int(0.8 * num_pairs)

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
    logger.info("\n" + "=" * 60)
    logger.info("Dataset prepared successfully!")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Total pairs:      {len(pairs)}")
    logger.info(f"Training:         {len(train_indices)} pairs")
    logger.info(f"Validation:       {len(val_indices)} pairs")
    logger.info("=" * 60)
    logger.info("\nTo start training, run:")
    logger.info(f"  python train.py --input-dir {input_dir} --target-dir {target_dir}")
    logger.info("\nOr with custom settings:")
    logger.info(f"  python train.py --input-dir {input_dir} --target-dir {target_dir} \\")
    logger.info(f"    --image-size 512 --batch-size 8 --epochs 50")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset from Hugging Face format (paired .tif and .jpg in same directory)"
    )
    parser.add_argument('source_dir', type=str,
                        help='Directory containing paired .tif and .jpg files')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for organized dataset (default: dataset)')
    parser.add_argument('--symlink', action='store_true',
                        help='Create symlinks instead of copying files (saves space)')

    args = parser.parse_args()

    prepare_paired_dataset(args.source_dir, args.output, args.symlink)


if __name__ == "__main__":
    main()
