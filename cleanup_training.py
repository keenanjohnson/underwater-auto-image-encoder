#!/usr/bin/env python3
"""
Cleanup script for training pipeline.

Removes all intermediate files and directories created during the training process,
allowing you to start fresh. Useful for:
- Starting training from scratch
- Freeing up disk space
- Testing different dataset configurations
- Cleaning up after experiments

This script is safe - it only removes known training artifacts and will prompt
before deleting anything unless --force is specified.
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default directories and files to clean up
DEFAULT_CLEANUP_TARGETS = {
    'prepared_datasets': [
        'training_dataset',
        'dataset',
    ],
    'checkpoints': [
        'checkpoints',
    ],
    'outputs': [
        'output',
    ],
}


def get_size_mb(path: Path) -> float:
    """Calculate total size of a directory or file in MB."""
    if not path.exists():
        return 0.0

    if path.is_file():
        return path.stat().st_size / (1024 * 1024)

    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            try:
                total += item.stat().st_size
            except (PermissionError, FileNotFoundError):
                pass

    return total / (1024 * 1024)


def remove_path(path: Path, dry_run: bool = False) -> tuple[bool, float]:
    """
    Remove a file or directory.

    Returns:
        (success, size_mb): Whether removal succeeded and size in MB
    """
    if not path.exists():
        return False, 0.0

    size_mb = get_size_mb(path)

    if dry_run:
        logger.info(f"  [DRY RUN] Would remove: {path} ({size_mb:.2f} MB)")
        return True, size_mb

    try:
        if path.is_file():
            path.unlink()
            logger.info(f"  ✓ Removed file: {path} ({size_mb:.2f} MB)")
        else:
            shutil.rmtree(path)
            logger.info(f"  ✓ Removed directory: {path} ({size_mb:.2f} MB)")
        return True, size_mb
    except Exception as e:
        logger.error(f"  ✗ Failed to remove {path}: {e}")
        return False, 0.0


def cleanup_training(
    keep_raw_dataset: bool = True,
    keep_checkpoints: bool = False,
    keep_outputs: bool = False,
    custom_targets: list[str] = None,
    dry_run: bool = False,
    force: bool = False
):
    """
    Clean up training artifacts.

    Args:
        keep_raw_dataset: Keep the raw downloaded dataset (default: True)
        keep_checkpoints: Keep model checkpoints (default: False)
        keep_outputs: Keep final outputs (default: False)
        custom_targets: Additional custom paths to remove
        dry_run: Show what would be removed without actually removing
        force: Skip confirmation prompt
    """
    logger.info("="*80)
    logger.info("TRAINING CLEANUP UTILITY")
    logger.info("="*80)

    # Determine what to clean up
    targets_to_remove = []

    # Prepared datasets (always clean unless explicitly kept)
    for target in DEFAULT_CLEANUP_TARGETS['prepared_datasets']:
        targets_to_remove.append(Path(target))

    # Checkpoints
    if not keep_checkpoints:
        for target in DEFAULT_CLEANUP_TARGETS['checkpoints']:
            targets_to_remove.append(Path(target))

    # Outputs
    if not keep_outputs:
        for target in DEFAULT_CLEANUP_TARGETS['outputs']:
            targets_to_remove.append(Path(target))

    # Custom targets
    if custom_targets:
        for target in custom_targets:
            targets_to_remove.append(Path(target))

    # Filter to only existing paths
    existing_targets = [p for p in targets_to_remove if p.exists()]

    if not existing_targets:
        logger.info("\n✓ Nothing to clean up - all directories are already clean!")
        return

    # Calculate total size
    logger.info("\nThe following items will be removed:")
    logger.info("-" * 80)

    total_size = 0.0
    for target in existing_targets:
        size_mb = get_size_mb(target)
        total_size += size_mb
        item_type = "DIR " if target.is_dir() else "FILE"
        logger.info(f"  [{item_type}] {target} ({size_mb:.2f} MB)")

    logger.info("-" * 80)
    logger.info(f"Total size to be freed: {total_size:.2f} MB ({total_size/1024:.2f} GB)")

    if keep_raw_dataset:
        logger.info("\n✓ Raw dataset will be preserved (dataset_raw/)")

    if keep_checkpoints:
        logger.info("✓ Checkpoints will be preserved (checkpoints/)")

    if keep_outputs:
        logger.info("✓ Outputs will be preserved (output/)")

    # Confirm before deletion
    if not force and not dry_run:
        logger.info("\n" + "="*80)
        response = input("⚠️  Continue with cleanup? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            logger.info("Cleanup cancelled.")
            return

    # Perform cleanup
    logger.info("\n" + "="*80)
    if dry_run:
        logger.info("DRY RUN - No files will actually be removed")
    else:
        logger.info("Starting cleanup...")
    logger.info("="*80)

    removed_count = 0
    freed_size = 0.0

    for target in existing_targets:
        success, size = remove_path(target, dry_run)
        if success:
            removed_count += 1
            freed_size += size

    # Summary
    logger.info("\n" + "="*80)
    if dry_run:
        logger.info("DRY RUN COMPLETE")
        logger.info(f"Would remove {removed_count} items")
        logger.info(f"Would free {freed_size:.2f} MB ({freed_size/1024:.2f} GB)")
    else:
        logger.info("CLEANUP COMPLETE!")
        logger.info(f"Removed {removed_count} items")
        logger.info(f"Freed {freed_size:.2f} MB ({freed_size/1024:.2f} GB)")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up training artifacts and intermediate files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe cleanup (keeps raw dataset, removes everything else)
  python cleanup_training.py

  # Dry run (see what would be removed without actually removing)
  python cleanup_training.py --dry-run

  # Keep checkpoints and outputs (only remove intermediate files)
  python cleanup_training.py --keep-checkpoints --keep-outputs

  # Force cleanup without confirmation prompt
  python cleanup_training.py --force

  # Complete cleanup (remove EVERYTHING including raw dataset)
  python cleanup_training.py --remove-raw-dataset --force

  # Remove specific custom directories
  python cleanup_training.py --remove my_experiment1 my_experiment2

Note: By default, the raw dataset (dataset_raw/) is preserved.
Use --remove-raw-dataset to also remove the downloaded dataset.
        """
    )

    parser.add_argument(
        '--remove-raw-dataset',
        action='store_true',
        help='Also remove the raw downloaded dataset (dataset_raw/). WARNING: You will need to re-download!'
    )

    parser.add_argument(
        '--keep-checkpoints',
        action='store_true',
        help='Keep model checkpoints (checkpoints/)'
    )

    parser.add_argument(
        '--keep-outputs',
        action='store_true',
        help='Keep final outputs (output/)'
    )

    parser.add_argument(
        '--remove',
        nargs='+',
        metavar='PATH',
        help='Additional custom paths to remove'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing anything'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    # Add raw dataset to cleanup targets if requested
    custom_targets = args.remove or []
    if args.remove_raw_dataset:
        custom_targets.append('dataset_raw')
        logger.warning("⚠️  Raw dataset will be removed! You will need to re-download.")

    cleanup_training(
        keep_raw_dataset=not args.remove_raw_dataset,
        keep_checkpoints=args.keep_checkpoints,
        keep_outputs=args.keep_outputs,
        custom_targets=custom_targets,
        dry_run=args.dry_run,
        force=args.force
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Cleanup cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n\n✗ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
