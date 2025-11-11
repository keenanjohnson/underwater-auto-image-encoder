#!/usr/bin/env python3
"""
Download dataset from Hugging Face Hub

Downloads the underwater image dataset from Hugging Face and organizes it
for use with the training pipeline.
"""

import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default dataset repository
DEFAULT_REPO_ID = "Seattle-Aquarium/Seattle_Aquarium_benthic_imagery"


def download_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: str = "dataset",
    repo_type: str = "dataset"
):
    """
    Download dataset from Hugging Face Hub

    Args:
        repo_id: Hugging Face repository ID (default: DEFAULT_REPO_ID)
        output_dir: Local directory to save the dataset
        repo_type: Type of repository ('dataset', 'model', or 'space')
    """
    output_path = Path(output_dir)

    logger.info(f"Downloading dataset '{repo_id}' from Hugging Face...")
    logger.info(f"Destination: {output_path.absolute()}")

    try:
        # Download the entire repository
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=output_path,
            local_dir_use_symlinks=False,  # Copy files instead of symlinking
        )

        logger.info(f"✓ Dataset downloaded successfully to: {download_path}")

        # Verify the expected structure
        input_dir = output_path / "input"
        target_dir = output_path / "target"

        if input_dir.exists() and target_dir.exists():
            # Count files
            input_files = list(input_dir.glob('*'))
            target_files = list(target_dir.glob('*'))

            logger.info(f"\nDataset structure verified:")
            logger.info(f"  Input images: {len(input_files)} files")
            logger.info(f"  Target images: {len(target_files)} files")
            logger.info(f"\nDataset ready for training!")
            logger.info(f"\nTo train the model, run:")
            logger.info(f"  python train.py --config config.yaml")
        else:
            logger.warning(f"\nWarning: Expected 'input/' and 'target/' directories not found")
            logger.warning(f"Please verify the dataset structure in {output_path}")

        return download_path

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error(f"\nPlease check:")
        logger.error(f"  1. Your internet connection")
        logger.error(f"  2. The repository ID is correct: {repo_id}")
        logger.error(f"  3. You have access to the repository (if private)")
        logger.error(f"\nFor private repositories, you may need to login first:")
        logger.error(f"  huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download underwater image dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the default dataset
  python download_dataset.py

  # Download to a custom directory
  python download_dataset.py --output my_dataset

  # Download a different dataset
  python download_dataset.py --repo-id username/dataset-name

  # For private datasets, login first:
  huggingface-cli login
  python download_dataset.py --repo-id username/private-dataset
        """
    )

    parser.add_argument(
        '--repo-id',
        type=str,
        default=DEFAULT_REPO_ID,
        help=f'Hugging Face repository ID (default: {DEFAULT_REPO_ID})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='dataset',
        help='Output directory for the dataset (default: dataset)'
    )

    parser.add_argument(
        '--repo-type',
        type=str,
        default='dataset',
        choices=['dataset', 'model', 'space'],
        help='Type of Hugging Face repository (default: dataset)'
    )

    args = parser.parse_args()

    download_dataset(
        repo_id=args.repo_id,
        output_dir=args.output,
        repo_type=args.repo_type
    )


if __name__ == "__main__":
    main()
