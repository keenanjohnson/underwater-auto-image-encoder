#!/usr/bin/env python3
"""
Download dataset from Hugging Face Hub

Downloads the underwater image dataset from Hugging Face and organizes it
for use with the training pipeline.
"""

import argparse
import logging
import os
from pathlib import Path
from huggingface_hub import snapshot_download

try:
    from huggingface_hub import HfFolder
except ImportError:
    # HfFolder was deprecated, use get_token instead
    from huggingface_hub import get_token
    HfFolder = None

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
    repo_type: str = "dataset",
    token: str = None
):
    """
    Download dataset from Hugging Face Hub

    Args:
        repo_id: Hugging Face repository ID (default: DEFAULT_REPO_ID)
        output_dir: Local directory to save the dataset
        repo_type: Type of repository ('dataset', 'model', or 'space')
        token: Hugging Face API token (optional, will use HF_TOKEN env var or saved token)
    """
    output_path = Path(output_dir)

    # Get token from various sources
    if token is None:
        # Try environment variable first
        token = os.environ.get("HF_TOKEN")
        if token is None:
            # Try saved token from huggingface-cli login
            try:
                if HfFolder is not None:
                    token = HfFolder.get_token()
                else:
                    # Use new API
                    token = get_token()
            except Exception:
                # If all else fails, token remains None
                pass

    if token:
        logger.info("Using Hugging Face authentication token")
    else:
        logger.warning("No Hugging Face token found - may hit rate limits for large downloads")
        logger.warning("To authenticate, run: huggingface-cli login")
        logger.warning("Or set HF_TOKEN environment variable")

    logger.info(f"Downloading dataset '{repo_id}' from Hugging Face...")
    logger.info(f"Destination: {output_path.absolute()}")

    try:
        # Download the entire repository
        # Note: snapshot_download automatically resumes interrupted downloads
        # and skips files that are already present and up-to-date
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=output_path,
            local_dir_use_symlinks=False,  # Copy files instead of symlinking
            resume_download=True,  # Resume interrupted downloads (default: True)
            token=token,  # Authentication token
        )

        logger.info(f"✓ Dataset downloaded successfully to: {download_path}")

        # Verify the expected structure - check for both old and new formats
        input_dir = output_path / "input"
        target_dir = output_path / "target"

        # Check for old structure (input/ and target/ at root)
        if input_dir.exists() and target_dir.exists():
            # Count files
            input_files = list(input_dir.glob('*'))
            target_files = list(target_dir.glob('*'))

            logger.info(f"\nDataset structure verified (legacy format):")
            logger.info(f"  Input images: {len(input_files)} files")
            logger.info(f"  Target images: {len(target_files)} files")
            logger.info(f"\nDataset ready for training!")
            logger.info(f"\nTo train the model, run:")
            logger.info(f"  python train.py --input-dir {input_dir} --target-dir {target_dir}")
        else:
            # Check for new structure (set01/, set02/, etc.)
            set_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('set')])

            if set_dirs:
                logger.info(f"\nDataset structure verified (set-based format):")
                logger.info(f"  Found {len(set_dirs)} set directories")

                total_input = 0
                total_output = 0

                for set_dir in set_dirs:
                    set_input = set_dir / "input"
                    set_output = set_dir / "output"

                    if set_input.exists() and set_output.exists():
                        n_input = len(list(set_input.glob('*')))
                        n_output = len(list(set_output.glob('*')))
                        total_input += n_input
                        total_output += n_output
                        logger.info(f"    {set_dir.name}: {n_input} input, {n_output} output images")

                logger.info(f"\n  Total: {total_input} input, {total_output} output images")
                logger.info(f"\nTo prepare the dataset for training, run:")
                logger.info(f"  python prepare_huggingface_dataset.py {output_path} --output training_dataset")
                logger.info(f"\nOr to use a specific set:")
                logger.info(f"  python prepare_huggingface_dataset.py {output_path}/set01 --output training_dataset")
            else:
                logger.warning(f"\nWarning: Could not detect dataset structure in {output_path}")
                logger.warning(f"Expected either:")
                logger.warning(f"  - Legacy format: input/ and target/ directories")
                logger.warning(f"  - Set-based format: set01/, set02/, etc. with input/ and output/ subdirectories")

        return download_path

    except Exception as e:
        error_msg = str(e).lower()

        # Check for specific error types
        if "rate limit" in error_msg or "429" in error_msg:
            logger.error(f"Rate limit error: {e}")
            logger.error(f"\n⚠️  RATE LIMIT EXCEEDED ⚠️")
            logger.error(f"\nYou need to authenticate with Hugging Face to continue:")
            logger.error(f"\nOption 1 - Login via CLI (recommended):")
            logger.error(f"  huggingface-cli login")
            logger.error(f"  # Then re-run this script")
            logger.error(f"\nOption 2 - Set environment variable:")
            logger.error(f"  export HF_TOKEN=your_token_here")
            logger.error(f"  python download_dataset.py --output {output_dir}")
            logger.error(f"\nOption 3 - Pass token as argument:")
            logger.error(f"  python download_dataset.py --token your_token_here")
            logger.error(f"\nGet your token at: https://huggingface.co/settings/tokens")
        else:
            logger.error(f"Error downloading dataset: {e}")
            logger.error(f"\nPlease check:")
            logger.error(f"  1. Your internet connection")
            logger.error(f"  2. The repository ID is correct: {repo_id}")
            logger.error(f"  3. You have access to the repository (if private)")
            logger.error(f"\nFor authentication issues, login first:")
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

  # For private datasets or to avoid rate limits, login first:
  huggingface-cli login
  python download_dataset.py --repo-id username/private-dataset

  # Or pass token directly:
  python download_dataset.py --token hf_xxxxxxxxxxxxx

  # Or use environment variable:
  export HF_TOKEN=hf_xxxxxxxxxxxxx
  python download_dataset.py
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

    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face API token (optional, can also use HF_TOKEN env var or huggingface-cli login)'
    )

    args = parser.parse_args()

    download_dataset(
        repo_id=args.repo_id,
        output_dir=args.output,
        repo_type=args.repo_type,
        token=args.token
    )


if __name__ == "__main__":
    main()
