#!/usr/bin/env python3
"""
Download dataset from Hugging Face Hub

Downloads the underwater image dataset from Hugging Face and organizes it
for use with the training pipeline.
"""

import argparse
import logging
import os
import time
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
    token: str = None,
    max_retries: int = 20,
    initial_retry_delay: int = 310
):
    """
    Download dataset from Hugging Face Hub with automatic retry on rate limits

    Args:
        repo_id: Hugging Face repository ID (default: DEFAULT_REPO_ID)
        output_dir: Local directory to save the dataset
        repo_type: Type of repository ('dataset', 'model', or 'space')
        token: Hugging Face API token (optional, will use HF_TOKEN env var or saved token)
        max_retries: Maximum number of retry attempts for rate limits (default: 20)
        initial_retry_delay: Initial delay in seconds before first retry (default: 310 - HF rate limit is 5 min)
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

    # Retry loop with exponential backoff for rate limits
    retry_count = 0
    retry_delay = initial_retry_delay
    download_path = None

    while retry_count <= max_retries:
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
            break  # Success! Exit retry loop

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limit errors
            if "rate limit" in error_msg or "429" in error_msg:
                retry_count += 1

                if retry_count > max_retries:
                    # Max retries exceeded - show error and give up
                    logger.error(f"Rate limit error: {e}")
                    logger.error(f"\n⚠️  RATE LIMIT EXCEEDED (Max retries: {max_retries}) ⚠️")
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
                    logger.error(f"\nAlternatively, upgrade to HuggingFace PRO for higher rate limits:")
                    logger.error(f"  https://huggingface.co/pricing")
                    raise
                else:
                    # Rate limited but we can retry
                    logger.warning(f"\n⚠️  Rate limit hit (attempt {retry_count}/{max_retries})")
                    logger.warning(f"HuggingFace rate limit: 1000 API requests per 5 minutes")
                    logger.warning(f"Waiting {retry_delay} seconds (~{retry_delay//60} minutes) before retrying...")
                    logger.warning(f"Download will resume from where it left off")
                    logger.info(f"\nTo avoid rate limits in the future:")
                    logger.info(f"  1. Authenticate with: huggingface-cli login")
                    logger.info(f"  2. Or upgrade to HuggingFace PRO: https://huggingface.co/pricing")

                    # Wait with progress indicator
                    remaining = retry_delay
                    while remaining > 0:
                        logger.info(f"Waiting... {remaining} seconds remaining")
                        sleep_time = min(30, remaining)
                        time.sleep(sleep_time)
                        remaining -= sleep_time

                    # Exponential backoff: double the delay for next retry
                    retry_delay = min(retry_delay * 2, 1200)  # Cap at 20 minutes
                    logger.info(f"\nRetrying download (attempt {retry_count + 1}/{max_retries + 1})...")
                    continue
            else:
                # Non-rate-limit error - fail immediately
                logger.error(f"Error downloading dataset: {e}")
                logger.error(f"\nPlease check:")
                logger.error(f"  1. Your internet connection")
                logger.error(f"  2. The repository ID is correct: {repo_id}")
                logger.error(f"  3. You have access to the repository (if private)")
                logger.error(f"\nFor authentication issues, login first:")
                logger.error(f"  huggingface-cli login")
                raise

    # Verify the download was successful
    if download_path is None:
        logger.error("\n❌ Download failed - no data was downloaded")
        logger.error("This may be due to:")
        logger.error("  1. Rate limiting (wait 5+ minutes and try again)")
        logger.error("  2. Authentication issues (run: huggingface-cli login)")
        logger.error("  3. Network connectivity problems")
        return None

    # Verify the expected structure - check for both old and new formats
    input_dir = output_path / "input"
    target_dir = output_path / "target"

    # Check for old structure (input/ and target/ at root)
    if input_dir.exists() and target_dir.exists():
        # Count files (recursively to catch all images)
        input_files = list(input_dir.rglob('*.tif')) + list(input_dir.rglob('*.tiff'))
        target_files = list(target_dir.rglob('*.tif')) + list(target_dir.rglob('*.tiff'))

        logger.info(f"\nDataset structure verified (legacy format):")
        logger.info(f"  Input images: {len(input_files)} files")
        logger.info(f"  Target images: {len(target_files)} files")

        if len(input_files) == 0 and len(target_files) == 0:
            logger.warning(f"\n⚠️  WARNING: Dataset directories exist but no images found!")
            logger.warning(f"This likely means the download was interrupted by rate limiting.")
            logger.warning(f"Please wait 5 minutes and run the download again - it will resume from where it left off.")
        else:
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
                # Support both 'output' and 'target' folder naming
                set_output = set_dir / "output"
                if not set_output.exists():
                    set_output = set_dir / "target"

                if set_input.exists() and set_output.exists():
                    # Count TIFF files specifically
                    input_files = list(set_input.glob('*.tif')) + list(set_input.glob('*.tiff'))
                    output_files = list(set_output.glob('*.tif')) + list(set_output.glob('*.tiff'))
                    n_input = len(input_files)
                    n_output = len(output_files)
                    total_input += n_input
                    total_output += n_output
                    logger.info(f"    {set_dir.name}: {n_input} input, {n_output} output images")

            logger.info(f"\n  Total: {total_input} input, {total_output} output images")

            if total_input == 0 and total_output == 0:
                logger.warning(f"\n⚠️  WARNING: Dataset directories exist but no images found!")
                logger.warning(f"This likely means the download was interrupted by rate limiting.")
                logger.warning(f"Please wait 5 minutes and run the download again - it will resume from where it left off.")
                logger.warning(f"\nAlternatively:")
                logger.warning(f"  1. Authenticate with: huggingface-cli login")
                logger.warning(f"  2. Or upgrade to HuggingFace PRO: https://huggingface.co/pricing")
            else:
                logger.info(f"\nTo prepare the dataset for training, run:")
                logger.info(f"  python prepare_huggingface_dataset.py {output_path} --output training_dataset")
                logger.info(f"\nOr to use a specific set:")
                logger.info(f"  python prepare_huggingface_dataset.py {output_path}/set01 --output training_dataset")
        else:
            logger.warning(f"\nWarning: Could not detect dataset structure in {output_path}")
            logger.warning(f"Expected either:")
            logger.warning(f"  - Legacy format: input/ and target/ directories")
            logger.warning(f"  - Set-based format: set01/, set02/, etc. with input/ and output/ (or target/) subdirectories")
            logger.warning(f"\nThe download may have been interrupted. Try running again to resume.")

    return download_path


def main():
    parser = argparse.ArgumentParser(
        description="Download underwater image dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the default dataset (auto-retries on rate limits)
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

  # Customize retry behavior:
  python download_dataset.py --max-retries 10 --retry-delay 310

Note: Downloads automatically resume from where they left off if interrupted.
Rate limits trigger automatic retries with exponential backoff (310s, 620s, 1200s, etc).
HuggingFace rate limit: 1000 API requests per 5 minutes for free accounts.
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

    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
        help='Maximum number of retry attempts for rate limits (default: 5)'
    )

    parser.add_argument(
        '--retry-delay',
        type=int,
        default=310,
        help='Initial delay in seconds before first retry (default: 310 - HF rate limit is 5 min, doubles each retry)'
    )

    args = parser.parse_args()

    download_dataset(
        repo_id=args.repo_id,
        output_dir=args.output,
        repo_type=args.repo_type,
        token=args.token,
        max_retries=args.max_retries,
        initial_retry_delay=args.retry_delay
    )


if __name__ == "__main__":
    main()
