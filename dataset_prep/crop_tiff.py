#!/usr/bin/env python3
"""
Crop TIFF images to specified dimensions (center crop)
Useful for preprocessing Hugging Face datasets to match training dimensions
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def center_crop_image(
    input_path: Path,
    output_path: Path,
    crop_size: Tuple[int, int],
    preserve_format: bool = True
) -> bool:
    """
    Center crop an image to specified dimensions

    Args:
        input_path: Path to input image
        output_path: Path to save cropped image
        crop_size: Target size (width, height)
        preserve_format: Keep original format, otherwise save as TIFF

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(input_path)

        # Extract metadata before any conversions
        metadata = {}

        # Preserve EXIF data
        if 'exif' in img.info:
            metadata['exif'] = img.info['exif']

        # Preserve ICC color profile
        if 'icc_profile' in img.info:
            metadata['icc_profile'] = img.info['icc_profile']

        # Preserve DPI information
        if 'dpi' in img.info:
            metadata['dpi'] = img.info['dpi']

        # Preserve any other metadata that PIL supports
        for key in ['transparency', 'gamma', 'chromaticity']:
            if key in img.info:
                metadata[key] = img.info[key]

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        crop_width, crop_height = crop_size

        # Check if image is large enough
        if width < crop_width or height < crop_height:
            logger.warning(
                f"Image {input_path.name} is too small ({width}x{height}), "
                f"cannot crop to {crop_width}x{crop_height}"
            )
            return False

        # Calculate center crop coordinates
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop image
        cropped = img.crop((left, top, right, bottom))

        # Save cropped image with preserved metadata
        if preserve_format:
            # Keep original format with metadata
            try:
                cropped.save(output_path, **metadata)
            except (TypeError, OSError) as e:
                logger.warning(f"Failed to save with full metadata: {e}. Retrying with basic metadata.")
                safe_metadata = {k: v for k, v in metadata.items() if k in ['exif', 'dpi']}
                cropped.save(output_path, **safe_metadata)
        else:
            # Save as TIFF with metadata
            cropped.save(output_path, 'TIFF', compression='none', **metadata)

        return True

    except Exception as e:
        logger.error(f"Error cropping {input_path.name}: {e}")
        return False


def crop_directory(
    input_dir: Path,
    output_dir: Path,
    crop_size: Tuple[int, int],
    extensions: List[str] = None,
    preserve_format: bool = True
) -> int:
    """
    Crop all images in a directory

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for cropped images
        crop_size: Target size (width, height)
        extensions: List of file extensions to process
        preserve_format: Keep original format

    Returns:
        Number of successfully cropped images
    """
    if extensions is None:
        extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))

    image_files = sorted(set(image_files))

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return 0

    logger.info(f"Found {len(image_files)} images to crop")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    success_count = 0
    for img_path in tqdm(image_files, desc="Cropping images"):
        # Determine output path
        if preserve_format:
            output_path = output_dir / img_path.name
        else:
            output_path = output_dir / f"{img_path.stem}.tif"

        if center_crop_image(img_path, output_path, crop_size, preserve_format):
            success_count += 1

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Center crop images to specified dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop single image
  python crop_tiff.py input.tif --output output.tif --width 4606 --height 4030

  # Crop all images in directory (default: 4606x4030 for underwater dataset)
  python crop_tiff.py my_dataset/ --output-dir cropped/

  # Crop with custom dimensions
  python crop_tiff.py my_dataset/ --output-dir cropped/ --width 2048 --height 2048

  # Crop and preserve original format (JPEG, PNG, etc.)
  python crop_tiff.py my_dataset/ --output-dir cropped/ --preserve-format
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input image file or directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file (for single image)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='cropped',
        help='Output directory (for directory processing, default: cropped)'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=4606,
        help='Crop width (default: 4606 for underwater dataset)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=4030,
        help='Crop height (default: 4030 for underwater dataset)'
    )

    parser.add_argument(
        '--preserve-format',
        action='store_true',
        help='Preserve original image format instead of converting to TIFF'
    )

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.tif', '.tiff', '.jpg', '.jpeg', '.png'],
        help='File extensions to process (default: .tif .tiff .jpg .jpeg .png)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    crop_size = (args.width, args.height)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Single file processing
    if input_path.is_file():
        if not args.output:
            logger.error("--output required for single file processing")
            return 1

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cropping {input_path.name} to {crop_size[0]}x{crop_size[1]}...")

        if center_crop_image(input_path, output_path, crop_size, args.preserve_format):
            logger.info(f"✓ Saved cropped image to: {output_path}")
            return 0
        else:
            logger.error("✗ Cropping failed")
            return 1

    # Directory processing
    elif input_path.is_dir():
        output_dir = Path(args.output_dir)

        logger.info(f"Cropping images in {input_path} to {crop_size[0]}x{crop_size[1]}...")
        logger.info(f"Output directory: {output_dir}")

        success_count = crop_directory(
            input_path,
            output_dir,
            crop_size,
            args.extensions,
            args.preserve_format
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Successfully cropped {success_count} images")
        logger.info(f"✓ Output directory: {output_dir}")
        logger.info("=" * 60)

        return 0 if success_count > 0 else 1

    else:
        logger.error(f"Invalid input: {input_path}")
        return 1


if __name__ == "__main__":
    exit(main())
