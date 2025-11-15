#!/usr/bin/env python3
"""
Test script for dataset preparation scripts.
Creates a mock dataset structure and tests the preparation pipeline.
"""

import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


def create_mock_image(path, size=(100, 100), color=(255, 0, 0)):
    """Create a simple test image."""
    img = Image.new('RGB', size, color)
    img.save(path)


def create_mock_set_dataset(root_dir, num_sets=2, images_per_set=5):
    """
    Create a mock dataset with set-based structure.

    Args:
        root_dir: Root directory for the mock dataset
        num_sets: Number of set directories to create
        images_per_set: Number of image pairs per set
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    for set_idx in range(1, num_sets + 1):
        set_dir = root / f"set{set_idx:02d}"
        input_dir = set_dir / "input"
        output_dir = set_dir / "output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(images_per_set):
            # Create input image (blue)
            input_path = input_dir / f"image_{img_idx:03d}.tif"
            create_mock_image(input_path, color=(0, 0, 255))

            # Create output image (green)
            output_path = output_dir / f"image_{img_idx:03d}.jpg"
            create_mock_image(output_path, color=(0, 255, 0))

    print(f"✓ Created mock dataset in {root}")
    print(f"  Sets: {num_sets}")
    print(f"  Images per set: {images_per_set}")
    print(f"  Total pairs: {num_sets * images_per_set}")


def test_prepare_script():
    """Test the prepare_hf_set_dataset.py script."""
    print("\n" + "=" * 80)
    print("Testing dataset preparation script")
    print("=" * 80)

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "mock_dataset"
        output_dir = Path(tmpdir) / "prepared_dataset"

        # Create mock dataset
        print("\n1. Creating mock dataset...")
        create_mock_set_dataset(dataset_dir, num_sets=3, images_per_set=10)

        # Test the preparation script
        print("\n2. Testing prepare_hf_set_dataset.py...")

        import sys
        import importlib.util

        # Import the prepare script
        spec = importlib.util.spec_from_file_location(
            "prepare_huggingface_dataset",
            "prepare_huggingface_dataset.py"
        )
        prepare_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare_module)

        # Run the preparation
        try:
            prepare_module.prepare_dataset(
                source_dirs=[str(dataset_dir)],
                output_dir=str(output_dir),
                symlink=False,
                split_ratio=0.8
            )

            # Verify output
            print("\n3. Verifying output structure...")

            input_dir = output_dir / "input"
            target_dir = output_dir / "target"
            split_file = output_dir / "split.txt"

            assert input_dir.exists(), "Input directory not created"
            assert target_dir.exists(), "Target directory not created"
            assert split_file.exists(), "Split file not created"

            input_files = list(input_dir.glob("img_*.tif"))
            target_files = list(target_dir.glob("img_*.jpg"))

            print(f"  ✓ Input directory: {len(input_files)} files")
            print(f"  ✓ Target directory: {len(target_files)} files")
            print(f"  ✓ Split file exists")

            assert len(input_files) == 30, f"Expected 30 input files, got {len(input_files)}"
            assert len(target_files) == 30, f"Expected 30 target files, got {len(target_files)}"

            # Check split file
            with open(split_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 4, "Split file should have 4 lines"

            print("\n" + "=" * 80)
            print("✓ All tests passed!")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_single_set():
    """Test processing a single set directory."""
    print("\n" + "=" * 80)
    print("Testing single set processing")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "mock_dataset"
        output_dir = Path(tmpdir) / "prepared_dataset"

        # Create mock dataset
        print("\n1. Creating mock dataset...")
        create_mock_set_dataset(dataset_dir, num_sets=3, images_per_set=10)

        # Test processing single set
        print("\n2. Testing single set processing...")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prepare_huggingface_dataset",
            "prepare_huggingface_dataset.py"
        )
        prepare_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare_module)

        try:
            # Process only set01
            prepare_module.prepare_dataset(
                source_dirs=[str(dataset_dir / "set01")],
                output_dir=str(output_dir),
                symlink=False,
                split_ratio=0.8
            )

            input_files = list((output_dir / "input").glob("img_*.tif"))

            print(f"\n3. Verification:")
            print(f"  ✓ Processed single set successfully")
            print(f"  ✓ Found {len(input_files)} files (expected 10)")

            assert len(input_files) == 10, f"Expected 10 files, got {len(input_files)}"

            print("\n" + "=" * 80)
            print("✓ Single set test passed!")
            print("=" * 80)

            return True

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("Dataset Preparation Script Tests")
    print("=" * 80)

    # Run tests
    test1_passed = test_prepare_script()
    test2_passed = test_single_set()

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Multi-set processing: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Single-set processing: {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Some tests failed")
        exit(1)
