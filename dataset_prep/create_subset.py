#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import random

def create_dataset_subset(num_pairs=1000):
    # Define paths
    base_dir = Path("/workspaces/auto-image-encoder")
    input_gpr_dir = base_dir / "photos" / "input_GPR"
    output_jpeg_dir = base_dir / "photos" / "human_output_JPEG"
    dataset_dir = base_dir / "dataset"
    
    # Create dataset directory structure
    dataset_input_dir = dataset_dir / "input_GPR"
    dataset_output_dir = dataset_dir / "human_output_JPEG"
    
    # Create directories if they don't exist
    dataset_input_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all GPR files
    gpr_files = list(input_gpr_dir.glob("*.GPR"))
    print(f"Found {len(gpr_files)} GPR files")
    
    # Find matching pairs
    matching_pairs = []
    for gpr_file in gpr_files:
        # Get the base name without extension
        base_name = gpr_file.stem
        
        # Check if corresponding JPEG exists
        jpeg_file = output_jpeg_dir / f"{base_name}.jpg"
        if jpeg_file.exists():
            matching_pairs.append((gpr_file, jpeg_file))
    
    print(f"Found {len(matching_pairs)} matching pairs")
    
    # Randomly select subset
    if len(matching_pairs) < num_pairs:
        print(f"Warning: Only {len(matching_pairs)} matching pairs available, using all of them")
        selected_pairs = matching_pairs
    else:
        # Set random seed for reproducibility
        random.seed(42)
        selected_pairs = random.sample(matching_pairs, num_pairs)
    
    print(f"Copying {len(selected_pairs)} pairs to dataset directory...")
    
    # Copy selected pairs
    for i, (gpr_file, jpeg_file) in enumerate(selected_pairs, 1):
        # Copy GPR file
        gpr_dest = dataset_input_dir / gpr_file.name
        shutil.copy2(gpr_file, gpr_dest)
        
        # Copy JPEG file
        jpeg_dest = dataset_output_dir / jpeg_file.name
        shutil.copy2(jpeg_file, jpeg_dest)
        
        if i % 100 == 0:
            print(f"Copied {i}/{len(selected_pairs)} pairs...")
    
    print(f"Successfully created dataset with {len(selected_pairs)} matching pairs")
    print(f"Dataset location: {dataset_dir}")
    
    # Verify the copy
    copied_gpr = len(list(dataset_input_dir.glob("*.GPR")))
    copied_jpeg = len(list(dataset_output_dir.glob("*.jpg")))
    print(f"Verification: {copied_gpr} GPR files and {copied_jpeg} JPEG files in dataset")

if __name__ == "__main__":
    create_dataset_subset(1000)