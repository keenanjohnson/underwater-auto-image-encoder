#!/usr/bin/env python3
"""
Setup script for Underwater Image Enhancer
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read version from app.py
import re
with open("app.py") as f:
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
    version = version_match.group(1) if version_match else "0.1.0"

# Read long description from README
readme_path = Path(__file__).parent / "GUI_README.md"
if readme_path.exists():
    long_description = readme_path.read_text()
else:
    long_description = "Underwater Image Enhancer - ML-powered image enhancement for marine surveys"

setup(
    name="underwater-enhancer",
    version=version,
    author="Seattle Aquarium",
    description="ML-powered underwater image enhancement application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Seattle-Aquarium/auto-image-encoder",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "customtkinter>=5.2.0",
        "darkdetect>=0.8.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "rawpy>=0.18.0",
        "imageio>=2.31.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pyinstaller>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "underwater-enhancer=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)