#!/bin/bash
# Installation script for gpr_tools (GoPro's GPR SDK)
# This provides lossless GPR to DNG conversion

set -e

echo "GPR Tools Installation Script"
echo "=============================="
echo

# Detect OS
OS="unknown"
ARCH="unknown"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    ARCH=$(uname -m)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    ARCH=$(uname -m)
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected: $OS ($ARCH)"

# Check if gpr_tools already exists
if command -v gpr_tools &> /dev/null; then
    echo "gpr_tools is already installed!"
    gpr_tools -h 2>/dev/null || true
    exit 0
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Downloading GPR SDK..."

# Download appropriate version
if [[ "$OS" == "macos" ]]; then
    # For macOS, we'll need to build from source
    echo "For macOS, you'll need to build from source:"
    echo
    echo "1. Install dependencies:"
    echo "   brew install cmake"
    echo
    echo "2. Clone and build:"
    echo "   git clone https://github.com/keenanjohnson/gpr_tools.git gpr"
    echo "   cd gpr"
    echo "   mkdir build && cd build"
    echo "   cmake .."
    echo "   make"
    echo "   sudo make install"
    echo
    echo "Alternatively, use other tools:"
    echo "   brew install dcraw exiftool"
    echo "   cargo install dnglab"
    
elif [[ "$OS" == "linux" ]]; then
    # Clone and build from source
    echo "Building from source..."
    
    # Check for required tools
    if ! command -v cmake &> /dev/null; then
        echo "cmake is required. Please install it first:"
        echo "  Ubuntu/Debian: sudo apt-get install cmake build-essential"
        echo "  Fedora: sudo dnf install cmake gcc-c++"
        exit 1
    fi
    
    git clone https://github.com/keenanjohnson/gpr_tools.git gpr
    cd gpr
    mkdir build && cd build
    cmake ..
    make
    
    echo
    echo "Build complete! To install system-wide:"
    echo "  sudo make install"
    echo
    echo "Or copy gpr_tools to your PATH:"
    echo "  cp source/app/gpr_tools/gpr_tools ~/bin/"
fi

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo
echo "For Docker containers, add this to your Dockerfile:"
echo "RUN git clone https://github.com/keenanjohnson/gpr_tools.git gpr && \\"
echo "    cd gpr && mkdir build && cd build && \\"
echo "    cmake .. && make && make install"
