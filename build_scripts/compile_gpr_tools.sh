#!/bin/bash
# Compile gpr_tools for current platform

set -e

echo "Compiling gpr_tools for $(uname -s)..."

# Clone gpr repository if not exists
if [ ! -d "temp/gpr" ]; then
    echo "Cloning GPR repository..."
    mkdir -p temp
    cd temp
    git clone https://github.com/gopro/gpr.git
    cd ..
fi

cd temp/gpr

# Update to latest
git pull

# Build based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Building for macOS..."
    # macOS
    mkdir -p build
    cd build
    cmake ..
    make
    
    # Copy binary
    mkdir -p ../../../binaries/darwin
    cp gpr_tools ../../../binaries/darwin/
    echo "✓ Binary copied to binaries/darwin/gpr_tools"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Building for Linux..."
    # Linux
    mkdir -p build
    cd build
    cmake ..
    make
    
    # Copy binary
    mkdir -p ../../../binaries/linux
    cp gpr_tools ../../../binaries/linux/
    echo "✓ Binary copied to binaries/linux/gpr_tools"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

cd ../../..
echo "✓ gpr_tools compiled successfully"