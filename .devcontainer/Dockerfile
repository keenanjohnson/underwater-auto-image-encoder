FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies for image processing and ML
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    libexiv2-dev \
    libraw-dev \
    imagemagick \
    exiftool \
    cmake \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install GoPro GPR tools with byte order fix
RUN git clone https://github.com/gopro/gpr.git /tmp/gpr \
    && cd /tmp/gpr \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_CXX_FLAGS="-DqDNGBigEndian=0" \
    && make -j$(nproc) \
    && cp source/app/gpr_tools/gpr_tools /usr/local/bin/ \
    && chmod +x /usr/local/bin/gpr_tools \
    && rm -rf /tmp/gpr \
    && which gpr_tools && (gpr_tools -h 2>&1 | head -5 || echo "gpr_tools installed")

# Create workspace directory
WORKDIR /workspace

# Switch to non-root user
USER vscode