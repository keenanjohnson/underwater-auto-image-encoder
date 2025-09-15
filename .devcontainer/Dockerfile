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


# Install GoPro GPR tools (using fork with MSVC and byte order fixes)
RUN git clone https://github.com/keenanjohnson/gpr_tools.git /tmp/gpr \
    && cd /tmp/gpr \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && cp source/app/gpr_tools/gpr_tools /usr/local/bin/ \
    && chmod +x /usr/local/bin/gpr_tools \
    && rm -rf /tmp/gpr \
    && which gpr_tools && (gpr_tools -h 2>&1 | head -5 || echo "gpr_tools installed")

# Install Python packages
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Create workspace directory
WORKDIR /workspace

# Switch to non-root user
USER vscode