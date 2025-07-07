# Multi-stage build for GPU-accelerated landmark detection
ARG BUILD_TYPE=gpu

# ----------------------------------------------------------------
# Stage 1: CUDA base image for GPU support
# Includes fixes for dlib compilation and adds cuDNN
# ----------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS gpu-base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python 3.11, and NVIDIA cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # Add NVIDIA cuDNN libraries required by dlib for GPU support
    libcudnn8 \
    libcudnn8-dev \
    # System libraries for OpenCV, dlib, etc.
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-dev \
    ffmpeg \
    wget \
    curl \
    # Build tools for compiling dlib
    cmake \
    libboost-all-dev \
    build-essential \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade Python packaging tools to prevent build errors
RUN python -m pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------------
# Stage 2: CPU-only fallback image
# ----------------------------------------------------------------
FROM python:3.11-slim AS cpu-base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for landmark detection (CPU-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-dev \
    ffmpeg \
    wget \
    curl \
    cmake \
    libboost-all-dev \
    build-essential \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade Python packaging tools
RUN python -m pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------------
# Stage 3: Final image (defaults to GPU, can be overridden)
# ----------------------------------------------------------------
FROM ${BUILD_TYPE}-base AS final

# Re-declare the build argument so it's available in this stage
ARG BUILD_TYPE

# Set working directory
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install dlib in a separate step to improve build reliability
# IMPORTANT: Remember to remove 'dlib' from your requirements.txt file
RUN pip install --no-cache-dir dlib

# Install Python dependencies from requirements.txt
# The fallback logic is kept in case other GPU-specific packages fail
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "GPU packages failed, installing CPU-only versions..." && \
     pip install --no-cache-dir flask>=2.3.0 flask-cors>=4.0.0 python-dotenv>=1.0.0 requests>=2.31.0 opencv-python>=4.8.0 numpy>=1.24.0 scipy>=1.11.0)

# Verify dlib installation and check for CUDA support
RUN python -c "import dlib; print(f'dlib version: {dlib.__version__}'); print(f'dlib is using CUDA: {dlib.DLIB_USE_CUDA}')"

# Copy the rest of the project files
COPY . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Download dlib's facial landmark predictor model if it doesn't exist
RUN if [ ! -f shape_predictor_68_face_landmarks.dat ]; then \
    echo "Downloading dlib facial landmark predictor..." && \
    wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2; \
    fi

# --- Environment Variables ---
# API configuration
ENV LANDMARK_PORT=${LANDMARK_PORT:-8002}
ENV LANDMARK_HOST=0.0.0.0

# System-specific settings
ENV LANDMARK_MAX_WORKERS=${LANDMARK_MAX_WORKERS:-35}
ENV LANDMARK_QUEUE_CHECK_INTERVAL=${LANDMARK_QUEUE_CHECK_INTERVAL:-5}
ENV LANDMARK_DB_PATH=/app/data/landmark_detection.db

# Detection thresholds
ENV LANDMARK_FRAME_SKIP=${LANDMARK_FRAME_SKIP:-2}
ENV LANDMARK_EAR_THRESHOLD=${LANDMARK_EAR_THRESHOLD:-0.25}
ENV LANDMARK_PERCLOS_THRESHOLD=${LANDMARK_PERCLOS_THRESHOLD:-0.30}
ENV LANDMARK_FATIGUE_THRESHOLD=${LANDMARK_FATIGUE_THRESHOLD:-0.60}
ENV LANDMARK_PERCLOS_WINDOW_SECONDS=${LANDMARK_PERCLOS_WINDOW_SECONDS:-1.5}

# GPU acceleration settings
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ENV DISABLE_GPU=${DISABLE_GPU:-false}
ENV CUDA_DEVICE_ID=${CUDA_DEVICE_ID:-0}

# Docker best practices
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE ${LANDMARK_PORT}

# Create volumes for persistent logs and data
VOLUME ["/app/logs", "/app/data"]

# Copy and set permissions for the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Add health check for the service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${LANDMARK_PORT}/ || exit 1

# Define the default command to run the application
CMD ["python", "start_landmark_system.py", "--port", ${LANDMARK_PORT}, "--workers", ${LANDMARK_MAX_WORKERS}]