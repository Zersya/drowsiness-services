# Multi-stage build for GPU-accelerated landmark detection
ARG BUILD_TYPE=gpu

# Stage 1: CUDA base image for GPU support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS gpu-base

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
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

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Stage 2: CPU-only fallback image
FROM python:3.11-slim AS cpu-base

# Install system dependencies for landmark detection (CPU-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    curl \
    cmake \
    libboost-all-dev \
    build-essential \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Final image (defaults to GPU, can be overridden)
FROM ${BUILD_TYPE}-base AS final

# To use the build argument within this stage, you must redeclare it without a value.
ARG BUILD_TYPE

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install additional dependencies for landmark detection first
RUN pip install --no-cache-dir opencv-python-headless

# Install Python dependencies for landmark system
# Try GPU-enabled packages first, fallback to CPU-only if GPU not available
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "GPU packages failed, installing CPU-only versions..." && \
     pip install --no-cache-dir flask>=2.3.0 flask-cors>=4.0.0 python-dotenv>=1.0.0 requests>=2.31.0 opencv-python>=4.8.0 numpy>=1.24.0 dlib>=19.24.0 scipy>=1.11.0)

# Verify dlib installation
RUN python -c "import dlib; print('dlib version:', dlib.__version__)"

# Copy all files in project
COPY . .

# Create directories for landmark system
RUN mkdir -p logs data

# Download dlib facial landmark predictor if not already present
RUN if [ ! -f shape_predictor_68_face_landmarks.dat ]; then \
    echo "Downloading dlib facial landmark predictor..." && \
    wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2; \
    fi

# Landmark API configuration
ENV LANDMARK_PORT=${LANDMARK_PORT:-8002}
ENV LANDMARK_HOST=0.0.0.0

# Landmark system specific settings
ENV LANDMARK_MAX_WORKERS=${LANDMARK_MAX_WORKERS:-35}
ENV LANDMARK_QUEUE_CHECK_INTERVAL=${LANDMARK_QUEUE_CHECK_INTERVAL:-5}
ENV LANDMARK_DB_PATH=${LANDMARK_DB_PATH:-/app/data/landmark_detection.db}

# Landmark detection thresholds
ENV LANDMARK_FRAME_SKIP=${LANDMARK_FRAME_SKIP:-2}
ENV LANDMARK_EAR_THRESHOLD=${LANDMARK_EAR_THRESHOLD:-0.25}
ENV LANDMARK_PERCLOS_THRESHOLD=${LANDMARK_PERCLOS_THRESHOLD:-0.30}
ENV LANDMARK_FATIGUE_THRESHOLD=${LANDMARK_FATIGUE_THRESHOLD:-0.60}
ENV LANDMARK_PERCLOS_WINDOW_SECONDS=${LANDMARK_PERCLOS_WINDOW_SECONDS:-1.5}

# GPU acceleration settings
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ENV DISABLE_GPU=${DISABLE_GPU:-false}
ENV CUDA_DEVICE_ID=${CUDA_DEVICE_ID:-0}

# Ensure proper signal handling in Docker
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output is unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# Expose landmark API port
EXPOSE ${LANDMARK_PORT}

# Create volume for persistent storage
# This ensures that the database file and logs persist across container restarts
VOLUME ["/app/logs", "/app/data"]

# Set database path to the persistent volume
ENV LANDMARK_DB_PATH=/app/data/landmark_detection.db

# Create landmark-specific directories
RUN mkdir -p /app/data

# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Add health check for landmark service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${LANDMARK_PORT}/ || exit 1

# Run the landmark system directly
CMD ["python", "start_landmark_system.py", "--port", ${LANDMARK_PORT}, "--workers", ${LANDMARK_MAX_WORKERS}]