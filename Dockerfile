# Stage 1: GPU Build
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

# Install dependencies for dlib with CUDA support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir dlib-cuda==19.22.99

# Stage 2: CPU Build
FROM python:3.11-slim as cpu

# Install dependencies for dlib on CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the appropriate build stage based on the target architecture
COPY --from=gpu /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=cpu /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Copy the application code
COPY . .

# Download dlib models
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && \
    wget http://dlib.net/files/mmod_human_face_detector.dat.bz2 && \
    bunzip2 mmod_human_face_detector.dat.bz2

# Set environment variables for landmark system
ENV ENABLE_ADAPTIVE_THRESHOLDS=true
ENV FORCE_ADAPTIVE_THRESHOLDS=false
ENV LOG_LEVEL=INFO
ENV LOG_FILE=landmark_detection.log
ENV ENABLE_GPU_ACCELERATION=false
ENV GPU_DEVICE=0
ENV LANDMARK_PORT=${LANDMARK_PORT:-8002}
ENV LANDMARK_HOST=0.0.0.0
ENV LANDMARK_MAX_WORKERS=${LANDMARK_MAX_WORKERS:-35}
ENV LANDMARK_QUEUE_CHECK_INTERVAL=${LANDMARK_QUEUE_CHECK_INTERVAL:-5}
ENV LANDMARK_DB_PATH=${LANDMARK_DB_PATH:-/app/data/landmark_detection.db}
ENV LANDMARK_FRAME_SKIP=${LANDMARK_FRAME_SKIP:-2}
ENV LANDMARK_EAR_THRESHOLD=${LANDMARK_EAR_THRESHOLD:-0.25}
ENV LANDMARK_PERCLOS_THRESHOLD=${LANDMARK_PERCLOS_THRESHOLD:-0.30}
ENV LANDMARK_FATIGUE_THRESHOLD=${LANDMARK_FATIGUE_THRESHOLD:-0.60}
ENV LANDMARK_PERCLOS_WINDOW_SECONDS=${LANDMARK_PERCLOS_WINDOW_SECONDS:-1.5}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE ${LANDMARK_PORT}

# Create volume for persistent storage
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

# Start the application
CMD ["bash", "start.sh"]