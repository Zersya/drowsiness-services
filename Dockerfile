FROM python:3.11-slim

# Using Python 3.11 for better package compatibility with landmark detection libraries

# Install system dependencies for landmark detection
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

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install additional dependencies for landmark detection first
RUN pip install --no-cache-dir opencv-python-headless

# Install Python dependencies for landmark system
RUN pip install --no-cache-dir -r requirements.txt

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
ENV LANDMARK_PORT=8003
ENV LANDMARK_HOST=0.0.0.0

# Landmark system specific settings
ENV LANDMARK_MAX_WORKERS=1
ENV LANDMARK_QUEUE_CHECK_INTERVAL=5
ENV LANDMARK_DB_PATH=/app/data/landmark_detection.db

# Landmark detection thresholds
ENV LANDMARK_FRAME_SKIP=2
ENV LANDMARK_EAR_THRESHOLD=0.25
ENV LANDMARK_PERCLOS_THRESHOLD=0.30
ENV LANDMARK_FATIGUE_THRESHOLD=0.60
ENV LANDMARK_PERCLOS_WINDOW_SECONDS=1.5

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
CMD ["python", "start_landmark_system.py", "--port", "8003", "--workers", "1"]
