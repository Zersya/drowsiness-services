FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Note: Using CUDA 12.1 base image which is compatible with CUDA 12.4 installations
# PyTorch doesn't have an official image with CUDA 12.4 yet, but 12.1 is compatible

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be needed for simplify.py
RUN pip install --no-cache-dir opencv-python-headless

# Copy all files in project
COPY . .

# Create directories
RUN mkdir -p models logs data

# Copy model files
COPY models/jingyeong-best.pt models/

# Download YOLOv8 pose model if not already present
RUN if [ ! -f models/yolov8l-pose.pt ]; then \
    echo "Downloading YOLOv8 pose model..." && \
    python -c "from ultralytics import YOLO; YOLO('yolov8l-pose.pt')" && \
    mv yolov8l-pose.pt models/ || \
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt -O models/yolov8l-pose.pt; \
    fi

# Set environment variables
ENV YOLO_MODEL_PATH=models/jingyeong-best.pt
ENV POSE_MODEL_PATH=models/yolov8l-pose.pt
# Enable CUDA support - will use CUDA 12.4 from the host if available
ENV USE_CUDA=true
# Use first GPU
ENV CUDA_VISIBLE_DEVICES=0

# Define ports for each application
# Ensure these environment variables are used by your Python applications to bind to the correct ports
ENV WEB_SERVER_PORT=8000         
ENV SIMPLIFY_PORT=8002           

# Configure ThreadPool settings
# Number of concurrent video processing workers
ENV MAX_WORKERS=4

# Seconds between queue checks
ENV QUEUE_CHECK_INTERVAL=5

# Ensure proper signal handling in Docker
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output is unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# Expose ports
# Replace 8000 and 8001 with the actual ports your applications use
EXPOSE ${WEB_SERVER_PORT}
EXPOSE ${SIMPLIFY_PORT}
# Alternatively, you can list them on one line: EXPOSE 8000 8001 8002

# Create volume for persistent storage
# This ensures that the database file, models, and logs persist across container restarts
VOLUME ["/app/models", "/app/logs", "/app/data"]

# Set database path to the persistent volume
ENV DB_PATH=/app/data/simplify_detection.db

# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run the startup script which will then run the application with ThreadPool support
CMD ["/app/start.sh"]
