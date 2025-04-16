#!/bin/bash

echo "Checking CUDA version..."
echo "Container is configured for CUDA 12.1, compatible with host CUDA 12.4"

nvcc --version || echo "NVCC not found, but container will still run with CPU support"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"

echo "Starting application with ThreadPool support (MAX_WORKERS=$MAX_WORKERS)..."

# Function to handle signals and pass them to the Python process
function handle_signal() {
    echo "Received signal $1, forwarding to Python process..."
    kill -$1 $PYTHON_PID
    wait $PYTHON_PID
}

# Set up signal handlers
trap 'handle_signal SIGTERM' SIGTERM
trap 'handle_signal SIGINT' SIGINT

# Start Python process and store its PID
python simplify.py &
PYTHON_PID=$!

# Wait for the Python process to finish
wait $PYTHON_PID
