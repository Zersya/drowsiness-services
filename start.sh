#!/bin/bash

echo "Checking CUDA version..."
echo "Container is configured for CUDA 12.1, compatible with host CUDA 12.4"

nvcc --version || echo "NVCC not found, but container will still run with CPU support"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"

# Ensure data directories exist and have the right permissions
mkdir -p /app/data
mkdir -p /app/logs
chmod -R 777 /app/data
chmod -R 777 /app/logs
echo "Data directories created and permissions set"

echo "Starting application with ThreadPool support (MAX_WORKERS=$MAX_WORKERS)..."

# Set environment variables to optimize for worker threads
export PYTHONTHREADDEBUG=1
export PYTHONFAULTHANDLER=1

# Function to handle signals and pass them to the Python process
function handle_signal() {
    echo "Received signal $1, forwarding to Python process..."
    kill -$1 $PYTHON_PID
    # Wait for the process to finish with a timeout
    for i in {1..30}; do
        if ! kill -0 $PYTHON_PID 2>/dev/null; then
            echo "Process has exited"
            break
        fi
        echo "Waiting for process to exit ($i/30)..."
        sleep 1
    done
    # Force kill if still running
    if kill -0 $PYTHON_PID 2>/dev/null; then
        echo "Process did not exit gracefully, force killing..."
        kill -9 $PYTHON_PID
    fi
}

# Set up signal handlers
trap 'handle_signal SIGTERM' SIGTERM
trap 'handle_signal SIGINT' SIGINT

# Start Python process and store its PID
echo "Starting Python process with worker threads..."
python simplify.py &
PYTHON_PID=$!
echo "Python process started with PID: $PYTHON_PID"

# Wait for the Python process to finish
wait $PYTHON_PID
