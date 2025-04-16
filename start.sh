#!/bin/bash

echo "Checking CUDA version..."
echo "Container is configured for CUDA 12.1, compatible with host CUDA 12.4"

nvcc --version || echo "NVCC not found, but container will still run with CPU support"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"

echo "Starting application with ThreadPool support (MAX_WORKERS=$MAX_WORKERS)..."

exec python simplify.py
