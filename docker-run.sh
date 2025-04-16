# Build the Docker image
docker build -t simplify-drowsiness-detector .

# Run the container with GPU support
docker run -d -p 8002:8002 \
  --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --network-opt "proxy.external=true" \
  --name simplify-drowsiness-detector \
  simplify-drowsiness-detector