# Docker Configuration for Landmark-based Drowsiness Detection System

This document describes the streamlined Docker configuration for the landmark-based drowsiness detection system, optimized for CPU-based facial landmark detection.

## Service Overview

### Landmark Drowsiness Detector
- **Container**: `landmark-drowsiness-detector`
- **Port**: 8003
- **Features**: Facial landmark-based detection, PERCLOS analysis, Eye Aspect Ratio (EAR) calculation
- **Architecture**: CPU-optimized, lightweight Python container
- **Dependencies**: dlib, OpenCV, Flask, scipy

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- 5GB free disk space

### Build and Run Service
```bash
# Build and start the landmark service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Alternative Startup Methods
```bash
# Start with custom configuration
docker-compose up -d landmark-drowsiness-detector

# Start with environment variables
LANDMARK_MAX_WORKERS=2 docker-compose up -d
```

## Configuration

### Environment Variables

#### Landmark Service Configuration
- `LANDMARK_PORT=8003` - API port
- `LANDMARK_HOST=0.0.0.0` - Bind address
- `LANDMARK_MAX_WORKERS=1` - Number of worker threads
- `LANDMARK_QUEUE_CHECK_INTERVAL=5` - Queue check interval in seconds
- `LANDMARK_DB_PATH=/app/data/landmark_detection.db` - Database path

#### Detection Thresholds
- `LANDMARK_FRAME_SKIP=2` - Process every nth frame for performance
- `LANDMARK_EAR_THRESHOLD=0.25` - Eye Aspect Ratio threshold
- `LANDMARK_PERCLOS_THRESHOLD=0.30` - PERCLOS threshold
- `LANDMARK_FATIGUE_THRESHOLD=0.60` - Overall fatigue threshold
- `LANDMARK_PERCLOS_WINDOW_SECONDS=1.5` - PERCLOS analysis window

### Volumes

#### Persistent Data
- `./logs:/app/logs` - Application logs
- `landmark_db_data:/app/data` - Landmark database
- `./data:/app/data_host` - Host data directory

## API Endpoints

### Landmark Service (Port 8003)
- `http://localhost:8003/` - API root and system information
- `http://localhost:8003/api/process` - Video processing endpoint
- `http://localhost:8003/api/queue/<id>` - Check processing status
- `http://localhost:8003/api/results` - Get all processed results
- `http://localhost:8003/api/result/<id>` - Get specific result
- `http://localhost:8003/api/webhook` - Webhook management
- `http://localhost:8003/api/download/db` - Download database
- `http://localhost:8003/api/precision` - Precision metrics

## Health Checks

The container includes health checks that verify the landmark service is running:
- Landmark service: Checks port 8003 every 30 seconds

View health status:
```bash
docker-compose ps
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check if port 8003 is in use
   netstat -tulpn | grep :8003
   ```

2. **Memory issues**
   ```bash
   # Monitor container memory usage
   docker stats
   ```

3. **dlib installation issues**
   ```bash
   # Check if dlib is properly installed
   docker exec -it landmark-drowsiness-detector python -c "import dlib; print('dlib version:', dlib.DLIB_VERSION)"
   ```

### Logs and Debugging

```bash
# View logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Access container shell
docker exec -it landmark-drowsiness-detector bash

# Check landmark system status
docker exec -it landmark-drowsiness-detector curl http://localhost:8003/
```

### Performance Tuning

#### For Multi-Core Systems
- Increase `LANDMARK_MAX_WORKERS` to 2-3 for better throughput
- Monitor CPU usage with `docker stats`

#### For Low-Memory Systems
- Keep `LANDMARK_MAX_WORKERS` at 1 for stability
- Increase `LANDMARK_FRAME_SKIP` to 3-4 to reduce processing load

## Maintenance

### Updating Services
```bash
# Rebuild containers
docker-compose build --no-cache

# Update and restart
docker-compose down
docker-compose up -d
```

### Backup Data
```bash
# Backup database
docker cp landmark-drowsiness-detector:/app/data ./backup/
```

### Clean Up
```bash
# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi landmark-drowsiness-detector

# Clean up Docker system
docker system prune -a
```

## Production Deployment

### Security Considerations
- Use environment files for sensitive configuration
- Implement proper network security
- Regular security updates

### Monitoring
- Set up log aggregation
- Monitor resource usage
- Implement alerting for service failures

### Scaling
- Use Docker Swarm or Kubernetes for multi-node deployment
- Implement load balancing for high availability
- Consider database clustering for large-scale deployments
