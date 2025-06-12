# Landmark-based Drowsiness Detection System

A modular drowsiness detection system based on facial landmarks and PERCLOS analysis, designed to maintain compatibility with the existing `simplify.py` architecture while using landmark-based detection instead of YOLO-based detection.

## Features

- **Facial Landmark Detection**: Uses dlib's 68-point facial landmark detector
- **PERCLOS Analysis**: Percentage of Eye Closure over time
- **Eye Aspect Ratio (EAR)**: Real-time eye closure detection
- **Blink Frequency Analysis**: Monitors blink patterns for drowsiness indicators
- **Queue-based Processing**: Handles multiple video processing requests
- **Webhook Notifications**: Real-time notifications for processing status
- **RESTful API**: Compatible with existing API interfaces
- **SQLite Database**: Persistent storage for results and queue management

## Architecture

The system is split into modular components for better maintainability:

```
landmark_api.py          # Main Flask API endpoints
landmark_database.py     # Database operations
landmark_processor.py    # Landmark-based drowsiness detection
landmark_analyzer.py     # Analysis logic for landmark results
landmark_queue.py        # Queue management
landmark_webhook.py      # Webhook dispatch
landmark_worker.py       # Worker logic for processing videos
```

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r landmark_requirements.txt
   ```

2. **Install dlib dependencies** (if not already installed):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install cmake libboost-all-dev

   # On macOS
   brew install cmake boost

   # On Windows
   # Install Visual Studio Build Tools and cmake
   ```

3. **Download the facial landmark predictor**:
   The system will automatically download `shape_predictor_68_face_landmarks.dat` on first run, or you can download it manually:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```

## Configuration

Copy `landmark_config.env` to `.env` and adjust settings as needed:

```bash
cp landmark_config.env .env
```

Key configuration options:
- `LANDMARK_PORT`: API server port (default: 8003)
- `LANDMARK_MAX_WORKERS`: Maximum concurrent video processing workers
- `EAR_THRESHOLD`: Eye Aspect Ratio threshold for closed eyes (default: 0.25)
- `PERCLOS_THRESHOLD`: PERCLOS threshold for drowsiness (default: 0.30)
- `FATIGUE_THRESHOLD`: Overall fatigue threshold (default: 0.60)

## Usage

### Starting the Server

```bash
python landmark_api.py
```

The API will be available at `http://localhost:8003`

### API Endpoints

#### Process a Video
```bash
POST /api/process
Content-Type: application/json

{
    "video_url": "https://example.com/video.mp4"
}
```

#### Check Processing Status
```bash
GET /api/queue/{queue_id}
```

#### Get Results
```bash
GET /api/result/{evidence_id}
```

#### Manage Webhooks
```bash
# List webhooks
GET /api/webhook

# Add webhook
POST /api/webhook
{
    "url": "https://your-webhook-url.com/endpoint"
}

# Delete webhook
DELETE /api/webhook
{
    "webhook_id": 1
}
```

## Detection Method

The landmark-based system uses:

1. **Facial Landmark Detection**: 68-point facial landmarks using dlib
2. **Eye Aspect Ratio (EAR)**: Calculated from eye landmark coordinates
3. **PERCLOS**: Percentage of time eyes are closed over a sliding window
4. **Blink Frequency**: Number of blinks per second
5. **Normal State Detection**: Frames where eyes are open and alert

### Drowsiness Indicators

- **High PERCLOS**: Eyes closed for more than 30% of time
- **Low EAR**: Eye aspect ratio below threshold (0.25)
- **Extended Eye Closure**: Eyes closed for more than 2 seconds
- **Low Blink Frequency**: Less than 0.5 blinks per second

## Compatibility

The system maintains API compatibility with `simplify.py`:

- Same endpoint structure and response format
- Compatible database schema
- Identical webhook payload format
- Same queue management system

## Database Schema

The system uses the same SQLite schema as `simplify.py`:

- `evidence_results`: Stores processing results
- `processing_queue`: Manages video processing queue
- `webhooks`: Stores webhook configurations

## Performance Considerations

- **Frame Skipping**: Processes every nth frame for better performance
- **Concurrent Processing**: Configurable worker threads
- **Memory Management**: Automatic cleanup of temporary files
- **Error Handling**: Robust error handling for corrupted videos

## Monitoring and Logging

- Comprehensive logging to console and file
- Processing time metrics
- Queue statistics
- Webhook delivery status
- Error tracking and reporting

## Comparison with YOLO-based System

| Feature | YOLO System | Landmark System |
|---------|-------------|-----------------|
| Detection Method | Object detection | Facial landmarks |
| Yawn Detection | ✅ | ❌ |
| Eye Closure | ✅ | ✅ |
| Head Pose | ✅ | ❌ |
| PERCLOS | ❌ | ✅ |
| Blink Analysis | Basic | Advanced |
| Model Size | Large | Small |
| Processing Speed | Moderate | Fast |
| Accuracy | High | High |

## Troubleshooting

### Common Issues

1. **dlib installation fails**:
   - Ensure cmake and boost are installed
   - Use pre-compiled wheels: `pip install dlib`

2. **Facial landmark predictor not found**:
   - The system will auto-download on first run
   - Ensure internet connectivity

3. **No face detected**:
   - Check video quality and lighting
   - Ensure face is clearly visible
   - Verify video format compatibility

4. **High memory usage**:
   - Reduce `LANDMARK_MAX_WORKERS`
   - Increase `LANDMARK_FRAME_SKIP`
   - Monitor video resolution

## Development

### Adding New Features

1. **New Detection Methods**: Extend `LandmarkDrowsinessProcessor`
2. **Custom Analyzers**: Implement `LandmarkDrowsinessAnalyzer`
3. **Additional Metrics**: Modify database schema and API responses
4. **Enhanced Webhooks**: Extend `LandmarkWebhookManager`

### Testing

```bash
# Test with a sample video
curl -X POST http://localhost:8003/api/process \
  -H "Content-Type: application/json" \
  -d '{"video_url": "path/to/test/video.mp4"}'
```

## License

This system is designed to integrate with the existing drowsiness detection infrastructure while providing landmark-based analysis capabilities.
