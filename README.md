# Drowsiness Detection Service

A service that analyzes video feeds to detect driver drowsiness using computer vision and customizable analysis algorithms.

## Requirements

- Python 3.12 or higher
- pip package manager
- Virtual environment (recommended)

## How to use

1. Clone the repository: `git clone https://github.com/yourusername/drowsiness-detection-service.git`
2. Navigate to the project directory: `cd drowsiness-detection-service`
3. Install the required packages: `pip install -r requirements.txt`
4. Set up your environment variables in a `.env` file. You can use the provided `.env.example` as a template.
5. Run the service: `python drowsiness_detector.py`
6. Start the web server: `python web_server.py`
7. Access the dashboard at `http://localhost:5000`

## Environment Variables

Create a `.env` file with the following variables:

```env
BASE_URL=https://your-api-endpoint.com
API_ENDPOINT=https://your-api-endpoint.com/api/v1
API_TOKEN=your_api_token
USERNAME=your_username
PASSWORD=your_password
YOLO_MODEL_PATH=path/to/your/model.pt
FETCH_INTERVAL_SECONDS=300
DROWSINESS_THRESHOLD_YAWN=3
DROWSINESS_THRESHOLD_EYE_CLOSED=5
```

## Customizing the Drowsiness Analyzer

The service uses a modular approach for drowsiness analysis, allowing you to implement and use different analysis algorithms.

### Available Analyzers

1. **ThresholdBasedAnalyzer** (Default)
   - Uses simple thresholds for yawns and closed eyes
   - Configuration parameters:
     - `yawn_threshold`: Number of yawns that indicates drowsiness
     - `eye_closed_threshold`: Number of frames with closed eyes that indicates drowsiness

### Creating a Custom Analyzer

1. Create a new class in `drowsiness_analyzer.py` that inherits from `DrowsinessAnalyzer`:

```python
from drowsiness_analyzer import DrowsinessAnalyzer

class CustomAnalyzer(DrowsinessAnalyzer):
    def __init__(self, custom_param1, custom_param2):
        self.param1 = custom_param1
        self.param2 = custom_param2
    
    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        # Implement your custom analysis logic here
        return {
            'is_drowsy': bool,  # Required
            'confidence': float,  # Required (0-1)
            'details': dict  # Required (custom details)
        }
```

2. Register your analyzer in the `create_analyzer` factory function:

```python
def create_analyzer(analyzer_type="threshold", **kwargs):
    if analyzer_type == "threshold":
        return ThresholdBasedAnalyzer(
            kwargs.get('yawn_threshold', 3),
            kwargs.get('eye_closed_threshold', 5)
        )
    elif analyzer_type == "custom":  # Add your analyzer type
        return CustomAnalyzer(
            kwargs.get('custom_param1'),
            kwargs.get('custom_param2')
        )
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
```

3. Use your custom analyzer in the detector:

```python
# In drowsiness_detector.py
analyzer = create_analyzer(
    analyzer_type="custom",
    custom_param1=value1,
    custom_param2=value2
)
```

### Analyzer Interface

Your custom analyzer must implement the `analyze` method with the following signature:

```python
def analyze(self, yawn_count, eye_closed_frames, total_frames):
    """
    Analyze drowsiness based on detection metrics.
    
    Args:
        yawn_count (int): Number of yawns detected
        eye_closed_frames (int): Number of frames with closed eyes
        total_frames (int): Total number of frames processed
        
    Returns:
        dict: Analysis results containing:
            - is_drowsy (bool): Whether drowsiness was detected
            - confidence (float): Confidence score (0-1)
            - details (dict): Additional detection details
    """
```

### Example: Machine Learning Based Analyzer

```python
class MLBasedAnalyzer(DrowsinessAnalyzer):
    def __init__(self, model_path):
        self.model = load_ml_model(model_path)
        
    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        features = self._extract_features(yawn_count, eye_closed_frames, total_frames)
        prediction = self.model.predict(features)
        
        return {
            'is_drowsy': prediction > 0.5,
            'confidence': float(prediction),
            'details': {
                'yawn_count': yawn_count,
                'eye_closed_frames': eye_closed_frames,
                'ml_features': features.tolist()
            }
        }
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-analyzer`)
3. Commit your changes (`git commit -m 'Add amazing analyzer'`)
4. Push to the branch (`git push origin feature/amazing-analyzer`)
5. Open a Pull Request
