# Drowsiness Detection Service

A service that analyzes video feeds to detect driver drowsiness using computer vision and customizable analysis algorithms.

![Dashboard](/images/dashboard.png)

## Requirements

- Python 3.12 or higher
- pip package manager
- Virtual environment (recommended)

## How to use

1. Clone the repository: `git clone https://github.com/Zersya/drowsiness-services.git`
2. Navigate to the project directory: `cd drowsiness-detection-service`
3. Install the required packages: `pip install -r requirements.txt`
4. Set up your environment variables in a `.env` file. You can use the provided `.env.example` as a template.
5. Run the service: `python drowsiness_detector.py`
6. Start the web server: `python web_server.py`
7. Access the dashboard at `http://localhost:5000`
8. Login using either PIN or Keycloak credentials (based on your configuration)

## Features

- Flexible authentication system (PIN-based or Keycloak SSO)
- Real-time drowsiness detection using YOLO-based computer vision
- Multiple analysis algorithms (Threshold-based, Rate-based, and Probabilistic)
- Performance monitoring dashboard with confusion matrix
- Support for multiple video sources and devices
- Fleet management capabilities
- Customizable drowsiness thresholds
- ML metrics tracking and analysis

## Environment Variables

Create a `.env` file with the following variables:

```env
# API Configuration
BASE_URL=you_url
API_ENDPOINT=third_party_services
API_TOKEN=your_api_token
USERNAME=your_username
PASSWORD=your_password

# Model Configuration
YOLO_MODEL_PATH=models/best.pt
USE_CUDA=false

# Detection Parameters
FETCH_INTERVAL_SECONDS=20
MIN_BLINK_FRAMES=1
BLINK_COOLDOWN=5
EYE_DETECTION_CONFIDENCE=0.5

# Authentication Configuration
AUTH_TYPE=PIN  # Options: PIN or KEYCLOAK
WEB_ACCESS_PIN=123456  # Only used if AUTH_TYPE=PIN

# Keycloak Configuration (Only used if AUTH_TYPE=KEYCLOAK)
KEYCLOAK_URL=https://your-keycloak-server/auth
KEYCLOAK_REALM=your-realm
KEYCLOAK_CLIENT_ID=your-client-id
KEYCLOAK_CLIENT_SECRET=your-client-secret
```

## Security Features

### Dashboard Authentication

The service supports two authentication methods:

#### 1. PIN-based Authentication
- Simple 6-digit PIN access
- Secure session management
- Automatic session expiration
- Protection against brute force attempts
- Configurable PIN through environment variables

To use PIN authentication:
```env
AUTH_TYPE=PIN
WEB_ACCESS_PIN=123456  # Your chosen PIN
```

#### 2. Keycloak SSO Authentication
- Enterprise-grade Single Sign-On
- Role-based access control
- Token-based authentication
- Automatic token refresh
- Secure session management

To use Keycloak authentication:
```env
AUTH_TYPE=KEYCLOAK
KEYCLOAK_URL=https://your-keycloak-server/auth
KEYCLOAK_REALM=your-realm
KEYCLOAK_CLIENT_ID=your-client-id
KEYCLOAK_CLIENT_SECRET=your-client-secret
```

### Keycloak Setup

1. Create a new client in your Keycloak realm
2. Set the client protocol to "openid-connect"
3. Configure the client settings:
   - Access Type: confidential
   - Valid Redirect URIs: http://localhost:5000/*
   - Web Origins: http://localhost:5000
4. Copy the client secret from the Credentials tab
5. Update your `.env` file with the Keycloak configuration

## Implementation Details

### Analyzer Configuration
The service uses the Rate-Based Analyzer by default, configured in `drowsiness_detector.py`:

```python
drowsiness_analyzer = create_analyzer(
    analyzer_type="rate",
    yawn_threshold=DROWSINESS_THRESHOLD_YAWN,
    eye_closed_threshold=DROWSINESS_THRESHOLD_EYE_CLOSED
)
```

This configuration uses environment variables to set the thresholds:
- `DROWSINESS_THRESHOLD_YAWN`: Number of yawns that indicate drowsiness
- `DROWSINESS_THRESHOLD_EYE_CLOSED`: Number of frames with closed eyes that indicate drowsiness

To use a different analyzer, modify the `analyzer_type` parameter to one of:
- `"threshold"` for ThresholdBasedAnalyzer
- `"rate"` for RateBasedAnalyzer
- `"probabilistic"` for ProbabilisticAnalyzer

## Available Analyzers

1. **ThresholdBasedAnalyzer** (Basic)
   - Uses simple thresholds for yawns and closed eyes
   - Configuration:
     - `yawn_threshold`: Number of yawns indicating drowsiness
     - `eye_closed_threshold`: Number of frames with closed eyes indicating drowsiness

2. **RateBasedAnalyzer** (Advanced)
   - Uses time-based analysis of eye closure and yawn frequency
   - Configuration:
     - `eye_closed_percentage_threshold`: Percentage of time eyes closed
     - `yawn_rate_threshold`: Yawns per minute
     - `fps`: Frames per second

3. **ProbabilisticAnalyzer** (Experimental)
   - Uses sigmoid function for probability-based detection
   - Configuration:
     - `a`: Weight for yawn rate
     - `b`: Weight for eye closure ratio
     - `c`: Threshold shift

## Creating a Custom Analyzer

1. Create a new class that inherits from `DrowsinessAnalyzer`:

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

2. Register your analyzer in `create_analyzer`:

```python
def create_analyzer(analyzer_type="threshold", **kwargs):
    if analyzer_type == "custom":
        return CustomAnalyzer(
            kwargs.get('custom_param1'),
            kwargs.get('custom_param2')
        )
```

## Performance Monitoring

The dashboard provides real-time monitoring of:
- Processed/Pending/Failed events
- Confusion matrix
- Accuracy and sensitivity metrics
- Device and fleet statistics
- Video playback with detection details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-analyzer`)
3. Commit your changes (`git commit -m 'Add amazing analyzer'`)
4. Push to the branch (`git push origin feature/amazing-analyzer`)
5. Open a Pull Request
