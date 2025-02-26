import logging
from abc import ABC, abstractmethod

class DrowsinessAnalyzer(ABC):
    """Abstract base class for drowsiness analysis implementations."""
    
    @abstractmethod
    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        """
        Analyze drowsiness based on detection metrics.
        
        Args:
            yawn_count (int): Number of yawns detected
            eye_closed_frames (int): Number of frames with closed eyes
            total_frames (int): Total number of frames processed
            
        Returns:
            dict: Analysis results containing at least:
                - is_drowsy (bool): Whether drowsiness was detected
                - confidence (float): Confidence score of the detection (0-1)
                - details (dict): Additional detection details
        """
        pass

class ThresholdBasedAnalyzer(DrowsinessAnalyzer):
    """Simple threshold-based drowsiness analysis."""
    
    def __init__(self, yawn_threshold, eye_closed_threshold):
        self.yawn_threshold = yawn_threshold
        self.eye_closed_threshold = eye_closed_threshold
        
    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        logging.info(f"Analyzing drowsiness: Yawns={yawn_count}, Eye Closed Frames={eye_closed_frames}, Total Frames={total_frames}")
        
        # Calculate basic metrics
        eye_closed_ratio = eye_closed_frames / total_frames if total_frames > 0 else 0
        yawn_rate = yawn_count / (total_frames / 30) if total_frames > 0 else 0  # Assuming 30 fps
        
        # Check thresholds
        is_drowsy_yawn = yawn_count > self.yawn_threshold
        is_drowsy_eyes = eye_closed_frames > self.eye_closed_threshold
        
        # Calculate confidence score (0-1)
        yawn_confidence = min(yawn_count / (self.yawn_threshold * 2), 1.0)
        eye_confidence = min(eye_closed_frames / (self.eye_closed_threshold * 2), 1.0)
        confidence = max(yawn_confidence, eye_confidence)
        
        return {
            'is_drowsy': is_drowsy_yawn or is_drowsy_eyes,
            'confidence': confidence,
            'details': {
                'yawn_count': yawn_count,
                'eye_closed_frames': eye_closed_frames,
                'eye_closed_ratio': eye_closed_ratio,
                'yawn_rate': yawn_rate,
                'yawn_threshold_exceeded': is_drowsy_yawn,
                'eye_threshold_exceeded': is_drowsy_eyes
            }
        }
        
class RateBasedAnalyzer(DrowsinessAnalyzer):
    """Rate-based drowsiness analysis using eye closure percentage and yawn frequency."""
    
    def __init__(self, eye_closed_percentage_threshold=90, yawn_rate_threshold=25, fps=30):
        """
        Initialize the analyzer with thresholds and frame rate.
        
        Args:
            eye_closed_percentage_threshold (float): Percentage of time eyes closed to indicate drowsiness
            yawn_rate_threshold (float): Yawns per minute to indicate drowsiness
            fps (int): Frames per second for time calculations
        """
        self.eye_closed_percentage_threshold = eye_closed_percentage_threshold
        self.yawn_rate_threshold = yawn_rate_threshold
        self.fps = fps

    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        """
        Analyze drowsiness based on yawn rate and eye closure percentage.
        
        Args:
            yawn_count (int): Number of yawns detected
            eye_closed_frames (int): Number of frames with closed eyes
            total_frames (int): Total number of frames processed
            
        Returns:
            dict: Analysis results with is_drowsy, confidence, and details
        """
        logging.info(f"Analyzing drowsiness: Yawns={yawn_count}, Eye Closed Frames={eye_closed_frames}, Total Frames={total_frames}")
        
        # Handle edge case where no frames are processed
        if total_frames == 0:
            return {
                'is_drowsy': False,
                'confidence': 0.0,
                'details': {
                    'eye_closed_percentage': 0.0,
                    'yawn_rate_per_minute': 0.0
                }
            }

        # Calculate time-based metrics
        time_in_seconds = total_frames / self.fps
        time_in_minutes = time_in_seconds / 60
        eye_closed_percentage = (eye_closed_frames / total_frames) * 100
        yawn_rate_per_minute = yawn_count / time_in_minutes if time_in_minutes > 0 else 0

        # Determine drowsiness
        is_drowsy_eyes = eye_closed_percentage > self.eye_closed_percentage_threshold
        is_drowsy_yawns = yawn_rate_per_minute > self.yawn_rate_threshold
        is_drowsy = is_drowsy_eyes or is_drowsy_yawns

        # Calculate confidence scores
        eye_confidence = min(eye_closed_percentage / self.eye_closed_percentage_threshold, 1.0)
        yawn_confidence = min(yawn_rate_per_minute / self.yawn_rate_threshold, 1.0)
        confidence = max(eye_confidence, yawn_confidence)

        # Return results
        return {
            'is_drowsy': is_drowsy,
            'confidence': confidence,
            'details': {
                'eye_closed_percentage': eye_closed_percentage,
                'yawn_rate_per_minute': yawn_rate_per_minute,
                'is_drowsy_eyes': is_drowsy_eyes,
                'is_drowsy_yawns': is_drowsy_yawns
            }
        }
        
# Factory function to create analyzers
def create_analyzer(analyzer_type="threshold", **kwargs):
    """
    Create and return a drowsiness analyzer instance.
    
    Args:
        analyzer_type (str): Type of analyzer to create ("threshold" or "rate")
        **kwargs: Configuration parameters for the analyzer
        
    Returns:
        DrowsinessAnalyzer: An instance of the requested analyzer
    """
    if analyzer_type == "threshold":
        yawn_threshold = kwargs.get('yawn_threshold', 3)
        eye_closed_threshold = kwargs.get('eye_closed_threshold', 5)
        return ThresholdBasedAnalyzer(yawn_threshold, eye_closed_threshold)
    elif analyzer_type == "rate":
        eye_closed_percentage_threshold = kwargs.get('eye_closed_percentage_threshold', 20)
        yawn_rate_threshold = kwargs.get('yawn_rate_threshold', 3)
        fps = kwargs.get('fps', 30)
        return RateBasedAnalyzer(eye_closed_percentage_threshold, yawn_rate_threshold, fps)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")