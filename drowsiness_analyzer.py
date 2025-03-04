import logging
import math
from abc import ABC, abstractmethod

class DrowsinessAnalyzer(ABC):
    """Abstract base class for drowsiness analysis implementations."""
    
    @abstractmethod
    def analyze(self, yawn_count, eye_closed_frames, normal_state_frames, total_frames):
        """
        Analyze drowsiness based on detection metrics.
        
        Args:
            yawn_count (int): Number of yawns detected
            eye_closed_frames (int): Number of frames with closed eyes
            normal_state_frames (int): Number of frames with normal state detected
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
    
    def __init__(self, eye_closed_percentage_threshold=10, yawn_rate_threshold=3, normal_state_threshold=60, fps=20, max_closure_duration_threshold=0.5):
        """
        Initialize with adjusted thresholds for better detection.
        
        Args:
            eye_closed_percentage_threshold (float): Percentage of frames with eyes closed to indicate drowsiness
            yawn_rate_threshold (float): Yawns per minute to indicate drowsiness
            normal_state_threshold (float): Percentage of normal state frames to override drowsiness
            fps (int): Frames per second (default, overridden by video FPS)
            max_closure_duration_threshold (float): Maximum eye closure duration in seconds to indicate drowsiness
        """
        self.eye_closed_percentage_threshold = eye_closed_percentage_threshold
        self.yawn_rate_threshold = yawn_rate_threshold
        self.normal_state_threshold = normal_state_threshold
        self.fps = fps
        self.max_closure_duration_threshold = max_closure_duration_threshold  # New threshold
        self.minimum_yawn_threshold = 2
        self.minimum_eye_closed_threshold = 2  # Retained but may be less relevant
        self.normal_state_ratio_threshold = 5
        self.minimum_frames_for_analysis = 10

    def analyze(self, detection_results):
        # Extract values from detection_results
        yawn_count = detection_results['yawn_count']
        eye_closed_frames = detection_results['eye_closed_frames']
        normal_state_frames = detection_results['normal_state_frames']
        total_frames = detection_results['total_frames']
        total_eye_closed_frames = detection_results.get('total_eye_closed_frames', 0)
        max_consecutive_eye_closed = detection_results.get('max_consecutive_eye_closed', 0)
        fps = detection_results['metrics']['fps']

        # Skip analysis if no detections
        if ((yawn_count == 0 or yawn_count is None) and 
            (eye_closed_frames == 0 or eye_closed_frames is None) and 
            (normal_state_frames == 0 or normal_state_frames is None)):
            return {
                'is_drowsy': None,
                'confidence': 0.0,
                'details': {
                    'eye_closed_percentage': 0.0,
                    'yawn_rate_per_minute': 0.0,
                    'normal_state_percentage': 0.0,
                    'reason': 'no_detection'
                }
            }

        logging.info(f"Analyzing drowsiness: Yawns={yawn_count}, Eye Closed Frames={eye_closed_frames}, "
                    f"Total Eye Closed Frames={total_eye_closed_frames}, Max Consecutive Closed={max_consecutive_eye_closed}, "
                    f"Normal State Frames={normal_state_frames}, Total Frames={total_frames}")

        if total_frames < self.minimum_frames_for_analysis:
            return {
                'is_drowsy': False,
                'confidence': 0.0,
                'details': {
                    'eye_closed_percentage': 0.0,
                    'yawn_rate_per_minute': 0.0,
                    'normal_state_percentage': 0.0,
                    'reason': 'insufficient_frames'
                }
            }

        # Calculate time-based metrics
        time_in_seconds = total_frames / fps if fps > 0 else 0
        time_in_minutes = time_in_seconds / 60
        
        # Calculate new eye closure metrics
        eye_closed_percentage = (total_eye_closed_frames / total_frames) * 100 if total_frames > 0 else 0
        max_closure_duration = max_consecutive_eye_closed / fps if fps > 0 else 0
        yawn_rate_per_minute = yawn_count / time_in_minutes if time_in_minutes > 0 else 0
        normal_state_percentage = (normal_state_frames / total_frames) * 100 if total_frames > 0 else 0

        # Drowsiness indicators
        is_drowsy_eyes = (eye_closed_percentage > self.eye_closed_percentage_threshold or 
                        max_closure_duration > self.max_closure_duration_threshold)
        is_drowsy_yawns = (yawn_count >= self.minimum_yawn_threshold and 
                        yawn_rate_per_minute > self.yawn_rate_threshold)
        is_drowsy_excessive_yawns = yawn_count > 10 or yawn_rate_per_minute > 100

        # Check normal state conditions
        is_normal_state_high = normal_state_percentage >= self.normal_state_threshold

        # Determine drowsiness with priority order
        is_drowsy = False
        confidence = 0.0
        reason = ''

        if is_drowsy_excessive_yawns:
            is_drowsy = True
            confidence = 1.0
            reason = 'excessive_yawns'
        elif is_drowsy_yawns and yawn_rate_per_minute > 60:
            is_drowsy = True
            confidence = 0.8
            reason = 'high_yawn_rate'
        elif is_normal_state_high and not (is_drowsy_yawns or is_drowsy_eyes):
            is_drowsy = False
            confidence = 0.1
            reason = 'high_normal_state'
        else:
            is_drowsy = is_drowsy_eyes or is_drowsy_yawns
            
            # Calculate confidence based on new metrics
            eye_percentage_confidence = min(eye_closed_percentage / self.eye_closed_percentage_threshold, 1.0) if eye_closed_percentage > self.eye_closed_percentage_threshold else 0
            eye_duration_confidence = min(max_closure_duration / self.max_closure_duration_threshold, 1.0) if max_closure_duration > self.max_closure_duration_threshold else 0
            eye_confidence = max(eye_percentage_confidence, eye_duration_confidence)
            yawn_confidence = min(yawn_count / self.minimum_yawn_threshold, 1.0) if yawn_rate_per_minute > self.yawn_rate_threshold else 0
            confidence = max(eye_confidence, yawn_confidence)
            
            # Adjust confidence based on normal state using a quadratic factor
            if normal_state_percentage > 0:
                normal_state_factor = (1 - normal_state_percentage / 100) ** 2
                confidence *= normal_state_factor
                
            reason = 'drowsy_indicators_present' if is_drowsy else 'no_significant_indicators'

            # Override drowsiness if confidence is low due to normal state
            if is_drowsy and confidence < 0.5:
                is_drowsy = False
                reason = 'low_confidence_due_to_normal_state'

        result = {
            'is_drowsy': is_drowsy,
            'confidence': confidence,
            'details': {
                'eye_closed_percentage': eye_closed_percentage,
                'max_closure_duration': max_closure_duration,
                'yawn_rate_per_minute': yawn_rate_per_minute,
                'normal_state_percentage': normal_state_percentage,
                'is_drowsy_eyes': is_drowsy_eyes,
                'is_drowsy_yawns': is_drowsy_yawns,
                'is_drowsy_excessive_yawns': is_drowsy_excessive_yawns,
                'is_normal_state_high': is_normal_state_high,
                'yawn_count': yawn_count,
                'eye_closed_frames': eye_closed_frames,
                'total_eye_closed_frames': total_eye_closed_frames,
                'max_consecutive_eye_closed': max_consecutive_eye_closed,
                'normal_state_frames': normal_state_frames,
                'reason': reason
            }
        }
        
        logging.info(f"Analysis result: {result}")
        return result
        
class ProbabilisticAnalyzer(DrowsinessAnalyzer):
    """Probabilistic drowsiness analysis using a sigmoid function."""
    
    def __init__(self, a=0.5, b=5, c=3, fps=30):
        """
        Initialize the analyzer with parameters for the sigmoid function.
        
        Args:
            a (float): Weight for yawn rate
            b (float): Weight for eye closure ratio
            c (float): Threshold parameter
            fps (int): Frames per second for time calculations
        """
        self.a = a  # Weight for yawn rate
        self.b = b  # Weight for eye closure ratio
        self.c = c  # Threshold shift for the sigmoid
        self.fps = fps  # Frames per second

    def analyze(self, yawn_count, eye_closed_frames, total_frames):
        """
        Analyze drowsiness based on yawn rate and eye closure ratio using a sigmoid function.
        
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
                    'probability': 0.0
                }
            }

        # Calculate time-based metrics
        time_in_seconds = total_frames / self.fps
        time_in_minutes = time_in_seconds / 60
        yawn_rate = yawn_count / time_in_minutes if time_in_minutes > 0 else 0
        eye_closed_ratio = eye_closed_frames / total_frames

        # Calculate the linear combination
        linear_comb = self.a * yawn_rate + self.b * eye_closed_ratio

        # Calculate the probability using sigmoid
        probability = 1 / (1 + math.exp(-(linear_comb - self.c)))

        # Determine drowsiness based on probability > 0.5
        is_drowsy = probability > 0.5

        return {
            'is_drowsy': is_drowsy,
            'confidence': probability,
            'details': {
                'yawn_rate': yawn_rate,
                'eye_closed_ratio': eye_closed_ratio,
                'probability': probability
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
        return RateBasedAnalyzer()
    elif analyzer_type == "probabilistic":
        a = kwargs.get('a', 0.5)
        b = kwargs.get('b', 0.5)
        c = kwargs.get('c', 3)
        fps = kwargs.get('fps', 20)
        return ProbabilisticAnalyzer(a, b, c, fps)
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
