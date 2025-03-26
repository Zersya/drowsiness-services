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
    """
    Revised rate-based drowsiness analysis using a scoring system based on 
    eye closure percentage (PERCLOS), maximum eye closure duration, and yawn frequency.
    """

    def __init__(self, 
                 # --- Thresholds for triggering drowsiness score calculation ---
                 perclos_threshold=15.0,          # Percentage of time eyes are closed (e.g., 15%)
                 max_closure_duration_threshold=0.4, # Max duration eyes stayed closed (e.g., 0.4 seconds)
                 yawn_rate_threshold=4.0,         # Yawns per minute (e.g., 4)
                 
                 # --- Parameters for Score Calculation ---
                 # How much exceeding the threshold contributes to the score? Higher value = steeper increase
                 perclos_scale=2.0, 
                 duration_scale=2.0,
                 yawn_rate_scale=1.0,
                 
                 # --- Weights for combining scores ---
                 eye_metric_weight=0.7,          # Weight for combined eye metrics (PERCLOS, Duration)
                 yawn_metric_weight=0.3,         # Weight for yawn metric
                 
                 # --- Normal State Influence ---
                 normal_state_damping_factor=0.5, # How much normal state reduces the score (0=no effect, 1=strong effect)

                 # --- Decision Making ---
                 drowsiness_decision_threshold=0.5, # Score above which drowsiness is triggered (0.0 to 1.0+)

                 # --- Minimum requirements ---
                 minimum_frames_for_analysis=30, # Require at least ~1.5 seconds of data at 20 FPS
                 fps=20                          # Default FPS, overridden by video info
                 ):
        """
        Initialize the analyzer with configurable thresholds and scoring parameters.

        Args:
            perclos_threshold (float): Minimum eye closure percentage to start contributing to score.
            max_closure_duration_threshold (float): Minimum max eye closure duration (seconds) to start contributing to score.
            yawn_rate_threshold (float): Minimum yawns per minute to start contributing to score.
            perclos_scale (float): Scaling factor for PERCLOS score contribution.
            duration_scale (float): Scaling factor for max closure duration score contribution.
            yawn_rate_scale (float): Scaling factor for yawn rate score contribution.
            eye_metric_weight (float): Weight for eye-based scores in the final score.
            yawn_metric_weight (float): Weight for yawn-based score in the final score.
            normal_state_damping_factor (float): Factor controlling how much normal state percentage reduces the final score.
            drowsiness_decision_threshold (float): Score threshold to classify as drowsy.
            minimum_frames_for_analysis (int): Minimum total frames required for a reliable analysis.
            fps (int): Default frames per second (used if not provided in detection_results).
        """
        # Input Validation (basic)
        assert 0 <= eye_metric_weight <= 1, "Eye weight must be between 0 and 1"
        assert 0 <= yawn_metric_weight <= 1, "Yawn weight must be between 0 and 1"
        # assert abs((eye_metric_weight + yawn_metric_weight) - 1.0) < 1e-6, "Weights must sum to 1.0" 
        # --> Relaxed this constraint: weights now signify relative importance, not fractions of a whole.

        self.perclos_threshold = perclos_threshold
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.yawn_rate_threshold = yawn_rate_threshold
        
        self.perclos_scale = perclos_scale
        self.duration_scale = duration_scale
        self.yawn_rate_scale = yawn_rate_scale
        
        self.eye_metric_weight = eye_metric_weight
        self.yawn_metric_weight = yawn_metric_weight
        
        self.normal_state_damping_factor = normal_state_damping_factor
        self.drowsiness_decision_threshold = drowsiness_decision_threshold
        
        self.minimum_frames_for_analysis = minimum_frames_for_analysis
        self.default_fps = fps
        
        logging.basicConfig(level=logging.INFO) # Ensure logging is configured

    def _calculate_metric_score(self, value, threshold, scale):
        """Calculates a score based on how much a value exceeds a threshold."""
        if value > threshold:
            # Simple linear scaling above threshold for now, capped at a reasonable score (e.g., 2.0)
            # Consider sigmoid or other non-linear functions for more nuanced scoring
            score = ((value - threshold) / threshold) * scale
            return min(score, 2.0) # Cap score to prevent extreme values dominating
        return 0.0

    def analyze(self, detection_results):
        """
        Analyzes detection results to determine drowsiness state and confidence.

        Args:
            detection_results (dict): Dictionary containing detection metrics:
                'yawn_count' (int): Number of yawns detected.
                'total_eye_closed_frames' (int): Total number of frames where eyes were closed.
                'max_consecutive_eye_closed' (int): Longest sequence of consecutive eye-closed frames.
                'normal_state_frames' (int): Number of frames classified as 'normal'.
                'total_frames' (int): Total number of frames analyzed.
                'metrics' (dict): Containing 'fps' (float).

        Returns:
            dict: Analysis result:
                'is_drowsy' (bool | None): True if drowsy, False if not, None if insufficient data.
                'confidence_score' (float): A score indicating the calculated drowsiness level (0.0+).
                                          Higher values indicate stronger drowsiness signals.
                                          This replaces the previous 'confidence' which was normalized (0-1).
                'details' (dict): Detailed metrics and intermediate scores.
        """
        # --- 1. Extract Data & Basic Checks ---
        yawn_count = detection_results.get('yawn_count', 0)
        total_eye_closed_frames = detection_results.get('total_eye_closed_frames', 0)
        max_consecutive_eye_closed = detection_results.get('max_consecutive_eye_closed', 0)
        normal_state_frames = detection_results.get('normal_state_frames', 0)
        total_frames = detection_results.get('total_frames', 0)
        fps = detection_results.get('metrics', {}).get('fps', self.default_fps)

        # Ensure FPS is valid
        if fps <= 0:
            fps = self.default_fps
            logging.warning(f"Invalid or missing FPS in detection_results. Using default FPS: {fps}")

        # Check for sufficient data
        if total_frames < self.minimum_frames_for_analysis:
            logging.info(f"Insufficient frames ({total_frames} < {self.minimum_frames_for_analysis}) for analysis.")
            return {
                'is_drowsy': None, # Use None to indicate uncertainty due to lack of data
                'confidence_score': 0.0,
                'details': {'reason': 'insufficient_frames', 'total_frames': total_frames}
            }
            
        # Check if any relevant detections occurred
        if total_eye_closed_frames == 0 and yawn_count == 0 and normal_state_frames == 0:
             logging.info("No relevant detections (eyes, yawns, normal state) found.")
             return {
                'is_drowsy': False, # No signals detected, assume not drowsy
                'confidence_score': 0.0,
                'details': {'reason': 'no_detection', 'total_frames': total_frames}
            }

        # --- 2. Calculate Primary Metrics ---
        time_in_seconds = total_frames / fps
        time_in_minutes = time_in_seconds / 60

        perclos = (total_eye_closed_frames / total_frames) * 100
        max_closure_duration = max_consecutive_eye_closed / fps
        yawn_rate_per_minute = yawn_count / time_in_minutes if time_in_minutes > 0 else 0
        normal_state_percentage = (normal_state_frames / total_frames) * 100

        logging.info(f"Calculated Metrics: PERCLOS={perclos:.2f}%, Max Closure={max_closure_duration:.2f}s, "
                     f"Yawn Rate={yawn_rate_per_minute:.2f}/min, Normal State={normal_state_percentage:.2f}%")

        # --- 3. Calculate Individual Scores ---
        perclos_score = self._calculate_metric_score(perclos, self.perclos_threshold, self.perclos_scale)
        duration_score = self._calculate_metric_score(max_closure_duration, self.max_closure_duration_threshold, self.duration_scale)
        yawn_score = self._calculate_metric_score(yawn_rate_per_minute, self.yawn_rate_threshold, self.yawn_rate_scale)

        # Combine eye scores - taking the max gives prominence to the strongest indicator
        # Alternative: weighted average if both are important `(perclos_score * w1 + duration_score * w2)`
        combined_eye_score = max(perclos_score, duration_score)

        # --- 4. Calculate Raw Drowsiness Score ---
        # Weighted sum of the component scores
        raw_drowsiness_score = (combined_eye_score * self.eye_metric_weight +
                                yawn_score * self.yawn_metric_weight)

        # --- 5. Apply Normal State Damping ---
        # Reduce score based on normal state percentage. 
        # The higher the normal state %, the more the score is reduced.
        damping = (normal_state_percentage / 100) * self.normal_state_damping_factor
        final_score = raw_drowsiness_score * (1.0 - damping)
        
        # Ensure score doesn't go below zero
        final_score = max(0.0, final_score)

        # --- 6. Make Drowsiness Decision ---
        is_drowsy = final_score >= self.drowsiness_decision_threshold

        # --- 7. Determine Reason ---
        reason = "alert"
        if is_drowsy:
            if combined_eye_score >= yawn_score and combined_eye_score > 0:
                 reason = f"eye_metrics_dominant (Score: {final_score:.2f})"
            elif yawn_score > combined_eye_score and yawn_score > 0:
                 reason = f"yawn_metrics_dominant (Score: {final_score:.2f})"
            else: # Should not happen if is_drowsy is True and scores are calculated correctly
                 reason = f"threshold_met (Score: {final_score:.2f})"
        elif final_score > 0: # Score is positive but below threshold
            reason = f"indicators_present_below_threshold (Score: {final_score:.2f})"
        elif total_frames >= self.minimum_frames_for_analysis : # Score is 0 and enough frames
             reason = "no_significant_indicators"
        # else: reason remains 'alert' or handled by initial checks

        # --- 8. Format Results ---
        result = {
            'is_drowsy': is_drowsy,
            # confidence_score to reflect it's a score, not probability
            'confidence': final_score, # Renamed from 'confidence' to reflect it's a score, not probability
            'details': {
                'perclos_%': perclos,
                'max_closure_duration_s': max_closure_duration,
                'yawn_rate_per_min': yawn_rate_per_minute,
                'normal_state_%': normal_state_percentage,
                'perclos_score': perclos_score,
                'duration_score': duration_score,
                'yawn_score': yawn_score,
                'combined_eye_score': combined_eye_score,
                'raw_drowsiness_score': raw_drowsiness_score,
                'normal_state_damping': damping,
                'reason': reason,
                # Include raw inputs for debugging
                'yawn_count': yawn_count,
                'total_eye_closed_frames': total_eye_closed_frames,
                'max_consecutive_eye_closed': max_consecutive_eye_closed,
                'normal_state_frames': normal_state_frames,
                'total_frames': total_frames,
                'fps': fps
            }
        }

        logging.info(f"Analysis result: is_drowsy={result['is_drowsy']}, score={result['confidence_score']:.3f}, reason={result['details']['reason']}")
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
