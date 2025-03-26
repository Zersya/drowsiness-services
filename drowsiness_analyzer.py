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
    Revised V2: Rate-based drowsiness analysis using a scoring system with
    overrides for extreme conditions and adjusted damping.
    """

    def __init__(self,
                 # --- Basic Thresholds (Moderate Starting Point) ---
                 perclos_threshold=12.0,          # % time eyes closed
                 max_closure_duration_threshold=0.3, # Max duration eyes closed (seconds)
                 yawn_rate_threshold=3.0,         # Yawns per minute

                 # --- Extreme Thresholds for Overrides ---
                 extreme_perclos_threshold=35.0,     # % PERCLOS indicating definite drowsiness
                 extreme_duration_threshold=1.0,     # Eye closure duration (sec) indicating definite drowsiness
                 extreme_yawn_rate_threshold=15.0,   # Yawn rate indicating definite drowsiness

                 # --- Score Calculation Parameters ---
                 perclos_scale=2.0,
                 duration_scale=2.5, # Slightly higher scale for duration
                 yawn_rate_scale=1.5,
                 score_cap=3.0, # Increased cap from 2.0

                 # --- Weights (Balanced) ---
                 eye_metric_weight=0.55,
                 yawn_metric_weight=0.45,

                 # --- Normal State Damping ---
                 normal_state_damping_factor=0.6, # Moderate damping factor
                 # Raw score above which damping effect is reduced
                 high_raw_score_threshold=1.2, 
                 damping_reduction_factor=0.5, # Reduce damping by this factor if raw score is high

                 # --- Decision Making ---
                 drowsiness_decision_threshold=0.45, # Moderate decision threshold

                 # --- Minimum requirements ---
                 minimum_frames_for_analysis=30, # ~1.5-2 seconds of data needed
                 fps=20                          # Default FPS
                 ):
        """
        Initialize V2 analyzer with balanced parameters, overrides, and adjusted damping.
        """
        # Store all parameters
        self.perclos_threshold = perclos_threshold
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.yawn_rate_threshold = yawn_rate_threshold
        self.extreme_perclos_threshold = extreme_perclos_threshold
        self.extreme_duration_threshold = extreme_duration_threshold
        self.extreme_yawn_rate_threshold = extreme_yawn_rate_threshold
        self.perclos_scale = perclos_scale
        self.duration_scale = duration_scale
        self.yawn_rate_scale = yawn_rate_scale
        self.score_cap = score_cap
        self.eye_metric_weight = eye_metric_weight
        self.yawn_metric_weight = yawn_metric_weight
        self.normal_state_damping_factor = normal_state_damping_factor
        self.high_raw_score_threshold = high_raw_score_threshold
        self.damping_reduction_factor = damping_reduction_factor
        self.drowsiness_decision_threshold = drowsiness_decision_threshold
        self.minimum_frames_for_analysis = minimum_frames_for_analysis
        self.default_fps = fps

        # Basic validation
        assert 0 <= eye_metric_weight <= 1
        assert 0 <= yawn_metric_weight <= 1
        # Weights don't strictly need to sum to 1 anymore with this scoring approach

        # Ensure logging is configured (ideally done once in the main application)
        # logging.basicConfig(level=logging.INFO) 
        # logging.getLogger().setLevel(logging.INFO) # Ensure root logger level is appropriate

    def _calculate_metric_score(self, value, threshold, scale):
        """Calculates a score based on how much a value exceeds a threshold, with capping."""
        if value > threshold and threshold > 0: # Avoid division by zero
            score = ((value - threshold) / threshold) * scale
            # Apply the configurable cap
            return min(score, self.score_cap) 
        return 0.0

    def analyze(self, detection_results):
        """
        Analyzes detection results using V2 logic with overrides and adjusted damping.
        """
        # --- 1. Extract Data & Basic Checks ---
        yawn_count = detection_results.get('yawn_count', 0)
        # Use the total count of eye closed *detections* if available and meaningful,
        # otherwise keep using frame-based metrics. Assuming 'eye_closed_frames' is the detection count.
        eye_closed_detection_count = detection_results.get('eye_closed_frames', 0) 
        total_eye_closed_frames = detection_results.get('total_eye_closed_frames', 0)
        max_consecutive_eye_closed = detection_results.get('max_consecutive_eye_closed', 0)
        normal_state_frames = detection_results.get('normal_state_frames', 0)
        total_frames = detection_results.get('total_frames', 0)
        fps = detection_results.get('metrics', {}).get('fps', self.default_fps)

        if fps <= 0:
            logging.warning(f"Invalid or missing FPS ({fps}). Using default FPS: {self.default_fps}")
            fps = self.default_fps

        if total_frames < self.minimum_frames_for_analysis:
            logging.info(f"Insufficient frames ({total_frames} < {self.minimum_frames_for_analysis}) for analysis.")
            return {'is_drowsy': None, 'confidence': 0.0, 'details': {'reason': 'insufficient_frames', 'total_frames': total_frames}}

        # Check if any relevant detections occurred - use frame counts here
        if total_eye_closed_frames == 0 and yawn_count == 0 and normal_state_frames == 0:
             logging.info("No relevant detections (closed eyes, yawns, normal state) found in frames.")
             return {'is_drowsy': False, 'confidence': 0.0, 'details': {'reason': 'no_detection', 'total_frames': total_frames}}

        # --- 2. Calculate Primary Metrics ---
        time_in_seconds = total_frames / fps
        time_in_minutes = time_in_seconds / 60 if time_in_seconds > 0 else 0

        perclos = (total_eye_closed_frames / total_frames) * 100 if total_frames > 0 else 0
        # Duration in seconds
        max_closure_duration = max_consecutive_eye_closed / fps if fps > 0 else 0 
        yawn_rate_per_minute = yawn_count / time_in_minutes if time_in_minutes > 0 else yawn_count * 60 # Estimate if time is very short
        normal_state_percentage = (normal_state_frames / total_frames) * 100 if total_frames > 0 else 0

        logging.info(f"V2 Metrics: PERCLOS={perclos:.2f}%, Max Closure={max_closure_duration:.2f}s, "
                     f"Yawn Rate={yawn_rate_per_minute:.2f}/min, Normal State={normal_state_percentage:.2f}%")

        # --- 3. Check for Extreme Overrides ---
        final_score = 0.0 # Initialize score
        is_drowsy = False
        reason = "checking_extremes"

        if perclos >= self.extreme_perclos_threshold:
            is_drowsy = True
            # Assign a high score, bypassing normal calculation and damping
            final_score = self.score_cap # Use max possible score
            reason = f"extreme_perclos (>{self.extreme_perclos_threshold}%)"
        elif max_closure_duration >= self.extreme_duration_threshold:
            is_drowsy = True
            final_score = self.score_cap 
            reason = f"extreme_duration (>{self.extreme_duration_threshold}s)"
        elif yawn_rate_per_minute >= self.extreme_yawn_rate_threshold:
             is_drowsy = True
             final_score = self.score_cap
             reason = f"extreme_yawn_rate (>{self.extreme_yawn_rate_threshold}/min)"

        # If an extreme override triggered, format and return result early
        if is_drowsy and reason != "checking_extremes":
            logging.info(f"Extreme Drowsiness Override Triggered: {reason}")
            details = self._create_details_dict(perclos, max_closure_duration, yawn_rate_per_minute, normal_state_percentage, 
                                                0, 0, 0, 0, 0, 0, # Scores/damping not applicable here
                                                reason, yawn_count, eye_closed_detection_count, total_eye_closed_frames, 
                                                max_consecutive_eye_closed, normal_state_frames, total_frames, fps)
            return {'is_drowsy': True, 'confidence': final_score, 'details': details}

        # --- 4. Calculate Individual Scores (If no extreme override) ---
        perclos_score = self._calculate_metric_score(perclos, self.perclos_threshold, self.perclos_scale)
        duration_score = self._calculate_metric_score(max_closure_duration, self.max_closure_duration_threshold, self.duration_scale)
        yawn_score = self._calculate_metric_score(yawn_rate_per_minute, self.yawn_rate_threshold, self.yawn_rate_scale)

        # Combine eye scores - using max focuses on the stronger signal
        combined_eye_score = max(perclos_score, duration_score)
        # Alternative: Average: (perclos_score + duration_score) / 2

        # --- 5. Calculate Raw Drowsiness Score ---
        raw_drowsiness_score = (combined_eye_score * self.eye_metric_weight +
                                yawn_score * self.yawn_metric_weight)

        # --- 6. Apply Adjusted Normal State Damping ---
        effective_damping_factor = self.normal_state_damping_factor
        # Reduce damping if raw score is already high
        if raw_drowsiness_score >= self.high_raw_score_threshold:
            effective_damping_factor *= self.damping_reduction_factor # e.g., 0.6 * 0.5 = 0.3
            logging.info(f"Raw score ({raw_drowsiness_score:.2f}) >= high threshold ({self.high_raw_score_threshold}), reducing damping factor to {effective_damping_factor:.2f}")

        damping_amount = (normal_state_percentage / 100.0) * effective_damping_factor
        # Ensure damping doesn't exceed 1.0 (100%)
        damping_amount = min(damping_amount, 1.0) 
        
        final_score = raw_drowsiness_score * (1.0 - damping_amount)
        final_score = max(0.0, final_score) # Ensure score doesn't go below zero

        # --- 7. Make Final Drowsiness Decision ---
        is_drowsy = final_score >= self.drowsiness_decision_threshold

        # --- 8. Determine Reason ---
        if is_drowsy:
            if combined_eye_score * self.eye_metric_weight > yawn_score * self.yawn_metric_weight + 0.1: # Add slight bias if equal
                 reason = f"eye_metrics_dominant (Score: {final_score:.2f})"
            elif yawn_score * self.yawn_metric_weight > combined_eye_score * self.eye_metric_weight + 0.1:
                 reason = f"yawn_metrics_dominant (Score: {final_score:.2f})"
            else: # Scores are close or only one is present
                 reason = f"threshold_met (Score: {final_score:.2f})"
        elif final_score > 0: # Score is positive but below threshold
            reason = f"indicators_present_below_threshold (Score: {final_score:.2f})"
        else: # Score is 0 (and no extreme overrides)
             reason = "no_significant_indicators"


        # --- 9. Format Results ---
        details = self._create_details_dict(perclos, max_closure_duration, yawn_rate_per_minute, normal_state_percentage, 
                                            perclos_score, duration_score, yawn_score, combined_eye_score, 
                                            raw_drowsiness_score, damping_amount, reason, 
                                            yawn_count, eye_closed_detection_count, total_eye_closed_frames, 
                                            max_consecutive_eye_closed, normal_state_frames, total_frames, fps)
        
        result = {
            'is_drowsy': is_drowsy,
            'confidence': final_score, # Use 'confidence' key as per user's last code
            'details': details
        }

        logging.info(f"V2 Analysis result: is_drowsy={result['is_drowsy']}, score={result['confidence']:.3f}, reason={result['details']['reason']}")
        return result

    def _create_details_dict(self, perclos, duration, yawn_rate, normal_perc,
                             p_score, dur_score, y_score, eye_score, raw_score, damping,
                             reason, yawn_cnt, eye_closed_det_cnt, tot_eye_frames, max_consec, 
                             norm_frames, tot_frames, fps):
        """Helper function to create the details dictionary."""
        return {
                'perclos_%': perclos,
                'max_closure_duration_s': duration,
                'yawn_rate_per_min': yawn_rate,
                'normal_state_%': normal_perc,
                'perclos_score': p_score,
                'duration_score': dur_score,
                'yawn_score': y_score,
                'combined_eye_score': eye_score,
                'raw_drowsiness_score': raw_score,
                'applied_damping_factor': damping, # This is the calculated damping amount (0 to 1)
                'reason': reason,
                # Raw Inputs
                'yawn_count': yawn_cnt,
                 # Include the detection count if you added it back to YoloProcessor
                'eye_closed_detection_count': eye_closed_det_cnt, 
                'total_eye_closed_frames': tot_eye_frames,
                'max_consecutive_eye_closed': max_consec,
                'normal_state_frames': norm_frames,
                'total_frames': tot_frames,
                'fps': fps
            }

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
