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
    Revised V3: Rate-based analysis with conditional overrides, non-linear damping,
    and averaged eye scores.
    """

    def __init__(self,
                 # --- Basic Thresholds ---
                 perclos_threshold=15.0,          # Increased slightly
                 max_closure_duration_threshold=0.4, # Increased slightly
                 yawn_rate_threshold=3.0,

                 # --- Extreme Thresholds for Conditional Overrides ---
                 extreme_perclos_threshold=45.0,     # Raised
                 extreme_duration_threshold=1.5,     # Raised
                 extreme_yawn_rate_threshold=15.0,
                 # Normal state MUST be below this for override to trigger
                 override_max_normal_perc=40.0,   

                 # --- Score Calculation Parameters ---
                 perclos_scale=1.2,           # Reduced scale
                 duration_scale=1.8,          # Reduced scale
                 yawn_rate_scale=1.5,
                 score_cap=2.5,               # Moderate cap

                 # --- Weights (Balanced) ---
                 eye_metric_weight=0.5,
                 yawn_metric_weight=0.5,

                 # --- Non-Linear Damping Parameters ---
                 # Damping = base * (normal_perc / 100) ^ power
                 damping_base_factor=0.8, # Max damping effect at 100% normal
                 damping_power=2.5,       # Power > 1 means steeper increase at high normal %

                 # --- Decision Making ---
                 drowsiness_decision_threshold=0.55, # Adjusted threshold

                 # --- Minimum requirements ---
                 minimum_frames_for_analysis=30, 
                 fps=20                          
                 ):
        """
        Initialize V3 analyzer with conditional overrides and non-linear damping.
        """
        # Store all parameters
        self.perclos_threshold = perclos_threshold
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.yawn_rate_threshold = yawn_rate_threshold
        self.extreme_perclos_threshold = extreme_perclos_threshold
        self.extreme_duration_threshold = extreme_duration_threshold
        self.extreme_yawn_rate_threshold = extreme_yawn_rate_threshold
        self.override_max_normal_perc = override_max_normal_perc
        self.perclos_scale = perclos_scale
        self.duration_scale = duration_scale
        self.yawn_rate_scale = yawn_rate_scale
        self.score_cap = score_cap
        self.eye_metric_weight = eye_metric_weight
        self.yawn_metric_weight = yawn_metric_weight
        self.damping_base_factor = damping_base_factor
        self.damping_power = damping_power
        self.drowsiness_decision_threshold = drowsiness_decision_threshold
        self.minimum_frames_for_analysis = minimum_frames_for_analysis
        self.default_fps = fps

        # Basic validation
        assert 0 <= eye_metric_weight <= 1
        assert 0 <= yawn_metric_weight <= 1
        assert 0 <= self.override_max_normal_perc <= 100
        assert self.damping_power > 0

        # Ensure logging is configured (ideally done once in the main application)
        # logging.basicConfig(level=logging.INFO) 
        # logging.getLogger().setLevel(logging.INFO) 

    def _calculate_metric_score(self, value, threshold, scale):
        """Calculates a score based on how much a value exceeds a threshold, with capping."""
        if value > threshold and threshold > 0: 
            score = ((value - threshold) / threshold) * scale
            return min(score, self.score_cap) 
        return 0.0

    def _calculate_damping(self, normal_state_percentage):
        """Calculates damping amount using a non-linear function."""
        if normal_state_percentage <= 0:
            return 0.0
        # Damping = base * (normal_perc / 100) ^ power
        damping_fraction = normal_state_percentage / 100.0
        damping_amount = self.damping_base_factor * math.pow(damping_fraction, self.damping_power)
        # Ensure damping doesn't exceed 1.0 (or slightly less to avoid zeroing out score)
        return min(damping_amount, 0.99) 


    def analyze(self, detection_results):
        """
        Analyzes detection results using V3 logic: conditional overrides, non-linear damping.
        """
        # --- 1. Extract Data & Basic Checks ---
        yawn_count = detection_results.get('yawn_count', 0)
        eye_closed_detection_count = detection_results.get('eye_closed_frames', 0) 
        total_eye_closed_frames = detection_results.get('total_eye_closed_frames', 0)
        max_consecutive_eye_closed = detection_results.get('max_consecutive_eye_closed', 0)
        normal_state_frames = detection_results.get('normal_state_frames', 0)
        total_frames = detection_results.get('total_frames', 0)
        fps = detection_results.get('metrics', {}).get('fps', self.default_fps)

        if fps <= 0:
            logging.warning(f"Invalid FPS ({fps}). Using default FPS: {self.default_fps}")
            fps = self.default_fps

        if total_frames < self.minimum_frames_for_analysis:
            logging.info(f"Insufficient frames ({total_frames} < {self.minimum_frames_for_analysis}).")
            return {'is_drowsy': None, 'confidence': 0.0, 'details': {'reason': 'insufficient_frames', 'total_frames': total_frames}}

        if total_eye_closed_frames == 0 and yawn_count == 0 and normal_state_frames == 0:
             logging.info("No relevant detections found.")
             return {'is_drowsy': False, 'confidence': 0.0, 'details': {'reason': 'no_detection', 'total_frames': total_frames}}

        # --- 2. Calculate Primary Metrics ---
        time_in_seconds = total_frames / fps
        time_in_minutes = time_in_seconds / 60 if time_in_seconds > 0 else 0

        perclos = (total_eye_closed_frames / total_frames) * 100 if total_frames > 0 else 0
        max_closure_duration = max_consecutive_eye_closed / fps if fps > 0 else 0 
        yawn_rate_per_minute = yawn_count / time_in_minutes if time_in_minutes > 0 else yawn_count * 60 
        normal_state_percentage = (normal_state_frames / total_frames) * 100 if total_frames > 0 else 0

        logging.info(f"V3 Metrics: PERCLOS={perclos:.2f}%, Max Closure={max_closure_duration:.2f}s, "
                     f"Yawn Rate={yawn_rate_per_minute:.2f}/min, Normal State={normal_state_percentage:.2f}%")

        # --- 3. Check for Conditional Extreme Overrides ---
        final_score = 0.0 
        is_drowsy = False
        reason = "checking_extremes"
        override_triggered = False

        # Check if normal state allows overrides
        allow_override = normal_state_percentage < self.override_max_normal_perc

        if allow_override:
            if perclos >= self.extreme_perclos_threshold:
                reason = f"extreme_perclos (>{self.extreme_perclos_threshold}%) & low_normal"
                override_triggered = True
            elif max_closure_duration >= self.extreme_duration_threshold:
                reason = f"extreme_duration (>{self.extreme_duration_threshold}s) & low_normal"
                override_triggered = True
            elif yawn_rate_per_minute >= self.extreme_yawn_rate_threshold:
                 reason = f"extreme_yawn_rate (>{self.extreme_yawn_rate_threshold}/min) & low_normal"
                 override_triggered = True

            if override_triggered:
                is_drowsy = True
                # Assign a high score, bypassing normal calculation and damping
                final_score = self.score_cap # Use max possible score
                logging.info(f"Conditional Drowsiness Override Triggered: {reason}")
                details = self._create_details_dict(perclos, max_closure_duration, yawn_rate_per_minute, normal_state_percentage, 
                                                    0, 0, 0, 0, 0, 0, # Scores/damping not applicable here
                                                    reason, yawn_count, eye_closed_detection_count, total_eye_closed_frames, 
                                                    max_consecutive_eye_closed, normal_state_frames, total_frames, fps)
                return {'is_drowsy': True, 'confidence': final_score, 'details': details}
        else:
             logging.info(f"Overrides skipped due to high normal state ({normal_state_percentage:.1f}% >= {self.override_max_normal_perc}%)")


        # --- 4. Calculate Individual Scores (If no override) ---
        perclos_score = self._calculate_metric_score(perclos, self.perclos_threshold, self.perclos_scale)
        duration_score = self._calculate_metric_score(max_closure_duration, self.max_closure_duration_threshold, self.duration_scale)
        yawn_score = self._calculate_metric_score(yawn_rate_per_minute, self.yawn_rate_threshold, self.yawn_rate_scale)

        # Combine eye scores using AVERAGE
        combined_eye_score = (perclos_score + duration_score) / 2.0

        # --- 5. Calculate Raw Drowsiness Score ---
        raw_drowsiness_score = (combined_eye_score * self.eye_metric_weight +
                                yawn_score * self.yawn_metric_weight)

        # --- 6. Apply Non-Linear Normal State Damping ---
        damping_amount = self._calculate_damping(normal_state_percentage)
        
        final_score = raw_drowsiness_score * (1.0 - damping_amount)
        final_score = max(0.0, final_score) 

        # --- 7. Make Final Drowsiness Decision ---
        is_drowsy = final_score >= self.drowsiness_decision_threshold

        # --- 8. Determine Reason ---
        if is_drowsy:
             # Check contributions before damping
             eye_contribution = combined_eye_score * self.eye_metric_weight
             yawn_contribution = yawn_score * self.yawn_metric_weight
             if eye_contribution > yawn_contribution + 0.05: # Add small tolerance
                 reason = f"eye_metrics_dominant (Score: {final_score:.2f})"
             elif yawn_contribution > eye_contribution + 0.05:
                 reason = f"yawn_metrics_dominant (Score: {final_score:.2f})"
             else: 
                 reason = f"threshold_met (Score: {final_score:.2f})"
        elif final_score > 0: 
            reason = f"indicators_present_below_threshold (Score: {final_score:.2f})"
        else: 
             reason = "no_significant_indicators"


        # --- 9. Format Results ---
        details = self._create_details_dict(perclos, max_closure_duration, yawn_rate_per_minute, normal_state_percentage, 
                                            perclos_score, duration_score, yawn_score, combined_eye_score, 
                                            raw_drowsiness_score, damping_amount, reason, 
                                            yawn_count, eye_closed_detection_count, total_eye_closed_frames, 
                                            max_consecutive_eye_closed, normal_state_frames, total_frames, fps)
        
        result = {
            'is_drowsy': is_drowsy,
            'confidence': final_score, 
            'details': details
        }

        logging.info(f"V3 Analysis result: is_drowsy={result['is_drowsy']}, score={result['confidence']:.3f}, reason={result['details']['reason']}")
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
                'combined_eye_score_avg': eye_score, # Renamed to reflect averaging
                'raw_drowsiness_score': raw_score,
                'applied_damping_factor': damping, 
                'reason': reason,
                # Raw Inputs
                'yawn_count': yawn_cnt,
                'eye_closed_detection_count': eye_closed_det_cnt, 
                'total_eye_closed_frames': tot_eye_frames,
                'max_consecutive_eye_closed_frames': max_consec, # Clarified name
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
