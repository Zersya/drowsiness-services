import logging
from abc import ABC, abstractmethod

class LandmarkDrowsinessAnalyzer(ABC):
    """Abstract base class for landmark-based drowsiness analysis implementations."""

    @abstractmethod
    def analyze(self, detection_results):
        """Analyze drowsiness based on landmark detection metrics."""
        pass

class FatigueResultAnalyzer(LandmarkDrowsinessAnalyzer):
    """
    Simple analyzer that extracts drowsiness results from the FatigueResult
    produced by the integrated FatigueDetectionSystem.
    """

    def __init__(self):
        """Initialize the FatigueResult analyzer."""
        pass

    def analyze(self, detection_results):
        """
        Extract drowsiness analysis from the FatigueResult embedded in detection_results.
        Returns results in the same format as simplify.py's RateBasedAnalyzer.
        """
        # Extract the FatigueResult from detection_results
        fatigue_result_dict = detection_results.get('fatigue_result', {})

        if not fatigue_result_dict:
            # Fallback if no FatigueResult available
            return {
                'is_drowsy': False,
                'confidence': 0.0,
                'details': {
                    'reason': 'no_fatigue_result_available',
                    'yawn_count': 0,
                    'yawn_frames': 0,
                    'eye_closed_frames': detection_results.get('eye_closed_frames', 0),
                    'total_eye_closed_frames': detection_results.get('total_eye_closed_frames', 0),
                    'max_consecutive_eye_closed': detection_results.get('max_consecutive_eye_closed', 0),
                    'normal_state_frames': detection_results.get('normal_state_frames', 0),
                    'total_frames': detection_results.get('total_frames', 0),
                    'fps': 20,  # Default FPS for fallback
                    'frames_with_face': 0,
                    'detected_blinks': 0,
                    'perclos_score': 0,
                    'avg_ear': 0,
                    'fatigue_percentage': 0.0,
                    'is_drowsy_eyes': False,
                    'is_drowsy_perclos': False,
                    'is_drowsy_blinks': False,
                    'is_normal_state_high': False,
                    'is_head_turned': False,
                    'is_head_down': False,
                    'head_turn_direction': 'center',
                    'yawn_rate_per_minute': 0.0,
                    'yawn_percentage': 0.0,
                    'is_drowsy_yawns': False,
                    'is_drowsy_excessive_yawns': False,
                }
            }

        # Extract core results from FatigueResult
        is_drowsy = fatigue_result_dict.get('is_fatigue', False)
        confidence = fatigue_result_dict.get('confidence', 0.0)
        percentage_fatigue = fatigue_result_dict.get('percentage_fatigue', 0.0)
        analysis_details = fatigue_result_dict.get('analysis_details', {})

        # Extract basic info from detection_results (no metrics section in simplified format)

        # Construct details dictionary compatible with simplify.py format
        details_output = {
            # Core FatigueResult metrics
            'perclos_score': analysis_details.get('average_perclos', 0),
            'avg_ear': analysis_details.get('average_ear', 0),
            'fatigue_percentage': percentage_fatigue,
            'detected_blinks': analysis_details.get('detected_blinks', 0),
            'frames_with_face': analysis_details.get('frames_with_face', 0),

            # Mapped metrics for compatibility
            'yawn_count': 0,  # Landmark system doesn't detect yawns
            'yawn_frames': 0,
            'eye_closed_frames': detection_results.get('eye_closed_frames', 0),
            'total_eye_closed_frames': detection_results.get('total_eye_closed_frames', 0),
            'max_consecutive_eye_closed': detection_results.get('max_consecutive_eye_closed', 0),
            'normal_state_frames': detection_results.get('normal_state_frames', 0),
            'total_frames': detection_results.get('total_frames', 0),
            'fps': analysis_details.get('video_fps', 20),

            # Analysis reason
            'reason': 'fatigue_detection_system_analysis',

            # Boolean flags based on FatigueResult
            'is_drowsy_eyes': is_drowsy,  # Map fatigue to eye drowsiness
            'is_drowsy_perclos': analysis_details.get('average_perclos', 0) > 0.3,
            'is_drowsy_blinks': analysis_details.get('detected_blinks', 0) < 5,  # Low blink count
            'is_normal_state_high': not is_drowsy,
            'is_head_turned': False,  # Landmark system doesn't detect head pose
            'is_head_down': False,
            'head_turn_direction': 'center',

            # Compatibility fields (set to 0 for landmark system)
            'yawn_rate_per_minute': 0.0,
            'yawn_percentage': 0.0,
            'is_drowsy_yawns': False,
            'is_drowsy_excessive_yawns': False,
            'eye_closed_percentage': 0.0,
            'max_closure_duration': 0.0,
            'normal_state_percentage': 0.0,
            'blink_frequency': 0.0,
        }

        logging.info(
            f"FatigueResultAnalyzer output: "
            f"is_drowsy:{is_drowsy}, confidence:{confidence:.3f}, "
            f"fatigue_percentage:{percentage_fatigue:.2f}%, "
            f"perclos:{analysis_details.get('average_perclos', 0):.3f}, "
            f"ear:{analysis_details.get('average_ear', 0):.3f}, "
            f"blinks:{analysis_details.get('detected_blinks', 0)}"
        )

        return {
            'is_drowsy': is_drowsy,
            'confidence': confidence,
            'details': details_output
        }

def create_landmark_analyzer(analyzer_type="landmark"):
    """Create and return a landmark-based drowsiness analyzer instance."""
    if analyzer_type == "landmark":
        return FatigueResultAnalyzer()
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")
