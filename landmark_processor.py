import cv2
import numpy as np
import dlib
import os
import logging
import time
import requests
import tempfile
from typing import Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class FatigueMetrics:
    """Data class to store fatigue analysis metrics from the front camera"""
    eye_aspect_ratio_left: float
    eye_aspect_ratio_right: float
    perclos_score: float
    blink_frequency: float
    frame_number: int
    timestamp: float

@dataclass
class FatigueResult:
    """Data class to store final fatigue detection results"""
    driver_name: str
    percentage_fatigue: float
    is_fatigue: bool
    confidence: float
    analysis_details: Dict
    analysis_timestamp: str

class FatigueDetectionSystem:
    """
    Main class for fatigue detection system - integrated from drowsiness_landmark.py
    """

    def __init__(self):
        """Initialize the fatigue detection system"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self._initialize_predictor()

        # --- THRESHOLDS for Facial Analysis ---
        self.EAR_THRESHOLD = 0.25
        self.PERCLOS_THRESHOLD = 0.30
        self.FATIGUE_THRESHOLD = 0.60 # Overall fatigue threshold (60%)
        self.PERCLOS_WINDOW_SECONDS = 1.5

        # Analysis parameters
        self.frame_buffer = []
        self.analysis_window_frames = 30
        self.blink_counter = 0
        self.closed_eye_frames = 0

    def _initialize_predictor(self):
        """Initialize facial landmark predictor"""
        try:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                print("Facial landmark predictor found and loaded.")
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                print("Warning: Facial landmark predictor not found. Attempting to download...")
                url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                response = requests.get(url, stream=True)
                response.raise_for_status()

                import bz2
                with open(predictor_path, "wb") as f:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(decompressor.decompress(chunk))

                self.predictor = dlib.shape_predictor(predictor_path)
                print("Predictor downloaded and loaded successfully.")

        except Exception as e:
            print(f"Error initializing or downloading predictor: {e}")
            self.predictor = None

    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        vertical_1 = euclidean(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = euclidean(eye_landmarks[2], eye_landmarks[4])
        horizontal = euclidean(eye_landmarks[0], eye_landmarks[3])

        if horizontal == 0: return 0.3 # Avoid division by zero
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def extract_eye_landmarks(self, face_landmarks) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract left and right eye landmarks from facial landmarks"""
        try:
            left_eye = np.array([(face_landmarks.part(i).x, face_landmarks.part(i).y) for i in range(36, 42)], dtype="double")
            right_eye = np.array([(face_landmarks.part(i).x, face_landmarks.part(i).y) for i in range(42, 48)], dtype="double")
            return left_eye, right_eye
        except Exception as e:
            print(f"Error extracting eye landmarks: {e}")
            return None, None

    def analyze_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FatigueMetrics:
        """Analyze a single frame for facial fatigue indicators"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left_ear, right_ear = -1.0, -1.0

        faces = self.detector(gray)

        if faces:
            face = faces[0]
            if self.predictor:
                landmarks = self.predictor(gray, face)
                left_eye, right_eye = self.extract_eye_landmarks(landmarks)
                if left_eye is not None and right_eye is not None:
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)

        avg_ear = -1.0
        if left_ear != -1.0:
            avg_ear = (left_ear + right_ear) / 2.0

        is_closed = avg_ear < self.EAR_THRESHOLD if avg_ear != -1.0 else False
        if is_closed:
            self.closed_eye_frames += 1

        self.frame_buffer.append(is_closed)
        if len(self.frame_buffer) > self.analysis_window_frames:
            self.frame_buffer.pop(0)

        perclos_score = sum(self.frame_buffer) / len(self.frame_buffer) if self.frame_buffer else 0

        if len(self.frame_buffer) > 1 and not self.frame_buffer[-2] and self.frame_buffer[-1]:
             self.blink_counter += 1

        blink_freq = self.blink_counter / max(1, timestamp)

        return FatigueMetrics(left_ear, right_ear, perclos_score, blink_freq, frame_number, timestamp)

    def analyze_video(self, video_path: str, driver_name: str = "Unknown") -> Optional[FatigueResult]:
        """Analyze a video file for fatigue using facial analysis."""
        print(f"\n🎬 Analyzing video for: {driver_name}")

        local_video_path = self.download_video(video_path)
        if not local_video_path:
            return None

        try:
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                print(f"❌ Error: Cannot open video file {local_video_path}")
                return None

            # Reset analysis state for each video
            self.frame_buffer, self.closed_eye_frames, self.blink_counter = [], 0, 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0 or fps == 0:
                print(f"❌ Error: Video file {local_video_path} has invalid properties.")
                cap.release()
                return None

            self.analysis_window_frames = int(self.PERCLOS_WINDOW_SECONDS * fps)
            print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS | PERCLOS window: {self.analysis_window_frames} frames")

            frame_metrics, timestamp = [], 0
            for frame_number in range(total_frames):
                ret, frame = cap.read()
                if not ret: break

                timestamp = frame_number / fps
                metrics = self.analyze_frame(frame, frame_number, timestamp)
                frame_metrics.append(metrics)

                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"⏳ Processing: {progress:.1f}%", end="\r")

            print("⏳ Processing: 100.0%")

            if not frame_metrics:
                print("\n❌ No frames could be analyzed")
                return None

            valid_ear_metrics = [m for m in frame_metrics if m.eye_aspect_ratio_left != -1.0]
            if not valid_ear_metrics:
                print("\n❌ Warning: No faces detected. Returning result with 0% fatigue.")
                fatigue_percentage = 0.0
                confidence = 0.5
                avg_perclos = 0
                avg_ear = 0
            else:
                avg_perclos = sum(m.perclos_score for m in frame_metrics) / len(frame_metrics)
                avg_ear = sum((m.eye_aspect_ratio_left + m.eye_aspect_ratio_right) / 2 for m in valid_ear_metrics) / len(valid_ear_metrics)

                perclos_factor = min(1.0, avg_perclos / self.PERCLOS_THRESHOLD)
                ear_factor = max(0.0, (self.EAR_THRESHOLD - avg_ear) / self.EAR_THRESHOLD) if avg_ear > 0 else 0
                fatigue_percentage = ((perclos_factor * 0.7) + (ear_factor * 0.3)) * 100
                confidence = min(0.95, 0.5 + (abs(fatigue_percentage - (self.FATIGUE_THRESHOLD * 100)) / 100))

            is_fatigue = fatigue_percentage > (self.FATIGUE_THRESHOLD * 100)

            analysis_details = {
                "video_fps": fps, "total_frames": len(frame_metrics),
                "frames_with_face": len(valid_ear_metrics), "average_perclos": round(avg_perclos, 4),
                "average_ear": round(avg_ear, 4), "analysis_duration_seconds": round(timestamp, 2),
                "detected_blinks": self.blink_counter
            }

            result = FatigueResult(
                driver_name=driver_name,
                percentage_fatigue=round(fatigue_percentage, 2),
                is_fatigue=is_fatigue,
                confidence=round(confidence, 3),
                analysis_details=analysis_details,
                analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            status = "🔴 FATIGUE DETECTED" if is_fatigue else "🟢 NO FATIGUE"
            print(f"\n✅ Analysis complete for {driver_name}: {status} ({fatigue_percentage:.2f}%)")
            return result

        except Exception as e:
            print(f"\n❌ An unexpected error during video analysis for {driver_name}: {e}")
            return None
        finally:
            if cap and cap.isOpened():
                cap.release()
            # Clean up downloaded file if it's a temp file
            if local_video_path and video_path != local_video_path and os.path.exists(local_video_path):
                os.unlink(local_video_path)
                print(f"🧹 Cleaned up temporary file: {local_video_path}")

    def download_video(self, url: str) -> Optional[str]:
        """Download video from URL to temporary file, or use local path."""
        try:
            if os.path.exists(url):
                return url

            print(f"Downloading video from URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

            temp_file_path = temp_file.name
            temp_file.close()
            return temp_file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading video from {url}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during video download: {e}")
            return None

class LandmarkDrowsinessProcessor:
    """
    Landmark-based drowsiness detection processor that integrates the FatigueDetectionSystem
    from drowsiness_landmark.py while maintaining compatibility with the simplify.py architecture.
    """

    def __init__(self):
        """Initialize the landmark-based drowsiness processor."""
        # Initialize the integrated FatigueDetectionSystem
        self.fatigue_system = FatigueDetectionSystem()

        # Processing parameters
        self.frame_skip = int(os.getenv('LANDMARK_FRAME_SKIP', '2'))  # Process every nth frame for performance

    def _cleanup_temp_file(self, temp_path):
        """Clean up temporary video file."""
        try:
            if temp_path and os.path.exists(temp_path) and temp_path.startswith("temp_"):
                os.remove(temp_path)
                logging.info(f"Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logging.warning(f"Failed to clean up temporary file {temp_path}: {e}")

    def process_video(self, video_url):
        """
        Process a video file for drowsiness detection using the integrated FatigueDetectionSystem.
        Returns results in the same format as simplify.py's YoloProcessor.
        """
        logging.info(f"Starting landmark-based drowsiness detection for: {video_url}")

        try:
            # Use the integrated FatigueDetectionSystem to analyze the video
            start_time = time.time()
            fatigue_result = self.fatigue_system.analyze_video(video_url, "driver")
            process_time = time.time() - start_time

            if not fatigue_result:
                return False, {'error': 'Failed to analyze video', 'reason': 'FatigueDetectionSystem returned None'}

            # Map FatigueResult to simplified format for database storage
            analysis_details = fatigue_result.analysis_details

            detection_results = {
                # Core metrics needed for database storage
                'yawn_frames': 0,  # Landmark system doesn't detect yawns
                'eye_closed_frames': analysis_details.get('detected_blinks', 0),  # Use blinks as eye closed events
                'max_consecutive_eye_closed': 0,  # Not directly available from FatigueResult
                'normal_state_frames': analysis_details.get('total_frames', 0) - analysis_details.get('detected_blinks', 0),
                'total_frames': analysis_details.get('total_frames', 0),
                'processing_status': 'processed',
                'process_time': process_time,  # Include process time for database storage

                # Include fatigue_result for analyzer (but this won't be stored in DB)
                'fatigue_result': {
                    'is_fatigue': fatigue_result.is_fatigue,
                    'confidence': fatigue_result.confidence,
                    'percentage_fatigue': fatigue_result.percentage_fatigue,
                    'analysis_details': fatigue_result.analysis_details
                }
            }

            logging.info(f"Landmark processing complete. Fatigue: {fatigue_result.is_fatigue}, "
                        f"Confidence: {fatigue_result.confidence:.3f}, "
                        f"Percentage: {fatigue_result.percentage_fatigue:.2f}%")

            return True, detection_results

        except Exception as e:
            logging.exception(f"Critical error in landmark processing: {e}")
            return False, {'error': str(e), 'reason': 'Exception during video processing'}
