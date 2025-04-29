import cv2
import os
import logging
import sqlite3
import time
import datetime
import json
import hashlib
import numpy as np
import torch
from ultralytics import YOLO
from abc import ABC, abstractmethod
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('simplify.log')  # Save to file
    ]
)

# Load environment variables
load_dotenv()

# Configuration
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/jingyeong-best.pt")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolov8l-pose.pt")
USE_CUDA = os.getenv('USE_CUDA', 'true').lower() == 'true'
# Use environment variable for database path to support Docker volumes
DB_PATH = os.getenv("DB_PATH", "simplify_detection.db")
PORT = int(os.getenv('SIMPLIFY_PORT', 8002))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 1))  # Maximum number of concurrent video processing workers
QUEUE_CHECK_INTERVAL = int(os.getenv('QUEUE_CHECK_INTERVAL', 5))  # Seconds between queue checks

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database Manager
class DatabaseManager:
    """Manages SQLite database operations for the simplified drowsiness detection system."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database and create necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create evidence_results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS evidence_results (
                        id INTEGER PRIMARY KEY,
                        video_url TEXT,
                        process_time REAL,
                        yawn_frames INTEGER,
                        eye_closed_frames INTEGER,
                        max_consecutive_eye_closed INTEGER,
                        normal_state_frames INTEGER,
                        total_frames INTEGER,
                        is_drowsy BOOLEAN,
                        confidence REAL,
                        is_head_turned BOOLEAN,
                        is_head_down BOOLEAN,
                        processing_status TEXT,
                        details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create processing_queue table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processing_queue (
                        id INTEGER PRIMARY KEY,
                        video_url TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create webhooks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS webhooks (
                        id INTEGER PRIMARY KEY,
                        url TEXT NOT NULL,
                        is_enabled BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                logging.info("Database initialized successfully")
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")

    def add_to_queue(self, video_url):
        """Add a video URL to the processing queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if this URL is already in the queue with status 'pending' or 'processing'
                cursor = conn.execute(
                    'SELECT id, status FROM processing_queue WHERE video_url = ? AND status IN ("pending", "processing")',
                    (video_url,)
                )
                existing = cursor.fetchone()

                if existing:
                    queue_id, status = existing
                    logging.info(f"Video already in queue: {video_url}, ID: {queue_id}, Status: {status}")
                    return queue_id

                # Add new entry to queue
                cursor = conn.execute(
                    'INSERT INTO processing_queue (video_url, status) VALUES (?, ?)',
                    (video_url, 'pending')
                )
                queue_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Added video to queue: {video_url}, ID: {queue_id}")
                return queue_id
        except sqlite3.Error as e:
            logging.error(f"Error adding to queue: {e}")
            return None

    def update_queue_status(self, queue_id, status):
        """Update the status of a queued item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE processing_queue SET status = ? WHERE id = ?',
                    (status, queue_id)
                )
                conn.commit()
                logging.info(f"Updated queue item {queue_id} status to {status}")
                return True
        except sqlite3.Error as e:
            logging.error(f"Error updating queue status: {e}")
            return False

    def get_queue_status(self, queue_id):
        """Get the status of a queued item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url, status, created_at FROM processing_queue WHERE id = ?',
                    (queue_id,)
                )
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
        except sqlite3.Error as e:
            logging.error(f"Error getting queue status: {e}")
            return None

    def get_next_pending_video(self):
        """Get the next pending video from the queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url FROM processing_queue WHERE status = "pending" ORDER BY created_at ASC LIMIT 1'
                )
                result = cursor.fetchone()
                if result:
                    # Update status to 'processing'
                    conn.execute(
                        'UPDATE processing_queue SET status = "processing" WHERE id = ?',
                        (result['id'],)
                    )
                    conn.commit()
                    return dict(result)
                return None
        except sqlite3.Error as e:
            logging.error(f"Error getting next pending video: {e}")
            return None

    def get_queue_stats(self):
        """Get statistics about the processing queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT status, COUNT(*) as count FROM processing_queue GROUP BY status'
                )
                results = cursor.fetchall()
                stats = {}
                for status, count in results:
                    stats[status] = count
                return stats
        except sqlite3.Error as e:
            logging.error(f"Error getting queue stats: {e}")
            return {}

    def store_evidence_result(self, video_url, detection_results, analysis_result, process_time, head_pose=None):
        """Store evidence result in the database."""
        try:
            is_drowsy = analysis_result.get('is_drowsy')
            confidence = analysis_result.get('confidence', 0.0)

            # Extract head pose information
            is_head_turned = False
            is_head_down = False
            if head_pose:
                is_head_turned = head_pose.get('head_turned', False)
                is_head_down = head_pose.get('head_down', False)

            # Combine all details for storage
            details = {
                'detection_results': detection_results,
                'analysis_result': analysis_result,
                'head_pose': head_pose
            }

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO evidence_results (
                        video_url, process_time, yawn_frames, eye_closed_frames,
                        max_consecutive_eye_closed,
                        normal_state_frames, total_frames, is_drowsy, confidence,
                        is_head_turned, is_head_down, processing_status, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_url,
                    process_time,
                    detection_results.get('yawn_frames', 0),
                    detection_results.get('eye_closed_frames', 0),
                    detection_results.get('max_consecutive_eye_closed', 0),
                    detection_results.get('normal_state_frames', 0),
                    detection_results.get('total_frames', 0),
                    is_drowsy,
                    confidence,
                    is_head_turned,
                    is_head_down,
                    'processed',
                    json.dumps(details)
                ))
                evidence_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Stored evidence result for {video_url}, ID: {evidence_id}")
                return evidence_id
        except sqlite3.Error as e:
            logging.error(f"Error storing evidence result: {e}")
            return None

    def get_evidence_result(self, evidence_id):
        """Get evidence result by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM evidence_results WHERE id = ?', (evidence_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            logging.error(f"Error getting evidence result: {e}")
            return None

    def get_all_evidence_results(self, limit=100, offset=0):
        """Get all evidence results with pagination."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM evidence_results ORDER BY created_at DESC LIMIT ? OFFSET ?',
                    (limit, offset)
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting all evidence results: {e}")
            return []

    def add_webhook(self, url):
        """Add a new webhook URL."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if this URL already exists
                cursor = conn.execute(
                    'SELECT id FROM webhooks WHERE url = ?',
                    (url,)
                )
                existing = cursor.fetchone()

                if existing:
                    # If it exists but is disabled, enable it
                    conn.execute(
                        'UPDATE webhooks SET is_enabled = 1 WHERE id = ?',
                        (existing[0],)
                    )
                    conn.commit()
                    logging.info(f"Re-enabled existing webhook: {url}, ID: {existing[0]}")
                    return existing[0]

                # Add new webhook
                cursor = conn.execute(
                    'INSERT INTO webhooks (url, is_enabled) VALUES (?, ?)',
                    (url, 1)
                )
                webhook_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Added new webhook: {url}, ID: {webhook_id}")
                return webhook_id
        except sqlite3.Error as e:
            logging.error(f"Error adding webhook: {e}")
            return None

    def delete_webhook(self, webhook_id):
        """Delete a webhook by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'DELETE FROM webhooks WHERE id = ?',
                    (webhook_id,)
                )
                conn.commit()
                logging.info(f"Deleted webhook with ID: {webhook_id}")
                return True
        except sqlite3.Error as e:
            logging.error(f"Error deleting webhook: {e}")
            return False

    def get_all_webhooks(self):
        """Get all registered webhooks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM webhooks ORDER BY created_at DESC'
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting all webhooks: {e}")
            return []

    def get_active_webhooks(self):
        """Get all active webhooks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM webhooks WHERE is_enabled = 1'
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting active webhooks: {e}")
            return []


# Pose Head Detector
class PoseHeadDetector:
    """Class for detecting head pose (head down or head turn) using YOLOv8 pose model."""

    def __init__(self, model_path=POSE_MODEL_PATH):
        """Initialize the pose head detector."""
        self.model_path = model_path
        self.use_cuda = USE_CUDA

        # Configuration parameters
        self.keypoint_conf_threshold = float(os.getenv('KEYPOINT_CONF_THRESHOLD', '0.5'))
        self.head_turn_ratio_threshold = float(os.getenv('HEAD_TURN_RATIO_THRESHOLD', '0.7'))
        self.head_down_ratio_threshold = float(os.getenv('HEAD_DOWN_RATIO_THRESHOLD', '0.3'))

        # Time thresholds (in seconds)
        self.head_turned_threshold_seconds = float(os.getenv('HEAD_TURNED_THRESHOLD_SECONDS', '1.5'))
        self.head_down_threshold_seconds = float(os.getenv('HEAD_DOWN_THRESHOLD_SECONDS', '1.5'))

        # Counters
        self.head_turned_counter = 0
        self.head_down_counter = 0

        # Status flags
        self.distracted_head_turn = False
        self.distracted_head_down = False

        # Define COCO keypoint indices (only need head-related ones)
        self.kp_indices = {
            "nose": 0, "left_eye": 1, "right_eye": 2,
            "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6
        }

        # Load the model
        self.model = self.load_model()

        # Set frames thresholds based on FPS and time thresholds
        self.fps = 20  # Default FPS, will be updated during processing
        self.update_frame_thresholds()

    def update_frame_thresholds(self, fps=None):
        """Update frame thresholds based on FPS."""
        if fps is not None:
            self.fps = fps

        # Convert time thresholds to frame counts
        self.head_turned_frames_threshold = int(self.head_turned_threshold_seconds * self.fps)
        self.head_down_frames_threshold = int(self.head_down_threshold_seconds * self.fps)

    def load_model(self):
        """Load the YOLOv8 pose model."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logging.warning(f"Pose model not found at {self.model_path}. Head pose detection will be disabled.")
                return None

            # Check if CUDA is available and enabled
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"Pose model: CUDA available: {cuda_available}, Using device: {device}")

            # Load the model
            model = YOLO(self.model_path)
            model.to(device)

            logging.info(f"Pose model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading pose model: {e}")
            return None

    def process_frame(self, frame):
        """Process a single frame to detect head pose."""
        if self.model is None:
            return {"head_turned": False, "head_down": False}

        try:
            # Reset per-frame flags
            frame_flag_head_turned = False
            frame_flag_head_down = False

            # Perform pose estimation
            results = self.model(frame, verbose=False)

            # Process results
            if results and len(results) > 0:
                result = results[0]

                # Get keypoints from the first person detected
                if len(result.keypoints.data) > 0:
                    keypoints = result.keypoints.data[0].cpu().numpy()

                    # Check if we have the necessary keypoints with sufficient confidence
                    if (keypoints[self.kp_indices["nose"]][2] > self.keypoint_conf_threshold and
                        keypoints[self.kp_indices["left_eye"]][2] > self.keypoint_conf_threshold and
                        keypoints[self.kp_indices["right_eye"]][2] > self.keypoint_conf_threshold):

                        # Get coordinates
                        nose = keypoints[self.kp_indices["nose"]][:2]
                        left_eye = keypoints[self.kp_indices["left_eye"]][:2]
                        right_eye = keypoints[self.kp_indices["right_eye"]][:2]

                        # Calculate eye midpoint
                        eye_midpoint = (left_eye + right_eye) / 2

                        # Check for head turn (asymmetry between eyes and nose)
                        # Calculate distances between facial landmarks
                        nose_to_left = np.linalg.norm(nose - left_eye)
                        nose_to_right = np.linalg.norm(nose - right_eye)

                        # Calculate ratio of distances
                        if min(nose_to_left, nose_to_right) > 0:
                            turn_ratio = max(nose_to_left, nose_to_right) / min(nose_to_left, nose_to_right)

                            # Check if head is turned
                            if turn_ratio > self.head_turn_ratio_threshold:
                                frame_flag_head_turned = True

                        # Check for head down (if shoulders are visible)
                        if (keypoints[self.kp_indices["left_shoulder"]][2] > self.keypoint_conf_threshold and
                            keypoints[self.kp_indices["right_shoulder"]][2] > self.keypoint_conf_threshold):

                            left_shoulder = keypoints[self.kp_indices["left_shoulder"]][:2]
                            right_shoulder = keypoints[self.kp_indices["right_shoulder"]][:2]
                            shoulder_midpoint = (left_shoulder + right_shoulder) / 2

                            # Vector from shoulder midpoint to eye midpoint
                            up_vector = eye_midpoint - shoulder_midpoint
                            up_vector_norm = np.linalg.norm(up_vector)

                            # Vector from eye midpoint to nose
                            gaze_vector = nose - eye_midpoint
                            gaze_vector_norm = np.linalg.norm(gaze_vector)

                            # Calculate angle between vectors if both have non-zero length
                            if up_vector_norm > 0 and gaze_vector_norm > 0:
                                # Normalize vectors
                                up_vector = up_vector / up_vector_norm
                                gaze_vector = gaze_vector / gaze_vector_norm

                                # Calculate dot product and angle
                                dot_product = np.clip(np.dot(up_vector, gaze_vector), -1.0, 1.0)
                                angle = np.arccos(dot_product)

                                # Convert to degrees
                                angle_degrees = np.degrees(angle)

                                # Check if head is down (angle > threshold)
                                if angle_degrees > 45:
                                    frame_flag_head_down = True

            # Update temporal counters
            self.head_turned_counter = (self.head_turned_counter + 1) if frame_flag_head_turned else 0
            self.head_down_counter = (self.head_down_counter + 1) if frame_flag_head_down else 0

            # Determine final status
            self.distracted_head_turn = self.head_turned_counter >= self.head_turned_frames_threshold
            self.distracted_head_down = self.head_down_counter >= self.head_down_frames_threshold

            return {
                "head_turned": self.distracted_head_turn,
                "head_down": self.distracted_head_down,
                "head_turned_counter": self.head_turned_counter,
                "head_down_counter": self.head_down_counter,
                "head_turned_threshold": self.head_turned_frames_threshold,
                "head_down_threshold": self.head_down_frames_threshold
            }

        except Exception as e:
            logging.error(f"Error in pose head detection: {e}")
            return {"head_turned": False, "head_down": False}


# Drowsiness Analyzer
class DrowsinessAnalyzer(ABC):
    """Abstract base class for drowsiness analysis implementations."""

    @abstractmethod
    def analyze(self, detection_results):
        """Analyze drowsiness based on detection metrics."""
        pass


class RateBasedAnalyzer(DrowsinessAnalyzer):
    """Rate-based drowsiness analysis using eye closure percentage and yawn frequency."""

    def __init__(self, eye_closed_percentage_threshold=5, yawn_percentage_threshold=5, normal_state_threshold=60, fps=20, max_closure_duration_threshold=0.3):
        """Initialize with adjusted thresholds for better detection."""
        self.eye_closed_percentage_threshold = eye_closed_percentage_threshold
        self.yawn_percentage_threshold = yawn_percentage_threshold
        self.normal_state_threshold = normal_state_threshold
        self.fps = fps
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.minimum_yawn_threshold = 1
        self.minimum_eye_closed_threshold = 3
        self.normal_state_ratio_threshold = 5
        self.minimum_frames_for_analysis = 10

    def analyze(self, detection_results):
        """Analyze drowsiness based on detection metrics."""
        # Extract values from detection_results
        yawn_frames = detection_results.get('yawn_frames', 0)
        eye_closed_frames = detection_results.get('eye_closed_frames', 0)
        normal_state_frames = detection_results.get('normal_state_frames', 0)
        total_frames = detection_results.get('total_frames', 0)
        # Use eye_closed_frames directly
        max_consecutive_eye_closed = detection_results.get('max_consecutive_eye_closed', 0)
        fps = detection_results.get('metrics', {}).get('fps', self.fps)

        # Extract head pose information if available
        head_pose = detection_results.get('head_pose', {})
        is_head_turned = head_pose.get('head_turned', False)
        is_head_down = head_pose.get('head_down', False)
        head_turned_frames = head_pose.get('head_turned_counter', 0)
        head_down_frames = head_pose.get('head_down_counter', 0)

        # Skip analysis if no detections
        if ((yawn_frames == 0 or yawn_frames is None) and
            (eye_closed_frames == 0 or eye_closed_frames is None) and
            (normal_state_frames == 0 or normal_state_frames is None)):
            return {
                'is_drowsy': None,
                'confidence': 0.0,
                'details': {
                    'eye_closed_percentage': 0.0,
                    'yawn_percentage': 0.0,
                    'normal_state_percentage': 0.0,
                    'reason': 'no_detection'
                }
            }

        logging.info(f"Analyzing drowsiness: Yawns={yawn_frames}, Eye Closed Frames={eye_closed_frames}, "
                    f"Max Consecutive Closed={max_consecutive_eye_closed}, "
                    f"Normal State Frames={normal_state_frames}, Total Frames={total_frames}")

        if total_frames < self.minimum_frames_for_analysis:
            return {
                'is_drowsy': False,
                'confidence': 0.0,
                'details': {
                    'eye_closed_percentage': 0.0,
                    'yawn_percentage': 0.0,
                    'normal_state_percentage': 0.0,
                    'reason': 'insufficient_frames'
                }
            }

        # Calculate metrics based on video properties

        # Calculate percentage metrics
        eye_closed_percentage = (eye_closed_frames / total_frames) * 100 if total_frames > 0 else 0
        max_closure_duration = max_consecutive_eye_closed / fps if fps > 0 else 0
        yawn_percentage = (yawn_frames / total_frames) * 100 if total_frames > 0 else 0
        normal_state_percentage = (normal_state_frames / total_frames) * 100 if total_frames > 0 else 0

        # Drowsiness indicators - more sensitive to eye closures
        is_drowsy_eyes = (eye_closed_percentage > self.eye_closed_percentage_threshold or
                        max_closure_duration > self.max_closure_duration_threshold or
                        eye_closed_frames >= self.minimum_eye_closed_threshold)
        # Check for drowsiness based on yawn percentage
        is_drowsy_yawns = (yawn_frames >= self.minimum_yawn_threshold and
                        yawn_percentage > self.yawn_percentage_threshold)
        is_drowsy_excessive_yawns = yawn_frames > 10 or yawn_percentage > 20  # 20% is excessive

        # Check if head pose indicates distraction (head down or turned)
        is_head_distracted = is_head_turned or is_head_down

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
        elif is_drowsy_yawns and yawn_percentage > 15:  # High percentage of yawning frames
            is_drowsy = True
            confidence = 0.8
            reason = 'high_yawn_rate'
        elif is_normal_state_high and not (is_drowsy_yawns or is_drowsy_eyes):
            is_drowsy = False
            confidence = 0.1
            reason = 'high_normal_state'
        else:
            # Consider head pose when determining drowsiness
            # If head is turned or down, we don't consider it drowsy even if other indicators are present
            if is_head_distracted:
                is_drowsy = False
                reason = 'head_pose_override'
                logging.info(f"Head pose override: Head turned={is_head_turned}, Head down={is_head_down}")
            else:
                is_drowsy = is_drowsy_eyes or is_drowsy_yawns

            # Calculate confidence based on new metrics - increased weight for eye closures
            eye_percentage_confidence = min(eye_closed_percentage / self.eye_closed_percentage_threshold, 1.0) if eye_closed_percentage > 0 else 0
            eye_duration_confidence = min(max_closure_duration / self.max_closure_duration_threshold, 1.0) if max_closure_duration > 0 else 0
            eye_count_confidence = min(eye_closed_frames / self.minimum_eye_closed_threshold, 1.0) if eye_closed_frames > 0 else 0
            # Use all three eye metrics for a more comprehensive eye confidence
            eye_confidence = max(eye_percentage_confidence, eye_duration_confidence, eye_count_confidence)
            yawn_confidence = min(yawn_percentage / self.yawn_percentage_threshold, 1.0) if yawn_percentage > 0 else 0
            # Give more weight to eye confidence (0.7) compared to yawn confidence (0.3)
            confidence = max(eye_confidence, yawn_confidence)

            # Adjust confidence based on normal state using a quadratic factor
            if normal_state_percentage > 0:
                normal_state_factor = (1 - normal_state_percentage / 100) ** 2
                confidence *= normal_state_factor

            reason = 'drowsy_indicators_present' if is_drowsy else 'no_significant_indicators'

            # Override drowsiness if confidence is very low due to normal state
            if is_drowsy and confidence < 0.15:
                is_drowsy = False
                reason = 'low_confidence_due_to_normal_state'

        result = {
            'is_drowsy': is_drowsy,
            'confidence': confidence,
            'details': {
                'eye_closed_percentage': eye_closed_percentage,
                'max_closure_duration': max_closure_duration,
                'yawn_percentage': yawn_percentage,
                'normal_state_percentage': normal_state_percentage,
                'is_drowsy_eyes': is_drowsy_eyes,
                'is_drowsy_yawns': is_drowsy_yawns,
                'is_drowsy_excessive_yawns': is_drowsy_excessive_yawns,
                'is_normal_state_high': is_normal_state_high,
                'is_head_turned': is_head_turned,
                'is_head_down': is_head_down,
                'head_turned_frames': head_turned_frames,
                'head_down_frames': head_down_frames,
                'yawn_frames': yawn_frames,
                'eye_closed_frames': eye_closed_frames,
                'max_consecutive_eye_closed': max_consecutive_eye_closed,
                'normal_state_frames': normal_state_frames,
                'reason': reason
            }
        }

        logging.info(f"Analysis result: {result}")
        return result


def create_analyzer(analyzer_type="rate"):
    """Create and return a drowsiness analyzer instance."""
    if analyzer_type == "rate":
        return RateBasedAnalyzer()
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")


# YOLO Processor
class YoloProcessor:
    """Processes videos with YOLO model for drowsiness detection."""

    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        # Load environment variables
        self.use_cuda = USE_CUDA
        # Extract model name from the path
        self.model_name = os.path.basename(model_path)
        self.model = self.load_model()
        # Add parameters for eye detection tuning
        self.min_blink_frames = int(os.getenv('MIN_BLINK_FRAMES', '1'))
        self.eye_closed_confidence = float(os.getenv('EYE_CLOSED_CONFIDENCE', '0.6'))
        self.yawn_confidence = float(os.getenv('YAWN_CONFIDENCE', '0.6'))
        self.normal_state_confidence = float(os.getenv('NORMAL_STATE_CONFIDENCE', '0.6'))
        # Initialize pose detector
        self.pose_detector = PoseHeadDetector()
        # Counters for consecutive frames
        self.consecutive_eye_closed = 0
        self.max_consecutive_eye_closed = 0
        # Tracking variables
        self.current_fps = 20  # Default FPS
        self.total_frames = 0
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.normal_state_frames = 0
        # Metrics
        self.metrics = {}

    def load_model(self):
        """Loads and returns a YOLO model for drowsiness detection."""
        logging.info("Loading YOLO model...")
        try:
            # Check if CUDA is available and enabled
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"CUDA available: {cuda_available}, Using device: {device}")

            # Load the model
            model = YOLO(self.model_path)

            # Move model to appropriate device
            model.to(device)

            # Set model parameters for inference
            model.conf = 0.25  # Confidence threshold
            model.iou = 0.45   # NMS IOU threshold

            if device == 'cuda':
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                # Set CUDA stream
                torch.cuda.set_stream(torch.cuda.Stream())

            logging.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None

    def reset_counters(self):
        """Reset all counters for a new video processing session."""
        self.consecutive_eye_closed = 0
        self.max_consecutive_eye_closed = 0
        self.total_frames = 0
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.normal_state_frames = 0
        self.metrics = {}

    def process_frame(self, frame):
        """Process a single frame with YOLO model."""
        try:
            # Ensure minimum frame size and proper aspect ratio
            min_size = 640
            height, width = frame.shape[:2]
            if height < min_size or width < min_size:
                scale = max(min_size/width, min_size/height)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Apply image enhancement
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

            # Process with main YOLO model
            results = self.model(frame, verbose=False)

            # Process with pose detector
            pose_results = self.pose_detector.process_frame(frame)

            if not results:
                return None

            # Return a dictionary with both results
            return {
                'detection_results': results,
                'pose_results': pose_results
            }
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None

    def download_video(self, video_url, temp_path="temp_video.mp4"):
        """Download video from URL to a temporary file."""
        try:
            # Create a hash of the URL to use as a unique filename
            url_hash = hashlib.md5(video_url.encode()).hexdigest()
            temp_path = f"temp_{url_hash}.mp4"

            # If file exists but is very small (likely corrupted), remove it
            if os.path.exists(temp_path):
                file_size = os.path.getsize(temp_path)
                if file_size < 1024:  # Less than 1KB
                    logging.warning(f"Found existing but potentially corrupted video file (size: {file_size} bytes). Removing it.")
                    os.remove(temp_path)
                else:
                    logging.info(f"Using cached video file: {temp_path} (size: {file_size / 1024:.2f} KB)")
                    return temp_path

            logging.info(f"Downloading video from {video_url}")
            response = requests.get(video_url, stream=True, timeout=30)  # Add timeout
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Check content type to ensure it's a video
            content_type = response.headers.get('content-type', '')
            if not ('video' in content_type or 'octet-stream' in content_type):
                logging.warning(f"Content type '{content_type}' may not be a video")

            # Get content length if available
            content_length = response.headers.get('content-length')
            if content_length:
                content_length = int(content_length)
                logging.info(f"Video file size: {content_length / 1024:.2f} KB")

            # Save the video to a temporary file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify the downloaded file
            if os.path.exists(temp_path):
                file_size = os.path.getsize(temp_path)
                if file_size < 1024:  # Less than 1KB
                    logging.error(f"Downloaded file is too small ({file_size} bytes), likely corrupted")
                    os.remove(temp_path)
                    return None
                logging.info(f"Successfully downloaded video to {temp_path} (size: {file_size / 1024:.2f} KB)")

            return temp_path
        except requests.exceptions.Timeout:
            logging.error(f"Timeout while downloading video from {video_url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error while downloading video: {e}")
            return None
        except Exception as e:
            logging.error(f"Error downloading video: {e}")
            return None

    def process_video(self, video_url):
        """Process a video for drowsiness detection."""
        # Reset counters for new video
        self.reset_counters()

        # Download the video
        temp_video_path = self.download_video(video_url)
        if not temp_video_path:
            logging.error("Failed to download video")
            return False, {}

        try:
            # Open the video file
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                logging.error(f"Error opening video file: {temp_video_path}")
                # Clean up the file if it can't be opened
                try:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        logging.info(f"Removed corrupted video file: {temp_video_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove corrupted video file: {e}")
                return False, {}

            # Get video properties
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_fps = cap.get(cv2.CAP_PROP_FPS)

            # Log video properties for debugging
            logging.info(f"Video properties - FPS: {self.current_fps}, Total frames: {self.total_frames}")

            # Validate video properties
            if self.total_frames <= 0 or self.current_fps <= 0:
                logging.error(f"Invalid video properties - FPS: {self.current_fps}, Total frames: {self.total_frames}")
                cap.release()
                # Clean up the file if it has invalid properties
                try:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        logging.info(f"Removed video file with invalid properties: {temp_video_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove video file with invalid properties: {e}")
                return False, {}

            # Use default FPS if needed
            if self.current_fps <= 0:
                self.current_fps = 20  # Default FPS if not available
                logging.warning(f"Using default FPS value: {self.current_fps}")

            # Update pose detector FPS
            self.pose_detector.update_frame_thresholds(self.current_fps)

            # Process frames
            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames for faster processing (process every 2nd frame)
                if frame_count % 2 != 0:
                    continue

                # Process the frame
                result = self.process_frame(frame)
                if result is None:
                    continue

                # Extract results
                detection_results = result['detection_results']
                pose_results = result['pose_results']

                # Update head pose information
                is_head_turned = pose_results.get('head_turned', False)
                is_head_down = pose_results.get('head_down', False)

                # Process YOLO detection results
                if len(detection_results) > 0:
                    # Get the first result (assuming single image input)
                    detection_result = detection_results[0]

                    # Check for detections
                    if len(detection_result.boxes) > 0:
                        # Get class IDs and confidences
                        cls_ids = detection_result.boxes.cls.cpu().numpy().astype(int)
                        confs = detection_result.boxes.conf.cpu().numpy()

                        # Check for eye closed detection
                        eye_closed_detected = False
                        for i, cls_id in enumerate(cls_ids):
                            if cls_id == 0 and confs[i] >= self.eye_closed_confidence:  # Assuming class 0 is 'eye_closed'
                                eye_closed_detected = True
                                self.eye_closed_frames += 1
                                self.consecutive_eye_closed += 1
                                break

                        if not eye_closed_detected:
                            self.consecutive_eye_closed = 0

                        # Update max consecutive eye closed frames
                        self.max_consecutive_eye_closed = max(self.max_consecutive_eye_closed, self.consecutive_eye_closed)

                        # Check for yawn detection
                        for i, cls_id in enumerate(cls_ids):
                            if cls_id == 2 and confs[i] >= self.yawn_confidence:  # Assuming class 2 is 'yawn'
                                self.yawn_frames += 1
                                break

                        # Check for normal state detection
                        for i, cls_id in enumerate(cls_ids):
                            if cls_id == 1 and confs[i] >= self.normal_state_confidence:  # Assuming class 1 is 'normal'
                                self.normal_state_frames += 1
                                break

            # Calculate processing time
            process_time = time.time() - start_time

            # Close the video file
            cap.release()

            # Prepare metrics
            self.metrics = {
                'fps': self.current_fps,
                'process_time': process_time,
                'processed_frames': frame_count
            }

            # Prepare detection results
            detection_results = {
                'yawn_frames': self.yawn_frames,
                'eye_closed_frames': self.eye_closed_frames,
                'max_consecutive_eye_closed': self.max_consecutive_eye_closed,
                'normal_state_frames': self.normal_state_frames,
                'total_frames': self.total_frames,
                'metrics': self.metrics,
                'head_pose': {
                    'head_turned': is_head_turned,
                    'head_down': is_head_down
                }
            }

            logging.info(f"Video processing complete: {detection_results}")

            # Clean up temporary file
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    logging.info(f"Removed temporary video file: {temp_video_path}")
            except Exception as e:
                logging.warning(f"Error removing temporary file: {e}")

            return True, detection_results

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            # Clean up temporary file
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except:
                pass
            return False, {}


# Initialize components
db_manager = DatabaseManager()
yolo_processor = YoloProcessor()
drowsiness_analyzer = create_analyzer(analyzer_type="rate")

# Create a thread pool for processing videos
# Use thread_name_prefix to make debugging easier
# Set max_workers explicitly to ensure all workers are created
processing_pool = ThreadPoolExecutor(
    max_workers=MAX_WORKERS,
    thread_name_prefix="VideoProcessor"
)
logging.info(f"Created ThreadPoolExecutor with {MAX_WORKERS} workers")

# Flag to control the background worker
shutdown_flag = threading.Event()

def send_webhook_notification(queue_id, evidence_id, status, video_url, results=None):
    """Send webhook notifications to all registered webhook URLs."""
    active_webhooks = db_manager.get_active_webhooks()
    if not active_webhooks:
        logging.info("No active webhooks found, skipping notifications")
        return

    # Prepare webhook payload
    payload = {
        'queue_id': queue_id,
        'evidence_id': evidence_id,
        'status': status,
        'video_url': video_url,
        'timestamp': datetime.datetime.now().isoformat()
    }

    # Add results if available
    if results:
        payload['results'] = results

    # Send to all active webhooks
    for webhook in active_webhooks:
        try:
            webhook_url = webhook['url']
            logging.info(f"Sending webhook notification to {webhook_url}")

            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10  # 10 second timeout
            )

            if response.status_code >= 200 and response.status_code < 300:
                logging.info(f"Webhook notification sent successfully to {webhook_url}")
            else:
                logging.warning(f"Webhook notification failed: {webhook_url}, Status: {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending webhook notification to {webhook['url']}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error sending webhook notification: {e}")

def process_video_task(queue_item):
    """Process a video from the queue."""
    queue_id = queue_item['id']
    video_url = queue_item['video_url']

    logging.info(f"Processing video from queue: {queue_id}, URL: {video_url}")

    try:
        # Process the video
        start_time = time.time()
        processing_success, detection_results = yolo_processor.process_video(video_url)
        process_time = time.time() - start_time

        if not processing_success:
            # Update queue status to failed
            db_manager.update_queue_status(queue_id, 'failed')

            # Store the failed evidence with status 'failed'
            try:
                # Create a minimal evidence record for the failed video
                failed_evidence = {
                    'video_url': video_url,
                    'processing_status': 'failed',
                    'process_time': process_time,
                    'details': json.dumps({
                        'error': 'Failed to process video',
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                }

                # Insert into database with a direct SQL query to handle the failed case
                with sqlite3.connect(db_manager.db_path) as conn:
                    conn.execute('''
                        INSERT INTO evidence_results (
                            video_url, process_time, processing_status, details
                        ) VALUES (?, ?, ?, ?)
                    ''', (
                        video_url,
                        process_time,
                        'failed',
                        failed_evidence['details']
                    ))
                    conn.commit()
                    logging.info(f"Stored failed evidence record for {video_url}")

                # Send webhook notification for failed processing
                send_webhook_notification(
                    queue_id=queue_id,
                    evidence_id=None,
                    status='failed',
                    video_url=video_url,
                    results={'error': 'Failed to process video'}
                )
            except Exception as e:
                logging.error(f"Error storing failed evidence record: {e}")

            return

        # Analyze drowsiness
        analysis_result = drowsiness_analyzer.analyze(detection_results)

        # Store results in database
        evidence_id = db_manager.store_evidence_result(
            video_url,
            detection_results,
            analysis_result,
            process_time,
            detection_results.get('head_pose')
        )

        # Update queue status
        db_manager.update_queue_status(queue_id, 'completed')

        logging.info(f"Completed processing video from queue: {queue_id}, Evidence ID: {evidence_id}")

        # Send webhook notification for successful processing
        # Get the full evidence result to include in the webhook
        evidence_result = db_manager.get_evidence_result(evidence_id)
        send_webhook_notification(
            queue_id=queue_id,
            evidence_id=evidence_id,
            status='completed',
            video_url=video_url,
            results=evidence_result
        )

    except Exception as e:
        logging.error(f"Error processing video from queue: {e}")
        db_manager.update_queue_status(queue_id, 'failed')

        # Send webhook notification for error
        send_webhook_notification(
            queue_id=queue_id,
            evidence_id=None,
            status='failed',
            video_url=video_url,
            results={'error': str(e)}
        )

def queue_worker():
    """Background worker to process videos from the queue."""
    logging.info("Starting queue worker thread")

    while not shutdown_flag.is_set():
        try:
            # Get current processing status
            stats = db_manager.get_queue_stats()
            processing_count = stats.get('processing', 0)
            pending_count = stats.get('pending', 0)

            # Calculate how many more tasks we can submit
            available_workers = MAX_WORKERS - processing_count

            logging.info(f"Queue worker check - Processing: {processing_count}, Pending: {pending_count}, Available workers: {available_workers}, Max workers: {MAX_WORKERS}")

            # If we have available workers and pending tasks, submit them
            if available_workers > 0 and pending_count > 0:
                # Submit up to available_workers tasks
                for _ in range(min(available_workers, pending_count)):
                    queue_item = db_manager.get_next_pending_video()
                    if queue_item:
                        # Submit the video for processing
                        processing_pool.submit(process_video_task, queue_item)
                        logging.info(f"Submitted video for processing: {queue_item['id']}")
                    else:
                        # This shouldn't happen, but just in case
                        logging.warning("Expected pending video but none found")
                        break
            elif pending_count == 0:
                logging.info("No pending videos in queue")
            else:
                logging.info(f"All workers busy ({processing_count}/{MAX_WORKERS})")

            # Wait before checking the queue again
            time.sleep(QUEUE_CHECK_INTERVAL)

        except Exception as e:
            logging.error(f"Error in queue worker: {e}")
            time.sleep(QUEUE_CHECK_INTERVAL)

    logging.info("Queue worker thread shutting down")

# Start the queue worker thread
# In Docker, daemon threads can be problematic, so we use a non-daemon thread
# and ensure proper cleanup on shutdown
queue_worker_thread = threading.Thread(target=queue_worker, daemon=False)
queue_worker_thread.start()
logging.info(f"Queue worker thread started with ID: {queue_worker_thread.ident}")


# API Endpoints
@app.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        'message': 'Simplified Drowsiness Detection API',
        'version': '1.1.0',
        'endpoints': [
            '/api/process',          # Add a video to the processing queue
            '/api/queue/<id>',       # Check the status of a queued video
            '/api/queue',            # Get queue statistics
            '/api/results',          # Get all processed videos
            '/api/result/<id>',      # Get details for a specific processed video
            '/api/webhook',          # Manage webhooks (GET, POST, DELETE)
        ]
    })


@app.route('/api/process', methods=['POST'])
def process_video():
    """Add a video to the processing queue."""
    try:
        # Get video URL from request
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing video_url in request'
            }), 400

        video_url = data['video_url']

        # Add to processing queue
        queue_id = db_manager.add_to_queue(video_url)
        if not queue_id:
            return jsonify({
                'success': False,
                'error': 'Failed to add video to processing queue'
            }), 500

        # Return queue ID immediately
        return jsonify({
            'success': True,
            'message': 'Video added to processing queue',
            'data': {
                'queue_id': queue_id,
                'status': 'pending',
                'status_url': f'/api/queue/{queue_id}'
            }
        })

    except Exception as e:
        logging.error(f"Error adding video to queue: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/queue/<int:queue_id>', methods=['GET'])
def get_queue_status(queue_id):
    """Get the status of a queued video."""
    try:
        # Get queue status
        queue_status = db_manager.get_queue_status(queue_id)
        if not queue_status:
            return jsonify({
                'success': False,
                'error': f'Queue item with ID {queue_id} not found'
            }), 404

        # Check if processing is complete
        if queue_status['status'] == 'completed':
            # Find the evidence result
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id FROM evidence_results WHERE video_url = ? ORDER BY created_at DESC LIMIT 1',
                    (queue_status['video_url'],)
                )
                evidence = cursor.fetchone()
                evidence_id = evidence['id'] if evidence else None

            return jsonify({
                'success': True,
                'data': {
                    'queue_id': queue_id,
                    'status': queue_status['status'],
                    'video_url': queue_status['video_url'],
                    'created_at': queue_status['created_at'],
                    'evidence_id': evidence_id,
                    'result_url': f'/api/result/{evidence_id}' if evidence_id else None
                }
            })
        elif queue_status['status'] == 'failed':
            return jsonify({
                'success': True,
                'data': {
                    'queue_id': queue_id,
                    'status': queue_status['status'],
                    'video_url': queue_status['video_url'],
                    'created_at': queue_status['created_at'],
                    'error': 'Video processing failed'
                }
            })
        else:  # pending or processing
            return jsonify({
                'success': True,
                'data': {
                    'queue_id': queue_id,
                    'status': queue_status['status'],
                    'video_url': queue_status['video_url'],
                    'created_at': queue_status['created_at']
                }
            })

    except Exception as e:
        logging.error(f"Error getting queue status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/queue', methods=['GET'])
def get_queue_stats():
    """Get statistics about the processing queue."""
    try:
        # Get queue stats
        stats = db_manager.get_queue_stats()

        # Get thread pool information
        active_workers = min(MAX_WORKERS, stats.get('processing', 0))
        pending_tasks = stats.get('pending', 0)
        completed_tasks = stats.get('completed', 0)
        failed_tasks = stats.get('failed', 0)

        # Calculate utilization percentage
        worker_utilization = (active_workers / MAX_WORKERS) * 100 if MAX_WORKERS > 0 else 0

        return jsonify({
            'success': True,
            'data': {
                'stats': stats,
                'worker_threads': {
                    'max': MAX_WORKERS,
                    'active': active_workers,
                    'utilization_percent': worker_utilization
                },
                'tasks': {
                    'pending': pending_tasks,
                    'processing': stats.get('processing', 0),
                    'completed': completed_tasks,
                    'failed': failed_tasks,
                    'total': pending_tasks + stats.get('processing', 0) + completed_tasks + failed_tasks
                },
                'queue_check_interval': QUEUE_CHECK_INTERVAL
            }
        })

    except Exception as e:
        logging.error(f"Error getting queue stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/results')
def get_results():
    """Get all evidence results with pagination."""
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Calculate offset
        offset = (page - 1) * per_page

        # Get results from database
        results = db_manager.get_all_evidence_results(per_page, offset)

        return jsonify({
            'success': True,
            'data': results
        })

    except Exception as e:
        logging.error(f"Error getting results: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/result/<int:evidence_id>')
def get_result(evidence_id):
    """Get a specific evidence result by ID."""
    try:
        # Get result from database
        result = db_manager.get_evidence_result(evidence_id)

        if not result:
            return jsonify({
                'success': False,
                'error': f'Evidence result with ID {evidence_id} not found'
            }), 404

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logging.error(f"Error getting result: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/webhook', methods=['GET', 'POST', 'DELETE'])
def manage_webhooks():
    """Manage webhooks for video processing notifications."""
    try:
        # GET: List all webhooks
        if request.method == 'GET':
            webhooks = db_manager.get_all_webhooks()
            return jsonify({
                'success': True,
                'data': webhooks
            })

        # POST: Add a new webhook
        elif request.method == 'POST':
            data = request.get_json()
            if not data or 'url' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing url in request'
                }), 400

            webhook_url = data['url']

            # Validate URL format
            try:
                parsed_url = requests.utils.urlparse(webhook_url)
                if not all([parsed_url.scheme, parsed_url.netloc]):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid URL format'
                    }), 400
            except Exception:
                return jsonify({
                    'success': False,
                    'error': 'Invalid URL format'
                }), 400

            # Add webhook to database
            webhook_id = db_manager.add_webhook(webhook_url)
            if not webhook_id:
                return jsonify({
                    'success': False,
                    'error': 'Failed to add webhook'
                }), 500

            return jsonify({
                'success': True,
                'message': 'Webhook added successfully',
                'data': {
                    'webhook_id': webhook_id,
                    'url': webhook_url
                }
            })

        # DELETE: Remove a webhook
        elif request.method == 'DELETE':
            data = request.get_json()
            if not data or 'webhook_id' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing webhook_id in request'
                }), 400

            webhook_id = data['webhook_id']

            # Delete webhook from database
            success = db_manager.delete_webhook(webhook_id)
            if not success:
                return jsonify({
                    'success': False,
                    'error': f'Failed to delete webhook with ID {webhook_id}'
                }), 500

            return jsonify({
                'success': True,
                'message': f'Webhook with ID {webhook_id} deleted successfully'
            })

    except Exception as e:
        logging.error(f"Error managing webhooks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Signal handler for graceful shutdown
def signal_handler(sig, _):
    logging.info(f"Received signal {sig}, shutting down...")
    # Signal the queue worker to stop
    shutdown_flag.set()
    # Wait for the queue worker to finish
    logging.info("Waiting for queue worker to finish...")
    queue_worker_thread.join(timeout=10)
    # Shutdown the thread pool
    logging.info("Shutting down thread pool...")
    processing_pool.shutdown(wait=True, cancel_futures=False)
    logging.info("Shutdown complete")
    import sys
    sys.exit(0)

# Main function
if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info(f"Starting Simplified Drowsiness Detection API on port {PORT}")
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        signal_handler(signal.SIGTERM, None)