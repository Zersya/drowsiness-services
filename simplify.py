import cv2
import os
import logging
import sqlite3
import time
import datetime
import json
import hashlib
import math
import numpy as np
import torch
from ultralytics import YOLO
from abc import ABC, abstractmethod
from flask import Flask, request, jsonify, send_file
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

    def get_processing_time_stats(self):
        """Calculate average processing time and queue wait time."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Calculate average processing time from completed evidence results
                cursor = conn.execute('''
                    SELECT AVG(process_time) as avg_processing_time
                    FROM evidence_results
                    WHERE processing_status = 'processed'
                    AND process_time IS NOT NULL
                ''')
                result = cursor.fetchone()
                avg_processing_time = result['avg_processing_time'] if result and result['avg_processing_time'] is not None else 0

                # Calculate average queue wait time
                # This is more complex as we need to join tables and calculate time differences
                cursor = conn.execute('''
                    SELECT
                        AVG(
                            (julianday(er.created_at) - julianday(pq.created_at)) * 24 * 60 * 60
                        ) as avg_queue_wait_time
                    FROM
                        evidence_results er
                    JOIN
                        processing_queue pq ON er.video_url = pq.video_url
                    WHERE
                        pq.status = 'completed'
                ''')
                result = cursor.fetchone()
                avg_queue_wait_time = result['avg_queue_wait_time'] if result and result['avg_queue_wait_time'] is not None else 0

                return {
                    'avg_processing_time': avg_processing_time,
                    'avg_queue_wait_time': avg_queue_wait_time
                }
        except sqlite3.Error as e:
            logging.error(f"Error calculating processing time stats: {e}")
            return {
                'avg_processing_time': 0,
                'avg_queue_wait_time': 0
            }

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
    """
    Class for detecting head pose (head down or head turn) using YOLOv8 pose model.
    Logic updated to align with the version from pose_head_detector.py for potentially improved performance.
    """

    def __init__(self, model_path=None): # Use global POSE_MODEL_PATH if None
        if model_path is None:
            model_path = os.getenv("POSE_MODEL_PATH", "yolov8l-pose.pt")

        self.model_path = model_path
        self.use_cuda = os.getenv('USE_CUDA', 'true').lower() == 'true'

        # Configuration parameters from pose_head_detector.py (and simplify.py uses similar env vars)
        self.keypoint_conf_threshold = float(os.getenv('KEYPOINT_CONF_THRESHOLD', '0.5'))
        # Head turn: ratio of nose deviation from eye center to inter-eye distance
        self.head_turn_ratio_threshold = float(os.getenv('HEAD_TURN_RATIO_THRESHOLD', '0.7')) # php.py uses 0.7
        # Head down: ratio of nose vertical drop from eye center to inter-eye distance (horizontal)
        self.head_down_ratio_threshold = float(os.getenv('HEAD_DOWN_RATIO_THRESHOLD', '0.3')) # php.py uses 0.3

        # Time thresholds (in seconds)
        self.head_turned_threshold_seconds = float(os.getenv('HEAD_TURNED_THRESHOLD_SECONDS', '1.5'))
        self.head_down_threshold_seconds = float(os.getenv('HEAD_DOWN_THRESHOLD_SECONDS', '1.5'))

        # Counters
        self.head_turned_counter = 0
        self.head_down_counter = 0
        
        # Status flags
        self.distracted_head_turn = False
        self.distracted_head_down = False
        
        # Define COCO keypoint indices (from pose_head_detector.py - only needs these three for its logic)
        self.kp_indices = {
            "nose": 0, "left_eye": 1, "right_eye": 2,
            # "left_ear": 3, "right_ear": 4, # Not used by php.py logic
            # "left_shoulder": 5, "right_shoulder": 6 # Not used by php.py logic
        }
        
        self.model = self.load_model()

        # Frame thresholds will be set by update_frame_thresholds() or a reset method
        self.fps = 20 # Default, should be updated
        self.head_turned_frames_threshold = int(self.head_turned_threshold_seconds * self.fps)
        self.head_down_frames_threshold = int(self.head_down_threshold_seconds * self.fps)


    def load_model(self):
        """Loads and returns a YOLO pose model."""
        logging.info(f"Loading YOLO pose model from {self.model_path}...")
        try:
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"Pose model: CUDA available: {cuda_available}, Using device: {device}")

            model = YOLO(self.model_path)
            model.to(device)
            # model.conf = 0.25 # General confidence threshold for detection (php.py sets this)
            # model.iou = 0.45  # NMS IOU threshold (php.py sets this)
            # Note: simplify.py's YoloProcessor sets these for its main model.
            # It's good practice for PoseHeadDetector to also set them if its YOLO instance is separate.
            logging.info(f"YOLO pose model loaded successfully from {self.model_path} on {device}")
            return model
        except Exception as e:
            logging.error(f"Error loading YOLO pose model: {e}")
            return None

    def update_frame_thresholds(self, fps=None):
        """
        Update frame thresholds based on FPS. Also resets counters and status flags.
        This combines simplify.py's update_frame_thresholds and php.py's reset_frame_counters.
        """
        if fps is not None and fps > 0:
            self.fps = fps
        
        self.head_turned_frames_threshold = max(1, int(self.head_turned_threshold_seconds * self.fps))
        self.head_down_frames_threshold = max(1, int(self.head_down_threshold_seconds * self.fps))
        
        # Reset counters and status flags (from php.py's reset_frame_counters)
        self.head_turned_counter = 0
        self.head_down_counter = 0
        self.distracted_head_turn = False
        self.distracted_head_down = False
        logging.debug(f"PoseHeadDetector frame thresholds updated for FPS {self.fps}: "
                     f"Turn Thresh={self.head_turned_frames_threshold} frames, "
                     f"Down Thresh={self.head_down_frames_threshold} frames")

    def process_frame(self, frame):
        """
        Process a single frame to detect head pose using logic from pose_head_detector.py.
        Returns a dictionary with head pose information.
        """
        if self.model is None:
            logging.warning("Pose model not loaded, returning default pose.")
            return {
                "head_turned": False, "head_down": False,
                "head_turned_counter": 0, "head_down_counter": 0,
                "head_turned_threshold": self.head_turned_frames_threshold,
                "head_down_threshold": self.head_down_frames_threshold
            }
        
        frame_flag_head_turned = False
        frame_flag_head_down = False
            
        try:
            # Perform pose estimation
            # By default, YOLO uses its internal confidence thresholds for detection.
            # If specific conf/iou needed for this model instance, set them on self.model after loading.
            results = self.model(frame, verbose=False) 
            
            if results and len(results) > 0:
                result_item = results[0] # Assuming single image, first result
                
                # Check if keypoints data is available and not empty
                if hasattr(result_item, 'keypoints') and result_item.keypoints is not None and \
                   hasattr(result_item.keypoints, 'data') and result_item.keypoints.data.shape[0] > 0:
                    
                    keypoints_data_tensor = result_item.keypoints.data[0] # Assuming first detected person
                    keypoints_cpu = keypoints_data_tensor.cpu().numpy()

                    def get_kp(name): # Helper to get keypoint coords and confidence
                        idx = self.kp_indices.get(name, -1)
                        if idx != -1 and idx < len(keypoints_cpu):
                            x, y, conf = keypoints_cpu[idx]
                            # Ensure coordinates are valid before int conversion
                            if not (np.isnan(x) or np.isnan(y)):
                                return (int(x), int(y)), float(conf)
                        return (0,0), 0.0
                    
                    nose_xy, n_conf = get_kp("nose")
                    left_eye_xy, le_conf = get_kp("left_eye")
                    right_eye_xy, re_conf = get_kp("right_eye")

                    if n_conf > self.keypoint_conf_threshold and \
                       le_conf > self.keypoint_conf_threshold and \
                       re_conf > self.keypoint_conf_threshold:

                        # --- Head Turn Check (from pose_head_detector.py) ---
                        eye_center_x = (left_eye_xy[0] + right_eye_xy[0]) / 2.0
                        eye_dist_x_pixels = abs(left_eye_xy[0] - right_eye_xy[0])
                        
                        # Check eye_dist_x_pixels to prevent division by zero or instability if eyes are too close/same point
                        if eye_dist_x_pixels > 5: # A small threshold for inter-eye distance
                            nose_deviation_from_center_x = abs(nose_xy[0] - eye_center_x)
                            if nose_deviation_from_center_x > (eye_dist_x_pixels * self.head_turn_ratio_threshold):
                                frame_flag_head_turned = True
                        
                        # --- Head Down Check (from pose_head_detector.py) ---
                        eye_center_y = (left_eye_xy[1] + right_eye_xy[1]) / 2.0
                        # Use horizontal eye distance as a stable reference for scaling vertical deviation
                        # eye_dist_x_pixels is already calculated above.
                        
                        if eye_dist_x_pixels > 5: # Again, ensure eyes are reasonably detected
                            nose_vertical_drop_from_eye_center = nose_xy[1] - eye_center_y # Positive if nose is below eye center
                            if nose_vertical_drop_from_eye_center > (eye_dist_x_pixels * self.head_down_ratio_threshold):
                                frame_flag_head_down = True
                    else:
                        logging.debug("Not all required keypoints (nose, eyes) found or confident enough for pose analysis.")
                else:
                    logging.debug("No keypoints detected in the frame or keypoints data is empty.")
            else:
                logging.debug("No pose estimation results from model.")

            # Update temporal counters
            self.head_turned_counter = (self.head_turned_counter + 1) if frame_flag_head_turned else 0
            self.head_down_counter = (self.head_down_counter + 1) if frame_flag_head_down else 0
            
            # Determine final status based on accumulated frames vs thresholds
            self.distracted_head_turn = self.head_turned_counter >= self.head_turned_frames_threshold
            self.distracted_head_down = self.head_down_counter >= self.head_down_frames_threshold
            
        except Exception as e:
            # Log the full error for debugging
            logging.exception(f"Error processing frame for head pose: {e}") 
            # In case of error, default to not distracted
            self.distracted_head_turn = False
            self.distracted_head_down = False
            # Counters might be in an intermediate state, but will be reset if the condition isn't met next frame.

        return {
            "head_turned": self.distracted_head_turn,
            "head_down": self.distracted_head_down,
            "head_turned_counter": self.head_turned_counter,
            "head_down_counter": self.head_down_counter,
            "head_turned_threshold": self.head_turned_frames_threshold,
            "head_down_threshold": self.head_down_frames_threshold
        }


# Drowsiness Analyzer
class DrowsinessAnalyzer(ABC):
    """Abstract base class for drowsiness analysis implementations."""

    @abstractmethod
    def analyze(self, detection_results):
        """Analyze drowsiness based on detection metrics."""
        pass
    
class RateBasedAnalyzer(DrowsinessAnalyzer):
    """
    Rate-based drowsiness analysis.
    Internal logic revised to align with drowsiness_analyzer.py for improved accuracy.
    Output structure of 'analyze' method maintained for backward compatibility.
    """

    def __init__(self, eye_closed_percentage_threshold=5, yawn_rate_threshold=3,
                 normal_state_threshold=60, fps=20, max_closure_duration_threshold=0.3,
                 minimum_yawn_threshold=1, minimum_eye_closed_threshold=3,
                 minimum_frames_for_analysis=10):
        """
        Initialize with thresholds. Parameters aligned with the new core logic.
        """
        self.eye_closed_percentage_threshold = eye_closed_percentage_threshold
        self.yawn_rate_threshold = yawn_rate_threshold # For internal logic
        self.normal_state_threshold = normal_state_threshold
        self.fps = fps
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.minimum_yawn_threshold = minimum_yawn_threshold
        self.minimum_eye_closed_threshold = minimum_eye_closed_threshold
        self.minimum_frames_for_analysis = minimum_frames_for_analysis
        
        # Original simplify.py thresholds that might be needed for populating 'details'
        # or for calculating backward-compatible boolean flags if their logic was tied to these exact names.
        # For instance, if the original 'is_drowsy_yawns' used 'yawn_percentage_threshold'.
        # We'll use the new core logic for these flags, but retain original names in details.
        # self.original_yawn_percentage_threshold = 5 # Example if needed for a specific detail field

    def analyze(self, detection_results):
        """
        Analyze drowsiness. Returns results with a structure compatible with the original simplify.py.
        """
        yawn_frames_input = detection_results.get('yawn_frames', 0)
        eye_closed_event_count = detection_results.get('eye_closed_frames', 0)
        normal_state_frames_input = detection_results.get('normal_state_frames', 0)
        total_frames_input = detection_results.get('total_frames', 0)
        max_consecutive_eye_closed_input = detection_results.get('max_consecutive_eye_closed', 0)
        
        current_fps = detection_results.get('metrics', {}).get('fps', self.fps)
        if current_fps <= 0:
            current_fps = self.fps

        head_pose = detection_results.get('head_pose', {})
        is_head_turned = head_pose.get('head_turned', False)
        is_head_down = head_pose.get('head_down', False)
        # For details, ensure these are available or set to 0 if not.
        head_turned_frames_val = head_pose.get('head_turned_counter', 0) 
        head_down_frames_val = head_pose.get('head_down_counter', 0)

        if total_frames_input < self.minimum_frames_for_analysis:
            # Construct details for insufficient frames, matching original keys
            details_output = {
                'eye_closed_percentage': 0.0,
                'max_closure_duration': 0.0,
                'yawn_percentage': 0.0, # Original key
                'normal_state_percentage': 0.0,
                'reason': 'insufficient_frames',
                'yawn_frames': yawn_frames_input,
                'eye_closed_frames': eye_closed_event_count,
                'max_consecutive_eye_closed': max_consecutive_eye_closed_input,
                'normal_state_frames': normal_state_frames_input,
                'total_frames': total_frames_input,
                'fps': current_fps,
                'is_drowsy_eyes': False,
                'is_drowsy_yawns': False,
                'is_drowsy_excessive_yawns': False,
                'is_normal_state_high': False, # Or based on normal_state_percentage if calculable
                'is_head_turned': is_head_turned,
                'is_head_down': is_head_down,
                'head_turned_frames': head_turned_frames_val,
                'head_down_frames': head_down_frames_val,
            }
            return {
                'is_drowsy': False,
                'confidence': 0.0,
                'details': details_output
            }

        time_in_seconds = total_frames_input / current_fps if current_fps > 0 else 0
        time_in_minutes = time_in_seconds / 60 if time_in_seconds > 0 else 0

        # Metrics for internal logic (drowsiness_analyzer.py inspired)
        internal_eye_closed_percentage = (eye_closed_event_count / total_frames_input) * 100 if total_frames_input > 0 else 0
        internal_max_closure_duration = max_consecutive_eye_closed_input / current_fps if current_fps > 0 else 0
        internal_yawn_rate_per_minute = yawn_frames_input / time_in_minutes if time_in_minutes > 0 else 0
        internal_normal_state_percentage = (normal_state_frames_input / total_frames_input) * 100 if total_frames_input > 0 else 0

        # Drowsiness indicators for internal logic
        internal_is_drowsy_eyes = (
            (internal_eye_closed_percentage > self.eye_closed_percentage_threshold) or
            (internal_max_closure_duration > self.max_closure_duration_threshold) or
            (eye_closed_event_count >= self.minimum_eye_closed_threshold)
        )
        internal_is_drowsy_yawns = (
            (yawn_frames_input >= self.minimum_yawn_threshold) and
            (internal_yawn_rate_per_minute > self.yawn_rate_threshold)
        )
        # Heuristic for excessive yawns based on drowsiness_analyzer.py
        # Adjust 10 if yawn_frames_input (frames) is very different from yawn_count (events)
        # Example: if a yawn event lasts avg 2 sec at 20fps (40 frames), 10 events = 400 frames.
        # Or keep it simpler:
        internal_is_drowsy_excessive_yawns = (yawn_frames_input > (10 * current_fps * 0.5) ) or (internal_yawn_rate_per_minute > 100) # ~10 events, assuming 0.5s of detected frames per event average

        internal_is_normal_state_high = internal_normal_state_percentage >= self.normal_state_threshold

        # Core drowsiness decision logic (inspired by drowsiness_analyzer.py)
        final_is_drowsy = False
        final_confidence = 0.0
        reason_for_drowsiness = ''

        if internal_is_drowsy_excessive_yawns:
            final_is_drowsy = True
            final_confidence = 1.0
            reason_for_drowsiness = 'excessive_yawns'
        elif internal_is_drowsy_yawns and internal_yawn_rate_per_minute > 60:
             final_is_drowsy = True
             final_confidence = 0.8
             reason_for_drowsiness = 'high_yawn_rate'
        elif internal_is_normal_state_high and not (internal_is_drowsy_yawns or internal_is_drowsy_eyes):
            final_is_drowsy = False
            final_confidence = 0.1 # Small confidence even if not drowsy, if normal state is high
            reason_for_drowsiness = 'high_normal_state_no_other_indicators'
        else:
            if is_head_turned or is_head_down: # Head pose override
                final_is_drowsy = False
                final_confidence = 0.0 # Or a small confidence indicating distraction
                reason_for_drowsiness = 'head_pose_override'
            else:
                final_is_drowsy = internal_is_drowsy_eyes or internal_is_drowsy_yawns

            # Confidence calculation
            eye_perc_conf = min(internal_eye_closed_percentage / self.eye_closed_percentage_threshold, 1.0) if self.eye_closed_percentage_threshold > 0 and internal_eye_closed_percentage > 0 else 0
            eye_dur_conf = min(internal_max_closure_duration / self.max_closure_duration_threshold, 1.0) if self.max_closure_duration_threshold > 0 and internal_max_closure_duration > 0 else 0
            eye_count_conf = min(eye_closed_event_count / self.minimum_eye_closed_threshold, 1.0) if self.minimum_eye_closed_threshold > 0 and eye_closed_event_count > 0 else 0
            
            current_eye_confidence = max(eye_perc_conf, eye_dur_conf, eye_count_conf)
            
            current_yawn_confidence = 0
            if self.minimum_yawn_threshold > 0 and yawn_frames_input >= self.minimum_yawn_threshold and internal_yawn_rate_per_minute > self.yawn_rate_threshold:
                 current_yawn_confidence = min(yawn_frames_input / self.minimum_yawn_threshold, 1.0) # Based on input count vs min count

            calculated_confidence = max(current_eye_confidence, current_yawn_confidence)

            if internal_normal_state_percentage > 0:
                normal_state_factor = (1 - (internal_normal_state_percentage / 100)) ** 2
                calculated_confidence *= normal_state_factor
            
            final_confidence = calculated_confidence

            if not reason_for_drowsiness: # If not set by head_pose_override
                reason_for_drowsiness = 'drowsy_indicators_present' if final_is_drowsy else 'no_significant_indicators'

            if final_is_drowsy and final_confidence < 0.15:
                final_is_drowsy = False
                reason_for_drowsiness = 'low_confidence_override'
        
        final_confidence = max(0.0, min(final_confidence, 1.0))

        # --- Construct details dictionary for backward compatibility ---
        # Calculate metrics for output details using original key names
        output_eye_closed_percentage = internal_eye_closed_percentage # Uses event-based calculation
        output_max_closure_duration = internal_max_closure_duration
        # For 'yawn_percentage', use the original calculation if different from internal logic.
        output_yawn_percentage = (yawn_frames_input / total_frames_input) * 100 if total_frames_input > 0 else 0
        output_normal_state_percentage = internal_normal_state_percentage

        details_output = {
            'eye_closed_percentage': output_eye_closed_percentage,
            'max_closure_duration': output_max_closure_duration,
            'yawn_percentage': output_yawn_percentage, # Original key, original calculation
            'normal_state_percentage': output_normal_state_percentage,
            'reason': reason_for_drowsiness,
            'yawn_frames': yawn_frames_input,
            'eye_closed_frames': eye_closed_event_count, # Event count
            'max_consecutive_eye_closed': max_consecutive_eye_closed_input, # In frames
            'normal_state_frames': normal_state_frames_input,
            'total_frames': total_frames_input,
            'fps': current_fps,
            # Boolean flags based on new internal logic
            'is_drowsy_eyes': internal_is_drowsy_eyes,
            'is_drowsy_yawns': internal_is_drowsy_yawns,
            'is_drowsy_excessive_yawns': internal_is_drowsy_excessive_yawns,
            'is_normal_state_high': internal_is_normal_state_high,
            'is_head_turned': is_head_turned,
            'is_head_down': is_head_down,
            'head_turned_frames': head_turned_frames_val,
            'head_down_frames': head_down_frames_val,
            # Add any other fields from original 'details' if they were simple pass-throughs
            # or calculated in a way that's still relevant and non-breaking.
            # OMITTING fields like 'perclos_score', 'raw_drowsiness_score' from the old complex system.
        }
        
        logging.info(
            f"RateBasedAnalyzer (simplify.py compat) output: "
            f"is_drowsy:{final_is_drowsy}, confidence:{final_confidence:.2f}, "
            f"reason:{reason_for_drowsiness}"
        )

        return {
            'is_drowsy': final_is_drowsy,
            'confidence': final_confidence,
            'details': details_output
        }

def create_analyzer(analyzer_type="rate"):
    """Create and return a drowsiness analyzer instance."""
    if analyzer_type == "rate":
        return RateBasedAnalyzer()
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")


class YoloProcessor:
    def __init__(self, model_path=None): # Use global YOLO_MODEL_PATH if None
        if model_path is None:
            model_path = os.getenv("YOLO_MODEL_PATH", "models/jingyeong-best.pt") # Match simplify.py global
        
        self.model_path = model_path
        self.use_cuda = os.getenv('USE_CUDA', 'true').lower() == 'true' # Match simplify.py global
        self.model_name = os.path.basename(self.model_path)
        self.model = self.load_model()

        # Confidence thresholds from simplify.py (can be tuned)
        self.eye_closed_confidence = float(os.getenv('EYE_CLOSED_CONFIDENCE', '0.6'))
        self.yawn_confidence = float(os.getenv('YAWN_CONFIDENCE', '0.6'))
        self.normal_state_confidence = float(os.getenv('NORMAL_STATE_CONFIDENCE', '0.6'))

        # Blink detection parameters (consider aligning with yolo_processor.py if those values are better)
        self.min_blink_frames = int(os.getenv('MIN_BLINK_FRAMES', '1'))  # yolo_processor.py uses 1
        self.blink_cooldown = int(os.getenv('BLINK_COOLDOWN', '2'))    # yolo_processor.py uses 2

        # Initialize pose detector (simplify.py's PoseHeadDetector is used)
        # It sources its own model path via os.getenv("POSE_MODEL_PATH", "yolov8l-pose.pt")
        self.pose_detector = PoseHeadDetector()
        
        # Initialize instance variables that will be reset by reset_counters()
        self.reset_counters()


    def load_model(self):
        """Loads and returns a YOLO model for drowsiness detection."""
        logging.info(f"Loading YOLO model from {self.model_path}...")
        try:
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"YOLO model: CUDA available: {cuda_available}, Using device: {device}")

            model = YOLO(self.model_path)
            model.to(device)
            # Set common inference parameters (adjust if needed)
            # model.conf = 0.25  # General confidence threshold for detection
            # model.iou = 0.45   # NMS IOU threshold
            logging.info(f"YOLO model loaded successfully from {self.model_path} on {device}")
            return model
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            return None

    def download_video(self, video_url, temp_path="temp_video.mp4"): # Matching simplify.py's signature
        """Download video from URL to a temporary file."""
        # Using simplify.py's existing robust download_video logic
        try:
            url_hash = hashlib.md5(video_url.encode()).hexdigest()
            temp_path = f"temp_{url_hash}.mp4"

            if os.path.exists(temp_path):
                file_size = os.path.getsize(temp_path)
                if file_size < 1024:
                    logging.warning(f"Found existing but potentially corrupted video file (size: {file_size} bytes). Removing it.")
                    os.remove(temp_path)
                else:
                    logging.info(f"Using cached video file: {temp_path} (size: {file_size / 1024:.2f} KB)")
                    return temp_path

            logging.info(f"Downloading video from {video_url}")
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            if not ('video' in content_type or 'octet-stream' in content_type):
                logging.warning(f"Content type '{content_type}' may not be a video")

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.exists(temp_path):
                file_size = os.path.getsize(temp_path)
                if file_size < 1024:
                    logging.error(f"Downloaded file is too small ({file_size} bytes), likely corrupted")
                    self._cleanup_temp_file(temp_path) # Use consistent cleanup
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
            if 'temp_path' in locals() and os.path.exists(temp_path): # Ensure cleanup on other exceptions
                self._cleanup_temp_file(temp_path)
            return None


    def process_frame(self, frame):
        """Process a single frame with YOLO model and PoseHeadDetector."""
        try:
            min_size = 640 # Standard processing size
            height, width = frame.shape[:2]
            if height < min_size or width < min_size: # Upscale if too small
                scale = max(min_size/width, min_size/height)
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
            # Image enhancement (from both simplify.py and yolo_processor.py)
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

            yolo_model_outputs = self.model(frame, verbose=False) 
            pose_detector_outputs = self.pose_detector.process_frame(frame)

            if not yolo_model_outputs: # No YOLO detections
                # Still return pose if available, or None if pose also failed.
                # For consistency with simplify.py, if yolo fails, frame processing failed.
                return None 
            
            return {
                'detection_results': yolo_model_outputs, # List of ultralytics.engine.results.Results
                'pose_results': pose_detector_outputs  # Dict from PoseHeadDetector
            }
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None

    def reset_counters(self):
        """Reset all relevant counters for a new video processing session."""
        # Counters from original simplify.py
        self.consecutive_eye_closed = 0
        self.max_consecutive_eye_closed = 0 
        self.eye_closed_frames = 0      # Blink/closure EVENTS
        self.yawn_frames = 0            # FRAMES with any yawn detection
        self.normal_state_frames = 0    # FRAMES with any normal state detection
        self.blink_cooldown_counter = 0 
        self.potential_blink_frames = 0 
        
        # Video properties to be set per video
        self.total_frames_from_video_file = 0 
        self.current_video_fps = 20 # Default, to be updated

        # These are effectively local to process_video but defined here for clarity if needed by other methods
        # self.yawn_detections_count_current_video = 0
        # self.total_eye_closed_duration_frames_current_video = 0


    def _cleanup_temp_file(self, file_path):
        """Safely remove a temporary file."""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logging.debug(f"Removed temporary video file: {file_path}")
        except Exception as e:
            logging.warning(f"Error removing temporary file {file_path}: {e}")

    def process_video(self, video_url):
        """
        Process a video for drowsiness detection.
        Output structure is kept compatible with original simplify.py,
        with necessary additions for improved analysis.
        """
        self.reset_counters() 

        temp_video_path = self.download_video(video_url)
        if not temp_video_path:
            return False, {'error': 'Failed to download video', 'reason': 'Download process failed or returned None'}

        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                logging.error(f"Error opening video file: {temp_video_path}")
                self._cleanup_temp_file(temp_video_path)
                return False, {'error': f'Error opening video file: {temp_video_path}'}

            self.total_frames_from_video_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_video_fps = cap.get(cv2.CAP_PROP_FPS)

            if self.total_frames_from_video_file <= 0 :
                logging.error(f"Invalid video properties: Total frames {self.total_frames_from_video_file}")
                cap.release()
                self._cleanup_temp_file(temp_video_path)
                return False, {'error': 'Invalid video properties (total frames zero or negative)'}
            if self.current_video_fps <= 0:
                logging.warning(f"Invalid video FPS {self.current_video_fps}, using default 20 FPS.")
                self.current_video_fps = 20 

            self.pose_detector.update_frame_thresholds(self.current_video_fps)

            # Local accumulators for metrics specific to yolo_processor.py's enhanced calculations
            yawn_detections_count_current_video = 0
            total_eye_closed_duration_frames_current_video = 0
            
            processed_frames_count = 0 
            video_processing_start_time = time.time()

            # Frame skipping logic from yolo_processor.py (targets ~10 FPS processing)
            frame_skip_interval = max(1, int(self.current_video_fps / 10))
            current_frame_index_in_video = -1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_index_in_video += 1

                if current_frame_index_in_video % frame_skip_interval != 0:
                    continue
                
                if frame is None or frame.size == 0: # Additional check
                    logging.warning(f"Skipping empty frame at index {current_frame_index_in_video}")
                    continue

                processed_frames_count += 1
                frame_analysis_result = self.process_frame(frame)

                if frame_analysis_result is None:
                    continue

                yolo_outputs_list = frame_analysis_result['detection_results'] # List of Results objects
                pose_info_dict = frame_analysis_result['pose_results']    # Dict from PoseHeadDetector

                frame_had_yawn_detected = False
                frame_had_eye_closed_detected = False
                # frame_had_normal_state_detected = False # Not strictly needed for counters if +=1 is per detection

                if yolo_outputs_list and len(yolo_outputs_list) > 0:
                    # Typically, for single frame processing, yolo_outputs_list contains one Results object
                    yolo_result_item = yolo_outputs_list[0]
                    
                    # Check if boxes attribute exists and is not None
                    if hasattr(yolo_result_item, 'boxes') and yolo_result_item.boxes is not None:
                        boxes = yolo_result_item.boxes
                        # Further check if cls and conf attributes exist
                        cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') and boxes.cls is not None else np.array([], dtype=int)
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') and boxes.conf is not None else np.array([], dtype=float)

                        # Normal State (class 1)
                        for i, cls_id in enumerate(cls_ids):
                            if cls_id == 1 and confs[i] >= self.normal_state_confidence:
                                self.normal_state_frames += 1 # simplify.py original logic: count frames with normal state
                                # frame_had_normal_state_detected = True # Not strictly needed if only counting once per frame
                                break 
                        
                        # Yawn (class 2)
                        current_frame_yawn_detections = 0
                        for i, cls_id in enumerate(cls_ids):
                            if cls_id == 2 and confs[i] >= self.yawn_confidence:
                                current_frame_yawn_detections += 1
                                frame_had_yawn_detected = True
                        yawn_detections_count_current_video += current_frame_yawn_detections
                        if frame_had_yawn_detected:
                            self.yawn_frames += 1 # simplify.py original: count frames with any yawn

                        # Closed Eyes (class 0)
                        num_confident_closed_eyes_this_frame = 0
                        for i, cls_id in enumerate(cls_ids):
                            if cls_id == 0 and confs[i] >= self.eye_closed_confidence:
                                num_confident_closed_eyes_this_frame += 1
                        
                        if num_confident_closed_eyes_this_frame > 0:
                            frame_had_eye_closed_detected = True
                            total_eye_closed_duration_frames_current_video += 1 # Crucial new metric
                            self.potential_blink_frames += 1
                            self.consecutive_eye_closed += 1
                        else: # Eyes not detected as closed in this frame
                            if self.potential_blink_frames >= self.min_blink_frames and self.blink_cooldown_counter == 0:
                                self.eye_closed_frames += 1 # Count ended blink event
                                self.blink_cooldown_counter = self.blink_cooldown
                            self.potential_blink_frames = 0
                            self.consecutive_eye_closed = 0
                        
                        if self.consecutive_eye_closed > self.max_consecutive_eye_closed:
                            self.max_consecutive_eye_closed = self.consecutive_eye_closed
                        
                        # Blink event detection (if eyes were closed and conditions met)
                        if frame_had_eye_closed_detected and \
                           self.potential_blink_frames >= self.min_blink_frames and \
                           self.blink_cooldown_counter == 0:
                            self.eye_closed_frames += 1 # Count event
                            self.blink_cooldown_counter = self.blink_cooldown
                    else:
                        logging.debug(f"Frame {current_frame_index_in_video}: No 'boxes' in yolo_result_item or it's None.")
                else:
                    logging.debug(f"Frame {current_frame_index_in_video}: No YOLO outputs or empty list.")

                if self.blink_cooldown_counter > 0:
                    self.blink_cooldown_counter -= 1
            
            # After loop, check for any pending blink event
            if self.potential_blink_frames >= self.min_blink_frames and self.blink_cooldown_counter == 0:
                self.eye_closed_frames += 1

            cap.release()
            video_processing_duration_seconds = time.time() - video_processing_start_time

            # --- Construct the output dictionary ---
            # This structure must be compatible with what the third-party platform expects from simplify.py
            
            # Head pose dict now includes counters for the analyzer
            output_head_pose_dict = {
                'head_turned': pose_info_dict.get('head_turned', False),
                'head_down': pose_info_dict.get('head_down', False),
                'head_turned_counter': pose_info_dict.get('head_turned_counter', 0),
                'head_down_counter': pose_info_dict.get('head_down_counter', 0),
                'head_turned_threshold': pose_info_dict.get('head_turned_threshold', 0), # For context
                'head_down_threshold': pose_info_dict.get('head_down_threshold', 0)   # For context
            }

            final_detection_results = {
                # --- Original simplify.py keys ---
                'yawn_frames': self.yawn_frames,
                'eye_closed_frames': self.eye_closed_frames,
                'max_consecutive_eye_closed': self.max_consecutive_eye_closed,
                'normal_state_frames': self.normal_state_frames,
                'total_frames': self.total_frames_from_video_file, # Total frames in original video

                'metrics': {
                    'fps': self.current_video_fps,
                    'process_time': video_processing_duration_seconds,
                    'processed_frames': processed_frames_count,
                    'consecutive_eye_closed': self.consecutive_eye_closed, # Value at end of processing
                    'potential_blink_frames': self.potential_blink_frames, # Value at end of processing
                    'processed_frame_ratio': processed_frames_count / self.total_frames_from_video_file if self.total_frames_from_video_file > 0 else 0
                },
                'head_pose': output_head_pose_dict,

                # --- Necessary ADDITIONS for improved RateBasedAnalyzer ---
                # These keys are new. If this breaks the third-party platform,
                # then RateBasedAnalyzer cannot be improved in the way that uses these metrics.
                'total_eye_closed_duration_frames': total_eye_closed_duration_frames_current_video,
                'yawn_detections_count': yawn_detections_count_current_video,

                # --- Optional informational additions (from yolo_processor.py) ---
                'model_name': self.model_name,
                # 'processing_status': 'processed' # Can be added if useful
            }
            
            logging.info(f"Video processing complete (YoloProcessor enhanced). Results: {final_detection_results}")
            self._cleanup_temp_file(temp_video_path)
            return True, final_detection_results

        except Exception as e:
            logging.exception(f"Critical error in YoloProcessor.process_video: {e}") # Use .exception for stack trace
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            self._cleanup_temp_file(temp_video_path)
            return False, {'error': str(e), 'reason': 'Exception during video processing'}

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
            '/api/download/db',      # Download the SQLite database file
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

        # Get processing time statistics
        time_stats = db_manager.get_processing_time_stats()

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
                'time_metrics': {
                    'average_processing_task_time': round(time_stats['avg_processing_time'], 2),
                    'average_queue_wait_time': round(time_stats['avg_queue_wait_time'], 2)
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


@app.route('/api/download/db', methods=['GET'])
def download_database():
    """Download the SQLite database file."""
    try:
        # Check if the database file exists
        if not os.path.exists(DB_PATH):
            return jsonify({
                'success': False,
                'error': f'Database file not found at {DB_PATH}'
            }), 404

        # Get the file size for logging
        file_size = os.path.getsize(DB_PATH)
        logging.info(f"Serving database file: {DB_PATH} (size: {file_size / 1024:.2f} KB)")

        # Generate a filename for the download
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"simplify_detection_{timestamp}.db"

        # Return the file as an attachment
        return send_file(
            DB_PATH,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        logging.error(f"Error downloading database file: {e}")
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
