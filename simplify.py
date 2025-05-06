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

                # Check if evidence_id column exists in processing_queue table
                cursor.execute("PRAGMA table_info(processing_queue)")
                columns = [column[1] for column in cursor.fetchall()]

                # Add evidence_id column if it doesn't exist
                if 'evidence_id' not in columns:
                    logging.info("Adding evidence_id column to processing_queue table")
                    cursor.execute('''
                        ALTER TABLE processing_queue
                        ADD COLUMN evidence_id INTEGER
                    ''')
                    logging.info("Added evidence_id column to processing_queue table")

                    # Populate evidence_id for existing completed queue items
                    self._populate_evidence_ids()

                conn.commit()
                logging.info("Database initialized successfully")
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")

    def _populate_evidence_ids(self):
        """Populate evidence_id for existing queue items that don't have it."""
        try:
            logging.info("Populating evidence_id for existing queue items")
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get all completed queue items without evidence_id
                cursor = conn.execute('''
                    SELECT id, video_url
                    FROM processing_queue
                    WHERE status = 'completed' AND (evidence_id IS NULL OR evidence_id = 0)
                ''')
                queue_items = cursor.fetchall()

                updated_count = 0
                for item in queue_items:
                    # Find the corresponding evidence result
                    cursor = conn.execute('''
                        SELECT id
                        FROM evidence_results
                        WHERE video_url = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (item['video_url'],))
                    evidence = cursor.fetchone()

                    if evidence:
                        # Update the queue item with the evidence_id
                        conn.execute('''
                            UPDATE processing_queue
                            SET evidence_id = ?
                            WHERE id = ?
                        ''', (evidence['id'], item['id']))
                        updated_count += 1

                conn.commit()
                logging.info(f"Updated evidence_id for {updated_count} existing queue items")
        except sqlite3.Error as e:
            logging.error(f"Error populating evidence_ids: {e}")

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

    def update_queue_evidence_id(self, queue_id, evidence_id):
        """Update the evidence_id of a queued item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE processing_queue SET evidence_id = ? WHERE id = ?',
                    (evidence_id, queue_id)
                )
                conn.commit()
                logging.info(f"Updated queue item {queue_id} evidence_id to {evidence_id}")
                return True
        except sqlite3.Error as e:
            logging.error(f"Error updating queue evidence_id: {e}")
            return False

    def update_queue_status_range(self, start_id, end_id, status):
        """Update the status of a range of queued items."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE processing_queue SET status = ? WHERE id >= ? AND id <= ?',
                    (status, start_id, end_id)
                )
                conn.commit()
                affected_rows = conn.total_changes
                logging.info(f"Updated {affected_rows} queue items from ID {start_id} to {end_id} with status {status}")
                return affected_rows
        except sqlite3.Error as e:
            logging.error(f"Error updating queue status range: {e}")
            return 0

    def get_queue_status(self, queue_id):
        """Get the status of a queued item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url, status, evidence_id, created_at FROM processing_queue WHERE id = ?',
                    (queue_id,)
                )
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
        except sqlite3.Error as e:
            logging.error(f"Error getting queue status: {e}")
            return None

    def get_queue_items_by_evidence_id(self, evidence_id):
        """Get all queue items that reference a specific evidence result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url, status, evidence_id, created_at FROM processing_queue WHERE evidence_id = ?',
                    (evidence_id,)
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting queue items by evidence_id: {e}")
            return []

    def get_next_pending_video(self):
        """Get the next pending video from the queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url, evidence_id FROM processing_queue WHERE status = "pending" ORDER BY created_at ASC LIMIT 1'
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

    def update_evidence_result(self, evidence_id, video_url, detection_results, analysis_result, process_time, head_pose=None):
        """Update an existing evidence result in the database."""
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
                'head_pose': head_pose,
                'reprocessed_at': datetime.datetime.now().isoformat()
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE evidence_results SET
                        process_time = ?,
                        yawn_frames = ?,
                        eye_closed_frames = ?,
                        max_consecutive_eye_closed = ?,
                        normal_state_frames = ?,
                        total_frames = ?,
                        is_drowsy = ?,
                        confidence = ?,
                        is_head_turned = ?,
                        is_head_down = ?,
                        processing_status = ?,
                        details = ?
                    WHERE id = ?
                ''', (
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
                    json.dumps(details),
                    evidence_id
                ))
                conn.commit()
                logging.info(f"Updated evidence result for {video_url}, ID: {evidence_id}")
                return evidence_id
        except sqlite3.Error as e:
            logging.error(f"Error updating evidence result: {e}")
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

    def get_evidence_result_by_video_url(self, video_url):
        """Get the most recent evidence result for a specific video URL."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM evidence_results WHERE video_url = ? ORDER BY created_at DESC LIMIT 1',
                    (video_url,)
                )
                result = cursor.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            logging.error(f"Error getting evidence result by video URL: {e}")
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
        # Keep original parameters for backward compatibility
        self.eye_closed_percentage_threshold = eye_closed_percentage_threshold
        self.yawn_percentage_threshold = yawn_percentage_threshold
        self.normal_state_threshold = normal_state_threshold
        self.fps = fps
        self.max_closure_duration_threshold = max_closure_duration_threshold
        self.minimum_yawn_threshold = 1
        self.minimum_eye_closed_threshold = 3
        self.normal_state_ratio_threshold = 5
        self.minimum_frames_for_analysis = 10

        # Add new parameters from drowsiness_analyzer.py
        # --- Score Calculation Parameters ---
        self.perclos_scale = 1.2
        self.duration_scale = 1.8
        self.yawn_rate_scale = 1.5
        self.score_cap = 2.5

        # --- Weights (Balanced) ---
        self.eye_metric_weight = 0.7  # More weight to eye metrics
        self.yawn_metric_weight = 0.3  # Less weight to yawn metrics

        # --- Non-Linear Damping Parameters ---
        self.damping_base_factor = 0.8
        self.damping_power = 2.5

        # --- Decision Making ---
        self.drowsiness_decision_threshold = 0.55

        # --- Extreme Thresholds for Conditional Overrides ---
        self.extreme_perclos_threshold = 45.0
        self.extreme_duration_threshold = 1.5
        self.extreme_yawn_rate_threshold = 15.0
        self.override_max_normal_perc = 40.0

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

    def _create_details_dict(self, perclos, duration, yawn_rate, normal_perc,
                             p_score, dur_score, y_score, eye_score, raw_score, damping,
                             reason, yawn_cnt, eye_closed_det_cnt, max_consec,
                             norm_frames, tot_frames, fps):
        """Helper function to create the details dictionary."""
        return {
            'eye_closed_percentage': perclos,
            'max_closure_duration': duration,
            'yawn_percentage': yawn_rate,
            'normal_state_percentage': normal_perc,
            # 'perclos_score': p_score,
            # 'duration_score': dur_score,
            # 'yawn_score': y_score,
            # 'combined_eye_score_avg': eye_score,
            # 'raw_drowsiness_score': raw_score,
            # 'applied_damping_factor': damping,
            'reason': reason,
            # Raw Inputs
            'yawn_frames': yawn_cnt,
            'eye_closed_frames': eye_closed_det_cnt,
            'max_consecutive_eye_closed': max_consec,
            'normal_state_frames': norm_frames,
            'total_frames': tot_frames,
            'fps': fps
        }

    def analyze(self, detection_results):
        """Analyze drowsiness based on detection metrics."""
        # Extract values from detection_results
        yawn_frames = detection_results.get('yawn_frames', 0)
        eye_closed_frames = detection_results.get('eye_closed_frames', 0)
        normal_state_frames = detection_results.get('normal_state_frames', 0)
        total_frames = detection_results.get('total_frames', 0)
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
        time_in_seconds = total_frames / fps if fps > 0 else 0
        time_in_minutes = time_in_seconds / 60 if time_in_seconds > 0 else 0

        # Calculate percentage metrics
        eye_closed_percentage = (eye_closed_frames / total_frames) * 100 if total_frames > 0 else 0
        max_closure_duration = max_consecutive_eye_closed / fps if fps > 0 else 0
        yawn_percentage = (yawn_frames / total_frames) * 100 if total_frames > 0 else 0
        normal_state_percentage = (normal_state_frames / total_frames) * 100 if total_frames > 0 else 0
        yawn_rate_per_minute = yawn_frames / time_in_minutes if time_in_minutes > 0 else 0

        # --- Check for Conditional Extreme Overrides ---
        final_score = 0.0
        is_drowsy = False
        reason = "checking_extremes"
        override_triggered = False

        # Check if normal state allows overrides
        allow_override = normal_state_percentage < self.override_max_normal_perc

        if allow_override:
            # Prioritize more reliable extreme indicators
            if max_closure_duration >= self.extreme_duration_threshold:
                reason = f"extreme_duration (>{self.extreme_duration_threshold}s) & low_normal ({normal_state_percentage:.1f}%)"
                override_triggered = True
            elif eye_closed_percentage >= self.extreme_perclos_threshold:
                reason = f"extreme_perclos (>{self.extreme_perclos_threshold}%) & low_normal ({normal_state_percentage:.1f}%)"
                override_triggered = True
            elif yawn_rate_per_minute >= self.extreme_yawn_rate_threshold:
                reason = f"extreme_yawn_rate (>{self.extreme_yawn_rate_threshold}/min) & low_normal ({normal_state_percentage:.1f}%)"
                override_triggered = True

            if override_triggered:
                is_drowsy = True
                # Assign a high score, bypassing normal calculation and damping
                final_score = self.score_cap  # Use max possible score
                logging.info(f"Conditional Drowsiness Override Triggered: {reason}")
                details = self._create_details_dict(
                    eye_closed_percentage, max_closure_duration, yawn_percentage, normal_state_percentage,
                    0, 0, 0, 0, 0, 0,  # Scores/damping not applicable here
                    reason, yawn_frames, eye_closed_frames, max_consecutive_eye_closed,
                    normal_state_frames, total_frames, fps
                )
                return {'is_drowsy': True, 'confidence': final_score, 'details': details}

        # --- Calculate Individual Scores (If no override) ---
        perclos_score = self._calculate_metric_score(eye_closed_percentage, self.eye_closed_percentage_threshold, self.perclos_scale)
        duration_score = self._calculate_metric_score(max_closure_duration, self.max_closure_duration_threshold, self.duration_scale)
        yawn_score = self._calculate_metric_score(yawn_percentage, self.yawn_percentage_threshold, self.yawn_rate_scale)

        # Combine eye scores using AVERAGE
        combined_eye_score = (perclos_score + duration_score) / 2.0

        # --- Calculate Raw Drowsiness Score ---
        raw_drowsiness_score = (combined_eye_score * self.eye_metric_weight +
                                yawn_score * self.yawn_metric_weight)

        # --- Apply Non-Linear Normal State Damping ---
        damping_amount = self._calculate_damping(normal_state_percentage)

        final_score = raw_drowsiness_score * (1.0 - damping_amount)
        final_score = max(0.0, min(final_score, self.score_cap))  # Ensure score stays within [0, score_cap]

        # --- Make Final Drowsiness Decision ---
        is_drowsy = final_score >= self.drowsiness_decision_threshold

        # --- Determine Reason ---
        if is_drowsy:
            # Check contributions before damping to see what pushed it over
            eye_contribution = combined_eye_score * self.eye_metric_weight
            yawn_contribution = yawn_score * self.yawn_metric_weight
            # Use a slightly larger tolerance or relative comparison if needed
            if eye_contribution > yawn_contribution + 0.05:
                reason = f"eye_metrics_dominant (Score: {final_score:.2f})"
            elif yawn_contribution > eye_contribution + 0.05:
                reason = f"yawn_metrics_dominant (Score: {final_score:.2f})"
            else:
                # Check if only one metric was non-zero
                if combined_eye_score > 0 and yawn_score == 0:
                    reason = f"eye_metrics_only (Score: {final_score:.2f})"
                elif yawn_score > 0 and combined_eye_score == 0:
                    reason = f"yawn_metrics_only (Score: {final_score:.2f})"
                else:
                    reason = f"combined_metrics_threshold_met (Score: {final_score:.2f})"
        elif final_score > 0:
            reason = f"indicators_present_below_threshold (Score: {final_score:.2f})"
        else:
            # Reached here means final_score is 0
            if raw_drowsiness_score > 0:  # Was non-zero before damping
                reason = f"indicators_damped_to_zero (Raw: {raw_drowsiness_score:.2f}, Damp: {damping_amount:.2f})"
            else:  # Raw score was already zero
                reason = "no_significant_indicators"

        # For backward compatibility, check head pose override
        # Check if head is in a distracted position (either turned or down)
        is_head_distracted = is_head_turned or is_head_down

        if is_head_distracted and is_drowsy:
            is_drowsy = False
            reason = 'head_pose_override'
            logging.info(f"Head pose override: Head turned={is_head_turned}, Head down={is_head_down}")

        # --- Format Results ---
        details = self._create_details_dict(
            eye_closed_percentage, max_closure_duration, yawn_percentage, normal_state_percentage,
            perclos_score, duration_score, yawn_score, combined_eye_score,
            raw_drowsiness_score, damping_amount, reason,
            yawn_frames, eye_closed_frames, max_consecutive_eye_closed,
            normal_state_frames, total_frames, fps
        )

        # Add backward compatibility fields
        details.update({
            'is_drowsy_eyes': eye_closed_percentage > self.eye_closed_percentage_threshold or
                             max_closure_duration > self.max_closure_duration_threshold or
                             eye_closed_frames >= self.minimum_eye_closed_threshold,
            'is_drowsy_yawns': yawn_frames >= self.minimum_yawn_threshold and
                              yawn_percentage > self.yawn_percentage_threshold,
            'is_drowsy_excessive_yawns': yawn_frames > 10 or yawn_percentage > 20,
            'is_normal_state_high': normal_state_percentage >= self.normal_state_threshold,
            'is_head_turned': is_head_turned,
            'is_head_down': is_head_down,
            'head_turned_frames': head_turned_frames,
            'head_down_frames': head_down_frames
        })

        result = {
            'is_drowsy': is_drowsy,
            'confidence': final_score,
            'details': details
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
        self.blink_cooldown = int(os.getenv('BLINK_COOLDOWN', '2'))  # Frames to wait before counting next blink
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
        self.blink_cooldown_counter = 0
        self.potential_blink_frames = 0
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
                                self.consecutive_eye_closed += 1
                                self.potential_blink_frames += 1
                                break

                        if eye_closed_detected:
                            # Count eye closed events if we have enough consecutive frames and not in cooldown
                            if self.potential_blink_frames >= self.min_blink_frames and self.blink_cooldown_counter == 0:
                                self.eye_closed_frames += 1
                                self.blink_cooldown_counter = self.blink_cooldown
                                logging.info(f"Eye closure detected with confidence")
                        else:
                            # Even when eyes are no longer detected as closed, if we had enough frames
                            # to consider it a valid eye closure, count it
                            if self.potential_blink_frames >= self.min_blink_frames and self.blink_cooldown_counter == 0:
                                self.eye_closed_frames += 1
                                self.blink_cooldown_counter = self.blink_cooldown
                                logging.info("Eye closure event counted after detection ended")
                            self.potential_blink_frames = 0
                            self.consecutive_eye_closed = 0

                        # Decrement cooldown counter if active
                        if self.blink_cooldown_counter > 0:
                            self.blink_cooldown_counter -= 1

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

            # Prepare detection results
            detection_results = {
                'yawn_frames': self.yawn_frames,
                'eye_closed_frames': self.eye_closed_frames,
                'max_consecutive_eye_closed': self.max_consecutive_eye_closed,
                'normal_state_frames': self.normal_state_frames,
                'total_frames': self.total_frames,
                'metrics': {
                    'fps': self.current_fps,
                    'process_time': process_time,
                    'processed_frames': frame_count,
                    'consecutive_eye_closed': self.consecutive_eye_closed,
                    'potential_blink_frames': self.potential_blink_frames,
                    'processed_frame_ratio': frame_count / self.total_frames if self.total_frames > 0 else 0
                },
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
        # Get the queue status to check if it has an associated evidence_id
        queue_status = db_manager.get_queue_status(queue_id)
        existing_evidence_id = queue_status.get('evidence_id') if queue_status else None

        # If the queue has an associated evidence_id, use that
        if existing_evidence_id:
            existing_evidence = db_manager.get_evidence_result(existing_evidence_id)
            is_reprocessing = existing_evidence is not None
            if is_reprocessing:
                logging.info(f"Reprocessing video from queue: {queue_id}, URL: {video_url}, Existing evidence ID from queue: {existing_evidence_id}")
        else:
            # Otherwise, check if this video has been processed before
            existing_evidence = db_manager.get_evidence_result_by_video_url(video_url)
            is_reprocessing = existing_evidence is not None
            if is_reprocessing:
                logging.info(f"Reprocessing video from queue: {queue_id}, URL: {video_url}, Existing evidence ID from lookup: {existing_evidence['id']}")
                # Update the queue with the evidence_id
                db_manager.update_queue_evidence_id(queue_id, existing_evidence['id'])

        # Process the video
        start_time = time.time()
        processing_success, detection_results = yolo_processor.process_video(video_url)
        process_time = time.time() - start_time

        if not processing_success:
            # Update queue status to failed
            db_manager.update_queue_status(queue_id, 'failed')

            # Store or update the failed evidence
            try:
                # Create a minimal evidence record for the failed video
                failed_details = {
                    'error': 'Failed to process video',
                    'timestamp': datetime.datetime.now().isoformat()
                }

                if is_reprocessing:
                    # Update existing evidence record
                    evidence_id = existing_evidence['id']
                    with sqlite3.connect(db_manager.db_path) as conn:
                        conn.execute('''
                            UPDATE evidence_results SET
                                process_time = ?,
                                processing_status = ?,
                                details = ?
                            WHERE id = ?
                        ''', (
                            process_time,
                            'failed',
                            json.dumps(failed_details),
                            evidence_id
                        ))
                        conn.commit()
                        logging.info(f"Updated failed evidence record for {video_url}, ID: {evidence_id}")
                else:
                    # Insert new evidence record
                    with sqlite3.connect(db_manager.db_path) as conn:
                        cursor = conn.execute('''
                            INSERT INTO evidence_results (
                                video_url, process_time, processing_status, details
                            ) VALUES (?, ?, ?, ?)
                        ''', (
                            video_url,
                            process_time,
                            'failed',
                            json.dumps(failed_details)
                        ))
                        evidence_id = cursor.lastrowid
                        conn.commit()
                        logging.info(f"Stored failed evidence record for {video_url}, ID: {evidence_id}")

                    # Update the queue with the evidence_id
                    db_manager.update_queue_evidence_id(queue_id, evidence_id)

                # Send webhook notification for failed processing
                send_webhook_notification(
                    queue_id=queue_id,
                    evidence_id=evidence_id,
                    status='failed',
                    video_url=video_url,
                    results={'error': 'Failed to process video'}
                )
            except Exception as e:
                logging.error(f"Error storing/updating failed evidence record: {e}")

            return

        # Analyze drowsiness
        analysis_result = drowsiness_analyzer.analyze(detection_results)

        # Store or update results in database
        if is_reprocessing:
            # Update existing evidence
            evidence_id = db_manager.update_evidence_result(
                existing_evidence['id'],
                video_url,
                detection_results,
                analysis_result,
                process_time,
                detection_results.get('head_pose')
            )
            logging.info(f"Updated existing evidence result for {video_url}, ID: {evidence_id}")
        else:
            # Store new evidence
            evidence_id = db_manager.store_evidence_result(
                video_url,
                detection_results,
                analysis_result,
                process_time,
                detection_results.get('head_pose')
            )
            logging.info(f"Stored new evidence result for {video_url}, ID: {evidence_id}")

            # Update the queue with the evidence_id
            db_manager.update_queue_evidence_id(queue_id, evidence_id)

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
            '/api/process',                # Add a video to the processing queue
            '/api/queue/<id>',             # Check the status of a queued video
            '/api/queue',                  # Get queue statistics
            '/api/queue/update-range',     # Update status of a range of queue items
            '/api/results',                # Get all processed videos
            '/api/result/<id>',            # Get details for a specific processed video
            '/api/webhook',                # Manage webhooks (GET, POST, DELETE)
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

        # Get evidence_id from queue status if available
        evidence_id = queue_status.get('evidence_id')

        # Check if processing is complete
        if queue_status['status'] == 'completed':
            # If evidence_id is not in the queue, try to find it
            if not evidence_id:
                # Find the evidence result
                with sqlite3.connect(db_manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        'SELECT id FROM evidence_results WHERE video_url = ? ORDER BY created_at DESC LIMIT 1',
                        (queue_status['video_url'],)
                    )
                    evidence = cursor.fetchone()
                    evidence_id = evidence['id'] if evidence else None

                    # Update the queue with the evidence_id if found
                    if evidence_id:
                        db_manager.update_queue_evidence_id(queue_id, evidence_id)

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
                    'evidence_id': evidence_id,
                    'result_url': f'/api/result/{evidence_id}' if evidence_id else None,
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
                    'created_at': queue_status['created_at'],
                    'evidence_id': evidence_id,
                    'result_url': f'/api/result/{evidence_id}' if evidence_id else None
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


@app.route('/api/queue/update-range', methods=['POST'])
def update_queue_range():
    """Update the status of a range of queue items."""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request data'
            }), 400

        # Validate required parameters
        if 'start_id' not in data or 'end_id' not in data or 'status' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: start_id, end_id, and status are required'
            }), 400

        start_id = data['start_id']
        end_id = data['end_id']
        status = data['status']

        # Validate IDs
        if not isinstance(start_id, int) or not isinstance(end_id, int) or start_id <= 0 or end_id <= 0:
            return jsonify({
                'success': False,
                'error': 'Invalid ID values: start_id and end_id must be positive integers'
            }), 400

        # Validate range
        if start_id > end_id:
            return jsonify({
                'success': False,
                'error': 'Invalid range: start_id must be less than or equal to end_id'
            }), 400

        # Validate status
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if status not in valid_statuses:
            return jsonify({
                'success': False,
                'error': f'Invalid status: must be one of {", ".join(valid_statuses)}'
            }), 400

        # If setting to pending, we need to get the queue items first to handle reprocessing
        if status == 'pending':
            # Get the queue items in the specified range
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url, evidence_id FROM processing_queue WHERE id >= ? AND id <= ?',
                    (start_id, end_id)
                )
                queue_items = [dict(row) for row in cursor.fetchall()]

            # Update queue items in the specified range (only update status, preserve evidence_id)
            affected_rows = db_manager.update_queue_status_range(start_id, end_id, status)

            # Return response with queue items that will be reprocessed
            return jsonify({
                'success': True,
                'message': f'Successfully updated {affected_rows} queue items to pending status. They will be reprocessed with their existing evidence results.',
                'data': {
                    'start_id': start_id,
                    'end_id': end_id,
                    'status': status,
                    'affected_rows': affected_rows,
                    'queue_items': queue_items
                }
            })
        else:
            # For other statuses, just update the queue items
            affected_rows = db_manager.update_queue_status_range(start_id, end_id, status)

            return jsonify({
                'success': True,
                'message': f'Successfully updated {affected_rows} queue items',
                'data': {
                    'start_id': start_id,
                    'end_id': end_id,
                    'status': status,
                    'affected_rows': affected_rows
                }
            })

    except Exception as e:
        logging.error(f"Error updating queue range: {e}")
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
