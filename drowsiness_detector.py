import time
import datetime
import os
import logging
import sqlite3
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import json
import hashlib
import sys

# Import our separated modules
from data_manager import DataManager
from yolo_processor import YoloProcessor
from api_client import ApiClient
from drowsiness_analyzer import create_analyzer
from ml_metrics_analyzer import MLMetricsAnalyzer

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('drowsiness_detector.log')  # Save to file
    ]
)

# Load environment variables
load_dotenv()

# Configuration
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_TOKEN = os.getenv("API_TOKEN")
# Default model path, will be overridden if active model is found in database
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/model-1.pt")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolov8l-pose.pt")
FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", "300"))  # Default 5 minutes
BASE_URL = os.getenv("BASE_URL")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Get active model from database if available
def get_active_model_path():
    try:
        conn = sqlite3.connect("drowsiness_detection.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT file_path FROM models WHERE is_active = 1')
        active_model = cursor.fetchone()
        conn.close()

        if active_model and active_model['file_path']:
            logging.info(f"Using active model from database: {active_model['file_path']}")
            return active_model['file_path']
        else:
            logging.info(f"No active model found in database, using default: {YOLO_MODEL_PATH}")
            return YOLO_MODEL_PATH
    except Exception as e:
        logging.error(f"Error getting active model from database: {e}")
        logging.info(f"Using default model path: {YOLO_MODEL_PATH}")
        return YOLO_MODEL_PATH

# Update model path from database
YOLO_MODEL_PATH = get_active_model_path()

# Validate configuration
if not all([API_ENDPOINT, API_TOKEN, YOLO_MODEL_PATH]):
    logging.error("Missing required environment variables. Please check your .env file")
    logging.error(f"API_ENDPOINT: {'Set' if API_ENDPOINT else 'Missing'}")
    logging.error(f"API_TOKEN: {'Set' if API_TOKEN else 'Missing'}")
    logging.error(f"YOLO_MODEL_PATH: {'Set' if YOLO_MODEL_PATH else 'Missing'}")
    exit(1)

# Check if pose model exists
if not os.path.exists(POSE_MODEL_PATH):
    logging.warning(f"Pose model not found at {POSE_MODEL_PATH}. Head pose detection will be disabled.")
    logging.warning("Please download a YOLOv8 pose model (e.g., yolov8l-pose.pt) to enable head pose detection.")

# Initialize components
data_manager = DataManager()
yolo_processor = YoloProcessor(YOLO_MODEL_PATH)
# Set the pose model path for the YoloProcessor
yolo_processor.pose_detector.model_path = POSE_MODEL_PATH
# Reload the pose model if needed
if not yolo_processor.pose_detector.model:
    logging.info(f"Loading pose model from {POSE_MODEL_PATH}")
    yolo_processor.pose_detector.model = yolo_processor.pose_detector.load_model()
api_client = ApiClient(
    BASE_URL,
    API_ENDPOINT,
    API_TOKEN,
    USERNAME,
    PASSWORD
)

# Initialize the drowsiness analyzer and ML metrics analyzer
drowsiness_analyzer = create_analyzer(
    analyzer_type="rate",
)
ml_metrics_analyzer = MLMetricsAnalyzer()

def analyze_drowsiness(detection_results):
    """Analyzes detection results to determine drowsiness and ML metrics."""
    # Get drowsiness analysis
    result = drowsiness_analyzer.analyze(detection_results)

    return result

def main():
    """Main processing loop."""
    logging.info("Starting Drowsiness Detection Service...")
    logging.info(f"API Endpoint: {API_ENDPOINT}")
    logging.info(f"YOLO Model Path: {YOLO_MODEL_PATH}")
    logging.info(f"YOLO Pose Model Path: {POSE_MODEL_PATH}")
    logging.info(f"Fetch Interval: {FETCH_INTERVAL_SECONDS} seconds")

    # Create PID file for process management
    pid = os.getpid()
    with open('drowsiness_detector.pid', 'w') as f:
        f.write(str(pid))
    logging.info(f"Created PID file with process ID: {pid}")

    try:
        while True:
            try:
                current_start_time = data_manager.get_last_fetch_time()
                current_end_time = datetime.datetime.now()

                # current_start_time = datetime.datetime.now() - datetime.timedelta(hours=72)
                # current_end_time = datetime.datetime.now() - datetime.timedelta(hours=48)

                if current_start_time >= current_end_time:
                    logging.info("No new time range to fetch. Waiting for next interval.")
                else:
                    # Fetch new evidence
                    logging.info(f"Fetching evidence from {current_start_time} to {current_end_time}")
                    result = api_client.fetch_video_evidence(current_start_time, current_end_time)

                    if result['status'] == 'success':
                        evidence_list = result['data']
                        logging.info(f"Processing {len(evidence_list)} evidence items")

                        for evidence in evidence_list:
                            # Store evidence in database
                            evidence_id = data_manager.store_evidence_result(evidence)

                            logging.info(f"Stored evidence for device: {evidence}")

                            # Process video if available
                            video_url = evidence.get('videoUrl')

                            if not ("Yawning" in evidence.get('alarmTypeValue', '') or "Eye closed" in evidence.get('alarmTypeValue', '')):
                                logging.info(f"Skipping processing for device: {evidence.get('deviceName')} due to alarm type: {evidence.get('alarmTypeValue')}")
                                data_manager.update_evidence_status(evidence_id, 'skipped')
                            elif video_url:
                                logging.info(f"Processing video for device: {evidence.get('deviceName')}")
                                processing_success, detection_results = yolo_processor.process_video(video_url)

                                if not processing_success:
                                    logging.error(f"Failed to process video for device: {evidence.get('deviceName')}")
                                    data_manager.update_evidence_status(evidence_id, 'failed')
                                elif processing_success and detection_results:
                                    # Add debug logging to check detection_results
                                    logging.debug(f"Detection results: {detection_results}")

                                    # Analyze drowsiness using the detection_results dictionary
                                    analysis_result = analyze_drowsiness(detection_results)

                                    # Update detection results with analysis
                                    detection_results.update({
                                        'is_drowsy': analysis_result['is_drowsy'],
                                        'confidence': analysis_result['confidence'],
                                        'analysis_details': analysis_result['details']
                                    })

                                    # Add ML metrics if available
                                    if 'ml_metrics' in analysis_result:
                                        detection_results['ml_metrics'] = analysis_result['ml_metrics']

                                    # Update evidence with detection results
                                    data_manager.update_evidence_result(evidence_id, detection_results)

                                    # Enhanced logging for debugging
                                    # Get head pose information
                                    head_pose = detection_results.get('head_pose', {})
                                    is_head_turned = head_pose.get('is_head_turned', False)
                                    is_head_down = head_pose.get('is_head_down', False)

                                    logging.info(f"Updated evidence result - ID: {evidence_id}, "
                                                 f"Yawns: {detection_results.get('yawn_count', 0)}, "
                                                 f"Eye Closed Events: {detection_results.get('eye_closed_frames', 0)}, "
                                                 f"Total Eye Closed Frames: {detection_results.get('total_eye_closed_frames', 0)}, "
                                                 f"Max Consecutive Closed: {detection_results.get('max_consecutive_eye_closed', 0)}, "
                                                 f"Normal State Frames: {detection_results.get('normal_state_frames', 0)}, "
                                                 f"Head Turned: {is_head_turned}, Head Down: {is_head_down}")

                                    # Log drowsiness detection
                                    if analysis_result['is_drowsy']:
                                        # Include head pose information in the log
                                        head_pose = detection_results.get('head_pose', {})
                                        is_head_turned = head_pose.get('is_head_turned', False)
                                        is_head_down = head_pose.get('is_head_down', False)

                                        logging.warning(
                                            f"Drowsiness detected for device {evidence.get('deviceName')} "
                                            f"with {detection_results.get('yawn_count')} yawns and "
                                            f"{detection_results.get('eye_closed_frames')} closed eye frames. "
                                            f"Confidence: {analysis_result['confidence']*100:.1f}%. "
                                            f"Head Turned: {is_head_turned}, Head Down: {is_head_down}"
                                        )
                                break  # Stop after processing first video file

                    # Update the last fetch time
                    data_manager.update_last_fetch_time(current_end_time)
                    logging.info(f"Updated fetch time to: {current_end_time}")

                # Process any pending evidence
                logging.info("Checking for pending evidence...")
                pending_evidence = data_manager.get_pending_evidence()
                logging.info(f"Found {len(pending_evidence)} pending evidence items")

                for evidence_id, video_url, device_name in pending_evidence:
                    if video_url:
                        logging.info(f"Processing pending evidence for device: {device_name}")
                        processing_success, detection_results = yolo_processor.process_video(video_url)

                        if not processing_success:
                            logging.error(f"Failed to process pending video for device: {device_name}")
                            data_manager.update_evidence_status(evidence_id, 'failed')
                        elif processing_success and detection_results:
                            # Analyze drowsiness using the detection_results dictionary
                            analysis_result = analyze_drowsiness(detection_results)

                            # Update detection results with analysis
                            detection_results.update({
                                'is_drowsy': analysis_result['is_drowsy'],
                                'confidence': analysis_result['confidence'],
                                'analysis_details': analysis_result['details']
                            })

                            # Update evidence with detection results
                            data_manager.update_evidence_result(evidence_id, detection_results)

                            # Enhanced logging for debugging
                            # Get head pose information for pending evidence
                            head_pose = detection_results.get('head_pose', {})
                            is_head_turned = head_pose.get('is_head_turned', False)
                            is_head_down = head_pose.get('is_head_down', False)

                            logging.info(f"Updated pending evidence result - ID: {evidence_id}, "
                                         f"Yawns: {detection_results.get('yawn_count', 0)}, "
                                         f"Eye Closed Events: {detection_results.get('eye_closed_frames', 0)}, "
                                         f"Total Eye Closed Frames: {detection_results.get('total_eye_closed_frames', 0)}, "
                                         f"Max Consecutive Closed: {detection_results.get('max_consecutive_eye_closed', 0)}, "
                                         f"Normal State Frames: {detection_results.get('normal_state_frames', 0)}, "
                                         f"Head Turned: {is_head_turned}, Head Down: {is_head_down}")

                # Wait for next interval
                logging.info(f"Waiting {FETCH_INTERVAL_SECONDS} seconds until next fetch...")
                time.sleep(FETCH_INTERVAL_SECONDS)

            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying after an error

    except KeyboardInterrupt:
        logging.info("Service stopped by user")

def cleanup():
    """Clean up resources before exiting."""
    try:
        if os.path.exists('drowsiness_detector.pid'):
            os.remove('drowsiness_detector.pid')
            logging.info("Removed PID file during cleanup")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
    finally:
        cleanup()