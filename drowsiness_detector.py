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
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", "300"))  # Default 5 minutes
DROWSINESS_THRESHOLD_YAWN = int(os.getenv("DROWSINESS_THRESHOLD_YAWN", "3"))
DROWSINESS_THRESHOLD_EYE_CLOSED = int(os.getenv("DROWSINESS_THRESHOLD_EYE_CLOSED", "10"))
BASE_URL = os.getenv("BASE_URL")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Validate configuration
if not all([API_ENDPOINT, API_TOKEN, YOLO_MODEL_PATH]):
    logging.error("Missing required environment variables. Please check your .env file")
    logging.error(f"API_ENDPOINT: {'Set' if API_ENDPOINT else 'Missing'}")
    logging.error(f"API_TOKEN: {'Set' if API_TOKEN else 'Missing'}")
    logging.error(f"YOLO_MODEL_PATH: {'Set' if YOLO_MODEL_PATH else 'Missing'}")
    exit(1)

# Initialize components
data_manager = DataManager()
yolo_processor = YoloProcessor(
    YOLO_MODEL_PATH, 
    DROWSINESS_THRESHOLD_YAWN, 
    DROWSINESS_THRESHOLD_EYE_CLOSED
)
api_client = ApiClient(
    BASE_URL,
    API_ENDPOINT,
    API_TOKEN,
    USERNAME,
    PASSWORD
)

# Initialize the drowsiness analyzer and ML metrics analyzer
drowsiness_analyzer = create_analyzer(
    analyzer_type="probabilistic",
    a=0.3, 
    b=0.7, 
    c=1.2, 
    fps=20 
)
ml_metrics_analyzer = MLMetricsAnalyzer()

def analyze_drowsiness(yawn_count, eye_closed_frames, total_frames_processed, take_type=None):
    """Analyzes detection counts to determine drowsiness and ML metrics."""
    # Get drowsiness analysis
    result = drowsiness_analyzer.analyze(yawn_count, eye_closed_frames, total_frames_processed)
    
    # Enhanced eye closure analysis
    if eye_closed_frames > 0:
        # Calculate eye closure rate and consecutive frames
        eye_closure_rate = eye_closed_frames / total_frames_processed
        
        # Multi-level threshold system for eye closure
        if eye_closure_rate >= 0.15:  # Severe drowsiness
            result['is_drowsy'] = True
            result['confidence'] = max(result['confidence'], 0.9)  # High confidence
            result['details']['drowsiness_level'] = 'severe'
        elif eye_closure_rate >= 0.08:  # Moderate drowsiness
            result['is_drowsy'] = True
            result['confidence'] = max(result['confidence'], 0.75)  # Moderate-high confidence
            result['details']['drowsiness_level'] = 'moderate'
        elif eye_closure_rate >= 0.05:  # Mild drowsiness
            result['is_drowsy'] = True
            result['confidence'] = max(result['confidence'], 0.6)  # Moderate confidence
            result['details']['drowsiness_level'] = 'mild'
        
        # Add detailed metrics
        result['details']['eye_closure_metrics'] = {
            'rate': eye_closure_rate,
            'total_frames_closed': eye_closed_frames,
            'total_frames': total_frames_processed,
            'percentage_closed': eye_closure_rate * 100
        }
    
    # Get ML metrics if take_type is provided
    if take_type is not None:
        ml_metrics = ml_metrics_analyzer.analyze(result['is_drowsy'], take_type)
        result['ml_metrics'] = ml_metrics
        
        if result['is_drowsy']:
            logging.warning(
                f"Drowsiness detected with {result['confidence']*100:.1f}% confidence!\n"
                f"Level: {result['details'].get('drowsiness_level', 'unknown')}\n"
                f"Eye Closure: {result['details']['eye_closure_metrics']['percentage_closed']:.1f}%\n"
                f"ML Metrics - Sensitivity: {ml_metrics['sensitivity']:.1f}%, "
                f"Accuracy: {ml_metrics['accuracy']:.1f}%\n"
                f"Details: {result['details']}"
            )
    
    return result

def main():
    """Main processing loop."""
    logging.info("Starting Drowsiness Detection Service...")
    logging.info(f"API Endpoint: {API_ENDPOINT}")
    logging.info(f"YOLO Model Path: {YOLO_MODEL_PATH}")
    logging.info(f"Fetch Interval: {FETCH_INTERVAL_SECONDS} seconds")
    
    try:
        while True:
            try:
                # current_start_time = data_manager.get_last_fetch_time()
                # current_end_time = datetime.datetime.now()
                
                current_start_time = datetime.datetime.now() - datetime.timedelta(hours=72)
                current_end_time = datetime.datetime.now() - datetime.timedelta(hours=48)
                
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
                                
                                if processing_success and detection_results:
                                    # Get take_type from evidence
                                    take_type = evidence.get('takeType') == 0  # Assuming 0 means True Alarm
                                    
                                    # Analyze drowsiness using the analyzer
                                    yawn_count = detection_results.get('yawn_count', 0)
                                    eye_closed_frames = detection_results.get('eye_closed_frames', 0)
                                    total_frames = detection_results.get('total_frames', 0)
                                    
                                    analysis_result = analyze_drowsiness(
                                        yawn_count, 
                                        eye_closed_frames, 
                                        total_frames,
                                        take_type
                                    )
                                    
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
                                    
                                    # Log drowsiness detection
                                    if analysis_result['is_drowsy']:
                                        logging.warning(
                                            f"Drowsiness detected for device {evidence.get('deviceName')} "
                                            f"with {detection_results.get('yawn_count')} yawns and "
                                            f"{detection_results.get('eye_closed_frames')} closed eye frames. "
                                            f"Confidence: {analysis_result['confidence']*100:.1f}%"
                                        )
                                break  # Stop after processing first video file
                    
                    # Update the last fetch time
                    data_manager.update_last_fetch_time(current_end_time)
                    logging.info(f"Updated fetch time to: {current_end_time}")
                
                # Process any pending evidence
                logging.info("Checking for pending evidence...")
                pending_evidence = data_manager.get_pending_evidence()
                logging.info(f"Found {len(pending_evidence)} pending evidence items")
                
                # Debug information about pending evidence
                if not pending_evidence:
                    logging.warning("No pending evidence found, but database may contain pending items.")
                    # Let's try to diagnose the issue
                    try:
                        # Get a direct count from the database for verification
                        pending_count = data_manager.get_pending_evidence_count()
                        logging.info(f"Direct database query shows {pending_count} pending items")
                        
                        if pending_count > 0:
                            logging.warning("Database contains pending items but query returned empty. Possible query issue.")
                    except Exception as e:
                        logging.error(f"Error checking pending evidence count: {e}")
                
                print(f'Pending evidence: {pending_evidence}')
                for evidence_id, video_url, device_name in pending_evidence:
                    if video_url:
                        logging.info(f"Processing pending evidence for device: {device_name}")
                        processing_success, detection_results = yolo_processor.process_video(video_url)
                        
                        if processing_success and detection_results:
                            # Analyze drowsiness for pending evidence
                            yawn_count = detection_results.get('yawn_count', 0)
                            eye_closed_frames = detection_results.get('eye_closed_frames', 0)
                            total_frames = detection_results.get('total_frames', 0)
                            
                            analysis_result = analyze_drowsiness(
                                yawn_count, 
                                eye_closed_frames, 
                                total_frames
                            )
                            
                            # Update detection results with analysis
                            detection_results.update({
                                'is_drowsy': analysis_result['is_drowsy'],
                                'confidence': analysis_result['confidence'],
                                'analysis_details': analysis_result['details']
                            })
                            
                            # Update evidence with detection results
                            data_manager.update_evidence_result(evidence_id, detection_results)
                
                # Wait for next interval
                logging.info(f"Waiting {FETCH_INTERVAL_SECONDS} seconds until next fetch...")
                time.sleep(FETCH_INTERVAL_SECONDS)
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying after an error
                
    except KeyboardInterrupt:
        logging.info("Service stopped by user")

if __name__ == "__main__":
    main()
