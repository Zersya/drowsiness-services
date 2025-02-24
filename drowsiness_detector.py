import requests
import cv2
import time
import datetime
import os
import logging
import sqlite3
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import json
from urllib.parse import urljoin
import hashlib

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

DB_PATH = "fetch_state.db"  # SQLite database path

# Database initialization
def init_database():
    """Initialize SQLite database and create necessary tables."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Existing fetch_state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fetch_state (
                    id INTEGER PRIMARY KEY,
                    last_fetch_time TIMESTAMP NOT NULL
                )
            ''')
            
            # New evidence_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evidence_results (
                    id INTEGER PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    device_name TEXT,
                    alarm_type TEXT,
                    alarm_type_value TEXT,
                    alarm_time TIMESTAMP,
                    location TEXT,
                    speed REAL,
                    video_url TEXT,
                    image_url TEXT,
                    is_drowsy BOOLEAN,
                    yawn_count INTEGER,
                    eye_closed_frames INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fleet_name TEXT,
                    alarm_guid TEXT UNIQUE,
                    processing_status TEXT DEFAULT 'pending'
                )
            ''')
            
            # New evidence_files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evidence_files (
                    id INTEGER PRIMARY KEY,
                    evidence_id INTEGER,
                    file_type TEXT,
                    file_url TEXT,
                    start_time TIMESTAMP,
                    stop_time TIMESTAMP,
                    channel INTEGER,
                    FOREIGN KEY (evidence_id) REFERENCES evidence_results (id)
                )
            ''')
            
            conn.commit()
            logging.info("Database initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        raise

def get_last_fetch_time():
    """Retrieve the last fetch time from the database."""
    # return datetime.datetime.now() - datetime.timedelta(minutes=30)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT last_fetch_time FROM fetch_state ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            
            if result:
                return datetime.datetime.fromisoformat(result[0])
            else:
                # If no record exists, insert default time (30 minutes ago) and return it
                default_time = datetime.datetime.now() - datetime.timedelta(minutes=30)
                cursor.execute('INSERT INTO fetch_state (last_fetch_time) VALUES (?)',
                             (default_time.isoformat(),))
                conn.commit()
                return default_time
                
    except sqlite3.Error as e:
        logging.error(f"Error retrieving last fetch time: {e}")
        # Fallback to 30 minutes ago if database access fails
        return datetime.datetime.now() - datetime.timedelta(minutes=30)

def update_last_fetch_time(fetch_time):
    """Update the last fetch time in the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO fetch_state (last_fetch_time) VALUES (?)',
                         (fetch_time.isoformat(),))
            conn.commit()
            logging.info(f"Updated last fetch time to {fetch_time}")
    except sqlite3.Error as e:
        logging.error(f"Error updating last fetch time: {e}")

def store_evidence_result(evidence_data, detection_results=None):
    """Store evidence and its processing results in the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Process video first if URL is available
            video_url = evidence_data.get('videoUrl')
            if video_url and not detection_results:
                processing_success, detection_results = process_video_for_drowsiness(video_url)
                processing_status = 'processed' if processing_success else 'failed'
            else:
                processing_status = 'processed' if detection_results else 'pending'

            # Insert main evidence record
            cursor.execute('''
                INSERT OR REPLACE INTO evidence_results (
                    device_id, device_name, alarm_type, alarm_type_value,
                    alarm_time, location, speed, video_url, image_url,
                    is_drowsy, yawn_count, eye_closed_frames,
                    fleet_name, alarm_guid, processing_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evidence_data.get('deviceID'),
                evidence_data.get('deviceName'),
                evidence_data.get('alarmType'),
                evidence_data.get('alarmTypeValue'),
                evidence_data.get('alarmTime'),
                evidence_data.get('location'),
                evidence_data.get('speed'),
                video_url,
                evidence_data.get('imageUrl'),
                detection_results.get('is_drowsy') if detection_results else None,
                detection_results.get('yawn_count') if detection_results else None,
                detection_results.get('eye_closed_frames') if detection_results else None,
                evidence_data.get('fleetName'),
                evidence_data.get('alarmGuid'),
                processing_status
            ))
            
            evidence_id = cursor.lastrowid
            
            # Insert associated files
            if evidence_data.get('files'):
                for file_data in evidence_data['files']:
                    cursor.execute('''
                        INSERT INTO evidence_files (
                            evidence_id, file_type, file_url,
                            start_time, stop_time, channel
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        evidence_id,
                        file_data.get('fileType'),
                        file_data.get('downUrl'),  # Changed from 'url' to 'downUrl'
                        file_data.get('fileStartTime'),  # Changed from 'startTime'
                        file_data.get('fileStopTime'),  # Changed from 'stopTime'
                        file_data.get('channel')
                    ))
            
            conn.commit()
            logging.info(f"Stored evidence result for device {evidence_data.get('deviceName')} with status {processing_status}")
            return evidence_id
            
    except sqlite3.Error as e:
        logging.error(f"Error storing evidence result: {e}")
        return None

def get_pending_evidence():
    """Retrieve pending evidence that needs processing."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, video_url, device_name
                FROM evidence_results
                WHERE processing_status = 'pending'
                AND video_url IS NOT NULL
                ORDER BY alarm_time ASC
            ''')
            return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Error retrieving pending evidence: {e}")
        return []

# Initialize database at startup
init_database()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize YOLO Model (Adapt based on your YOLO framework - this is a placeholder)
def load_yolo_model():
    """Loads and returns a YOLO model for drowsiness detection."""
    logging.info("Loading YOLO model...")
    try:
        from ultralytics import YOLO
        import torch

        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")

        # Load the model
        model = YOLO(YOLO_MODEL_PATH)
        
        # Move model to appropriate device
        model.to(device)
        
        # Set model parameters for inference
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # NMS IOU threshold
        
        logging.info("YOLO model loaded successfully")
        return model
        
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return None

yolo_model = load_yolo_model() # Load the model at startup (important for efficiency)
if yolo_model is None:
    logging.error("YOLO model failed to load. Exiting.")
    exit()

def perform_login():
    """Performs login and returns new token."""
    login_url = urljoin(BASE_URL, "/vss/user/login.action")
    
    login_data = {
        'username': USERNAME,
        'password': PASSWORD,
        'lang': 'en',
        'platform': 'web',
        'version': 'v2'
    }
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'platform': 'web',
        'version': 'v2'
    }
    
    try:
        response = requests.post(login_url, data=login_data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        if response_data.get('status') == 10000:  # Success status
            new_token = response_data.get('data', {}).get('token')
            if new_token:
                # Update the environment variable and the global API_TOKEN
                global API_TOKEN
                API_TOKEN = new_token
                # Optionally save to .env file for persistence
                update_env_file('API_TOKEN', new_token)
                logging.info("Successfully logged in and updated token")
                return True
        
        logging.error(f"Login failed: {response_data.get('msg')}")
        return False
        
    except Exception as e:
        logging.error(f"Login error: {e}")
        return False

def update_env_file(key, value):
    """Updates a single value in the .env file."""
    try:
        # Read existing .env file
        if os.path.exists('.env'):
            with open('.env', 'r') as file:
                lines = file.readlines()
        else:
            lines = []

        # Find and replace the line with the key
        key_found = False
        new_lines = []
        for line in lines:
            if line.startswith(f'{key}='):
                new_lines.append(f'{key}={value}\n')
                key_found = True
            else:
                new_lines.append(line)

        # Add the key if it wasn't found
        if not key_found:
            new_lines.append(f'{key}={value}\n')

        # Write back to .env file
        with open('.env', 'w') as file:
            file.writelines(new_lines)
            
    except Exception as e:
        logging.error(f"Error updating .env file: {e}")

def fetch_video_evidence(start_time, end_time, retry_count=0):
    """Fetches video evidence from the API with session handling."""
    if retry_count >= 3:  # Limit retry attempts
        logging.error("Maximum retry attempts reached")
        return {'status': 'error', 'error': 'Maximum retry attempts reached'}
        
    logging.info(f"Fetching video evidence from {start_time} to {end_time}")
    formatted_start_time = start_time.strftime('%Y-%m-%d+%H:%M:%S')
    formatted_end_time = end_time.strftime('%Y-%m-%d+%H:%M:%S')

    data = {
        'alarmType': '',
        'startTime': formatted_start_time,
        'endTime': formatted_end_time,
        'pageNum': '1',
        'pageSize': 30,
        'token': API_TOKEN,
        'scheme': 'https',
        'lang': 'en'
    }
    
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'platform': 'web',
        'version': 'v2'
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=data)
        response.raise_for_status()
        response_data = response.json()
        
        # logging.info(f"API response: {response_data}")
        
        # Handle session expiration
        if response_data.get('status') == 10023:
            logging.info("Session expired. Attempting to login again...")
            if perform_login():
                # Retry the fetch with new token
                return fetch_video_evidence(start_time, end_time, retry_count + 1)
            else:
                return {'status': 'error', 'error': 'Failed to refresh session'}
        
        # Check response structure based on the evidence format
        if (response_data.get('status') == 10000 and 
            response_data.get('data') and 
            response_data['data'].get('list')):
            
            evidence_list = response_data['data']['list']
            if evidence_list:
                logging.info(f"Found {len(evidence_list)} evidence items")
                processed_evidence = []
                
                for evidence in evidence_list:
                    # Extract relevant information from evidence
                    evidence_data = {
                        'deviceID': evidence.get('deviceID'),
                        'deviceName': evidence.get('deviceName'),
                        'alarmType': evidence.get('alarmType'),
                        'alarmTypeValue': evidence.get('alarmTypeValue'),
                        'location': evidence.get('location'),
                        'speed': evidence.get('speed'),
                        'files': []
                    }
                    
                    # Process file list if available
                    if 'alarmFile' in evidence:
                        for file_item in evidence['alarmFile']:
                            if file_item.get('downUrl') and file_item.get('fileType'):
                                file_data = {
                                    'url': file_item['downUrl'],
                                    'type': file_item['fileType'],
                                    'startTime': file_item.get('fileStartTime'),
                                    'stopTime': file_item.get('fileStopTime'),
                                    'channel': file_item.get('channel')
                                }
                                evidence_data['files'].append(file_data)
                    
                    # Add main video and image URLs if available
                    if evidence.get('videoUrl'):
                        evidence_data['files'].append({
                            'url': evidence['videoUrl'],
                            'type': 'video',
                            'startTime': evidence.get('alarmTime'),
                            'stopTime': evidence.get('alarmTimeEnd')
                        })
                    
                    if evidence.get('imageUrl'):
                        evidence_data['files'].append({
                            'url': evidence['imageUrl'],
                            'type': 'image'
                        })
                    
                    processed_evidence.append(evidence_data)
                
                return {'status': response_data.get('status') == 10000, 'data': processed_evidence}
            
        logging.info("No evidence found in the response")
        return {'status': 'no_data', 'data': []}
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching video evidence: {e}")
        return {'status': 'error', 'error': str(e)}

def download_video(url, temp_dir="temp_videos"):
    """Downloads video from URL and returns local path."""
    try:
        # Create temp directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Generate unique filename
        video_id = hashlib.md5(url.encode()).hexdigest()
        local_path = os.path.join(temp_dir, f"{video_id}.mp4")
        
        # Download file if it doesn't exist
        if not os.path.exists(local_path):
            logging.info(f"Downloading video from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
        return local_path
        
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return None

def process_video_for_drowsiness(evidence_data):
    """Process video for drowsiness detection and return results."""
    try:
        # Find video file with type '2' from alarmFile
        video_file = None
        if 'files' in evidence_data:
            for file in evidence_data['files']:
                if file.get('type') == '2':
                    video_file = file
                    break

        if not video_file:
            logging.error("No valid video file found in evidence")
            return False, None
            
        video_url = video_file.get('url')
        if not video_url:
            logging.error("No download URL found for video")
            return False, None
            
        # Download video to temporary location
        local_video_path = download_video(video_url)
        if not local_video_path:
            logging.error("Failed to download video")
            return False, None

        # Initialize counters
        yawn_count = 0
        eye_closed_frames = 0
        total_processed_frames = 0
        consecutive_eye_closed = 0  # Track consecutive frames with closed eyes

        cap = cv2.VideoCapture(local_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Processing video: {total_frames} frames at {fps} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with YOLO model
            results = yolo_model(frame)
            
            # Process detections
            for result in results:
                # Count yawns
                yawn_detections = result.boxes[result.boxes.cls == 0]  # Assuming class 0 is yawn
                yawn_count += len(yawn_detections)
                
                # Count closed eyes
                closed_eyes = result.boxes[result.boxes.cls == 1]  # Assuming class 1 is closed eyes
                if len(closed_eyes) > 0:
                    eye_closed_frames += 1
                    consecutive_eye_closed += 1
                else:
                    consecutive_eye_closed = 0
                
                # Early detection of severe drowsiness
                if (yawn_count >= DROWSINESS_THRESHOLD_YAWN or 
                    consecutive_eye_closed >= DROWSINESS_THRESHOLD_EYE_CLOSED):
                    cap.release()
                    
                    # Clean up downloaded video
                    try:
                        os.remove(local_video_path)
                    except Exception as e:
                        logging.warning(f"Failed to clean up video file: {e}")
                    
                    detection_results = {
                        'is_drowsy': True,
                        'yawn_count': yawn_count,
                        'eye_closed_frames': eye_closed_frames,
                        'total_frames': total_processed_frames,
                        'early_detection': True
                    }
                    return True, detection_results
                
            total_processed_frames += 1

            # Log progress periodically
            if total_processed_frames % 100 == 0:
                progress = (total_processed_frames / total_frames) * 100
                logging.info(f"Processing progress: {progress:.2f}%")

        cap.release()

        # Clean up downloaded video
        try:
            os.remove(local_video_path)
        except Exception as e:
            logging.warning(f"Failed to clean up video file: {e}")

        # Calculate final drowsiness metrics
        is_drowsy = (yawn_count >= DROWSINESS_THRESHOLD_YAWN or 
                    eye_closed_frames >= DROWSINESS_THRESHOLD_EYE_CLOSED)

        detection_results = {
            'is_drowsy': is_drowsy,
            'yawn_count': yawn_count,
            'eye_closed_frames': eye_closed_frames,
            'total_frames': total_processed_frames,
            'early_detection': False
        }

        return True, detection_results

    except Exception as e:
        logging.error(f"Error processing video for drowsiness: {e}")
        if 'local_video_path' in locals() and os.path.exists(local_video_path):
            try:
                os.remove(local_video_path)
            except Exception as cleanup_error:
                logging.warning(f"Failed to clean up video file after error: {cleanup_error}")
        return False, None

def perform_yolo_detection(frame, model):
    """Performs YOLO detection on a frame and returns relevant detections.
       **Placeholder -  Implement based on your chosen YOLO framework.**
    """
    # --- Example using YOLOv5 (ultralytics) - Adjust for your model and framework ---
    # results = model(frame)
    # detections = []
    # for *xyxy, conf, cls in results.xyxy[0]: # xyxy detections
    #     label_index = int(cls)
    #     label = model.names[label_index] # Get class name
    #     if label in ['Yawning', 'Eye Closed']: # Filter for relevant labels
    #         detections.append({'label': label, 'confidence': float(conf), 'bbox': xyxy})
    # return detections
    logging.warning("YOLO detection is a placeholder. Implement your YOLO inference here and return relevant detections.")
    return [] # Placeholder - return empty list if no detections


def analyze_drowsiness(yawn_count, eye_closed_frames, total_frames_processed):
    """Analyzes detection counts to determine drowsiness."""
    logging.info(f"Analyzing drowsiness: Yawns={yawn_count}, Eye Closed Frames={eye_closed_frames}, Total Frames={total_frames_processed}")
    if yawn_count > DROWSINESS_THRESHOLD_YAWN:
        logging.warning("Drowsiness detected: Excessive yawning!")
        return True
    if eye_closed_frames > DROWSINESS_THRESHOLD_EYE_CLOSED: # You might want to normalize this by frame rate/total frames
        logging.warning("Drowsiness detected: Prolonged eye closure!")
        return True
    return False # Not drowsy based on thresholds

def update_evidence_result(evidence_id, detection_results):
    """Updates an existing evidence record with detection results."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE evidence_results
                SET is_drowsy = ?,
                    yawn_count = ?,
                    eye_closed_frames = ?,
                    processing_status = ?
                WHERE id = ?
            ''', (
                detection_results.get('is_drowsy') if detection_results else None,
                detection_results.get('yawn_count') if detection_results else None,
                detection_results.get('eye_closed_frames') if detection_results else None,
                'processed' if detection_results else 'failed',
                evidence_id
            ))
            conn.commit()
    except Exception as e:
        logging.error(f"Error updating evidence result: {e}")

def main():
    """Main processing loop."""
    logging.info("Starting Drowsiness Detection Service...")
    logging.info(f"API Endpoint: {API_ENDPOINT}")
    logging.info(f"YOLO Model Path: {YOLO_MODEL_PATH}")
    logging.info(f"Fetch Interval: {FETCH_INTERVAL_SECONDS} seconds")
    
    try:
        while True:
            try:
                current_start_time = get_last_fetch_time()
                current_end_time = datetime.datetime.now()
                
                if current_start_time >= current_end_time:
                    logging.info("No new time range to fetch. Waiting for next interval.")
                else:
                    result = fetch_video_evidence(current_start_time, current_end_time)
                    logging.info(f"Fetch result: {result}")
                    if result.get('status') and isinstance(result.get('data'), list):
                        evidence_list = result['data']
                        
                        for evidence in evidence_list:
                            logging.info(f"Processing evidence for device: {evidence.get('deviceName')}")
                            
                            # Skip if no files array or empty files
                            if not evidence.get('files'):
                                logging.warning(f"No files found for device: {evidence.get('deviceName')}")
                                continue

                            # Store evidence even if no video to track all events
                            evidence_id = store_evidence_result(evidence, None)
                            
                            # Log alarm type information
                            logging.info(f"Alarm Type: {evidence.get('alarmType')} - {evidence.get('alarmTypeValue')}")
                            
                            # Additional info logging
                            if evidence.get('speed') is not None:
                                logging.info(f"Vehicle Speed: {evidence.get('speed')} km/h")
                            
                            # Process files if they exist
                            for file in evidence['files']:
                                if file.get('type') == '2':  # Video file
                                    video_url = file.get('url')
                                    if video_url:
                                        logging.info(f"Processing video for device: {evidence.get('deviceName')}")
                                        processing_success, detection_results = process_video_for_drowsiness(evidence)
                                        
                                        if processing_success:
                                            # Update the existing evidence record with detection results
                                            update_evidence_result(evidence_id, detection_results)
                                            
                                            if detection_results.get('is_drowsy'):
                                                logging.critical(
                                                    f"DROWSINESS DETECTED - Device: {evidence.get('deviceName')}\n"
                                                    f"Location: {evidence.get('location')}\n"
                                                    f"Speed: {evidence.get('speed')}\n"
                                                    f"Yawns: {detection_results['yawn_count']}\n"
                                                    f"Eye Closed Frames: {detection_results['eye_closed_frames']}"
                                                )
                                        break  # Stop after processing first video file
                    
                    # Update the last fetch time
                    update_last_fetch_time(current_end_time)
                    logging.info(f"Updated fetch time to: {current_end_time}")
                
                # Process any pending evidence
                pending_evidence = get_pending_evidence()
                for evidence_id, video_url, device_name in pending_evidence:
                    if video_url:
                        logging.info(f"Processing pending evidence for device: {device_name}")
                        processing_success, detection_results = process_video_for_drowsiness(video_url)
                        
                        status = 'processed' if processing_success else 'failed'
                        
                        with sqlite3.connect(DB_PATH) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE evidence_results
                                SET is_drowsy = ?,
                                    yawn_count = ?,
                                    eye_closed_frames = ?,
                                    processing_status = ?
                                WHERE id = ?
                            ''', (
                                detection_results.get('is_drowsy') if detection_results else None,
                                detection_results.get('yawn_count') if detection_results else None,
                                detection_results.get('eye_closed_frames') if detection_results else None,
                                status,
                                evidence_id
                            ))
                            conn.commit()
                
                time.sleep(FETCH_INTERVAL_SECONDS)
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(FETCH_INTERVAL_SECONDS)
                
    except KeyboardInterrupt:
        logging.info("Service stopped by user")

if __name__ == "__main__":
    main()
