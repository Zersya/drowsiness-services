import cv2
import os
import logging
import hashlib
import torch
from ultralytics import YOLO
import requests

class YoloProcessor:
    def __init__(self, model_path, drowsiness_threshold_yawn, drowsiness_threshold_eye_closed):
        self.model_path = model_path
        self.drowsiness_threshold_yawn = drowsiness_threshold_yawn
        self.drowsiness_threshold_eye_closed = drowsiness_threshold_eye_closed
        self.model = self.load_model()
        
    def load_model(self):
        """Loads and returns a YOLO model for drowsiness detection."""
        logging.info("Loading YOLO model...")
        try:
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {device}")

            # Load the model
            model = YOLO(self.model_path)
            
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
    
    def download_video(self, url, temp_dir="temp_videos"):
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
    
    def process_video(self, video_url):
        """Process video for drowsiness detection and return results."""
        try:
            # Download video to temporary location
            local_video_path = self.download_video(video_url)
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
                results = self.model(frame)
                
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
                    if (yawn_count >= self.drowsiness_threshold_yawn or 
                        consecutive_eye_closed >= self.drowsiness_threshold_eye_closed):
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
            is_drowsy = (yawn_count >= self.drowsiness_threshold_yawn or 
                        eye_closed_frames >= self.drowsiness_threshold_eye_closed)

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
    
    def process_video_from_evidence(self, evidence_data):
        """Process video from evidence data structure."""
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
                
            return self.process_video(video_url)
            
        except Exception as e:
            logging.error(f"Error processing evidence video: {e}")
            return False, None