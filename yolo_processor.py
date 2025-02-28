import cv2
import os
import logging
import hashlib
import torch
from ultralytics import YOLO
import requests
from dotenv import load_dotenv

class YoloProcessor:
    def __init__(self, model_path, drowsiness_threshold_yawn, drowsiness_threshold_eye_closed):
        self.model_path = model_path
        self.drowsiness_threshold_yawn = drowsiness_threshold_yawn
        self.drowsiness_threshold_eye_closed = drowsiness_threshold_eye_closed
        # Load environment variables
        load_dotenv()
        self.use_cuda = os.getenv('USE_CUDA', 'true').lower() == 'true'
        self.model = self.load_model()
        # Add parameters for eye detection tuning
        self.min_blink_frames = int(os.getenv('MIN_BLINK_FRAMES', '3'))  # Minimum frames for a blink
        self.blink_cooldown = int(os.getenv('BLINK_COOLDOWN', '15'))  # Frames to wait before counting next blink
        self.confidence_threshold = float(os.getenv('EYE_DETECTION_CONFIDENCE', '0.6'))  # Minimum confidence for eye closed detection
        
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
                
            logging.info(f"YOLO model loaded successfully on {device}")
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

    def process_frame(self, frame):
        """Process a single frame with appropriate device placement."""
        try:
            if self.use_cuda and torch.cuda.is_available():
                # Convert frame to tensor and move to GPU
                frame_tensor = torch.from_numpy(frame).cuda()
                results = self.model(frame_tensor)
                # Move results back to CPU if needed
                results = [r.cpu() for r in results]
            else:
                results = self.model(frame)
            return results
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
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
            blink_cooldown_counter = 0
            potential_blink_frames = 0

            cap = cv2.VideoCapture(local_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logging.info(f"Processing video: {total_frames} frames at {fps} FPS")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame with YOLO model using the new process_frame method
                results = self.process_frame(frame)
                if results is None:
                    continue
                
                # Process detections
                for result in results:
                    # Count yawns
                    yawn_detections = result.boxes[result.boxes.cls == 2]  # Assuming class 2 is yawn
                    yawn_count += len(yawn_detections)
                    
                    # Improved closed eyes detection
                    closed_eyes = result.boxes[result.boxes.cls == 0]
                    confident_detections = closed_eyes[closed_eyes.conf >= self.confidence_threshold]
                    
                    if len(confident_detections) > 0:
                        potential_blink_frames += 1
                        consecutive_eye_closed += 1
                    else:
                        if potential_blink_frames >= self.min_blink_frames and blink_cooldown_counter == 0:
                            eye_closed_frames += 1
                            blink_cooldown_counter = self.blink_cooldown
                        potential_blink_frames = 0
                        consecutive_eye_closed = 0

                    if blink_cooldown_counter > 0:
                        blink_cooldown_counter -= 1

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
