import cv2
import os
import logging
import hashlib
import torch
from ultralytics import YOLO
import requests
import time
from dotenv import load_dotenv

# Configure logging if not already done elsewhere
# logging.basicConfig(level=logging.INFO)

class YoloProcessor:
    # Removed obsolete drowsiness_threshold arguments
    def __init__(self, model_path):
        self.model_path = model_path
        
        # Load environment variables
        load_dotenv()
        self.use_cuda = os.getenv('USE_CUDA', 'true').lower() == 'true'
        # Confidence threshold for ALL detections (eyes, yawn, normal)
        self.confidence_threshold = float(os.getenv('DETECTION_CONFIDENCE', '0.5')) # Use a general name, default 0.5
        
        # Extract model name from the path
        self.model_name = os.path.basename(model_path)
        self.model = self.load_model()

    def load_model(self):
        """Loads and returns a YOLO model for drowsiness detection."""
        logging.info("Loading YOLO model...")
        try:
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"CUDA available: {cuda_available}, Using device: {device}")

            model = YOLO(self.model_path)
            model.to(device)

            if device == 'cuda':
                torch.backends.cudnn.benchmark = True

            logging.info(f"YOLO model loaded successfully from {self.model_path} on {device}")
            return model

        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}", exc_info=True)
            return None

    def download_video(self, url, temp_dir="temp_videos"):
        """Downloads video from URL and returns local path."""
        try:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            video_id = hashlib.md5(url.encode()).hexdigest()
            local_path = os.path.join(temp_dir, f"{video_id}.mp4")

            if not os.path.exists(local_path):
                logging.info(f"Downloading video from {url} to {local_path}")
                response = requests.get(url, stream=True, timeout=30) 
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"Video downloaded successfully.")
            else:
                logging.info(f"Using existing video file: {local_path}")

            return local_path

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading video from URL {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error during video download/saving: {e}", exc_info=True)
            return None

    def process_frame(self, frame):
        """Process a single frame with YOLO model."""
        try:
            # Perform inference
            # Lowering internal conf might find more boxes initially, before our threshold filters them
            results = self.model(frame, verbose=False, conf=0.2) 

            if not results:
                return None 

            return results[0] 

        except Exception as e:
            logging.error(f"Error processing frame: {e}", exc_info=True) 
            return None

    def process_video(self, video_url):
        """Process video for drowsiness detection and return results for the analyzer."""
        try:
            start_time = time.time()

            local_video_path = self.download_video(video_url)
            if not local_video_path:
                logging.error("Failed to download or find video.")
                return False, {'processing_status': 'download_failed'} 

            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video file: {local_video_path}")
                return False, {'processing_status': 'open_failed'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0: 
                logging.warning(f"Invalid FPS ({fps}) detected for {local_video_path}. Setting to 30 FPS default.")
                fps = 30 
            if total_video_frames <= 0:
                 logging.warning(f"Invalid total frame count ({total_video_frames}) for {local_video_path}. Progress reporting may be inaccurate.")


            logging.info(f"Processing video: {local_video_path}, Total Frames: {total_video_frames}, FPS: {fps:.2f}")

            # --- Initialize counters ---
            yawn_count = 0              # Total confident yawn detections
            eye_closed_frames = 0       # <<< ADDED BACK: Total confident eye closed detections
            total_eye_closed_frames = 0 # Total frames where eyes were detected closed
            normal_state_frames = 0     # Total frames where normal state was detected
            max_consecutive_eye_closed = 0 # Longest run of closed eyes (in frames)
            
            # --- Internal tracking variables ---
            total_processed_frames = 0 # Count frames actually processed after skipping
            consecutive_eye_closed_current = 0

            # Define class IDs (ensure these match your trained model!)
            CLASS_ID_EYE_CLOSED = 0
            CLASS_ID_NORMAL = 1
            CLASS_ID_YAWN = 2

            # Frame skipping
            target_processing_fps = 15 
            frame_skip = max(1, int(fps / target_processing_fps))
            logging.info(f"Processing approx. 1 frame every {frame_skip} frames.")
            
            frame_index = -1 

            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                frame_index += 1
                
                if frame_index % frame_skip != 0:
                    continue

                if frame is None or frame.size == 0:
                    logging.warning(f"Empty frame encountered at index {frame_index}. Skipping.")
                    continue
                
                total_processed_frames += 1 

                result = self.process_frame(frame)

                if result is None or result.boxes is None:
                    consecutive_eye_closed_current = 0 
                    continue

                # --- Check Detections ---
                frame_has_confident_closed_eyes = False
                frame_has_confident_normal_state = False
                # Keep track if we already counted eye closure for this frame
                counted_eye_closed_this_frame = False

                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()

                    if confidence >= self.confidence_threshold:
                        if class_id == CLASS_ID_EYE_CLOSED:
                            frame_has_confident_closed_eyes = True
                            # Increment eye_closed_frames for EACH confident detection
                            eye_closed_frames += 1 # <<< INCREMENT HERE
                            # Mark that we found at least one for the total_eye_closed_frames counter
                            counted_eye_closed_this_frame = True 
                        elif class_id == CLASS_ID_NORMAL:
                            frame_has_confident_normal_state = True
                        elif class_id == CLASS_ID_YAWN:
                            # Increment yawn_count for EACH confident detection
                            yawn_count += 1 

                # --- Update Frame-Based Counters ---
                
                if frame_has_confident_normal_state:
                    normal_state_frames += 1
                    # Optional: Reset consecutive eye closed if normal state detected? 
                    # consecutive_eye_closed_current = 0 

                # Use the flag set during box iteration for frame-level counts
                if counted_eye_closed_this_frame: 
                    total_eye_closed_frames += 1
                    consecutive_eye_closed_current += 1
                else:
                    # If no confident eye closure detected in this frame, reset consecutive counter
                    consecutive_eye_closed_current = 0

                # Update the maximum consecutive count
                if consecutive_eye_closed_current > max_consecutive_eye_closed:
                    max_consecutive_eye_closed = consecutive_eye_closed_current
                
                # --- Logging Progress ---
                if total_processed_frames % 100 == 0: 
                    progress_percent = (frame_index / total_video_frames) * 100 if total_video_frames > 0 else 0
                    logging.info(f"Progress: ~{progress_percent:.1f}% (Frame {frame_index}/{total_video_frames}) - "
                                 f"Yawns: {yawn_count}, EyeClosed Det: {eye_closed_frames}, " # Log new counter
                                 f"Total Closed Frames: {total_eye_closed_frames}, Max Consec: {max_consecutive_eye_closed}, Normal: {normal_state_frames}")

            # --- End of Video Processing ---
            cap.release()
            logging.info("Video capture released.")

            try:
                if local_video_path and os.path.exists(local_video_path):
                    os.remove(local_video_path)
                    logging.info(f"Temporary video file {local_video_path} removed.")
            except OSError as e:
                logging.warning(f"Failed to clean up video file {local_video_path}: {e}")

            if total_processed_frames == 0:
                logging.error("No frames were processed successfully.")
                return False, {'processing_status': 'no_frames_processed'}

            end_time = time.time()
            process_time = end_time - start_time

            # --- Prepare results dictionary ---
            detection_results = {
                'yawn_count': yawn_count,
                'eye_closed_frames': eye_closed_frames, # <<< ADDED BACK
                'total_eye_closed_frames': total_eye_closed_frames,
                'max_consecutive_eye_closed': max_consecutive_eye_closed,
                'normal_state_frames': normal_state_frames,
                'total_frames': total_processed_frames, 
                'model_name': self.model_name,
                'process_time': process_time,
                'processing_status': 'processed_successfully',
                'metrics': {
                    'fps': fps, 
                    'total_original_frames': total_video_frames,
                    'frames_actually_processed': total_processed_frames,
                    'frame_skip_ratio': frame_skip,
                    'detection_confidence_threshold': self.confidence_threshold
                }
            }

            logging.info(f"Video processing completed. Time: {process_time:.2f}s. Results: {detection_results}")
            return True, detection_results

        except Exception as e:
            logging.error(f"Unhandled error during video processing: {e}", exc_info=True)
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'local_video_path' in locals() and local_video_path and os.path.exists(local_video_path):
                try: os.remove(local_video_path); logging.info(f"Cleaned up {local_video_path} after error.")
                except OSError as cleanup_error: logging.warning(f"Failed cleanup {local_video_path}: {cleanup_error}")
            return False, {'processing_status': 'processing_error', 'error_message': str(e)}

    def process_video_from_evidence(self, evidence_data):
        """Process video from evidence data structure."""
        # This method remains the same, calling the updated process_video
        try:
            video_file = None
            if 'files' in evidence_data:
                for file_info in evidence_data['files']:
                    if isinstance(file_info, dict) and file_info.get('type') == '2': 
                        video_file = file_info
                        break 

            if not video_file:
                logging.error("No dictionary entry with type '2' (video) found in evidence 'files' list.")
                return False, {'processing_status': 'no_video_file_in_evidence'}

            video_url = video_file.get('url')
            if not video_url:
                logging.error("Video file entry found, but 'url' is missing.")
                return False, {'processing_status': 'video_url_missing'}

            logging.info(f"Processing video from evidence URL: {video_url}")
            return self.process_video(video_url)

        except Exception as e:
            logging.error(f"Error processing evidence video: {e}", exc_info=True)
            return False, {'processing_status': 'evidence_processing_error', 'error_message': str(e)}
