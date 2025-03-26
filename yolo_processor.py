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

        # --- Removed obsolete blink detection parameters ---
        # self.min_blink_frames = int(os.getenv('MIN_BLINK_FRAMES', '1')) 
        # self.blink_cooldown = int(os.getenv('BLINK_COOLDOWN', '2')) 

    def load_model(self):
        """Loads and returns a YOLO model for drowsiness detection."""
        logging.info("Loading YOLO model...")
        try:
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"CUDA available: {cuda_available}, Using device: {device}")

            model = YOLO(self.model_path)
            model.to(device)

            # Set model parameters (these might be internal defaults, but good to be explicit)
            # model.conf = 0.25 # This might be overridden by specific calls if verbose=False isn't enough
            # model.iou = 0.45   
            
            # Note: self.confidence_threshold is applied *after* model inference below

            if device == 'cuda':
                torch.backends.cudnn.benchmark = True
                # torch.backends.cudnn.deterministic = False # Usually False is faster
                # torch.cuda.set_stream(torch.cuda.Stream()) # Often managed internally

            logging.info(f"YOLO model loaded successfully from {self.model_path} on {device}")
            return model

        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            # Consider raising the exception or handling it more robustly
            return None

    def download_video(self, url, temp_dir="temp_videos"):
        """Downloads video from URL and returns local path."""
        try:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Use a simpler filename if preferred, hash ensures uniqueness
            video_id = hashlib.md5(url.encode()).hexdigest()
            local_path = os.path.join(temp_dir, f"{video_id}.mp4")

            if not os.path.exists(local_path):
                logging.info(f"Downloading video from {url} to {local_path}")
                response = requests.get(url, stream=True, timeout=30) # Added timeout
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        # if chunk: # Filter out keep-alive new chunks
                        f.write(chunk)
                logging.info(f"Video downloaded successfully.")
            else:
                logging.info(f"Using existing video file: {local_path}")

            return local_path

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading video from URL {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error during video download/saving: {e}")
            return None

    def process_frame(self, frame):
        """Process a single frame with YOLO model."""
        try:
            # Optional: Resize if needed, but YOLOv8 handles various sizes. 
            # Ensure consistency if resizing. Consider padding instead of stretching.
            # min_size = 640 
            # height, width = frame.shape[:2]
            # if height < min_size or width < min_size:
            #     scale = max(min_size/width, min_size/height)
            #     frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Optional: Image enhancement - Use cautiously, can sometimes hurt accuracy.
            # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5) 

            # Perform inference
            results = self.model(frame, verbose=False, conf=0.2) # Lower internal conf slightly? Test performance.

            if not results:
                return None # No results list returned

            # Return the first result object (usually contains boxes, masks, etc.)
            return results[0] 

        except Exception as e:
            logging.error(f"Error processing frame: {e}", exc_info=True) # Add traceback
            return None

    def process_video(self, video_url):
        """Process video for drowsiness detection and return results for the analyzer."""
        try:
            start_time = time.time()

            local_video_path = self.download_video(video_url)
            if not local_video_path:
                logging.error("Failed to download or find video.")
                # Return status and None for results
                return False, {'processing_status': 'download_failed'} 

            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video file: {local_video_path}")
                return False, {'processing_status': 'open_failed'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or total_video_frames <= 0:
                logging.warning(f"Invalid video properties for {local_video_path}: FPS={fps}, TotalFrames={total_video_frames}. Trying to process anyway.")
                # Handle potential division by zero later if fps is needed and still 0
                if fps <= 0: fps = 30 # Assign a default FPS if invalid
                # Fall through to attempt processing

            logging.info(f"Processing video: {local_video_path}, Total Frames: {total_video_frames}, FPS: {fps:.2f}")

            # --- Initialize counters needed by RateBasedAnalyzerRevised ---
            yawn_count = 0
            total_eye_closed_frames = 0 # Total frames eyes detected closed
            normal_state_frames = 0    # Total frames normal state detected
            max_consecutive_eye_closed = 0 # Longest run of closed eyes
            
            # --- Internal tracking variables ---
            total_processed_frames = 0 # Count frames actually processed after skipping
            consecutive_eye_closed_current = 0
            # consecutive_normal_state = 0 # Not strictly needed by analyzer

            # Define class IDs (ensure these match your trained model!)
            CLASS_ID_EYE_CLOSED = 0
            CLASS_ID_NORMAL = 1
            CLASS_ID_YAWN = 2

            # Frame skipping (process around 10-15 FPS for efficiency)
            target_processing_fps = 15 
            frame_skip = max(1, int(fps / target_processing_fps)) if fps > 0 else 1
            logging.info(f"Processing approx. 1 frame every {frame_skip} frames.")
            
            frame_index = -1 # Use index for tracking position

            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                frame_index += 1
                
                # Apply frame skipping
                if frame_index % frame_skip != 0:
                    continue

                if frame is None or frame.size == 0:
                    logging.warning(f"Empty frame encountered at index {frame_index}. Skipping.")
                    continue
                
                total_processed_frames += 1 # Increment for each frame sent to model

                # Process the frame using YOLO
                result = self.process_frame(frame)

                if result is None or result.boxes is None:
                    # Reset consecutive count if frame processing failed or no boxes
                    consecutive_eye_closed_current = 0 
                    continue

                # --- Check Detections (using self.confidence_threshold) ---
                frame_has_closed_eyes = False
                frame_has_normal_state = False
                frame_has_yawn = False

                # Iterate through detected boxes
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()

                    if confidence >= self.confidence_threshold:
                        if class_id == CLASS_ID_EYE_CLOSED:
                            frame_has_closed_eyes = True
                        elif class_id == CLASS_ID_NORMAL:
                            frame_has_normal_state = True
                        elif class_id == CLASS_ID_YAWN:
                            frame_has_yawn = True 
                            # Count every confident yawn detection individually
                            yawn_count += 1 

                # --- Update Counters Based on Frame Detections ---
                
                if frame_has_normal_state:
                    normal_state_frames += 1
                    # Optional: Reset consecutive eye closed if normal state detected? Needs careful thought.
                    # If 'normal' means 'alert', it might make sense.
                    # consecutive_eye_closed_current = 0 # Uncomment if 'normal' should override 'closed' status

                if frame_has_closed_eyes:
                    total_eye_closed_frames += 1
                    consecutive_eye_closed_current += 1
                else:
                    # If eyes are not closed in this frame, reset consecutive counter
                    consecutive_eye_closed_current = 0

                # Update the maximum consecutive count seen so far
                if consecutive_eye_closed_current > max_consecutive_eye_closed:
                    max_consecutive_eye_closed = consecutive_eye_closed_current
                
                # --- Logging Progress ---
                if total_processed_frames % 100 == 0: # Log every 100 processed frames
                    approx_original_frame = frame_index # frame_index corresponds to original video
                    progress_percent = (approx_original_frame / total_video_frames) * 100 if total_video_frames > 0 else 0
                    logging.info(f"Progress: ~{progress_percent:.1f}% (Frame {approx_original_frame}/{total_video_frames}) - "
                                 f"Yawns: {yawn_count}, Total Closed: {total_eye_closed_frames}, Max Consec Closed: {max_consecutive_eye_closed}, Normal: {normal_state_frames}")

            # --- End of Video Processing ---
            cap.release()
            logging.info("Video capture released.")

            # Attempt to clean up the downloaded video file
            try:
                if local_video_path and os.path.exists(local_video_path):
                    os.remove(local_video_path)
                    logging.info(f"Temporary video file {local_video_path} removed.")
            except OSError as e:
                logging.warning(f"Failed to clean up video file {local_video_path}: {e}")

            if total_processed_frames == 0:
                logging.error("No frames were processed successfully (video might be empty or invalid after skipping).")
                return False, {'processing_status': 'no_frames_processed'}

            end_time = time.time()
            process_time = end_time - start_time

            # --- Prepare results dictionary for RateBasedAnalyzerRevised ---
            detection_results = {
                'yawn_count': yawn_count,
                'total_eye_closed_frames': total_eye_closed_frames,
                'max_consecutive_eye_closed': max_consecutive_eye_closed,
                'normal_state_frames': normal_state_frames,
                'total_frames': total_processed_frames, # Use the count of frames actually analyzed
                'model_name': self.model_name,
                'process_time': process_time,
                'processing_status': 'processed_successfully',
                'metrics': {
                    'fps': fps, # Original video FPS
                    'total_original_frames': total_video_frames,
                    'frames_actually_processed': total_processed_frames,
                    'frame_skip_ratio': frame_skip,
                    'detection_confidence_threshold': self.confidence_threshold
                }
                # Removed obsolete/unused fields:
                # 'eye_closed_frames': eye_closed_frames_event_count, 
                # 'early_detection': False, 
                # 'metrics': {'consecutive_eye_closed', 'consecutive_normal_state', 'potential_blink_frames'}
            }

            logging.info(f"Video processing completed. Time: {process_time:.2f}s. Results: {detection_results}")
            return True, detection_results

        except Exception as e:
            logging.error(f"Unhandled error during video processing: {e}", exc_info=True)
            # Ensure cleanup happens even on unexpected errors
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'local_video_path' in locals() and local_video_path and os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                    logging.info(f"Cleaned up video file {local_video_path} after error.")
                except OSError as cleanup_error:
                    logging.warning(f"Failed to clean up video file {local_video_path} after error: {cleanup_error}")
            return False, {'processing_status': 'processing_error', 'error_message': str(e)}

    def process_video_from_evidence(self, evidence_data):
        """Process video from evidence data structure."""
        # This method seems okay, just ensure it calls the updated process_video
        try:
            video_file = None
            if 'files' in evidence_data:
                for file_info in evidence_data['files']:
                     # Assuming 'type' is string '2' for video
                    if isinstance(file_info, dict) and file_info.get('type') == '2': 
                        video_file = file_info
                        break # Found the video file

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

