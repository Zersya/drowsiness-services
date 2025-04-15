import cv2
import os
import logging
import hashlib
import torch
from ultralytics import YOLO
import requests
import time
from dotenv import load_dotenv
from pose_head_detector import PoseHeadDetector

class YoloProcessor:
    def __init__(self, model_path, drowsiness_threshold_yawn=6, drowsiness_threshold_eye_closed=35):
        self.model_path = model_path
        self.drowsiness_threshold_yawn = drowsiness_threshold_yawn
        self.drowsiness_threshold_eye_closed = drowsiness_threshold_eye_closed
        # Load environment variables
        load_dotenv()
        self.use_cuda = os.getenv('USE_CUDA', 'true').lower() == 'true'
        # Extract model name from the path
        self.model_name = os.path.basename(model_path)
        self.model = self.load_model()
        # Add parameters for eye detection tuning
        self.min_blink_frames = int(os.getenv('MIN_BLINK_FRAMES', '1'))  # Minimum frames for a blink (reduced from 3 to 1)
        self.blink_cooldown = int(os.getenv('BLINK_COOLDOWN', '2'))  # Frames to wait before counting next blink (reduced from 15 to 5)
        self.confidence_threshold = float(os.getenv('EYE_DETECTION_CONFIDENCE', '0.6'))  # Minimum confidence for eye closed detection (reduced from 0.6 to 0.5)

        # Initialize pose head detector
        self.pose_model_path = os.getenv('POSE_MODEL_PATH', 'models\yolov8l-pose.pt')
        self.pose_detector = PoseHeadDetector(self.pose_model_path)

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

            # If file exists but is very small (likely corrupted), remove it
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                if file_size < 1024:  # Less than 1KB
                    logging.warning(f"Found existing but potentially corrupted video file (size: {file_size} bytes). Removing it.")
                    os.remove(local_path)

            # Download file if it doesn't exist
            if not os.path.exists(local_path):
                logging.info(f"Downloading video from {url}")
                response = requests.get(url, stream=True, timeout=30)  # Add timeout
                response.raise_for_status()

                # Check content type to ensure it's a video
                content_type = response.headers.get('content-type', '')
                if not ('video' in content_type or 'octet-stream' in content_type):
                    logging.warning(f"Content type '{content_type}' may not be a video")

                # Get content length if available
                content_length = response.headers.get('content-length')
                if content_length:
                    content_length = int(content_length)
                    logging.info(f"Video file size: {content_length / 1024:.2f} KB")

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Verify the downloaded file
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    if file_size < 1024:  # Less than 1KB
                        logging.error(f"Downloaded file is too small ({file_size} bytes), likely corrupted")
                        os.remove(local_path)
                        return None
                    logging.info(f"Successfully downloaded video to {local_path} (size: {file_size / 1024:.2f} KB)")

            return local_path

        except requests.exceptions.Timeout:
            logging.error(f"Timeout while downloading video from {url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error while downloading video: {e}")
            return None
        except Exception as e:
            logging.error(f"Error downloading video: {e}")
            return None

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
            # We don't modify the results object directly since it's a list
            return {
                'detection_results': results,
                'pose_results': pose_results
            }

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None

    def process_video(self, video_url):
        """Process video for drowsiness detection and return results."""
        try:
            # Start timing the processing
            start_time = time.time()

            local_video_path = self.download_video(video_url)
            if not local_video_path:
                logging.error("Failed to download video")
                return False, None

            # Initialize counters
            yawn_count = 0
            eye_closed_frames = 0
            normal_state_frames = 0
            total_processed_frames = 0
            consecutive_eye_closed = 0
            consecutive_normal_state = 0
            blink_cooldown_counter = 0
            potential_blink_frames = 0
            # New variables for improved metrics
            total_eye_closed_frames = 0
            max_consecutive_eye_closed = 0
            # Head pose detection counters
            head_turned_frames = 0
            head_down_frames = 0

            # Try to open the video file
            cap = cv2.VideoCapture(local_video_path)

            # Check if the video file was opened successfully
            if not cap.isOpened():
                logging.error(f"Failed to open video file: {local_video_path}")
                return False, None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Log video properties for debugging
            logging.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}")

            # Reset pose detector frame counters with video FPS
            self.pose_detector.reset_frame_counters(fps)

            # Validate video properties
            if fps <= 0 or total_frames <= 0:
                logging.error("Invalid video properties - FPS or frame count is zero or negative")
                cap.release()
                return False, None

            logging.info(f"Processing video: {total_frames} frames at {fps} FPS")

            frame_skip = max(1, int(fps / 10))  # Process 10 frames per second
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                if frame is None or frame.size == 0:
                    continue

                results = self.process_frame(frame)

                if results is None:
                    continue

                # Process detections
                # Extract detection and pose results
                detection_results = results.get('detection_results', [])
                pose_results = results.get('pose_results', {})

                # Process pose results first
                if pose_results:
                    if pose_results.get('head_turned', False):
                        head_turned_frames += 1
                        logging.info(f"Head turned detected: {pose_results.get('head_turned_counter', 0)}/{pose_results.get('head_turned_threshold', 0)}")
                    if pose_results.get('head_down', False):
                        head_down_frames += 1
                        logging.info(f"Head down detected: {pose_results.get('head_down_counter', 0)}/{pose_results.get('head_down_threshold', 0)}")

                # Process object detection results
                for result in detection_results:
                    # Normal state detection (class 1)
                    normal_state = result.boxes[result.boxes.cls == 1]
                    confident_normal = normal_state[normal_state.conf >= self.confidence_threshold]

                    if len(confident_normal) > 0:
                        normal_state_frames += 1
                        consecutive_normal_state += 1
                        consecutive_eye_closed = 0
                    else:
                        consecutive_normal_state = 0

                    # Yawn detection (class 2)
                    yawn_detections = result.boxes[result.boxes.cls == 2]
                    confident_yawns = yawn_detections[yawn_detections.conf >= self.confidence_threshold]
                    yawn_count += len(confident_yawns)

                    # Closed eyes detection (class 0)
                    closed_eyes = result.boxes[result.boxes.cls == 0]
                    confident_detections = closed_eyes[closed_eyes.conf >= self.confidence_threshold]

                    if len(confident_detections) > 0:
                        # Increment total eye closed frames
                        total_eye_closed_frames += 1
                        potential_blink_frames += 1
                        consecutive_eye_closed += 1
                        # Update maximum consecutive eye closure
                        if consecutive_eye_closed > max_consecutive_eye_closed:
                            max_consecutive_eye_closed = consecutive_eye_closed
                        consecutive_normal_state = 0

                        # Count eye closed events more aggressively
                        # If we have at least min_blink_frames consecutive frames with closed eyes
                        # and we're not in cooldown, count it as an eye closed event
                        if potential_blink_frames >= self.min_blink_frames and blink_cooldown_counter == 0:
                            eye_closed_frames += 1
                            blink_cooldown_counter = self.blink_cooldown
                            logging.info(f"Eye closure detected with confidence: {max(confident_detections.conf):.2f}")
                    else:
                        # Even when eyes are no longer detected as closed, if we had enough frames
                        # to consider it a valid eye closure, count it
                        if potential_blink_frames >= self.min_blink_frames and blink_cooldown_counter == 0:
                            eye_closed_frames += 1
                            blink_cooldown_counter = self.blink_cooldown
                            logging.info("Eye closure event counted after detection ended")
                        potential_blink_frames = 0
                        consecutive_eye_closed = 0

                    if blink_cooldown_counter > 0:
                        blink_cooldown_counter -= 1

                total_processed_frames += 1

                if total_processed_frames % 50 == 0:
                    progress = (total_processed_frames / (total_frames/frame_skip)) * 100
                    logging.info(f"Processing progress: {progress:.2f}% - "
                                f"Yawns: {yawn_count}, Closed Eyes: {eye_closed_frames}, "
                                f"Normal State: {normal_state_frames}")

            cap.release()

            try:
                os.remove(local_video_path)
            except Exception as e:
                logging.warning(f"Failed to clean up video file: {e}")

            if total_processed_frames == 0:
                logging.error("No frames were processed successfully")
                return False, None

            # Calculate total processing time
            end_time = time.time()
            process_time = end_time - start_time

            detection_results = {
                'yawn_count': yawn_count,
                'eye_closed_frames': eye_closed_frames,  # Retained for compatibility
                'normal_state_frames': normal_state_frames,
                'total_frames': total_processed_frames,
                'total_eye_closed_frames': total_eye_closed_frames,  # New metric
                'max_consecutive_eye_closed': max_consecutive_eye_closed,  # New metric
                'early_detection': False,
                'process_time': process_time,  # Add processing time in seconds
                'model_name': self.model_name,  # Add model name
                'head_pose': {
                    'head_turned_frames': head_turned_frames,
                    'head_down_frames': head_down_frames,
                    'is_head_turned': head_turned_frames > 0,
                    'is_head_down': head_down_frames > 0
                },
                'metrics': {
                    'consecutive_eye_closed': consecutive_eye_closed,
                    'consecutive_normal_state': consecutive_normal_state,
                    'potential_blink_frames': potential_blink_frames,
                    'fps': fps,
                    'processed_frame_ratio': total_processed_frames / total_frames
                },
                'processing_status': 'processed'
            }

            logging.info(f"Video processing completed in {process_time:.2f} seconds: {detection_results}")
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
