import cv2
import numpy as np
import logging
from ultralytics import YOLO
import torch
import os
from dotenv import load_dotenv

class PoseHeadDetector:
    """Class for detecting head pose (head down or head turn) using YOLOv8 pose model."""
    
    def __init__(self, model_path='yolov8l-pose.pt'):
        """
        Initialize the pose head detector.
        
        Args:
            model_path (str): Path to the YOLOv8 pose model
        """
        # Load environment variables
        load_dotenv()
        self.model_path = model_path
        self.use_cuda = os.getenv('USE_CUDA', 'true').lower() == 'true'
        
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
            # "left_ear": 3, "right_ear": 4, # Optional for other logic
            # "left_shoulder": 5, "right_shoulder": 6 # Optional for context
        }
        
        # Load the model
        self.model = self.load_model()
        
    def load_model(self):
        """Loads and returns a YOLO pose model."""
        logging.info("Loading YOLO pose model...")
        try:
            # Check if CUDA is available and enabled
            cuda_available = torch.cuda.is_available()
            device = 'cuda' if (cuda_available and self.use_cuda) else 'cpu'
            logging.info(f"CUDA available: {cuda_available}, Using device: {device} for pose model")

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

            logging.info(f"YOLO pose model loaded successfully on {device}")
            return model

        except Exception as e:
            logging.error(f"Error loading YOLO pose model: {e}")
            return None
    
    def reset_frame_counters(self, fps=30):
        """Reset frame counters and convert time thresholds to frame counts."""
        # Convert time thresholds to frame counts
        self.head_turned_frames_threshold = max(1, int(self.head_turned_threshold_seconds * fps))
        self.head_down_frames_threshold = max(1, int(self.head_down_threshold_seconds * fps))
        
        # Reset counters
        self.head_turned_counter = 0
        self.head_down_counter = 0
        
        # Reset status flags
        self.distracted_head_turn = False
        self.distracted_head_down = False
    
    def process_frame(self, frame):
        """
        Process a single frame to detect head pose.
        
        Args:
            frame: The video frame to process
            
        Returns:
            dict: Detection results with head pose information
        """
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
                
                if result.keypoints and result.keypoints.data.shape[0] > 0:
                    # Assuming the first detected person is the target
                    keypoints_data = result.keypoints.data[0].cpu().numpy()
                    
                    # Helper to get keypoint coords and confidence
                    def get_kp(name):
                        idx = self.kp_indices.get(name, -1)
                        if idx != -1 and idx < len(keypoints_data):
                            x, y, conf = keypoints_data[idx]
                            xy = (int(x), int(y))
                            return xy, conf
                        return (0,0), 0.0
                    
                    try:
                        # Get key points needed for head pose
                        (nose_xy, n_conf) = get_kp("nose")
                        (left_eye_xy, le_conf) = get_kp("left_eye")
                        (right_eye_xy, re_conf) = get_kp("right_eye")
                        
                        # --- Head Turn Check ---
                        if (n_conf > self.keypoint_conf_threshold and 
                            le_conf > self.keypoint_conf_threshold and 
                            re_conf > self.keypoint_conf_threshold):
                            
                            eye_center_x = (left_eye_xy[0] + right_eye_xy[0]) / 2
                            eye_dist_x = abs(left_eye_xy[0] - right_eye_xy[0])
                            
                            if eye_dist_x > 0 and abs(nose_xy[0] - eye_center_x) > eye_dist_x * self.head_turn_ratio_threshold:
                                frame_flag_head_turned = True
                        
                        # --- Head Down Check ---
                        if (n_conf > self.keypoint_conf_threshold and 
                            le_conf > self.keypoint_conf_threshold and 
                            re_conf > self.keypoint_conf_threshold):
                            
                            eye_center_y = (left_eye_xy[1] + right_eye_xy[1]) / 2
                            eye_dist_x = abs(left_eye_xy[0] - right_eye_xy[0])  # Use horizontal dist as scale ref
                            
                            if eye_dist_x > 0 and (nose_xy[1] - eye_center_y) > eye_dist_x * self.head_down_ratio_threshold:
                                frame_flag_head_down = True
                    
                    except Exception as e:
                        logging.error(f"An error occurred during keypoint logic: {e}")
            
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
            logging.error(f"Error processing frame for head pose: {e}")
            return {"head_turned": False, "head_down": False}
