#!/usr/bin/env python3
"""
Fatigue Detection System with CUDA GPU Acceleration
==================================================

A system for detecting driver fatigue by analyzing facial features
from a single front-facing video. It analyzes eye closure patterns (PERCLOS)
to determine fatigue levels.

Enhanced with CUDA GPU acceleration for improved performance while maintaining
backward compatibility with CPU-only environments.

Author: Fellou AI Agent
Date: December 6, 2025 (Updated: June 12, 2025)
CUDA Enhancement: July 7, 2025
"""

import cv2
import numpy as np
import dlib
import json
import os
import argparse
import time
import requests
import tempfile
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import euclidean

# CUDA and GPU acceleration imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    # Check if OpenCV was compiled with CUDA support
    OPENCV_CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
except (AttributeError, cv2.error):
    OPENCV_CUDA_AVAILABLE = False

class CUDAManager:
    """
    Manages CUDA GPU acceleration for the drowsiness detection system.
    Provides automatic fallback to CPU processing when GPU is not available.
    """

    def __init__(self, enable_gpu: bool = True, device_id: int = 0):
        """
        Initialize CUDA manager with GPU acceleration settings.

        Args:
            enable_gpu: Whether to attempt GPU acceleration
            device_id: CUDA device ID to use (default: 0)
        """
        self.enable_gpu = enable_gpu
        self.device_id = device_id
        self.gpu_available = False
        self.cupy_available = CUPY_AVAILABLE
        self.opencv_cuda_available = OPENCV_CUDA_AVAILABLE

        # Performance tracking
        self.gpu_processing_times = []
        self.cpu_processing_times = []

        if self.enable_gpu:
            self._initialize_gpu()

        # Log GPU status
        self._log_gpu_status()

    def _initialize_gpu(self):
        """Initialize GPU acceleration if available."""
        try:
            if self.cupy_available:
                # Set CUDA device
                cp.cuda.Device(self.device_id).use()
                # Test GPU availability with a simple operation
                test_array = cp.array([1, 2, 3])
                _ = cp.sum(test_array)
                self.gpu_available = True
                logging.info(f"✅ CUDA GPU acceleration initialized on device {self.device_id}")
            else:
                logging.warning("⚠️ CuPy not available, GPU acceleration disabled")
        except Exception as e:
            logging.warning(f"⚠️ Failed to initialize GPU acceleration: {e}")
            self.gpu_available = False

    def _log_gpu_status(self):
        """Log the current GPU acceleration status."""
        status_lines = [
            "🔧 GPU Acceleration Status:",
            f"  • GPU Enabled: {'✅' if self.enable_gpu else '❌'}",
            f"  • CuPy Available: {'✅' if self.cupy_available else '❌'}",
            f"  • OpenCV CUDA Available: {'✅' if self.opencv_cuda_available else '❌'}",
            f"  • GPU Ready: {'✅' if self.gpu_available else '❌'}"
        ]

        if self.gpu_available:
            try:
                device_name = cp.cuda.Device(self.device_id).attributes['name']
                memory_info = cp.cuda.MemoryInfo()
                status_lines.extend([
                    f"  • Device: {device_name}",
                    f"  • Memory: {memory_info.total // (1024**3)} GB total"
                ])
            except:
                pass

        for line in status_lines:
            print(line)

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available and enabled."""
        return self.enable_gpu and self.gpu_available

    def to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """
        Transfer numpy array to GPU memory.

        Args:
            array: NumPy array to transfer

        Returns:
            CuPy array on GPU or original array if GPU not available
        """
        if self.is_gpu_available():
            try:
                return cp.asarray(array)
            except Exception as e:
                logging.warning(f"Failed to transfer array to GPU: {e}")
        return array

    def to_cpu(self, array) -> np.ndarray:
        """
        Transfer array from GPU to CPU memory.

        Args:
            array: Array to transfer (CuPy or NumPy)

        Returns:
            NumPy array on CPU
        """
        if self.is_gpu_available() and hasattr(array, 'get'):
            try:
                return array.get()
            except Exception as e:
                logging.warning(f"Failed to transfer array to CPU: {e}")
        return np.asarray(array)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for GPU vs CPU processing."""
        stats = {
            'gpu_available': self.gpu_available,
            'gpu_processing_times': self.gpu_processing_times.copy(),
            'cpu_processing_times': self.cpu_processing_times.copy(),
            'avg_gpu_time': np.mean(self.gpu_processing_times) if self.gpu_processing_times else 0,
            'avg_cpu_time': np.mean(self.cpu_processing_times) if self.cpu_processing_times else 0
        }

        if stats['avg_gpu_time'] > 0 and stats['avg_cpu_time'] > 0:
            stats['speedup_ratio'] = stats['avg_cpu_time'] / stats['avg_gpu_time']
        else:
            stats['speedup_ratio'] = 1.0

        return stats

    def benchmark_gpu_vs_cpu(self, test_frame: np.ndarray, iterations: int = 10) -> Dict:
        """
        Benchmark GPU vs CPU performance for image processing operations.

        Args:
            test_frame: Test frame for benchmarking
            iterations: Number of iterations to run for each test

        Returns:
            Dictionary with benchmark results
        """
        results = {
            'gpu_available': self.gpu_available,
            'iterations': iterations,
            'operations': {}
        }

        if not self.gpu_available:
            print("⚠️ GPU not available for benchmarking")
            return results

        print(f"🏁 Starting GPU vs CPU benchmark ({iterations} iterations)...")

        # Test color conversion
        print("  Testing color conversion...")
        cpu_times = []
        gpu_times = []

        for i in range(iterations):
            # CPU test
            start_time = time.time()
            _ = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            cpu_times.append(time.time() - start_time)

            # GPU test
            start_time = time.time()
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(test_frame)
                gpu_gray = cv2.cuda_GpuMat()
                cv2.cuda.cvtColor(gpu_frame, gpu_gray, cv2.COLOR_BGR2GRAY)
                _ = gpu_gray.download()
                gpu_times.append(time.time() - start_time)
            except:
                gpu_times.append(float('inf'))

        results['operations']['color_conversion'] = {
            'cpu_avg_time': np.mean(cpu_times),
            'gpu_avg_time': np.mean(gpu_times),
            'speedup': np.mean(cpu_times) / np.mean(gpu_times) if np.mean(gpu_times) > 0 else 0
        }

        # Test histogram equalization
        print("  Testing histogram equalization...")
        gray_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        cpu_times = []
        gpu_times = []

        for i in range(iterations):
            # CPU test
            start_time = time.time()
            _ = cv2.equalizeHist(gray_frame)
            cpu_times.append(time.time() - start_time)

            # GPU test
            start_time = time.time()
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(gray_frame)
                gpu_result = cv2.cuda_GpuMat()
                cv2.cuda.equalizeHist(gpu_frame, gpu_result)
                _ = gpu_result.download()
                gpu_times.append(time.time() - start_time)
            except:
                gpu_times.append(float('inf'))

        results['operations']['histogram_equalization'] = {
            'cpu_avg_time': np.mean(cpu_times),
            'gpu_avg_time': np.mean(gpu_times),
            'speedup': np.mean(cpu_times) / np.mean(gpu_times) if np.mean(gpu_times) > 0 else 0
        }

        print("✅ Benchmark completed")
        return results

@dataclass
class FatigueMetrics:
    """Data class to store fatigue analysis metrics from the front camera"""
    eye_aspect_ratio_left: float
    eye_aspect_ratio_right: float
    perclos_score: float
    blink_frequency: float
    frame_number: int
    timestamp: float

@dataclass
class FatigueResult:
    """Data class to store final fatigue detection results"""
    driver_name: str
    percentage_fatigue: float
    is_fatigue: bool
    confidence: float
    analysis_details: Dict
    analysis_timestamp: str

class FatigueDetectionSystem:
    """
    Main class for fatigue detection system with CUDA GPU acceleration
    """

    def __init__(self, enable_gpu: bool = True, gpu_device_id: int = 0):
        """
        Initialize the fatigue detection system with optional GPU acceleration.

        Args:
            enable_gpu: Whether to enable GPU acceleration (default: True)
            gpu_device_id: CUDA device ID to use (default: 0)
        """
        # Initialize CUDA manager first
        self.cuda_manager = CUDAManager(enable_gpu=enable_gpu, device_id=gpu_device_id)

        # Initialize face detection and landmark prediction
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self._initialize_predictor()

        # Initialize GPU-accelerated OpenCV objects if available
        self._initialize_gpu_cv_objects()

        # --- INFRARED-OPTIMIZED THRESHOLDS for High Sensitivity ---
        # Adaptive EAR thresholds optimized for infrared video characteristics
        self.EAR_THRESHOLD_BASE = 0.25  # Higher base threshold for infrared
        self.EAR_THRESHOLD_ADAPTIVE = 0.25  # Will be calibrated per video
        self.EAR_CALIBRATION_FRAMES = 90  # More frames for stable infrared calibration

        # PERCLOS thresholds - highly sensitive for infrared drowsiness detection
        self.PERCLOS_THRESHOLD_MILD = 0.08    # 8% for early detection in infrared
        self.PERCLOS_THRESHOLD_MODERATE = 0.15 # 15% for moderate fatigue
        self.PERCLOS_THRESHOLD_SEVERE = 0.25   # 25% for severe fatigue
        self.PERCLOS_WINDOW_SECONDS = 3.0      # Longer window for infrared stability

        # Multi-level fatigue detection - optimized for infrared sensitivity
        self.FATIGUE_THRESHOLD_MILD = 0.15     # 15% for early infrared detection
        self.FATIGUE_THRESHOLD_MODERATE = 0.35 # 35% for moderate fatigue
        self.FATIGUE_THRESHOLD_SEVERE = 0.60   # 60% for severe fatigue

        # Confidence thresholds for precision control
        self.MIN_CONFIDENCE_THRESHOLD = 0.60   # Minimum confidence for fatigue detection
        
        # Blink analysis parameters
        self.MIN_BLINK_DURATION = 3    # Minimum frames for valid blink
        self.MAX_BLINK_DURATION = 15   # Maximum frames for valid blink
        self.MICROSLEEP_THRESHOLD = 30 # Frames indicating microsleep
        
        # Yawning detection parameters
        self.YAWN_THRESHOLD = 0.6      # Mouth aspect ratio threshold for yawning
        self.MIN_YAWN_DURATION = 10    # Minimum frames for valid yawn
        self.MAX_YAWN_DURATION = 60    # Maximum frames for valid yawn
        self.YAWN_FATIGUE_WEIGHT = 0.8 # High weight for yawning in fatigue calculation
        
        # Mask detection parameters
        self.MASK_DETECTION_THRESHOLD = 0.4  # Threshold for detecting masks
        self.MASK_COMPENSATION_FACTOR = 1.2  # Increase eye-based detection when mask present
        
        # Analysis parameters
        self.frame_buffer = []
        self.ear_history = []
        self.blink_history = []
        self.yawn_history = []
        self.analysis_window_frames = 60  # Increased for better analysis
        self.blink_counter = 0
        self.closed_eye_frames = 0
        self.consecutive_closed_frames = 0
        self.max_consecutive_closed = 0
        self.microsleep_events = 0
        
        # Yawning tracking
        self.yawn_counter = 0
        self.consecutive_yawn_frames = 0
        self.max_consecutive_yawn = 0
        self.yawn_frames = 0
        
        # Mask detection
        self.mask_detected_frames = 0
        self.total_analyzed_frames = 0
        self.is_mask_present = False
        
        # Infrared-optimized calibration data
        self.calibration_ears = []
        self.calibration_mars = []
        self.baseline_ear = None
        self.baseline_mar = None  # Mouth aspect ratio baseline
        self.is_calibrated = False
        self.last_ear = None
        self.infrared_mode = True  # Flag for infrared-specific processing

        # Infrared video quality assessment
        self.frame_quality_scores = []
        self.low_quality_frame_count = 0
        self.infrared_enhancement_factor = 1.0

        # Multi-modal detection for infrared
        self.head_pose_history = []
        self.micro_movement_scores = []
        self.temporal_patterns = []
        self.previous_landmarks = None
        
    def _initialize_predictor(self):
        """Initialize facial landmark predictor"""
        try:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                print("Facial landmark predictor found and loaded.")
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                print("Warning: Facial landmark predictor not found. Attempting to download...")
                url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                import bz2
                with open(predictor_path, "wb") as f:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(decompressor.decompress(chunk))
                
                self.predictor = dlib.shape_predictor(predictor_path)
                print("Predictor downloaded and loaded successfully.")

        except Exception as e:
            print(f"Error initializing or downloading predictor: {e}")
            self.predictor = None

    def _initialize_gpu_cv_objects(self):
        """Initialize GPU-accelerated OpenCV objects if CUDA is available."""
        self.gpu_clahe = None
        self.gpu_bilateral_filter = None
        self.gpu_morph_kernel = None

        if self.cuda_manager.opencv_cuda_available:
            try:
                # Initialize GPU-accelerated CLAHE
                self.gpu_clahe = cv2.cuda.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

                # Create morphological kernel on GPU
                kernel_cpu = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                self.gpu_morph_kernel = cv2.cuda_GpuMat()
                self.gpu_morph_kernel.upload(kernel_cpu)

                print("✅ GPU-accelerated OpenCV objects initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize GPU OpenCV objects: {e}")
                self.cuda_manager.opencv_cuda_available = False
        else:
            print("ℹ️ OpenCV CUDA not available, using CPU fallback")

    def download_video(self, url: str) -> Optional[str]:
        """Download video from URL to temporary file, or use local path."""
        try:
            if os.path.exists(url):
                return url
            
            print(f"Downloading video from URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            temp_file_path = temp_file.name
            temp_file.close()
            return temp_file_path
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading video from {url}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during video download: {e}")
            return None

    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """Calculate Enhanced Eye Aspect Ratio (EAR) with noise reduction and GPU acceleration"""
        try:
            # Try GPU-accelerated calculation first
            if self.cuda_manager.is_gpu_available():
                ear = self._calculate_ear_gpu(eye_landmarks)
                if ear is not None:
                    return ear

            # Fallback to CPU calculation
            return self._calculate_ear_cpu(eye_landmarks)

        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.3

    def _calculate_ear_gpu(self, eye_landmarks: np.ndarray) -> Optional[float]:
        """GPU-accelerated EAR calculation using CuPy"""
        try:
            # Transfer landmarks to GPU
            gpu_landmarks = self.cuda_manager.to_gpu(eye_landmarks)

            # Calculate vertical distances using GPU
            vertical_1 = float(cp.linalg.norm(gpu_landmarks[1] - gpu_landmarks[5]))
            vertical_2 = float(cp.linalg.norm(gpu_landmarks[2] - gpu_landmarks[4]))

            # Calculate horizontal distance using GPU
            horizontal = float(cp.linalg.norm(gpu_landmarks[0] - gpu_landmarks[3]))

            if horizontal == 0:
                return 0.3  # Avoid division by zero

            # Enhanced EAR calculation with weighted verticals
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

            # Apply smoothing to reduce noise
            if hasattr(self, 'last_ear') and self.last_ear is not None:
                # Simple exponential smoothing
                alpha = 0.3  # Smoothing factor
                ear = alpha * ear + (1 - alpha) * self.last_ear

            self.last_ear = ear
            return max(0.1, min(0.6, ear))  # Clamp to reasonable range

        except Exception as e:
            print(f"GPU EAR calculation failed: {e}")
            return None

    def _calculate_ear_cpu(self, eye_landmarks: np.ndarray) -> float:
        """CPU fallback for EAR calculation"""
        # Calculate vertical distances
        vertical_1 = euclidean(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = euclidean(eye_landmarks[2], eye_landmarks[4])

        # Calculate horizontal distance
        horizontal = euclidean(eye_landmarks[0], eye_landmarks[3])

        if horizontal == 0:
            return 0.3  # Avoid division by zero

        # Enhanced EAR calculation with weighted verticals
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        # Apply smoothing to reduce noise
        if hasattr(self, 'last_ear') and self.last_ear is not None:
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing factor
            ear = alpha * ear + (1 - alpha) * self.last_ear

        self.last_ear = ear
        return max(0.1, min(0.6, ear))  # Clamp to reasonable range
    
    def extract_eye_landmarks(self, face_landmarks) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract left and right eye landmarks from facial landmarks"""
        try:
            left_eye = np.array([(face_landmarks.part(i).x, face_landmarks.part(i).y) for i in range(36, 42)], dtype="double")
            right_eye = np.array([(face_landmarks.part(i).x, face_landmarks.part(i).y) for i in range(42, 48)], dtype="double")
            return left_eye, right_eye
        except Exception as e:
            print(f"Error extracting eye landmarks: {e}")
            return None, None
    
    def extract_mouth_landmarks(self, face_landmarks) -> Optional[np.ndarray]:
        """Extract mouth landmarks from facial landmarks"""
        try:
            # Extract inner mouth landmarks (points 60-67)
            mouth = np.array([(face_landmarks.part(i).x, face_landmarks.part(i).y) for i in range(60, 68)], dtype="double")
            return mouth
        except Exception as e:
            print(f"Error extracting mouth landmarks: {e}")
            return None
    
    def calculate_mar(self, mouth_landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for yawn detection with GPU acceleration"""
        try:
            # Try GPU-accelerated calculation first
            if self.cuda_manager.is_gpu_available():
                mar = self._calculate_mar_gpu(mouth_landmarks)
                if mar is not None:
                    return mar

            # Fallback to CPU calculation
            return self._calculate_mar_cpu(mouth_landmarks)

        except Exception as e:
            print(f"Error calculating MAR: {e}")
            return 0.3

    def _calculate_mar_gpu(self, mouth_landmarks: np.ndarray) -> Optional[float]:
        """GPU-accelerated MAR calculation using CuPy"""
        try:
            # Transfer landmarks to GPU
            gpu_landmarks = self.cuda_manager.to_gpu(mouth_landmarks)

            # Calculate vertical distances (mouth height) using GPU
            vertical_1 = float(cp.linalg.norm(gpu_landmarks[2] - gpu_landmarks[6]))
            vertical_2 = float(cp.linalg.norm(gpu_landmarks[3] - gpu_landmarks[7]))

            # Calculate horizontal distance (mouth width) using GPU
            horizontal = float(cp.linalg.norm(gpu_landmarks[0] - gpu_landmarks[4]))

            if horizontal == 0:
                return 0.3  # Avoid division by zero

            # MAR calculation - higher values indicate open mouth (yawning)
            mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

            # Apply smoothing to reduce noise
            if hasattr(self, 'last_mar') and self.last_mar is not None:
                alpha = 0.3  # Smoothing factor
                mar = alpha * mar + (1 - alpha) * self.last_mar

            self.last_mar = mar
            return max(0.1, min(2.0, mar))  # Clamp to reasonable range

        except Exception as e:
            print(f"GPU MAR calculation failed: {e}")
            return None

    def _calculate_mar_cpu(self, mouth_landmarks: np.ndarray) -> float:
        """CPU fallback for MAR calculation"""
        # Calculate vertical distances (mouth height)
        vertical_1 = euclidean(mouth_landmarks[2], mouth_landmarks[6])  # Top to bottom
        vertical_2 = euclidean(mouth_landmarks[3], mouth_landmarks[7])  # Top to bottom (inner)

        # Calculate horizontal distance (mouth width)
        horizontal = euclidean(mouth_landmarks[0], mouth_landmarks[4])  # Left to right

        if horizontal == 0:
            return 0.3  # Avoid division by zero

        # MAR calculation - higher values indicate open mouth (yawning)
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

        # Apply smoothing to reduce noise
        if hasattr(self, 'last_mar') and self.last_mar is not None:
            alpha = 0.3  # Smoothing factor
            mar = alpha * mar + (1 - alpha) * self.last_mar

        self.last_mar = mar
        return max(0.1, min(2.0, mar))  # Clamp to reasonable range

    def detect_micro_movements(self, current_landmarks) -> float:
        """Detect micro-movements that indicate drowsiness in infrared video"""
        try:
            if self.previous_landmarks is None:
                self.previous_landmarks = current_landmarks
                return 0.0

            # Calculate movement of key facial points
            movement_score = 0.0
            key_points = [30, 33, 36, 39, 42, 45]  # Nose tip, eye corners

            for point_idx in key_points:
                prev_point = (self.previous_landmarks.part(point_idx).x, self.previous_landmarks.part(point_idx).y)
                curr_point = (current_landmarks.part(point_idx).x, current_landmarks.part(point_idx).y)
                movement = euclidean(prev_point, curr_point)
                movement_score += movement

            # Normalize by number of points
            movement_score /= len(key_points)

            # Store for temporal analysis
            self.micro_movement_scores.append(movement_score)
            if len(self.micro_movement_scores) > 30:  # Keep last 30 frames
                self.micro_movement_scores.pop(0)

            self.previous_landmarks = current_landmarks

            # Low movement indicates potential drowsiness
            return max(0.0, 1.0 - (movement_score / 10.0))  # Invert and normalize

        except Exception as e:
            print(f"Error detecting micro-movements: {e}")
            return 0.0

    def analyze_head_pose(self, face_landmarks) -> dict:
        """Analyze head pose for drowsiness indicators in infrared video"""
        try:
            # Get key facial points for pose estimation
            nose_tip = (face_landmarks.part(30).x, face_landmarks.part(30).y)
            chin = (face_landmarks.part(8).x, face_landmarks.part(8).y)
            left_eye = (face_landmarks.part(36).x, face_landmarks.part(36).y)
            right_eye = (face_landmarks.part(45).x, face_landmarks.part(45).y)

            # Calculate head tilt (roll)
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_center_y = (left_eye[1] + right_eye[1]) / 2

            # Head nod detection (pitch) - distance from eyes to chin
            face_height = euclidean((eye_center_x, eye_center_y), chin)

            # Head turn detection (yaw) - nose position relative to eye center
            nose_offset = abs(nose_tip[0] - eye_center_x)
            face_width = euclidean(left_eye, right_eye)

            pose_data = {
                'head_down': face_height < 80,  # Threshold for head down
                'head_turned': (nose_offset / face_width) > 0.3 if face_width > 0 else False,
                'face_height': face_height,
                'nose_offset_ratio': nose_offset / face_width if face_width > 0 else 0
            }

            self.head_pose_history.append(pose_data)
            if len(self.head_pose_history) > 30:
                self.head_pose_history.pop(0)

            return pose_data

        except Exception as e:
            print(f"Error analyzing head pose: {e}")
            return {'head_down': False, 'head_turned': False, 'face_height': 0, 'nose_offset_ratio': 0}

    def detect_mask(self, face_landmarks) -> bool:
        """Detect if person is wearing a mask based on landmark visibility"""
        try:
            # Check if lower face landmarks are obscured or distorted
            # Points around nose and mouth area
            nose_tip = face_landmarks.part(33)  # Nose tip
            mouth_center = face_landmarks.part(62)  # Mouth center
            chin = face_landmarks.part(8)  # Chin
            
            # Calculate distances to detect mask presence
            nose_to_mouth = euclidean((nose_tip.x, nose_tip.y), (mouth_center.x, mouth_center.y))
            mouth_to_chin = euclidean((mouth_center.x, mouth_center.y), (chin.x, chin.y))
            
            # If these distances are unusually small, likely wearing mask
            face_height = euclidean((face_landmarks.part(19).x, face_landmarks.part(19).y), 
                                  (face_landmarks.part(8).x, face_landmarks.part(8).y))
            
            # Normalize distances by face height
            nose_mouth_ratio = nose_to_mouth / face_height if face_height > 0 else 0
            mouth_chin_ratio = mouth_to_chin / face_height if face_height > 0 else 0
            
            # Mask detection logic - if lower face ratios are small, likely masked
            is_masked = (nose_mouth_ratio < 0.15 or mouth_chin_ratio < 0.2)
            
            return is_masked
            
        except Exception as e:
            print(f"Error detecting mask: {e}")
            return False
    
    def calibrate_thresholds(self, avg_ear: float, avg_mar: float, frame_number: int):
        """Infrared-optimized calibration for EAR and MAR thresholds"""
        if frame_number < self.EAR_CALIBRATION_FRAMES:
            if avg_ear > 0:
                self.calibration_ears.append(avg_ear)
            if avg_mar > 0:
                if not hasattr(self, 'calibration_mars'):
                    self.calibration_mars = []
                self.calibration_mars.append(avg_mar)
        elif frame_number == self.EAR_CALIBRATION_FRAMES:
            # Calculate baseline EAR with infrared-specific adjustments
            if self.calibration_ears:
                self.baseline_ear = np.mean(self.calibration_ears)
                # More aggressive threshold for infrared - use 85% of baseline instead of 80%
                self.EAR_THRESHOLD_ADAPTIVE = max(0.20, self.baseline_ear * 0.85)

            # Calculate baseline MAR with infrared sensitivity
            if hasattr(self, 'calibration_mars') and self.calibration_mars:
                self.baseline_mar = np.mean(self.calibration_mars)
                # Lower yawn threshold for infrared - use 1.6x instead of 1.8x
                self.YAWN_THRESHOLD = max(0.5, self.baseline_mar * 1.6)

            self.is_calibrated = True
            print(f"📊 Infrared Calibrated - EAR: baseline={self.baseline_ear:.3f}, threshold={self.EAR_THRESHOLD_ADAPTIVE:.3f}")
            if self.baseline_mar:
                print(f"📊 Infrared Calibrated - MAR: baseline={self.baseline_mar:.3f}, yawn_threshold={self.YAWN_THRESHOLD:.3f}")
    
    def detect_blink_patterns(self, is_closed: bool, frame_number: int):
        """Enhanced blink pattern detection"""
        if is_closed:
            self.consecutive_closed_frames += 1
            self.closed_eye_frames += 1
        else:
            if self.consecutive_closed_frames > 0:
                # Analyze the closed period
                if self.MIN_BLINK_DURATION <= self.consecutive_closed_frames <= self.MAX_BLINK_DURATION:
                    # Valid blink detected
                    self.blink_counter += 1
                    self.blink_history.append({
                        'frame': frame_number,
                        'duration': self.consecutive_closed_frames,
                        'type': 'normal_blink'
                    })
                elif self.consecutive_closed_frames > self.MICROSLEEP_THRESHOLD:
                    # Microsleep detected
                    self.microsleep_events += 1
                    self.blink_history.append({
                        'frame': frame_number,
                        'duration': self.consecutive_closed_frames,
                        'type': 'microsleep'
                    })
                
                # Update max consecutive
                self.max_consecutive_closed = max(self.max_consecutive_closed, self.consecutive_closed_frames)
                self.consecutive_closed_frames = 0
    
    def detect_yawn_patterns(self, is_yawning: bool, frame_number: int, mar_value: float):
        """Enhanced yawn pattern detection"""
        if is_yawning:
            self.consecutive_yawn_frames += 1
            self.yawn_frames += 1
        else:
            if self.consecutive_yawn_frames > 0:
                # Analyze the yawning period
                if self.MIN_YAWN_DURATION <= self.consecutive_yawn_frames <= self.MAX_YAWN_DURATION:
                    # Valid yawn detected
                    self.yawn_counter += 1
                    self.yawn_history.append({
                        'frame': frame_number,
                        'duration': self.consecutive_yawn_frames,
                        'max_mar': mar_value,
                        'type': 'yawn'
                    })
                    print(f"😴 Yawn detected at frame {frame_number} (duration: {self.consecutive_yawn_frames} frames)")
                
                # Update max consecutive yawn
                self.max_consecutive_yawn = max(self.max_consecutive_yawn, self.consecutive_yawn_frames)
                self.consecutive_yawn_frames = 0
    
    def calculate_enhanced_perclos(self, timestamp: float) -> float:
        """Calculate PERCLOS with temporal weighting"""
        if not self.frame_buffer:
            return 0.0
        
        # Recent frames have higher weight
        total_weight = 0
        weighted_closed = 0
        
        for i, is_closed in enumerate(self.frame_buffer):
            # Linear weighting - recent frames weighted more
            weight = (i + 1) / len(self.frame_buffer)
            total_weight += weight
            if is_closed:
                weighted_closed += weight
        
        return weighted_closed / total_weight if total_weight > 0 else 0.0
    
    def calculate_fatigue_score(self, perclos_score: float, avg_ear: float, blink_freq: float, timestamp: float) -> tuple:
        """Enhanced multi-factor fatigue scoring with infrared-specific multi-modal detection"""
        fatigue_factors = {}

        # Calculate yawning metrics
        yawn_rate = self.yawn_counter / max(1, timestamp)  # yawns per second
        yawn_percentage = (self.yawn_frames / max(1, self.total_analyzed_frames)) * 100

        # Calculate multi-modal scores for infrared
        micro_movement_factor = 0.0
        head_pose_factor = 0.0

        if self.micro_movement_scores:
            avg_micro_movement = np.mean(self.micro_movement_scores[-10:])  # Last 10 frames
            micro_movement_factor = min(1.0, avg_micro_movement)

        if self.head_pose_history:
            recent_poses = self.head_pose_history[-10:]  # Last 10 frames
            head_down_ratio = sum(1 for pose in recent_poses if pose['head_down']) / len(recent_poses)
            head_pose_factor = head_down_ratio
        
        # Factor 1: PERCLOS analysis (30% weight - reduced to accommodate yawning)
        if perclos_score >= self.PERCLOS_THRESHOLD_SEVERE:
            perclos_factor = 1.0
            fatigue_factors['perclos_level'] = 'severe'
        elif perclos_score >= self.PERCLOS_THRESHOLD_MODERATE:
            perclos_factor = 0.75
            fatigue_factors['perclos_level'] = 'moderate'
        elif perclos_score >= self.PERCLOS_THRESHOLD_MILD:
            perclos_factor = 0.5
            fatigue_factors['perclos_level'] = 'mild'
        else:
            perclos_factor = 0.0
            fatigue_factors['perclos_level'] = 'normal'
        
        # Factor 2: EAR analysis (25% weight)
        if self.is_calibrated and self.baseline_ear:
            ear_ratio = avg_ear / self.baseline_ear if avg_ear > 0 else 1.0
            if ear_ratio < 0.7:  # Significantly below baseline
                ear_factor = 1.0
                fatigue_factors['ear_level'] = 'severe'
            elif ear_ratio < 0.8:
                ear_factor = 0.7
                fatigue_factors['ear_level'] = 'moderate'
            elif ear_ratio < 0.9:
                ear_factor = 0.4
                fatigue_factors['ear_level'] = 'mild'
            else:
                ear_factor = 0.0
                fatigue_factors['ear_level'] = 'normal'
        else:
            # Fallback to absolute threshold
            if avg_ear < self.EAR_THRESHOLD_ADAPTIVE:
                ear_factor = 0.8
                fatigue_factors['ear_level'] = 'below_threshold'
            else:
                ear_factor = 0.0
                fatigue_factors['ear_level'] = 'normal'
        
        # Factor 3: Yawning analysis (25% weight - HIGH PRIORITY)
        if self.yawn_counter > 0:
            # Any yawning indicates fatigue
            if self.yawn_counter >= 3:  # Multiple yawns = severe fatigue
                yawn_factor = 1.0
                fatigue_factors['yawn_level'] = 'severe'
            elif self.yawn_counter >= 2:  # 2 yawns = moderate fatigue
                yawn_factor = 0.85
                fatigue_factors['yawn_level'] = 'moderate'
            else:  # 1 yawn = mild fatigue
                yawn_factor = 0.7
                fatigue_factors['yawn_level'] = 'mild'
        else:
            yawn_factor = 0.0
            fatigue_factors['yawn_level'] = 'normal'
        
        # Factor 4: Blink frequency analysis (15% weight)
        expected_blink_rate = 0.3  # blinks per second (18 per minute)
        if blink_freq < expected_blink_rate * 0.3:  # Very low blink rate
            blink_factor = 0.8
            fatigue_factors['blink_level'] = 'very_low'
        elif blink_freq < expected_blink_rate * 0.6:  # Low blink rate
            blink_factor = 0.5
            fatigue_factors['blink_level'] = 'low'
        else:
            blink_factor = 0.0
            fatigue_factors['blink_level'] = 'normal'
        
        # Factor 5: Microsleep events (5% weight)
        microsleep_factor = min(1.0, self.microsleep_events * 0.3)
        fatigue_factors['microsleep_events'] = self.microsleep_events
        
        # Add yawning details to factors
        fatigue_factors['yawn_count'] = self.yawn_counter
        fatigue_factors['yawn_rate_per_minute'] = yawn_rate * 60
        fatigue_factors['yawn_percentage'] = yawn_percentage
        fatigue_factors['mask_detected'] = self.is_mask_present
        
        # Infrared-optimized weighted combination with multi-modal detection
        base_fatigue = (
            perclos_factor * 0.25 +        # Reduced weight for PERCLOS
            ear_factor * 0.20 +             # Reduced weight for EAR
            yawn_factor * 0.20 +            # Maintained weight for yawning
            blink_factor * 0.15 +           # Maintained weight for blinking
            microsleep_factor * 0.05 +      # Maintained weight for microsleep
            micro_movement_factor * 0.10 +  # New: micro-movement detection
            head_pose_factor * 0.05         # New: head pose analysis
        ) * 100
        
        # Apply mask compensation if detected
        if self.is_mask_present:
            # Increase weight of eye-based metrics when mask is present
            mask_compensated_fatigue = base_fatigue * self.MASK_COMPENSATION_FACTOR
            fatigue_percentage = min(100, mask_compensated_fatigue)
            fatigue_factors['mask_compensation_applied'] = True
        else:
            fatigue_percentage = base_fatigue
            fatigue_factors['mask_compensation_applied'] = False
        
        # CRITICAL: Any yawning should result in at least mild fatigue (30%)
        if self.yawn_counter > 0 and fatigue_percentage < 30:
            fatigue_percentage = max(30, fatigue_percentage)
            fatigue_factors['yawn_override'] = True
        else:
            fatigue_factors['yawn_override'] = False
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(perclos_score, avg_ear, timestamp)
        
        return fatigue_percentage, confidence, fatigue_factors
    
    def _calculate_confidence(self, perclos_score: float, avg_ear: float, timestamp: float) -> float:
        """Calculate confidence score based on data quality"""
        confidence_factors = []
        
        # Factor 1: Calibration quality
        if self.is_calibrated:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Factor 2: Data consistency
        if len(self.ear_history) > 10:
            ear_std = np.std(self.ear_history[-30:])  # Last 30 values
            if ear_std < 0.05:  # Consistent readings
                confidence_factors.append(0.9)
            elif ear_std < 0.1:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.6)
        
        # Factor 3: Analysis duration
        if timestamp > 10:  # At least 10 seconds
            confidence_factors.append(0.9)
        elif timestamp > 5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Factor 4: Face detection quality
        face_detection_rate = len([ear for ear in self.ear_history if ear > 0]) / max(1, len(self.ear_history))
        confidence_factors.append(min(0.9, face_detection_rate))
        
        return min(0.95, np.mean(confidence_factors))

    def assess_infrared_frame_quality(self, gray_frame: np.ndarray) -> float:
        """Assess the quality of an infrared frame for drowsiness detection with GPU acceleration"""
        try:
            # Try GPU-accelerated quality assessment first
            if self.cuda_manager.is_gpu_available():
                quality_score = self._assess_frame_quality_gpu(gray_frame)
                if quality_score is not None:
                    return quality_score

            # Fallback to CPU processing
            return self._assess_frame_quality_cpu(gray_frame)

        except Exception as e:
            print(f"Error assessing frame quality: {e}")
            return 0.5  # Default medium quality

    def _assess_frame_quality_gpu(self, gray_frame: np.ndarray) -> Optional[float]:
        """GPU-accelerated frame quality assessment using CuPy"""
        try:
            # Transfer frame to GPU
            gpu_frame = self.cuda_manager.to_gpu(gray_frame)

            # Calculate contrast using standard deviation (GPU)
            contrast = float(cp.std(gpu_frame))

            # Calculate sharpness using Laplacian variance (GPU)
            # Create Laplacian kernel
            laplacian_kernel = cp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=cp.float32)

            # Apply convolution for Laplacian
            gpu_frame_float = gpu_frame.astype(cp.float32)
            laplacian = cp.abs(cp.convolve2d(gpu_frame_float, laplacian_kernel, mode='same'))
            sharpness = float(cp.var(laplacian))

            # Calculate brightness distribution (GPU)
            hist, _ = cp.histogram(gpu_frame, bins=256, range=(0, 256))
            hist_mean = float(cp.mean(hist))
            hist_std = float(cp.std(hist))
            brightness_uniformity = 1.0 - (hist_std / max(hist_mean, 1e-6))

            # Combine metrics for overall quality score (0-1)
            quality_score = min(1.0, (contrast / 50.0) * 0.4 + (sharpness / 500.0) * 0.4 + brightness_uniformity * 0.2)

            return quality_score

        except Exception as e:
            print(f"GPU quality assessment failed: {e}")
            return None

    def _assess_frame_quality_cpu(self, gray_frame: np.ndarray) -> float:
        """CPU fallback for frame quality assessment"""
        # Calculate contrast using standard deviation
        contrast = np.std(gray_frame)

        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        sharpness = laplacian.var()

        # Calculate brightness distribution
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        brightness_uniformity = 1.0 - (np.std(hist) / np.mean(hist))

        # Combine metrics for overall quality score (0-1)
        quality_score = min(1.0, (contrast / 50.0) * 0.4 + (sharpness / 500.0) * 0.4 + brightness_uniformity * 0.2)

        return quality_score

    def enhance_infrared_frame(self, gray_frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing specifically for infrared video frames with GPU acceleration"""
        start_time = time.time()

        try:
            # Assess frame quality first
            quality_score = self.assess_infrared_frame_quality(gray_frame)
            self.frame_quality_scores.append(quality_score)

            if quality_score < 0.3:
                self.low_quality_frame_count += 1
                # Apply more aggressive enhancement for low quality frames
                enhancement_factor = 1.5
            else:
                enhancement_factor = 1.0

            # Try GPU-accelerated processing first
            if self.cuda_manager.opencv_cuda_available:
                enhanced = self._enhance_infrared_frame_gpu(gray_frame, enhancement_factor)
                if enhanced is not None:
                    processing_time = time.time() - start_time
                    self.cuda_manager.gpu_processing_times.append(processing_time)
                    return enhanced

            # Fallback to CPU processing
            enhanced = self._enhance_infrared_frame_cpu(gray_frame, enhancement_factor)
            processing_time = time.time() - start_time
            self.cuda_manager.cpu_processing_times.append(processing_time)
            return enhanced

        except Exception as e:
            print(f"Error in infrared enhancement: {e}")
            # Fallback to basic histogram equalization
            return cv2.equalizeHist(gray_frame)

    def _enhance_infrared_frame_gpu(self, gray_frame: np.ndarray, enhancement_factor: float) -> Optional[np.ndarray]:
        """GPU-accelerated infrared frame enhancement using OpenCV CUDA"""
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(gray_frame)

            # Step 1: Advanced histogram equalization with CLAHE (GPU)
            if self.gpu_clahe is not None:
                clip_limit = 3.0 * enhancement_factor
                # Update CLAHE parameters if needed
                self.gpu_clahe.setClipLimit(clip_limit)
                gpu_enhanced = cv2.cuda_GpuMat()
                self.gpu_clahe.apply(gpu_frame, gpu_enhanced)
            else:
                gpu_enhanced = gpu_frame

            # Step 2: Gamma correction for infrared contrast enhancement
            gamma = 1.2 * enhancement_factor
            gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gpu_gamma_table = cv2.cuda_GpuMat()
            gpu_gamma_table.upload(gamma_table)
            cv2.cuda.LUT(gpu_enhanced, gpu_gamma_table, gpu_enhanced)

            # Step 3: Bilateral filtering (GPU) - Note: Limited GPU support, may fallback
            try:
                gpu_filtered = cv2.cuda_GpuMat()
                cv2.cuda.bilateralFilter(gpu_enhanced, gpu_filtered, 9, 75, 75)
                gpu_enhanced = gpu_filtered
            except:
                # Bilateral filter not available on GPU, continue with current result
                pass

            # Step 4: Adaptive histogram equalization for local contrast (GPU)
            cv2.cuda.equalizeHist(gpu_enhanced, gpu_enhanced)

            # Step 5: Morphological operations (GPU)
            if self.gpu_morph_kernel is not None:
                gpu_morphed = cv2.cuda_GpuMat()
                cv2.cuda.morphologyEx(gpu_enhanced, gpu_morphed, cv2.MORPH_CLOSE, self.gpu_morph_kernel)
                gpu_enhanced = gpu_morphed

            # Download result from GPU
            enhanced = gpu_enhanced.download()
            return enhanced

        except Exception as e:
            print(f"GPU enhancement failed, falling back to CPU: {e}")
            return None

    def _enhance_infrared_frame_cpu(self, gray_frame: np.ndarray, enhancement_factor: float) -> np.ndarray:
        """CPU fallback for infrared frame enhancement"""
        # Step 1: Advanced histogram equalization with CLAHE
        clip_limit = 3.0 * enhancement_factor
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_frame)

        # Step 2: Gamma correction for infrared contrast enhancement
        gamma = 1.2 * enhancement_factor  # Adjust gamma based on quality
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, gamma_table)

        # Step 3: Bilateral filtering for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Step 4: Adaptive histogram equalization for local contrast
        enhanced = cv2.equalizeHist(enhanced)

        # Step 5: Morphological operations to enhance facial features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        return enhanced

    def _convert_to_grayscale_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated color conversion to grayscale"""
        try:
            if self.cuda_manager.opencv_cuda_available:
                # Upload frame to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)

                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda_GpuMat()
                cv2.cuda.cvtColor(gpu_frame, gpu_gray, cv2.COLOR_BGR2GRAY)

                # Download result
                gray = gpu_gray.download()
                return gray
            else:
                # Fallback to CPU
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"GPU color conversion failed, using CPU: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def analyze_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FatigueMetrics:
        """Enhanced frame analysis with GPU-accelerated preprocessing and adaptive thresholds"""
        # GPU-accelerated color conversion if available
        gray = self._convert_to_grayscale_gpu(frame)
        left_ear, right_ear = -1.0, -1.0
        mar_value = -1.0

        # Apply enhanced infrared preprocessing (already GPU-accelerated)
        gray = self.enhance_infrared_frame(gray)
        
        faces = self.detector(gray)
        self.total_analyzed_frames += 1
        
        if faces:
            # Use the largest face if multiple detected
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            if self.predictor:
                landmarks = self.predictor(gray, face)
                
                # Extract eye landmarks
                left_eye, right_eye = self.extract_eye_landmarks(landmarks)
                if left_eye is not None and right_eye is not None:
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                
                # Extract mouth landmarks and detect yawning
                mouth = self.extract_mouth_landmarks(landmarks)
                if mouth is not None:
                    mar_value = self.calculate_mar(mouth)

                # Multi-modal detection for infrared
                micro_movement_score = self.detect_micro_movements(landmarks)
                head_pose_data = self.analyze_head_pose(landmarks)

                # Detect mask presence
                is_masked = self.detect_mask(landmarks)
                if is_masked:
                    self.mask_detected_frames += 1
        
        # Update mask detection status
        if self.total_analyzed_frames > 30:  # After some frames
            mask_ratio = self.mask_detected_frames / self.total_analyzed_frames
            self.is_mask_present = mask_ratio > 0.3  # If 30%+ frames show mask
        
        # Process EAR data
        avg_ear = -1.0
        if left_ear != -1.0:
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(avg_ear)
            
            # Keep only recent EAR history
            if len(self.ear_history) > 300:  # ~10 seconds at 30fps
                self.ear_history.pop(0)
        
        # Calibrate thresholds if not done yet
            if not self.is_calibrated:
                self.calibrate_thresholds(avg_ear if avg_ear > 0 else 0, mar_value if mar_value > 0 else 0, frame_number)        
        # Determine if eyes are closed using adaptive threshold
        threshold = self.EAR_THRESHOLD_ADAPTIVE if self.is_calibrated else self.EAR_THRESHOLD_BASE
        is_closed = avg_ear < threshold if avg_ear != -1.0 else False
        
        # Determine if yawning
        yawn_threshold = self.YAWN_THRESHOLD if hasattr(self, 'YAWN_THRESHOLD') else 0.6
        is_yawning = mar_value > yawn_threshold if mar_value != -1.0 else False
        
        # Enhanced pattern detection
        self.detect_blink_patterns(is_closed, frame_number)
        self.detect_yawn_patterns(is_yawning, frame_number, mar_value)
        
        # Update frame buffer
        self.frame_buffer.append(is_closed)
        if len(self.frame_buffer) > self.analysis_window_frames:
            self.frame_buffer.pop(0)
        
        # Calculate enhanced PERCLOS
        perclos_score = self.calculate_enhanced_perclos(timestamp)
        
        # Calculate blink frequency
        blink_freq = self.blink_counter / max(1, timestamp)
        
        return FatigueMetrics(left_ear, right_ear, perclos_score, blink_freq, frame_number, timestamp)
    
    def analyze_video(self, video_path: str, driver_name: str = "Unknown") -> Optional[FatigueResult]:
        """Analyze a video file for fatigue using facial analysis."""
        print(f"\n🎬 Analyzing video for: {driver_name}")
        
        local_video_path = self.download_video(video_path)
        if not local_video_path:
            return None

        try:
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                print(f"❌ Error: Cannot open video file {local_video_path}")
                return None
            
            # Reset analysis state for each video
            self.frame_buffer = []
            self.ear_history = []
            self.blink_history = []
            self.yawn_history = []
            self.closed_eye_frames = 0
            self.blink_counter = 0
            self.consecutive_closed_frames = 0
            self.max_consecutive_closed = 0
            self.microsleep_events = 0
            
            # Reset yawning tracking
            self.yawn_counter = 0
            self.consecutive_yawn_frames = 0
            self.max_consecutive_yawn = 0
            self.yawn_frames = 0
            
            # Reset mask detection
            self.mask_detected_frames = 0
            self.total_analyzed_frames = 0
            self.is_mask_present = False
            
            # Reset calibration
            self.calibration_ears = []
            if hasattr(self, 'calibration_mars'):
                self.calibration_mars = []
            self.baseline_ear = None
            self.baseline_mar = None
            self.is_calibrated = False
            self.last_ear = None
            if hasattr(self, 'last_mar'):
                self.last_mar = None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0 or fps == 0:
                print(f"❌ Error: Video file {local_video_path} has invalid properties.")
                cap.release()
                return None

            self.analysis_window_frames = int(self.PERCLOS_WINDOW_SECONDS * fps)
            print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS | PERCLOS window: {self.analysis_window_frames} frames")

            frame_metrics, timestamp = [], 0
            for frame_number in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                timestamp = frame_number / fps
                metrics = self.analyze_frame(frame, frame_number, timestamp)
                frame_metrics.append(metrics)
                
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"⏳ Processing: {progress:.1f}%", end="\r")
            
            print("⏳ Processing: 100.0%")
            
            if not frame_metrics:
                print("\n❌ No frames could be analyzed")
                return None
                
            valid_ear_metrics = [m for m in frame_metrics if m.eye_aspect_ratio_left != -1.0]
            if not valid_ear_metrics:
                print("\n❌ Warning: No faces detected. Returning result with 0% fatigue.")
                fatigue_percentage = 0.0
                confidence = 0.3
                avg_perclos = 0
                avg_ear = 0
                fatigue_factors = {'reason': 'no_face_detected'}
                fatigue_level = 'unknown'
            else:
                # Calculate enhanced metrics
                avg_perclos = sum(m.perclos_score for m in frame_metrics) / len(frame_metrics)
                avg_ear = sum((m.eye_aspect_ratio_left + m.eye_aspect_ratio_right) / 2 for m in valid_ear_metrics) / len(valid_ear_metrics)
                avg_blink_freq = sum(m.blink_frequency for m in frame_metrics) / len(frame_metrics)
                
                # Use enhanced fatigue scoring
                fatigue_percentage, confidence, fatigue_factors = self.calculate_fatigue_score(
                    avg_perclos, avg_ear, avg_blink_freq, timestamp
                )
                
                # Determine fatigue level
                if fatigue_percentage >= self.FATIGUE_THRESHOLD_SEVERE * 100:
                    fatigue_level = 'severe'
                elif fatigue_percentage >= self.FATIGUE_THRESHOLD_MODERATE * 100:
                    fatigue_level = 'moderate'
                elif fatigue_percentage >= self.FATIGUE_THRESHOLD_MILD * 100:
                    fatigue_level = 'mild'
                else:
                    fatigue_level = 'normal'

            # Enhanced fatigue detection with confidence threshold for precision
            fatigue_detected = fatigue_percentage >= (self.FATIGUE_THRESHOLD_MILD * 100)
            confidence_met = confidence >= self.MIN_CONFIDENCE_THRESHOLD
            is_fatigue = fatigue_detected and confidence_met
            
            analysis_details = {
                "video_fps": fps, 
                "total_frames": len(frame_metrics),
                "frames_with_face": len(valid_ear_metrics), 
                "average_perclos": round(avg_perclos, 4),
                "average_ear": round(avg_ear, 4), 
                "analysis_duration_seconds": round(timestamp, 2),
                "detected_blinks": self.blink_counter,
                "microsleep_events": self.microsleep_events,
                "max_consecutive_closed_frames": self.max_consecutive_closed,
                
                # Yawning metrics
                "detected_yawns": self.yawn_counter,
                "max_consecutive_yawn_frames": self.max_consecutive_yawn,
                "yawn_frames": self.yawn_frames,
                "yawn_rate_per_minute": (self.yawn_counter / max(1, timestamp)) * 60,
                
                # Mask detection
                "mask_detected": self.is_mask_present,
                "mask_detection_confidence": round(self.mask_detected_frames / max(1, self.total_analyzed_frames), 3),
                
                # Calibration info
                "baseline_ear": round(self.baseline_ear, 4) if self.baseline_ear else None,
                "baseline_mar": round(self.baseline_mar, 4) if hasattr(self, 'baseline_mar') and self.baseline_mar else None,
                "ear_threshold_used": round(self.EAR_THRESHOLD_ADAPTIVE if self.is_calibrated else self.EAR_THRESHOLD_BASE, 4),
                "yawn_threshold_used": round(self.YAWN_THRESHOLD, 4) if hasattr(self, 'YAWN_THRESHOLD') else 0.6,
                "calibration_status": "calibrated" if self.is_calibrated else "default_thresholds",
                
                # Results
                "fatigue_level": fatigue_level,
                "fatigue_factors": fatigue_factors,
                "detection_method": "enhanced_landmark_v3.0_with_yawning_gpu",

                # GPU Performance Statistics
                "gpu_performance": self.cuda_manager.get_performance_stats()
            }
            
            result = FatigueResult(
                driver_name=driver_name,
                percentage_fatigue=round(fatigue_percentage, 2), 
                is_fatigue=is_fatigue, 
                confidence=round(confidence, 3), 
                analysis_details=analysis_details,
                analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            status = "🔴 FATIGUE DETECTED" if is_fatigue else "🟢 NO FATIGUE"
            print(f"\n✅ Analysis complete for {driver_name}: {status} ({fatigue_percentage:.2f}%)")
            return result
                
        except Exception as e:
            print(f"\n❌ An unexpected error during video analysis for {driver_name}: {e}")
            return None
        finally:
            if cap and cap.isOpened():
                cap.release()
            # Clean up downloaded file if it's a temp file
            if local_video_path and video_path != local_video_path and os.path.exists(local_video_path):
                os.unlink(local_video_path)
                print(f"🧹 Cleaned up temporary file: {local_video_path}")

    def get_gpu_status(self) -> Dict:
        """Get comprehensive GPU status and capabilities"""
        return {
            'gpu_available': self.cuda_manager.is_gpu_available(),
            'cupy_available': self.cuda_manager.cupy_available,
            'opencv_cuda_available': self.cuda_manager.opencv_cuda_available,
            'device_info': self.cuda_manager.get_device_info(),
            'performance_stats': self.cuda_manager.get_performance_stats()
        }

    def run_benchmark(self, test_frame: np.ndarray = None, iterations: int = 10) -> Dict:
        """Run GPU vs CPU performance benchmark"""
        if test_frame is None:
            # Create a test frame if none provided
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        return self.cuda_manager.benchmark_gpu_vs_cpu(test_frame, iterations)


def main(args):
    """Main function to run fatigue detection on a single video with GPU acceleration support."""
    # Initialize detector with GPU acceleration settings
    enable_gpu = not args.disable_gpu and os.getenv('DISABLE_GPU', 'false').lower() != 'true'
    gpu_device_id = args.gpu_device if hasattr(args, 'gpu_device') else int(os.getenv('CUDA_DEVICE_ID', '0'))

    print(f"🚀 Initializing Fatigue Detection System (GPU: {'enabled' if enable_gpu else 'disabled'})")
    if enable_gpu:
        print(f"🎮 Using CUDA device: {gpu_device_id}")

    detector = FatigueDetectionSystem(enable_gpu=enable_gpu, gpu_device_id=gpu_device_id)
    driver_name = os.path.splitext(os.path.basename(args.output_json))[0]

    # --- Correctly call the single video analyzer ---
    result_object = detector.analyze_video(args.front_video, driver_name)
    
    print("\n" + "=" * 60)
    print("🎯 FATIGUE DETECTION ANALYSIS COMPLETE")
    print("=" * 60)

    # --- Process the result object directly ---
    if result_object:
        status = "🔴 FATIGUE" if result_object.is_fatigue else "🟢 ALERT"
        print(f"Driver: {result_object.driver_name}")
        print(f"Result: {status}")
        print(f"Fatigue Score: {result_object.percentage_fatigue}%")
        print(f"Confidence: {result_object.confidence:.3f}")

        # Convert result object to dictionary for saving
        result_data = asdict(result_object)
    else:
        print(f"❌ Analysis failed for {driver_name}. No report will be generated.")
        result_data = {
            "driver_name": driver_name, 
            "error": "Analysis failed to produce a result.",
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    output_file = args.output_json
    try:
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"\n💾 Full analysis report saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving results to {output_file}: {e}")
    
    if args.output_video:
        print(f"\nℹ️ Note: Video output argument is present but not implemented in this version.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a merged video and JSON fatigue report with GPU acceleration support.")
    parser.add_argument('--front_video', type=str, required=True, help='Path to the front-facing driver video.')
    parser.add_argument('--rear_video', type=str, help='(Optional) Path to the rear-facing driver video.')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save the output merged video file (e.g., output.mp4).')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the final JSON report file (e.g., report.json).')

    # GPU acceleration options
    parser.add_argument('--disable-gpu', action='store_true', help='Disable GPU acceleration and use CPU only.')
    parser.add_argument('--gpu-device', type=int, default=0, help='CUDA device ID to use for GPU acceleration (default: 0).')
    
    parsed_args = parser.parse_args()
    main(parsed_args)