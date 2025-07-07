#!/usr/bin/env python3
"""
Fatigue Detection System
========================

A system for detecting driver fatigue by analyzing facial features
from a single front-facing video. It analyzes eye closure patterns (PERCLOS)
to determine fatigue levels.

Author: Fellou AI Agent
Date: December 6, 2025 (Updated: June 12, 2025)
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
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import euclidean

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
    Main class for fatigue detection system
    """
    
    def __init__(self):
        """Initialize the fatigue detection system"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self._initialize_predictor()
        
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
        """Calculate Enhanced Eye Aspect Ratio (EAR) with noise reduction"""
        try:
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
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.3
    
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
        """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
        try:
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

        except Exception as e:
            print(f"Error calculating MAR: {e}")
            return 0.3

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
        """Assess the quality of an infrared frame for drowsiness detection"""
        try:
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

        except Exception as e:
            print(f"Error assessing frame quality: {e}")
            return 0.5  # Default medium quality

    def enhance_infrared_frame(self, gray_frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing specifically for infrared video frames"""
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

        except Exception as e:
            print(f"Error in infrared enhancement: {e}")
            # Fallback to basic histogram equalization
            return cv2.equalizeHist(gray_frame)

    def analyze_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> FatigueMetrics:
        """Enhanced frame analysis with infrared-optimized preprocessing and adaptive thresholds"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left_ear, right_ear = -1.0, -1.0
        mar_value = -1.0

        # Apply enhanced infrared preprocessing
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
                "detection_method": "enhanced_landmark_v3.0_with_yawning"
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


def main(args):
    """Main function to run fatigue detection on a single video."""
    detector = FatigueDetectionSystem()
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
    parser = argparse.ArgumentParser(description="Generate a merged video and JSON fatigue report.")
    parser.add_argument('--front_video', type=str, required=True, help='Path to the front-facing driver video.')
    parser.add_argument('--rear_video', type=str, help='(Optional) Path to the rear-facing driver video.')
    parser.add_argument('--output_video', type=str, required=True, help='Path to save the output merged video file (e.g., output.mp4).')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the final JSON report file (e.g., report.json).')
    
    parsed_args = parser.parse_args()
    main(parsed_args)