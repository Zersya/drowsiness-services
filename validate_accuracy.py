#!/usr/bin/env python3
"""
Accuracy validation script for GPU-accelerated drowsiness detection.
This script ensures that GPU acceleration doesn't affect detection accuracy.
"""

import sys
import numpy as np
import cv2
from drowsiness_landmark import FatigueDetectionSystem

def generate_test_landmarks():
    """Generate realistic test landmarks for validation."""
    # Eye landmarks (6 points per eye)
    left_eye = np.array([
        [36, 108], [39, 106], [42, 106], [45, 108], [42, 110], [39, 110]
    ], dtype=np.float64)
    
    right_eye = np.array([
        [54, 108], [57, 106], [60, 106], [63, 108], [60, 110], [57, 110]
    ], dtype=np.float64)
    
    # Mouth landmarks (8 points)
    mouth = np.array([
        [48, 132], [51, 129], [54, 128], [57, 129], [60, 132], [57, 135], [54, 136], [51, 135]
    ], dtype=np.float64)
    
    return left_eye, right_eye, mouth

def test_ear_accuracy():
    """Test EAR calculation accuracy between GPU and CPU."""
    print("🔍 Testing EAR Calculation Accuracy...")
    
    left_eye, right_eye, _ = generate_test_landmarks()
    
    # Test with GPU enabled
    detector_gpu = FatigueDetectionSystem(enable_gpu=True)
    ear_gpu_left = detector_gpu.calculate_ear(left_eye)
    ear_gpu_right = detector_gpu.calculate_ear(right_eye)
    
    # Test with CPU only
    detector_cpu = FatigueDetectionSystem(enable_gpu=False)
    ear_cpu_left = detector_cpu.calculate_ear(left_eye)
    ear_cpu_right = detector_cpu.calculate_ear(right_eye)
    
    # Compare results
    tolerance = 1e-6  # Very small tolerance for floating point comparison
    
    left_diff = abs(ear_gpu_left - ear_cpu_left)
    right_diff = abs(ear_gpu_right - ear_cpu_right)
    
    print(f"   Left Eye EAR - GPU: {ear_gpu_left:.6f}, CPU: {ear_cpu_left:.6f}, Diff: {left_diff:.8f}")
    print(f"   Right Eye EAR - GPU: {ear_gpu_right:.6f}, CPU: {ear_cpu_right:.6f}, Diff: {right_diff:.8f}")
    
    if left_diff < tolerance and right_diff < tolerance:
        print("   ✅ EAR calculations match between GPU and CPU")
        return True
    else:
        print(f"   ❌ EAR calculations differ beyond tolerance ({tolerance})")
        return False

def test_mar_accuracy():
    """Test MAR calculation accuracy between GPU and CPU."""
    print("\n🔍 Testing MAR Calculation Accuracy...")
    
    _, _, mouth = generate_test_landmarks()
    
    # Test with GPU enabled
    detector_gpu = FatigueDetectionSystem(enable_gpu=True)
    mar_gpu = detector_gpu.calculate_mar(mouth)
    
    # Test with CPU only
    detector_cpu = FatigueDetectionSystem(enable_gpu=False)
    mar_cpu = detector_cpu.calculate_mar(mouth)
    
    # Compare results
    tolerance = 1e-6
    diff = abs(mar_gpu - mar_cpu)
    
    print(f"   MAR - GPU: {mar_gpu:.6f}, CPU: {mar_cpu:.6f}, Diff: {diff:.8f}")
    
    if diff < tolerance:
        print("   ✅ MAR calculations match between GPU and CPU")
        return True
    else:
        print(f"   ❌ MAR calculations differ beyond tolerance ({tolerance})")
        return False

def test_frame_processing_accuracy():
    """Test frame processing accuracy between GPU and CPU."""
    print("\n🔍 Testing Frame Processing Accuracy...")
    
    # Create a deterministic test frame
    np.random.seed(42)  # For reproducible results
    test_frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    
    # Test with GPU enabled
    detector_gpu = FatigueDetectionSystem(enable_gpu=True)
    
    # Test frame enhancement
    enhanced_gpu = detector_gpu.enhance_infrared_frame(test_frame)
    quality_gpu = detector_gpu.assess_infrared_frame_quality(test_frame)
    
    # Test with CPU only
    detector_cpu = FatigueDetectionSystem(enable_gpu=False)
    
    enhanced_cpu = detector_cpu.enhance_infrared_frame(test_frame)
    quality_cpu = detector_cpu.assess_infrared_frame_quality(test_frame)
    
    # Compare enhanced frames
    if enhanced_gpu is not None and enhanced_cpu is not None:
        frame_diff = np.mean(np.abs(enhanced_gpu.astype(float) - enhanced_cpu.astype(float)))
        print(f"   Enhanced Frame Mean Difference: {frame_diff:.6f}")
        
        # Allow small differences due to different processing paths
        frame_tolerance = 1.0  # Allow 1 pixel value difference on average
        if frame_diff < frame_tolerance:
            print("   ✅ Frame enhancement results are similar")
            frame_ok = True
        else:
            print(f"   ❌ Frame enhancement differs beyond tolerance ({frame_tolerance})")
            frame_ok = False
    else:
        print("   ❌ Frame enhancement failed on one or both modes")
        frame_ok = False
    
    # Compare quality scores
    quality_diff = abs(quality_gpu - quality_cpu)
    quality_tolerance = 0.01  # 1% tolerance
    
    print(f"   Quality Score - GPU: {quality_gpu:.6f}, CPU: {quality_cpu:.6f}, Diff: {quality_diff:.6f}")
    
    if quality_diff < quality_tolerance:
        print("   ✅ Quality assessment results are similar")
        quality_ok = True
    else:
        print(f"   ❌ Quality assessment differs beyond tolerance ({quality_tolerance})")
        quality_ok = False
    
    return frame_ok and quality_ok

def test_color_conversion_accuracy():
    """Test color conversion accuracy between GPU and CPU."""
    print("\n🔍 Testing Color Conversion Accuracy...")
    
    # Create a deterministic test color image
    np.random.seed(42)
    test_color_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Test with GPU enabled
    detector_gpu = FatigueDetectionSystem(enable_gpu=True)
    gray_gpu = detector_gpu._convert_to_grayscale_gpu(test_color_frame)
    
    # Test with standard OpenCV (CPU)
    gray_cpu = cv2.cvtColor(test_color_frame, cv2.COLOR_BGR2GRAY)
    
    if gray_gpu is not None:
        # Compare results
        diff = np.mean(np.abs(gray_gpu.astype(float) - gray_cpu.astype(float)))
        print(f"   Grayscale Conversion Mean Difference: {diff:.6f}")
        
        # Allow very small differences due to different processing paths
        tolerance = 0.1  # Very small tolerance
        if diff < tolerance:
            print("   ✅ Color conversion results match")
            return True
        else:
            print(f"   ❌ Color conversion differs beyond tolerance ({tolerance})")
            return False
    else:
        print("   ❌ GPU color conversion failed")
        return False

def test_consistency_across_runs():
    """Test that GPU results are consistent across multiple runs."""
    print("\n🔍 Testing Consistency Across Multiple Runs...")
    
    left_eye, _, _ = generate_test_landmarks()
    detector = FatigueDetectionSystem(enable_gpu=True)
    
    # Run EAR calculation multiple times
    ear_results = []
    for i in range(5):
        ear = detector.calculate_ear(left_eye)
        ear_results.append(ear)
    
    # Check consistency
    ear_std = np.std(ear_results)
    print(f"   EAR Results: {ear_results}")
    print(f"   Standard Deviation: {ear_std:.8f}")
    
    # Results should be identical (std dev should be 0 or very close)
    tolerance = 1e-10
    if ear_std < tolerance:
        print("   ✅ GPU results are consistent across runs")
        return True
    else:
        print(f"   ❌ GPU results vary across runs (std dev: {ear_std})")
        return False

def main():
    """Run all accuracy validation tests."""
    print("🧪 GPU Acceleration Accuracy Validation")
    print("=" * 50)
    
    tests = [
        ("EAR Calculation Accuracy", test_ear_accuracy),
        ("MAR Calculation Accuracy", test_mar_accuracy),
        ("Frame Processing Accuracy", test_frame_processing_accuracy),
        ("Color Conversion Accuracy", test_color_conversion_accuracy),
        ("Consistency Across Runs", test_consistency_across_runs),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! GPU acceleration preserves accuracy.")
        return 0
    else:
        print("⚠️ Some validation tests failed. GPU acceleration may affect accuracy.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
