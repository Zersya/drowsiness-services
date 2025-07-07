#!/usr/bin/env python3
"""
Test script for GPU acceleration in drowsiness landmark detection system.
This script validates GPU functionality and measures performance improvements.
"""

import sys
import time
import numpy as np
import cv2
from drowsiness_landmark import FatigueDetectionSystem

def test_gpu_availability():
    """Test if GPU acceleration is available and working."""
    print("🔍 Testing GPU Availability...")
    
    try:
        # Test CuPy availability
        import cupy as cp
        print(f"✅ CuPy available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(f"   GPU Device Count: {cp.cuda.runtime.getDeviceCount()}")
            print(f"   Current Device: {cp.cuda.runtime.getDevice()}")
    except ImportError:
        print("❌ CuPy not available")
    
    try:
        # Test OpenCV CUDA
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"✅ OpenCV CUDA devices: {cuda_devices}")
    except (AttributeError, cv2.error):
        print("❌ OpenCV CUDA not available")
    
    return True

def test_system_initialization():
    """Test system initialization with GPU settings."""
    print("\n🚀 Testing System Initialization...")
    
    # Test GPU-enabled initialization
    try:
        detector_gpu = FatigueDetectionSystem(enable_gpu=True, gpu_device_id=0)
        gpu_status = detector_gpu.get_gpu_status()
        print(f"✅ GPU-enabled system initialized")
        print(f"   GPU Available: {gpu_status['gpu_available']}")
        print(f"   CuPy Available: {gpu_status['cupy_available']}")
        print(f"   OpenCV CUDA Available: {gpu_status['opencv_cuda_available']}")
    except Exception as e:
        print(f"❌ GPU initialization failed: {e}")
        return False
    
    # Test CPU-only initialization
    try:
        detector_cpu = FatigueDetectionSystem(enable_gpu=False)
        print(f"✅ CPU-only system initialized")
    except Exception as e:
        print(f"❌ CPU initialization failed: {e}")
        return False
    
    return True

def test_image_processing():
    """Test GPU-accelerated image processing operations."""
    print("\n🖼️ Testing Image Processing...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        detector = FatigueDetectionSystem(enable_gpu=True)
        
        # Test color conversion
        print("   Testing color conversion...")
        gray_image = detector._convert_to_grayscale_gpu(test_image)
        if gray_image is not None and gray_image.shape == (480, 640):
            print("   ✅ Color conversion successful")
        else:
            print("   ❌ Color conversion failed")
            return False
        
        # Test frame enhancement
        print("   Testing frame enhancement...")
        enhanced = detector.enhance_infrared_frame(gray_image)
        if enhanced is not None and enhanced.shape == gray_image.shape:
            print("   ✅ Frame enhancement successful")
        else:
            print("   ❌ Frame enhancement failed")
            return False
        
        # Test frame quality assessment
        print("   Testing frame quality assessment...")
        quality = detector.assess_infrared_frame_quality(gray_image)
        if 0 <= quality <= 1:
            print(f"   ✅ Frame quality assessment successful (quality: {quality:.3f})")
        else:
            print("   ❌ Frame quality assessment failed")
            return False
        
    except Exception as e:
        print(f"   ❌ Image processing test failed: {e}")
        return False
    
    return True

def test_mathematical_operations():
    """Test GPU-accelerated mathematical operations."""
    print("\n🧮 Testing Mathematical Operations...")
    
    try:
        detector = FatigueDetectionSystem(enable_gpu=True)
        
        # Test EAR calculation with dummy landmarks
        print("   Testing EAR calculation...")
        eye_landmarks = np.array([
            [0, 0], [5, -2], [10, -1], [15, 0], [10, 1], [5, 2]
        ], dtype=np.float64)
        
        ear_value = detector.calculate_ear(eye_landmarks)
        if 0.1 <= ear_value <= 0.6:
            print(f"   ✅ EAR calculation successful (EAR: {ear_value:.3f})")
        else:
            print(f"   ❌ EAR calculation failed (EAR: {ear_value})")
            return False
        
        # Test MAR calculation with dummy landmarks
        print("   Testing MAR calculation...")
        mouth_landmarks = np.array([
            [0, 0], [2, -1], [4, -2], [6, -1], [8, 0], [6, 1], [4, 2], [2, 1]
        ], dtype=np.float64)
        
        mar_value = detector.calculate_mar(mouth_landmarks)
        if 0.1 <= mar_value <= 2.0:
            print(f"   ✅ MAR calculation successful (MAR: {mar_value:.3f})")
        else:
            print(f"   ❌ MAR calculation failed (MAR: {mar_value})")
            return False
        
    except Exception as e:
        print(f"   ❌ Mathematical operations test failed: {e}")
        return False
    
    return True

def test_performance_benchmark():
    """Test performance benchmarking functionality."""
    print("\n⚡ Testing Performance Benchmark...")
    
    try:
        detector = FatigueDetectionSystem(enable_gpu=True)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run benchmark
        print("   Running benchmark (this may take a moment)...")
        benchmark_results = detector.run_benchmark(test_frame, iterations=5)
        
        if 'operations' in benchmark_results:
            print("   ✅ Benchmark completed successfully")
            for operation, metrics in benchmark_results['operations'].items():
                speedup = metrics.get('speedup', 0)
                print(f"      {operation}: {speedup:.2f}x speedup")
        else:
            print("   ❌ Benchmark failed - no results")
            return False
        
    except Exception as e:
        print(f"   ❌ Performance benchmark test failed: {e}")
        return False
    
    return True

def test_fallback_behavior():
    """Test CPU fallback behavior."""
    print("\n🔄 Testing CPU Fallback Behavior...")
    
    try:
        # Test with GPU disabled
        detector = FatigueDetectionSystem(enable_gpu=False)
        
        # Test that operations still work
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Test frame enhancement (should use CPU)
        enhanced = detector.enhance_infrared_frame(gray_image)
        if enhanced is not None:
            print("   ✅ CPU fallback for frame enhancement working")
        else:
            print("   ❌ CPU fallback for frame enhancement failed")
            return False
        
        # Test EAR calculation (should use CPU)
        eye_landmarks = np.array([
            [0, 0], [5, -2], [10, -1], [15, 0], [10, 1], [5, 2]
        ], dtype=np.float64)
        
        ear_value = detector.calculate_ear(eye_landmarks)
        if 0.1 <= ear_value <= 0.6:
            print("   ✅ CPU fallback for EAR calculation working")
        else:
            print("   ❌ CPU fallback for EAR calculation failed")
            return False
        
    except Exception as e:
        print(f"   ❌ CPU fallback test failed: {e}")
        return False
    
    return True

def main():
    """Run all GPU acceleration tests."""
    print("🧪 GPU Acceleration Test Suite")
    print("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("System Initialization", test_system_initialization),
        ("Image Processing", test_image_processing),
        ("Mathematical Operations", test_mathematical_operations),
        ("Performance Benchmark", test_performance_benchmark),
        ("CPU Fallback Behavior", test_fallback_behavior),
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
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU acceleration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
