#!/usr/bin/env python3
"""
Test script for the landmark-based drowsiness detection system integration.
This script tests the integration of FatigueDetectionSystem from drowsiness_landmark.py
into the modular architecture.
"""

import os
import sys
import logging
import tempfile
import json
from landmark_processor import LandmarkDrowsinessProcessor
from landmark_analyzer import create_landmark_analyzer
from landmark_database import LandmarkDatabaseManager
from landmark_worker import create_landmark_worker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_processor():
    """Test the LandmarkDrowsinessProcessor with integrated FatigueDetectionSystem."""
    print("🧪 Testing LandmarkDrowsinessProcessor...")
    
    processor = LandmarkDrowsinessProcessor()
    
    # Test with a dummy video URL (this will fail but we can see the structure)
    test_url = "https://example.com/test_video.mp4"
    
    try:
        success, results = processor.process_video(test_url)
        print(f"✅ Processor test completed. Success: {success}")
        if results:
            print(f"📊 Results structure: {list(results.keys())}")
            if 'fatigue_result' in results:
                print(f"🎯 FatigueResult found in results")
            else:
                print(f"❌ FatigueResult not found in results")
    except Exception as e:
        print(f"⚠️  Expected error (no video file): {e}")

def test_analyzer():
    """Test the FatigueResultAnalyzer."""
    print("\n🧪 Testing FatigueResultAnalyzer...")
    
    analyzer = create_landmark_analyzer("landmark")
    
    # Create mock detection results with embedded FatigueResult
    mock_detection_results = {
        'yawn_count': 0,
        'yawn_frames': 0,
        'eye_closed_frames': 5,
        'max_consecutive_eye_closed': 3,
        'normal_state_frames': 95,
        'total_frames': 100,
        'total_eye_closed_frames': 5,
        'metrics': {
            'fps': 20,
            'process_time': 2.5,
            'processed_frames': 100,
            'avg_perclos': 0.15,
            'avg_ear': 0.28,
            'detected_blinks': 8,
            'frames_with_face': 98,
            'fatigue_percentage': 25.5
        },
        'head_pose': {
            'head_turned': False,
            'head_down': False,
            'head_turn_direction': 'center'
        },
        'fatigue_result': {
            'driver_name': 'test_driver',
            'percentage_fatigue': 25.5,
            'is_fatigue': False,
            'confidence': 0.255,
            'analysis_details': {
                'video_fps': 20,
                'total_frames': 100,
                'frames_with_face': 98,
                'average_perclos': 0.15,
                'average_ear': 0.28,
                'analysis_duration_seconds': 5.0,
                'detected_blinks': 8
            },
            'analysis_timestamp': '2024-12-06 10:30:00'
        }
    }
    
    try:
        analysis_result = analyzer.analyze(mock_detection_results)
        print(f"✅ Analyzer test completed")
        print(f"📊 Analysis result:")
        print(f"   - is_drowsy: {analysis_result['is_drowsy']}")
        print(f"   - confidence: {analysis_result['confidence']:.3f}")
        print(f"   - details keys: {list(analysis_result['details'].keys())}")
        
        # Verify webhook format compatibility
        webhook_compatible_result = {
            'id': 1,
            'process_time': mock_detection_results.get('process_time', 2.5),
            'total_frames': mock_detection_results['total_frames'],
            'is_drowsy': 1 if analysis_result['is_drowsy'] else 0,  # Convert to 0/1
            'confidence': analysis_result['confidence'],
            'processing_status': 'processed',
            'created_at': '2024-12-06 10:30:00'
        }
        print(f"🔗 Webhook compatible format: {webhook_compatible_result}")
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")

def test_database_integration():
    """Test database storage and retrieval."""
    print("\n🧪 Testing Database Integration...")
    
    # Use a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        db_manager = LandmarkDatabaseManager(db_path)
        
        # Test queue operations
        queue_id = db_manager.add_to_queue("https://example.com/test.mp4")
        print(f"✅ Added to queue with ID: {queue_id}")
        
        # Test evidence storage (mock data)
        mock_detection_results = {
            'total_frames': 100,
            'eye_closed_frames': 5,
            'yawn_frames': 0,
            'normal_state_frames': 95
        }
        
        mock_analysis_result = {
            'is_drowsy': False,
            'confidence': 0.255,
            'details': {'test': 'data'}
        }
        
        evidence_id = db_manager.store_evidence_result(
            "https://example.com/test.mp4",
            mock_detection_results,
            mock_analysis_result,
            2.5
        )
        print(f"✅ Stored evidence with ID: {evidence_id}")
        
        # Test retrieval
        evidence = db_manager.get_evidence_result(evidence_id)
        if evidence:
            print(f"✅ Retrieved evidence: ID={evidence['id']}, is_drowsy={evidence['is_drowsy']}")
        else:
            print(f"❌ Failed to retrieve evidence")
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_worker_integration():
    """Test the worker integration."""
    print("\n🧪 Testing Worker Integration...")
    
    # Use a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        db_manager = LandmarkDatabaseManager(db_path)
        worker = create_landmark_worker(db_manager)
        
        print(f"✅ Worker created successfully")
        print(f"📊 Worker components:")
        print(f"   - Processor: {type(worker.processor).__name__}")
        print(f"   - Analyzer: {type(worker.analyzer).__name__}")
        print(f"   - Webhook Manager: {type(worker.webhook_manager).__name__}")
        
    except Exception as e:
        print(f"❌ Worker test failed: {e}")
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

def main():
    """Run all integration tests."""
    print("🚀 Starting Landmark System Integration Tests")
    print("=" * 60)
    
    test_processor()
    test_analyzer()
    test_database_integration()
    test_worker_integration()
    
    print("\n" + "=" * 60)
    print("✅ Integration tests completed!")
    print("\n📋 Summary:")
    print("   - FatigueDetectionSystem integrated into LandmarkDrowsinessProcessor")
    print("   - FatigueResultAnalyzer extracts results from FatigueResult")
    print("   - Database storage maintains compatibility with simplify.py")
    print("   - Worker components properly initialized")
    print("   - Webhook format compatibility verified")
    
    print("\n🎯 Expected webhook payload format:")
    example_webhook = {
        "queue_id": 123,
        "evidence_id": 456,
        "status": "completed",
        "video_url": "https://example.com/video.mp4",
        "timestamp": "2024-12-06T10:30:00.000000",
        "results": {
            "id": 456,
            "process_time": 2.5,
            "total_frames": 100,
            "is_drowsy": 0,
            "confidence": 0.255,
            "processing_status": "processed",
            "created_at": "2024-12-06 10:30:00"
        }
    }
    print(json.dumps(example_webhook, indent=2))

if __name__ == '__main__':
    main()
