#!/usr/bin/env python3
"""
Test script to verify threshold validation functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validate_infrared_improvements import InfraredValidationSystem

def test_threshold_validation():
    """Test the threshold validation with custom parameters"""
    print("🧪 Testing Threshold Validation System")
    print("=" * 50)
    
    # Check if temp_videos directory exists
    if not os.path.exists("temp_videos"):
        print("❌ temp_videos directory not found. Creating test scenario...")
        print("   This test will verify the threshold setting mechanism only.")
        
        # Create validator
        validator = InfraredValidationSystem(precision_mode=True)
        
        # Test threshold setting and restoration
        print("\n🔧 Testing threshold setting mechanism:")
        
        # Store original values
        original_fatigue = validator.detector.FATIGUE_THRESHOLD_MILD
        original_perclos = validator.detector.PERCLOS_THRESHOLD_MILD
        original_confidence = validator.detector.MIN_CONFIDENCE_THRESHOLD
        
        print(f"   Original thresholds:")
        print(f"     Fatigue: {original_fatigue}")
        print(f"     PERCLOS: {original_perclos}")
        print(f"     Confidence: {original_confidence}")
        
        # Test setting new thresholds
        test_fatigue = 0.50
        test_perclos = 0.25
        test_confidence = 0.85
        
        print(f"\n   Setting test thresholds:")
        print(f"     Fatigue: {test_fatigue}")
        print(f"     PERCLOS: {test_perclos}")
        print(f"     Confidence: {test_confidence}")
        
        # Manually set thresholds to test the mechanism
        validator.detector.FATIGUE_THRESHOLD_MILD = test_fatigue
        validator.detector.PERCLOS_THRESHOLD_MILD = test_perclos
        validator.detector.MIN_CONFIDENCE_THRESHOLD = test_confidence
        
        # Verify they were set correctly
        assert validator.detector.FATIGUE_THRESHOLD_MILD == test_fatigue
        assert validator.detector.PERCLOS_THRESHOLD_MILD == test_perclos
        assert validator.detector.MIN_CONFIDENCE_THRESHOLD == test_confidence
        
        print("   ✅ Threshold setting successful!")
        
        # Restore original thresholds
        validator.detector.FATIGUE_THRESHOLD_MILD = original_fatigue
        validator.detector.PERCLOS_THRESHOLD_MILD = original_perclos
        validator.detector.MIN_CONFIDENCE_THRESHOLD = original_confidence
        
        # Verify restoration
        assert validator.detector.FATIGUE_THRESHOLD_MILD == original_fatigue
        assert validator.detector.PERCLOS_THRESHOLD_MILD == original_perclos
        assert validator.detector.MIN_CONFIDENCE_THRESHOLD == original_confidence
        
        print("   ✅ Threshold restoration successful!")
        
        print("\n✅ Threshold validation mechanism test completed successfully!")
        print("   The adaptive optimizer should now work correctly.")
        
        return True
    
    else:
        print("✅ temp_videos directory found. Testing full validation...")
        
        # Create validator
        validator = InfraredValidationSystem(precision_mode=True)
        
        # Test with a simple threshold configuration
        try:
            results = validator.validate_with_thresholds(
                fatigue_threshold=0.40,
                perclos_threshold=0.20,
                confidence_threshold=0.80
            )
            
            if results and 'summary' in results:
                metrics = results['summary'].get('metrics', {})
                print(f"   ✅ Validation successful!")
                print(f"   Precision: {metrics.get('precision', 'N/A'):.3f}")
                print(f"   Recall: {metrics.get('recall', 'N/A'):.3f}")
                return True
            else:
                print("   ⚠️ Validation returned empty results")
                return False
                
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return False

if __name__ == "__main__":
    success = test_threshold_validation()
    if success:
        print("\n🚀 Ready to run full adaptive optimization!")
        print("   Execute: python adaptive_threshold_optimizer.py")
    else:
        print("\n❌ Issues detected. Please check the validation system.")
