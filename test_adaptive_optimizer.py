#!/usr/bin/env python3
"""
Test script for the Adaptive Threshold Optimizer
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer

def test_optimizer():
    """Test the adaptive optimizer with a small number of iterations"""
    print("🧪 Testing Adaptive Threshold Optimizer")
    print("=" * 50)
    
    # Create optimizer with limited iterations for testing
    optimizer = AdaptiveThresholdOptimizer(
        target_precision=0.90,
        max_iterations=5  # Small number for testing
    )
    
    print(f"🎯 Target precision: {optimizer.target_precision*100:.1f}%")
    print(f"🔄 Max iterations: {optimizer.max_iterations}")
    print(f"📊 Parameter bounds: {optimizer.param_bounds}")
    
    # Test parameter generation
    print("\n🔍 Testing parameter generation:")
    for i in range(3):
        params = optimizer.generate_next_parameters()
        print(f"  Iteration {i+1}: {params}")
    
    print("\n✅ Basic functionality test completed")
    print("Run 'python adaptive_threshold_optimizer.py' for full optimization")

if __name__ == "__main__":
    test_optimizer()
