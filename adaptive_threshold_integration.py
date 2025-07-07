#!/usr/bin/env python3
"""
Adaptive Threshold Integration System
====================================

This module provides seamless integration of machine learning-discovered optimal thresholds
from the adaptive optimization process into the production landmark drowsiness detection system.

Features:
- Load optimal thresholds from adaptive_optimization_history.json
- Apply thresholds to production FatigueDetectionSystem
- Maintain backward compatibility with existing functionality
- Environment variable configuration support
- Automatic fallback to default thresholds if optimization results unavailable

Usage:
    from adaptive_threshold_integration import apply_adaptive_thresholds
    
    # Apply to existing FatigueDetectionSystem instance
    fatigue_system = FatigueDetectionSystem()
    apply_adaptive_thresholds(fatigue_system)
    
    # Or create with adaptive thresholds automatically applied
    fatigue_system = create_adaptive_fatigue_system()
"""

import json
import os
import logging
from typing import Dict, Optional, Any
from landmark_processor import FatigueDetectionSystem

# Configuration constants
OPTIMIZATION_HISTORY_FILE = "adaptive_optimization_history.json"
FALLBACK_THRESHOLDS = {
    'fatigue_threshold': 0.1749126021093799,      
    'perclos_threshold': 0.14893937787674166,      
    'confidence_threshold': 0.50    
}

class AdaptiveThresholdManager:
    """Manager class for handling adaptive threshold integration"""
    
    def __init__(self, optimization_file: str = OPTIMIZATION_HISTORY_FILE):
        self.optimization_file = optimization_file
        self.optimal_thresholds = None
        self.optimization_metadata = None
        self._load_optimization_results()
    
    def _load_optimization_results(self) -> bool:
        """Load optimization results from JSON file"""
        try:
            if not os.path.exists(self.optimization_file):
                logging.warning(f"Optimization file {self.optimization_file} not found. Using fallback thresholds.")
                return False
            
            with open(self.optimization_file, 'r') as f:
                data = json.load(f)
            
            # Extract best iteration results
            if 'best_iteration' in data and 'thresholds' in data['best_iteration']:
                self.optimal_thresholds = data['best_iteration']['thresholds']
                self.optimization_metadata = {
                    'iteration': data['best_iteration']['iteration'],
                    'precision': data['best_iteration'].get('precision', 'N/A'),
                    'recall': data['best_iteration'].get('recall', 'N/A'),
                    'improvement_score': data['best_iteration'].get('improvement_score', 'N/A'),
                    'target_precision': data.get('optimization_config', {}).get('target_precision', 0.9)
                }
                
                logging.info(f"✅ Loaded optimal thresholds from iteration {self.optimization_metadata['iteration']}")
                logging.info(f"   Precision: {self.optimization_metadata['precision']:.3f}")
                logging.info(f"   Recall: {self.optimization_metadata['recall']:.3f}")
                logging.info(f"   Thresholds: {self.optimal_thresholds}")
                return True
            else:
                logging.warning("Invalid optimization file format. Missing best_iteration or thresholds.")
                return False
                
        except Exception as e:
            logging.error(f"Error loading optimization results: {e}")
            return False
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get optimal thresholds or fallback defaults"""
        if self.optimal_thresholds:
            return self.optimal_thresholds.copy()
        else:
            logging.info("Using fallback thresholds (no optimization results available)")
            return FALLBACK_THRESHOLDS.copy()
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get optimization metadata"""
        return self.optimization_metadata
    
    def apply_to_fatigue_system(self, fatigue_system: FatigueDetectionSystem) -> bool:
        """Apply optimal thresholds to a FatigueDetectionSystem instance"""
        try:
            thresholds = self.get_thresholds()
            
            # Map adaptive optimization parameter names to FatigueDetectionSystem attributes
            # Note: FatigueDetectionSystem uses different attribute names than InfraredOptimizedDetector
            fatigue_system.FATIGUE_THRESHOLD = thresholds['fatigue_threshold']
            fatigue_system.PERCLOS_THRESHOLD = thresholds['perclos_threshold']
            
            # For confidence threshold, we need to add this as a new attribute since
            # the base FatigueDetectionSystem doesn't have it
            fatigue_system.MIN_CONFIDENCE_THRESHOLD = thresholds['confidence_threshold']
            
            # Log the applied thresholds
            logging.info("🎯 Applied adaptive thresholds to FatigueDetectionSystem:")
            logging.info(f"   FATIGUE_THRESHOLD: {fatigue_system.FATIGUE_THRESHOLD}")
            logging.info(f"   PERCLOS_THRESHOLD: {fatigue_system.PERCLOS_THRESHOLD}")
            logging.info(f"   MIN_CONFIDENCE_THRESHOLD: {fatigue_system.MIN_CONFIDENCE_THRESHOLD}")
            
            if self.optimization_metadata:
                logging.info(f"   Source: Adaptive optimization iteration {self.optimization_metadata['iteration']}")
                logging.info(f"   Achieved precision: {self.optimization_metadata['precision']:.3f}")
            else:
                logging.info("   Source: Fallback defaults (no optimization data)")
            
            return True
            
        except Exception as e:
            logging.error(f"Error applying thresholds to FatigueDetectionSystem: {e}")
            return False

# Global threshold manager instance
_threshold_manager = None

def get_threshold_manager() -> AdaptiveThresholdManager:
    """Get or create the global threshold manager instance"""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = AdaptiveThresholdManager()
    return _threshold_manager

def apply_adaptive_thresholds(fatigue_system: FatigueDetectionSystem) -> bool:
    """
    Apply adaptive optimization results to an existing FatigueDetectionSystem instance.
    
    Args:
        fatigue_system: The FatigueDetectionSystem instance to modify
        
    Returns:
        bool: True if thresholds were applied successfully, False otherwise
    """
    manager = get_threshold_manager()
    return manager.apply_to_fatigue_system(fatigue_system)

def create_adaptive_fatigue_system() -> FatigueDetectionSystem:
    """
    Create a new FatigueDetectionSystem with adaptive thresholds automatically applied.
    
    Returns:
        FatigueDetectionSystem: New instance with optimal thresholds applied
    """
    fatigue_system = FatigueDetectionSystem()
    apply_adaptive_thresholds(fatigue_system)
    return fatigue_system

def get_current_thresholds() -> Dict[str, float]:
    """Get the current optimal thresholds without applying them"""
    manager = get_threshold_manager()
    return manager.get_thresholds()

def get_optimization_metadata() -> Optional[Dict[str, Any]]:
    """Get metadata about the optimization process"""
    manager = get_threshold_manager()
    return manager.get_metadata()

def enable_adaptive_thresholds_via_env() -> bool:
    """
    Check if adaptive thresholds should be enabled via environment variable.
    Set ENABLE_ADAPTIVE_THRESHOLDS=true to enable automatic threshold application.
    
    Returns:
        bool: True if adaptive thresholds should be enabled
    """
    return os.getenv('ENABLE_ADAPTIVE_THRESHOLDS', 'false').lower() in ('true', '1', 'yes', 'on')

# Example usage and testing
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 Testing Adaptive Threshold Integration")
    print("=" * 50)
    
    # Test threshold manager
    manager = AdaptiveThresholdManager()
    thresholds = manager.get_thresholds()
    metadata = manager.get_metadata()
    
    print(f"📊 Current optimal thresholds: {thresholds}")
    if metadata:
        print(f"🎯 Optimization metadata: {metadata}")
    else:
        print("⚠️  No optimization metadata available")
    
    # Test applying to FatigueDetectionSystem
    print("\n🔧 Testing threshold application...")
    fatigue_system = FatigueDetectionSystem()
    
    print(f"Before: FATIGUE_THRESHOLD={fatigue_system.FATIGUE_THRESHOLD}")
    print(f"Before: PERCLOS_THRESHOLD={fatigue_system.PERCLOS_THRESHOLD}")
    
    success = apply_adaptive_thresholds(fatigue_system)
    
    print(f"After: FATIGUE_THRESHOLD={fatigue_system.FATIGUE_THRESHOLD}")
    print(f"After: PERCLOS_THRESHOLD={fatigue_system.PERCLOS_THRESHOLD}")
    print(f"After: MIN_CONFIDENCE_THRESHOLD={getattr(fatigue_system, 'MIN_CONFIDENCE_THRESHOLD', 'Not set')}")
    
    print(f"\n✅ Threshold application {'successful' if success else 'failed'}")
    
    # Test creating new system with adaptive thresholds
    print("\n🏭 Testing adaptive system creation...")
    adaptive_system = create_adaptive_fatigue_system()
    print(f"New system FATIGUE_THRESHOLD: {adaptive_system.FATIGUE_THRESHOLD}")
    print(f"New system PERCLOS_THRESHOLD: {adaptive_system.PERCLOS_THRESHOLD}")
    
    print("\n🎉 Integration testing complete!")
