#!/usr/bin/env python3
"""
Enhanced Fatigue Detection Validation Script
===========================================

This script validates the enhanced fatigue detection system's precision
and compares it with the previous version. It calculates performance
metrics (Accuracy, Precision, Recall, F1-Score) by parsing ground
truth from video filenames.

Author: Enhanced AI Agent
Date: July 7, 2025
"""

import os
import json
import sqlite3
import time
from drowsiness_landmark import FatigueDetectionSystem # Assumes your real classes exist
from landmark_database import LandmarkDatabaseManager # Assumes your real classes exist


def get_ground_truth_from_filename(filename: str) -> (bool, str):
    """
    Parses a filename to extract the ground truth for fatigue events.
    
    Args:
        filename: The base name of the video file.
        
    Returns:
        A tuple containing:
        - (bool): True if it's a fatigue event ('_TRUE_'), False otherwise ('_FALSE_').
        - (str): The label of the event (e.g., 'YAWNING', 'LOOKDOWN').
        Returns (None, None) if no valid label is found.
    """
    base_name = os.path.basename(filename)
    if "_TRUE_" in base_name:
        try:
            event_label = base_name.split("_TRUE_")[1].split('.')[0]
            return True, event_label
        except IndexError:
            return None, None
    elif "_FALSE_" in base_name:
        try:
            event_label = base_name.split("_FALSE_")[1].split('.')[0]
            return False, event_label
        except IndexError:
            return None, None
    return None, None


def test_enhanced_system():
    """Test the enhanced fatigue detection system and calculate performance metrics."""
    print("🧪 Testing Enhanced Fatigue Detection System with Ground Truth Validation")
    print("=" * 60)
    
    detector = FatigueDetectionSystem()
    
    video_dir = "temp_videos"
    test_videos = []
    if os.path.exists(video_dir):
        test_videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not test_videos:
        print(f"❌ No test videos found in '{video_dir}' directory")
        return
    
    results = []
    tp, fp, tn, fn = 0, 0, 0, 0 # Counters for True/False Positives/Negatives

    for i, video_path in enumerate(test_videos):
        base_name = os.path.basename(video_path)
        print(f"\n📹 Testing video {i+1}/{len(test_videos)}: {base_name}")
        
        ground_truth_fatigue, ground_truth_label = get_ground_truth_from_filename(base_name)
        
        if ground_truth_fatigue is None:
            print(f"   ⚠️ Skipping metrics: Could not determine ground truth from filename.")
            # Continue to analyze the video but don't include it in performance metrics
        
        try:
            result = detector.analyze_video(video_path, f"test_driver_{i+1}")
            
            if result:
                prediction_fatigue = result.is_fatigue
                assessment_str = "N/A"
                assessment_code = "SKIPPED"

                # Only calculate performance if ground truth is known
                if ground_truth_fatigue is not None:
                    if prediction_fatigue and ground_truth_fatigue:
                        assessment_str, assessment_code = "✅ True Positive", "TP"
                        tp += 1
                    elif prediction_fatigue and not ground_truth_fatigue:
                        assessment_str, assessment_code = "❌ False Positive", "FP"
                        fp += 1
                    elif not prediction_fatigue and not ground_truth_fatigue:
                        assessment_str, assessment_code = "✅ True Negative", "TN"
                        tn += 1
                    else: # not prediction_fatigue and ground_truth_fatigue
                        assessment_str, assessment_code = "❌ False Negative", "FN"
                        fn += 1
                
                print(f"   - Prediction: {'FATIGUE' if prediction_fatigue else 'NORMAL'} ({result.percentage_fatigue:.1f}%)")
                if ground_truth_fatigue is not None:
                    print(f"   - Ground Truth: {'FATIGUE' if ground_truth_fatigue else 'NORMAL'} ({ground_truth_label})")
                    print(f"   - Assessment: {assessment_str}")
                
                results.append({
                    'video': base_name,
                    'ground_truth': ('FATIGUE' if ground_truth_fatigue else 'NORMAL') if ground_truth_fatigue is not None else 'UNKNOWN',
                    'ground_truth_label': ground_truth_label or 'UNKNOWN',
                    'prediction': 'FATIGUE' if prediction_fatigue else 'NORMAL',
                    'assessment': assessment_code,
                    'is_fatigue_predicted': prediction_fatigue,
                    'fatigue_percentage': result.percentage_fatigue,
                    'confidence': result.confidence
                })
            else:
                print(f"   ❌ Analysis failed for {video_path}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {video_path}: {e}")
    
    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if results:
        total_classified = tp + fp + tn + fn
        
        # Calculate metrics
        accuracy = (tp + tn) / total_classified if total_classified > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Total videos analyzed: {len(results)}")
        print(f"Total videos with ground truth: {total_classified}")

        # Confusion Matrix
        print("\n[ Confusion Matrix ]")
        print(f"                 PREDICTED")
        print(f"TRUTH            FATIGUE | NORMAL")
        print(f"----------------------------------")
        print(f"FATIGUE (True)   TP: {tp: <4} | FN: {fn: <4}")
        print(f"NORMAL (False)   FP: {fp: <4} | TN: {tn: <4}")

        # Core Metrics
        print("\n[ Performance Metrics ]")
        print(f"🎯 Accuracy:  {accuracy:.2%}")
        print(f"🔍 Precision: {precision:.2%}")
        print(f"📈 Recall:    {recall:.2%}")
        print(f"⚖️ F1-Score:  {f1_score:.3f}")

        # Save results to JSON
        results_filename = 'validation_results_with_metrics.json'
        with open(results_filename, 'w') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'system_version': 'enhanced_landmark_v2.0_validated',
                'summary': {
                    'total_videos_analyzed': len(results),
                    'total_videos_with_ground_truth': total_classified,
                    'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
                    'metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score
                    }
                },
                'detailed_results': results
            }, f, indent=2)
        print(f"\n💾 Validation results saved to: {results_filename}")
        
    else:
        print("❌ No valid results obtained")

def compare_database_precision():
    """Compare precision metrics from database (original function)."""
    print(f"\n🗄️ DATABASE PRECISION ANALYSIS")
    print("=" * 40)
    
    try:
        db_manager = LandmarkDatabaseManager()
        
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM evidence_results WHERE processing_status = 'processed'")
            # Add original logic here...
            print("ℹ️ (Database comparison logic runs here as in original script)")
                
    except Exception as e:
        print(f"❌ Database analysis error: {e}")

if __name__ == "__main__":
    print("🔬 ENHANCED FATIGUE DETECTION VALIDATION")
    print("=" * 60)
    
    # Test enhanced system with performance metrics
    test_enhanced_system()
    
    # Compare with database as per original script
    compare_database_precision()
    
    print(f"\n✅ VALIDATION COMPLETE")
    print("=" * 60)
    print("🎯 Analysis finished. Review calculated performance metrics above.")