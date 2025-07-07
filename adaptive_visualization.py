#!/usr/bin/env python3
"""
Visualization system for Adaptive Threshold Optimization results
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_optimization_history(filepath: str = "adaptive_optimization_history.json") -> Dict:
    """Load optimization history from JSON file"""
    if not os.path.exists(filepath):
        print(f"❌ History file not found: {filepath}")
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_optimization_progress(history: Dict):
    """Analyze and display optimization progress"""
    if not history or 'iterations' not in history:
        print("❌ No iteration data found")
        return
    
    iterations = history['iterations']
    config = history.get('optimization_config', {})
    best = history.get('best_iteration', {})
    
    print("📊 ADAPTIVE OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Configuration summary
    print(f"🎯 Target Precision: {config.get('target_precision', 'N/A')*100:.1f}%")
    print(f"🔄 Total Iterations: {len(iterations)}")
    print(f"📈 Parameter Bounds: {config.get('param_bounds', 'N/A')}")
    
    # Best result summary
    if best:
        print(f"\n🏆 BEST RESULT (Iteration {best['iteration']}):")
        print(f"   Precision: {best['precision']:.3f} ({best['precision']*100:.1f}%)")
        print(f"   Recall: {best['recall']:.3f} ({best['recall']*100:.1f}%)")
        print(f"   Improvement Score: {best['improvement_score']:.3f}")
        print(f"   Thresholds: {best['thresholds']}")
    
    # Progress analysis
    print(f"\n📈 OPTIMIZATION PROGRESS:")
    precisions = [iter['precision'] for iter in iterations]
    recalls = [iter['recall'] for iter in iterations]
    scores = [iter['improvement_score'] for iter in iterations]
    
    print(f"   Initial → Final Precision: {precisions[0]:.3f} → {precisions[-1]:.3f}")
    print(f"   Initial → Final Recall: {recalls[0]:.3f} → {recalls[-1]:.3f}")
    print(f"   Initial → Final Score: {scores[0]:.3f} → {scores[-1]:.3f}")
    print(f"   Max Precision Achieved: {max(precisions):.3f}")
    print(f"   Max Recall Achieved: {max(recalls):.3f}")
    
    # Exploration strategy analysis
    exploration_types = [iter['exploration_type'] for iter in iterations]
    exploration_counts = {}
    for exp_type in exploration_types:
        exploration_counts[exp_type] = exploration_counts.get(exp_type, 0) + 1
    
    print(f"\n🔍 EXPLORATION STRATEGY BREAKDOWN:")
    for exp_type, count in exploration_counts.items():
        percentage = (count / len(iterations)) * 100
        print(f"   {exp_type.capitalize()}: {count} iterations ({percentage:.1f}%)")

def analyze_parameter_evolution(history: Dict):
    """Analyze how parameters evolved during optimization"""
    if not history or 'iterations' not in history:
        return
    
    iterations = history['iterations']
    
    print(f"\n🔄 PARAMETER EVOLUTION:")
    print("-" * 40)
    
    # Track parameter changes
    param_names = ['fatigue_threshold', 'perclos_threshold', 'confidence_threshold']
    
    for param in param_names:
        values = [iter['thresholds'][param] for iter in iterations]
        print(f"\n📊 {param.replace('_', ' ').title()}:")
        print(f"   Initial: {values[0]:.3f}")
        print(f"   Final: {values[-1]:.3f}")
        print(f"   Range: {min(values):.3f} - {max(values):.3f}")
        print(f"   Std Dev: {np.std(values):.3f}")

def find_convergence_point(history: Dict) -> int:
    """Find the iteration where optimization converged"""
    if not history or 'iterations' not in history:
        return -1
    
    iterations = history['iterations']
    scores = [iter['improvement_score'] for iter in iterations]
    
    # Look for convergence (minimal improvement over last 3 iterations)
    convergence_threshold = 0.01
    window_size = 3
    
    for i in range(window_size, len(scores)):
        recent_scores = scores[i-window_size:i]
        if max(recent_scores) - min(recent_scores) < convergence_threshold:
            return i - window_size + 1
    
    return len(iterations)  # No convergence detected

def generate_optimization_report(history: Dict):
    """Generate comprehensive optimization report"""
    if not history:
        print("❌ No optimization history available")
        return
    
    iterations = history['iterations']
    config = history.get('optimization_config', {})
    best = history.get('best_iteration', {})
    
    print(f"\n📋 COMPREHENSIVE OPTIMIZATION REPORT")
    print("=" * 60)
    
    # Efficiency metrics
    target_precision = config.get('target_precision', 0.9)
    target_achieved = best.get('precision', 0) >= target_precision if best else False
    convergence_iter = find_convergence_point(history)
    
    print(f"🎯 TARGET ACHIEVEMENT:")
    print(f"   Target Precision: {target_precision*100:.1f}%")
    print(f"   Achieved: {'✅ YES' if target_achieved else '❌ NO'}")
    if best:
        print(f"   Best Precision: {best['precision']*100:.1f}%")
        print(f"   Gap to Target: {(target_precision - best['precision'])*100:.1f}%")
    
    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"   Total Iterations: {len(iterations)}")
    print(f"   Convergence Point: Iteration {convergence_iter}")
    print(f"   Efficiency: {(convergence_iter/len(iterations))*100:.1f}% of max iterations")
    
    # Performance trends
    precisions = [iter['precision'] for iter in iterations]
    improvement_rate = (precisions[-1] - precisions[0]) / len(iterations)
    
    print(f"\n📈 PERFORMANCE TRENDS:")
    print(f"   Average Improvement Rate: {improvement_rate:.4f} per iteration")
    print(f"   Total Precision Gain: {(precisions[-1] - precisions[0])*100:.1f}%")
    
    # Top performing iterations
    sorted_iterations = sorted(iterations, key=lambda x: x['improvement_score'], reverse=True)
    
    print(f"\n🏆 TOP 3 PERFORMING ITERATIONS:")
    for i, iter_data in enumerate(sorted_iterations[:3]):
        print(f"   #{i+1}: Iteration {iter_data['iteration']} - "
              f"Precision: {iter_data['precision']:.3f}, "
              f"Score: {iter_data['improvement_score']:.3f}")

def create_simple_plots(history: Dict):
    """Create simple text-based plots for optimization progress"""
    if not history or 'iterations' not in history:
        return
    
    iterations = history['iterations']
    
    print(f"\n📊 OPTIMIZATION PROGRESS VISUALIZATION")
    print("-" * 50)
    
    # Simple precision progress plot
    precisions = [iter['precision'] for iter in iterations]
    max_precision = max(precisions)
    
    print(f"📈 Precision Progress (Max: {max_precision:.3f}):")
    for i, precision in enumerate(precisions):
        bar_length = int((precision / max_precision) * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   Iter {i+1:2d}: {bar} {precision:.3f}")
    
    # Simple improvement score plot
    scores = [iter['improvement_score'] for iter in iterations]
    max_score = max(scores) if scores else 1
    
    print(f"\n🎯 Improvement Score Progress (Max: {max_score:.3f}):")
    for i, score in enumerate(scores):
        bar_length = int((score / max_score) * 30) if max_score > 0 else 0
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   Iter {i+1:2d}: {bar} {score:.3f}")

def main():
    """Main analysis function"""
    print("📊 ADAPTIVE OPTIMIZATION ANALYSIS TOOL")
    print("=" * 60)
    
    # Load optimization history
    history = load_optimization_history()
    
    if not history:
        print("❌ No optimization history found. Run the optimizer first.")
        return
    
    # Perform comprehensive analysis
    analyze_optimization_progress(history)
    analyze_parameter_evolution(history)
    generate_optimization_report(history)
    create_simple_plots(history)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 History file: adaptive_optimization_history.json")

if __name__ == "__main__":
    main()
