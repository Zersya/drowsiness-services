#!/usr/bin/env python3
"""
Adaptive Threshold Optimizer for Infrared Drowsiness Detection
Uses intelligent parameter exploration to discover optimal thresholds toward 90% precision target.
"""

import json
import os
import sys
import time
import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validate_infrared_improvements import InfraredValidationSystem

@dataclass
class OptimizationIteration:
    """Results from a single optimization iteration"""
    iteration: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    thresholds: Dict[str, float]
    confusion_matrix: Dict[str, int]
    timestamp: str
    improvement_score: float  # Combined metric for optimization
    exploration_type: str  # Type of parameter exploration used

class AdaptiveThresholdOptimizer:
    """Adaptive learning system for threshold optimization"""
    
    def __init__(self, target_precision: float = 0.90, max_iterations: int = 50):
        """Initialize adaptive optimizer"""
        self.target_precision = target_precision
        self.max_iterations = max_iterations
        self.iteration_history = []
        self.best_iteration = None
        self.current_iteration = 0
        
        # Parameter bounds
        self.param_bounds = {
            'fatigue_threshold': (0.15, 0.80),
            'perclos_threshold': (0.05, 0.40),
            'confidence_threshold': (0.50, 0.95)
        }
        
        # Learning parameters
        self.exploration_rate = 0.3  # Balance between exploration and exploitation
        self.convergence_threshold = 0.01  # Minimum improvement to continue
        self.stagnation_limit = 5  # Max iterations without improvement
        
        # Initialize validation system
        self.validator = InfraredValidationSystem()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🤖 Adaptive Threshold Optimizer initialized")
        self.logger.info(f"🎯 Target precision: {target_precision*100:.1f}%")
        self.logger.info(f"🔄 Max iterations: {max_iterations}")

    def calculate_improvement_score(self, precision: float, recall: float) -> float:
        """Calculate improvement score prioritizing precision while maintaining recall"""
        if precision == 0:
            return 0.0
        
        # Heavily weight precision toward target, but penalize very low recall
        precision_score = min(precision / self.target_precision, 1.0)
        recall_penalty = max(0, 0.3 - recall) * 2  # Penalty if recall < 30%
        
        return precision_score - recall_penalty

    def generate_initial_parameters(self) -> Dict[str, float]:
        """Generate initial parameter set using domain knowledge"""
        # Start with moderate values based on previous analysis
        return {
            'fatigue_threshold': 0.35,
            'perclos_threshold': 0.15,
            'confidence_threshold': 0.70
        }

    def generate_next_parameters(self) -> Dict[str, float]:
        """Intelligently generate next parameter combination"""
        if len(self.iteration_history) == 0:
            return self.generate_initial_parameters()
        
        # Determine exploration strategy
        if random.random() < self.exploration_rate:
            return self._explore_parameters()
        else:
            return self._exploit_best_parameters()

    def _explore_parameters(self) -> Dict[str, float]:
        """Explore new parameter space regions"""
        exploration_strategies = [
            self._random_exploration,
            self._gradient_exploration,
            self._boundary_exploration
        ]
        
        strategy = random.choice(exploration_strategies)
        return strategy()

    def _random_exploration(self) -> Dict[str, float]:
        """Random parameter exploration within bounds"""
        params = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            params[param] = random.uniform(min_val, max_val)
        return params

    def _gradient_exploration(self) -> Dict[str, float]:
        """Gradient-based exploration around best parameters"""
        if not self.best_iteration:
            return self._random_exploration()
        
        base_params = self.best_iteration.thresholds.copy()
        params = {}
        
        for param, (min_val, max_val) in self.param_bounds.items():
            # Add small random perturbation
            perturbation = random.gauss(0, 0.05)  # 5% standard deviation
            new_val = base_params[param] + perturbation
            params[param] = max(min_val, min(max_val, new_val))
        
        return params

    def _boundary_exploration(self) -> Dict[str, float]:
        """Explore parameter boundaries for edge cases"""
        params = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            # Randomly choose boundary or near-boundary values
            if random.random() < 0.5:
                params[param] = min_val + random.uniform(0, 0.1) * (max_val - min_val)
            else:
                params[param] = max_val - random.uniform(0, 0.1) * (max_val - min_val)
        return params

    def _exploit_best_parameters(self) -> Dict[str, float]:
        """Exploit around best known parameters"""
        if not self.best_iteration:
            return self._random_exploration()
        
        # Get top 3 iterations
        top_iterations = sorted(self.iteration_history, 
                              key=lambda x: x.improvement_score, reverse=True)[:3]
        
        # Weighted average of top parameters
        params = {}
        weights = [0.5, 0.3, 0.2]  # Decreasing weights
        
        for param in self.param_bounds.keys():
            weighted_sum = sum(w * iter.thresholds[param] 
                             for w, iter in zip(weights, top_iterations))
            
            # Add small exploration noise
            noise = random.gauss(0, 0.02)
            min_val, max_val = self.param_bounds[param]
            params[param] = max(min_val, min(max_val, weighted_sum + noise))
        
        return params

    def evaluate_parameters(self, params: Dict[str, float]) -> OptimizationIteration:
        """Evaluate parameter set and return results"""
        self.current_iteration += 1
        
        self.logger.info(f"🔍 Iteration {self.current_iteration}/{self.max_iterations}")
        self.logger.info(f"📊 Testing parameters: {params}")
        
        # Run validation with current parameters
        start_time = time.time()
        results = self.validator.validate_with_thresholds(
            fatigue_threshold=params['fatigue_threshold'],
            perclos_threshold=params['perclos_threshold'],
            confidence_threshold=params['confidence_threshold']
        )
        processing_time = time.time() - start_time
        
        # Extract metrics
        metrics = results['summary']['metrics']
        confusion = results['summary']['confusion_matrix']
        
        # Handle both uppercase and lowercase confusion matrix keys
        tp = confusion.get('TP', confusion.get('tp', 0))
        fp = confusion.get('FP', confusion.get('fp', 0))
        tn = confusion.get('TN', confusion.get('tn', 0))
        fn = confusion.get('FN', confusion.get('fn', 0))
        
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1_score = metrics.get('f1_score', 0.0)
        accuracy = metrics.get('accuracy', 0.0)
        
        # Calculate improvement score
        improvement_score = self.calculate_improvement_score(precision, recall)
        
        # Create iteration result
        iteration = OptimizationIteration(
            iteration=self.current_iteration,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            thresholds=params.copy(),
            confusion_matrix={'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            improvement_score=improvement_score,
            exploration_type=self._get_exploration_type()
        )
        
        self.logger.info(f"📈 Results: Precision={precision:.3f}, Recall={recall:.3f}, "
                        f"F1={f1_score:.3f}, Score={improvement_score:.3f}")
        
        return iteration

    def _get_exploration_type(self) -> str:
        """Determine exploration type for logging"""
        if len(self.iteration_history) == 0:
            return "initial"
        elif random.random() < self.exploration_rate:
            return "exploration"
        else:
            return "exploitation"

    def update_best_iteration(self, iteration: OptimizationIteration):
        """Update best iteration if current is better"""
        if (not self.best_iteration or 
            iteration.improvement_score > self.best_iteration.improvement_score):
            self.best_iteration = iteration
            self.logger.info(f"🎉 New best iteration! Score: {iteration.improvement_score:.3f}")

    def check_convergence(self) -> bool:
        """Check if optimization has converged"""
        # Check if target precision reached
        if (self.best_iteration and 
            self.best_iteration.precision >= self.target_precision):
            self.logger.info(f"🎯 Target precision {self.target_precision*100:.1f}% achieved!")
            return True
        
        # Check for stagnation
        if len(self.iteration_history) >= self.stagnation_limit:
            recent_scores = [iter.improvement_score 
                           for iter in self.iteration_history[-self.stagnation_limit:]]
            if max(recent_scores) - min(recent_scores) < self.convergence_threshold:
                self.logger.info("📉 Optimization stagnated - no significant improvement")
                return True
        
        return False

    def save_iteration_history(self, filepath: str = "adaptive_optimization_history.json"):
        """Save complete iteration history"""
        history_data = {
            'optimization_config': {
                'target_precision': self.target_precision,
                'max_iterations': self.max_iterations,
                'param_bounds': self.param_bounds
            },
            'iterations': []
        }
        
        for iteration in self.iteration_history:
            history_data['iterations'].append({
                'iteration': iteration.iteration,
                'precision': float(iteration.precision),
                'recall': float(iteration.recall),
                'f1_score': float(iteration.f1_score),
                'accuracy': float(iteration.accuracy),
                'thresholds': iteration.thresholds,
                'confusion_matrix': iteration.confusion_matrix,
                'timestamp': iteration.timestamp,
                'improvement_score': float(iteration.improvement_score),
                'exploration_type': iteration.exploration_type
            })
        
        # Add best iteration summary
        if self.best_iteration:
            history_data['best_iteration'] = {
                'iteration': self.best_iteration.iteration,
                'precision': float(self.best_iteration.precision),
                'recall': float(self.best_iteration.recall),
                'thresholds': self.best_iteration.thresholds,
                'improvement_score': float(self.best_iteration.improvement_score)
            }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"💾 Iteration history saved to {filepath}")

    def optimize(self) -> OptimizationIteration:
        """Run adaptive optimization process"""
        self.logger.info("🚀 Starting adaptive threshold optimization")
        
        for iteration_num in range(self.max_iterations):
            # Generate next parameter set
            params = self.generate_next_parameters()
            
            # Evaluate parameters
            iteration = self.evaluate_parameters(params)
            
            # Update history and best result
            self.iteration_history.append(iteration)
            self.update_best_iteration(iteration)
            
            # Save progress
            self.save_iteration_history()
            
            # Check convergence
            if self.check_convergence():
                break
            
            # Adaptive exploration rate (decrease over time)
            self.exploration_rate = max(0.1, self.exploration_rate * 0.95)
        
        self.logger.info("✅ Optimization completed")
        if self.best_iteration:
            self.logger.info(f"🏆 Best result: Precision={self.best_iteration.precision:.3f}, "
                           f"Recall={self.best_iteration.recall:.3f}")
            self.logger.info(f"🎯 Best thresholds: {self.best_iteration.thresholds}")
        
        return self.best_iteration

def main():
    """Main optimization execution"""
    print("🤖 ADAPTIVE THRESHOLD OPTIMIZER FOR INFRARED DROWSINESS DETECTION")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = AdaptiveThresholdOptimizer(
        target_precision=0.90,
        max_iterations=50
    )
    
    # Run optimization
    best_result = optimizer.optimize()
    
    if best_result:
        print(f"\n🏆 OPTIMIZATION COMPLETE")
        print(f"Best Precision: {best_result.precision:.3f}")
        print(f"Best Recall: {best_result.recall:.3f}")
        print(f"Best Thresholds: {best_result.thresholds}")
        print(f"Achieved in {best_result.iteration} iterations")
    else:
        print("\n❌ Optimization failed to find suitable parameters")

if __name__ == "__main__":
    main()
