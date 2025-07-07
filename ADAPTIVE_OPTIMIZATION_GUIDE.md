# Adaptive Threshold Optimization System

## Overview

The new **Adaptive Threshold Optimizer** replaces the static grid search approach with a true machine learning system that intelligently discovers optimal thresholds for infrared drowsiness detection.

## Key Improvements

### 🧠 Intelligent Parameter Exploration
- **Dynamic Generation**: Parameters are generated based on performance feedback, not pre-defined lists
- **Adaptive Learning**: Each iteration learns from previous results to propose better parameters
- **Multi-Strategy Exploration**: Uses exploration vs exploitation balance for optimal search

### 🎯 Performance-Driven Optimization
- **Target-Oriented**: Specifically optimizes toward 90% precision target
- **Improvement Scoring**: Custom scoring function prioritizes precision while maintaining recall
- **Convergence Detection**: Automatically stops when target is reached or no improvement possible

### 🔄 Iterative Learning Process
1. **Initial Parameters**: Starts with domain-knowledge based parameters
2. **Evaluation**: Tests parameters on full 28-video dataset
3. **Learning**: Analyzes results and generates improved parameters
4. **Adaptation**: Adjusts exploration strategy based on progress
5. **Convergence**: Continues until target achieved or stagnation detected

## Architecture

### Core Components

#### `AdaptiveThresholdOptimizer`
- Main optimization engine
- Manages learning process and parameter generation
- Tracks iteration history and best results

#### `OptimizationIteration`
- Data structure for each optimization iteration
- Stores parameters, results, and metadata
- Enables learning from historical performance

### Parameter Exploration Strategies

#### 1. **Exploration Phase** (30% of iterations)
- **Random Exploration**: Explores new parameter space regions
- **Gradient Exploration**: Small perturbations around best parameters
- **Boundary Exploration**: Tests edge cases and parameter limits

#### 2. **Exploitation Phase** (70% of iterations)
- **Best Parameter Refinement**: Focuses on promising parameter regions
- **Weighted Averaging**: Combines top-performing parameter sets
- **Local Optimization**: Fine-tunes around best known solutions

### Learning Algorithm

```python
# Improvement Score Calculation
precision_score = min(precision / target_precision, 1.0)
recall_penalty = max(0, 0.3 - recall) * 2
improvement_score = precision_score - recall_penalty
```

## Usage

### Basic Usage
```python
from adaptive_threshold_optimizer import AdaptiveThresholdOptimizer

# Initialize optimizer
optimizer = AdaptiveThresholdOptimizer(
    target_precision=0.90,
    max_iterations=30
)

# Run optimization
best_result = optimizer.optimize()

# Get optimal thresholds
optimal_thresholds = best_result.thresholds
```

### Advanced Configuration
```python
optimizer = AdaptiveThresholdOptimizer(
    target_precision=0.90,
    max_iterations=50
)

# Customize parameter bounds
optimizer.param_bounds = {
    'fatigue_threshold': (0.10, 0.90),
    'perclos_threshold': (0.03, 0.50),
    'confidence_threshold': (0.40, 0.98)
}

# Adjust learning parameters
optimizer.exploration_rate = 0.4  # More exploration
optimizer.convergence_threshold = 0.005  # Stricter convergence
```

## Output Files

### `adaptive_optimization_history.json`
Complete iteration history with:
- All tested parameter combinations
- Performance metrics for each iteration
- Exploration strategy used
- Timestamp and metadata

### Example Output Structure
```json
{
  "optimization_config": {
    "target_precision": 0.90,
    "max_iterations": 30,
    "param_bounds": {...}
  },
  "iterations": [
    {
      "iteration": 1,
      "precision": 0.75,
      "recall": 0.45,
      "thresholds": {
        "fatigue_threshold": 0.35,
        "perclos_threshold": 0.15,
        "confidence_threshold": 0.70
      },
      "improvement_score": 0.73,
      "exploration_type": "initial"
    }
  ],
  "best_iteration": {
    "iteration": 15,
    "precision": 0.92,
    "recall": 0.38,
    "thresholds": {...}
  }
}
```

## Advantages Over Static Grid Search

### 🎯 **Efficiency**
- Tests only promising parameter combinations
- Converges faster to optimal solutions
- Avoids exhaustive search of poor parameter regions

### 🧠 **Intelligence**
- Learns from each iteration
- Adapts search strategy based on results
- Balances exploration of new regions with exploitation of good regions

### 📈 **Performance**
- Specifically optimized for precision target
- Maintains recall while maximizing precision
- Provides convergence guarantees

### 🔄 **Adaptability**
- Automatically adjusts exploration rate
- Detects stagnation and convergence
- Handles different optimization landscapes

## Running the Optimizer

### Full Optimization
```bash
python adaptive_threshold_optimizer.py
```

### Test Basic Functionality
```bash
python test_adaptive_optimizer.py
```

## Expected Results

The adaptive optimizer should:
1. **Start** with reasonable initial parameters
2. **Explore** parameter space intelligently
3. **Learn** from each iteration's results
4. **Converge** to optimal thresholds achieving 90% precision
5. **Complete** in 15-30 iterations (vs 15 static configurations)

## Monitoring Progress

The optimizer provides detailed logging:
- 🔍 Current iteration and parameters being tested
- 📊 Performance results (precision, recall, F1)
- 🎉 New best results discovered
- 🎯 Target achievement or convergence detection

## Integration

Once optimal thresholds are discovered, update the production system:

```python
# Apply optimal thresholds to detector
detector.FATIGUE_THRESHOLD = best_result.thresholds['fatigue_threshold']
detector.PERCLOS_THRESHOLD = best_result.thresholds['perclos_threshold']
detector.MIN_CONFIDENCE_THRESHOLD = best_result.thresholds['confidence_threshold']
```

This adaptive system represents a significant advancement from static configuration testing to true machine learning-based parameter optimization.
