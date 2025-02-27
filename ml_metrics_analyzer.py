import logging
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MLMetrics:
    sensitivity: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

class MLMetricsAnalyzer:
    """Analyzer for ML metrics separate from drowsiness detection."""
    
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
    def calculate_metrics(self) -> MLMetrics:
        """Calculate sensitivity (recall) and accuracy."""
        try:
            sensitivity = self.true_positives / (self.true_positives + self.false_negatives)
            accuracy = (self.true_positives + self.true_negatives) / (
                self.true_positives + self.true_negatives + 
                self.false_positives + self.false_negatives
            )
        except ZeroDivisionError:
            sensitivity = 0.0
            accuracy = 0.0
            
        return MLMetrics(
            sensitivity=sensitivity * 100,  # Convert to percentage
            accuracy=accuracy * 100,  # Convert to percentage
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            true_negatives=self.true_negatives,
            false_negatives=self.false_negatives
        )
        
    def analyze(self, is_drowsy: bool, take_type: Optional[bool] = None) -> Dict:
        """
        Analyze ML metrics based on drowsiness detection and take type.
        
        Args:
            is_drowsy (bool): Whether drowsiness was detected
            take_type (bool): True for true alarm, False for false alarm, None if unknown
            
        Returns:
            dict: Analysis results with metrics
        """
        metrics_updated = False
        
        if take_type is not None:
            if take_type and is_drowsy:
                self.true_positives += 1
            elif take_type and not is_drowsy:
                self.false_negatives += 1
            elif not take_type and is_drowsy:
                self.false_positives += 1
            else:
                self.true_negatives += 1
            metrics_updated = True
            
        metrics = self.calculate_metrics()
        
        return {
            'sensitivity': metrics.sensitivity,
            'accuracy': metrics.accuracy,
            'metrics_updated': metrics_updated,
            'confusion_matrix': {
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'true_negatives': metrics.true_negatives,
                'false_negatives': metrics.false_negatives
            }
        }

    def reset_metrics(self):
        """Reset all metrics counters."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0