"""
Model Drift Detection

Monitors for:
- Data drift: Input feature distribution changes
- Prediction drift: Model output distribution changes
- Concept drift: Relationship between features and target changes

Uses statistical tests to detect significant drift.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog

log = structlog.get_logger()


@dataclass
class DriftResult:
    """Result of a drift detection test."""
    metric_name: str
    drift_detected: bool
    p_value: float
    threshold: float
    severity: str  # 'none', 'low', 'medium', 'high'
    timestamp: str
    details: Dict
    
    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "drift_detected": self.drift_detected,
            "p_value": round(self.p_value, 4),
            "threshold": self.threshold,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class DriftDetector:
    """
    Drift detection using statistical tests.
    
    Methods:
    - Kolmogorov-Smirnov test for continuous distributions
    - Chi-squared test for categorical distributions
    - Population Stability Index (PSI) for overall drift
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: p-value threshold for drift detection (default 0.05)
        """
        self.threshold = threshold
        self.reference_data: Dict[str, np.ndarray] = {}
    
    def set_reference(self, data: Dict[str, np.ndarray]):
        """Set reference (baseline) data for comparison."""
        self.reference_data = data
        log.info("drift_reference_set", features=list(data.keys()))
    
    def detect_feature_drift(
        self, 
        current_data: Dict[str, np.ndarray]
    ) -> List[DriftResult]:
        """
        Detect drift in input features using KS test.
        
        Args:
            current_data: Current feature distributions
            
        Returns:
            List of DriftResult for each feature
        """
        results = []
        
        for feature, current_values in current_data.items():
            if feature not in self.reference_data:
                continue
            
            reference_values = self.reference_data[feature]
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(reference_values, current_values)
            
            drift_detected = p_value < self.threshold
            
            # Determine severity
            if not drift_detected:
                severity = "none"
            elif p_value < 0.001:
                severity = "high"
            elif p_value < 0.01:
                severity = "medium"
            else:
                severity = "low"
            
            results.append(DriftResult(
                metric_name=f"feature_drift_{feature}",
                drift_detected=drift_detected,
                p_value=p_value,
                threshold=self.threshold,
                severity=severity,
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "ks_statistic": round(statistic, 4),
                    "reference_mean": round(float(reference_values.mean()), 4),
                    "current_mean": round(float(current_values.mean()), 4),
                    "reference_std": round(float(reference_values.std()), 4),
                    "current_std": round(float(current_values.std()), 4),
                }
            ))
        
        return results
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> DriftResult:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Baseline prediction distribution
            current_predictions: Current prediction distribution
            
        Returns:
            DriftResult for prediction drift
        """
        statistic, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        
        drift_detected = p_value < self.threshold
        
        if not drift_detected:
            severity = "none"
        elif p_value < 0.001:
            severity = "high"
        elif p_value < 0.01:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftResult(
            metric_name="prediction_drift",
            drift_detected=drift_detected,
            p_value=p_value,
            threshold=self.threshold,
            severity=severity,
            timestamp=datetime.utcnow().isoformat(),
            details={
                "ks_statistic": round(statistic, 4),
                "reference_mean": round(float(reference_predictions.mean()), 4),
                "current_mean": round(float(current_predictions.mean()), 4),
            }
        )
    
    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        buckets: int = 10
    ) -> Tuple[float, str]:
        """
        Calculate Population Stability Index (PSI).
        
        PSI interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change
        
        Args:
            reference: Reference distribution
            current: Current distribution
            buckets: Number of buckets for binning
            
        Returns:
            (PSI value, interpretation)
        """
        # Create buckets from reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate frequencies
        ref_counts = np.histogram(reference, breakpoints)[0]
        cur_counts = np.histogram(current, breakpoints)[0]
        
        # Convert to proportions (add small value to avoid log(0))
        ref_prop = (ref_counts + 0.0001) / len(reference)
        cur_prop = (cur_counts + 0.0001) / len(current)
        
        # Calculate PSI
        psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
        
        if psi < 0.1:
            interpretation = "No significant change"
        elif psi < 0.25:
            interpretation = "Moderate change - investigate"
        else:
            interpretation = "Significant change - action required"
        
        return round(psi, 4), interpretation


def run_drift_check(
    symbol: str,
    reference_prices: np.ndarray,
    current_prices: np.ndarray,
    reference_predictions: Optional[np.ndarray] = None,
    current_predictions: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run full drift detection suite.
    
    Args:
        symbol: Asset symbol
        reference_prices: Historical baseline prices
        current_prices: Recent prices
        reference_predictions: Optional baseline predictions
        current_predictions: Optional current predictions
        
    Returns:
        Comprehensive drift report
    """
    detector = DriftDetector(threshold=0.05)
    
    results = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": [],
        "overall_status": "healthy",
    }
    
    # Feature drift (price distribution)
    detector.set_reference({"price": reference_prices})
    feature_results = detector.detect_feature_drift({"price": current_prices})
    
    for r in feature_results:
        results["checks"].append(r.to_dict())
        if r.drift_detected and r.severity in ["medium", "high"]:
            results["overall_status"] = "drift_detected"
    
    # PSI
    psi, psi_interpretation = detector.calculate_psi(reference_prices, current_prices)
    results["psi"] = {
        "value": psi,
        "interpretation": psi_interpretation,
    }
    
    if psi >= 0.25:
        results["overall_status"] = "significant_drift"
    
    # Prediction drift (if provided)
    if reference_predictions is not None and current_predictions is not None:
        pred_result = detector.detect_prediction_drift(
            reference_predictions, current_predictions
        )
        results["checks"].append(pred_result.to_dict())
    
    log.info("drift_check_complete", symbol=symbol, status=results["overall_status"])
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate reference data (stable period)
    reference = np.random.normal(95000, 2000, 1000)
    
    # Simulate current data (with drift)
    current = np.random.normal(98000, 3000, 200)  # Mean and variance shifted
    
    result = run_drift_check("BTCUSDT", reference, current)
    
    import json
    print(json.dumps(result, indent=2))
