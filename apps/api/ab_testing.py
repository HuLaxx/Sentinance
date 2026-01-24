"""
A/B Testing Framework for ML Models

Enables controlled experiments to compare model versions:
- Traffic splitting between model variants
- Statistical significance testing
- Automatic winner detection
- Experiment logging and analysis
"""

import hashlib
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from scipy import stats
import numpy as np

log = structlog.get_logger()


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class Variant:
    """A variant in an A/B test."""
    name: str
    model_version: str
    traffic_percent: float  # 0-100
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    
    @property
    def sample_size(self) -> int:
        return len(self.predictions)
    
    @property
    def mean_error(self) -> float:
        if not self.predictions or not self.actuals:
            return 0.0
        errors = [abs(p - a) for p, a in zip(self.predictions, self.actuals)]
        return sum(errors) / len(errors)
    
    @property
    def mean_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)


@dataclass
class Experiment:
    """An A/B experiment configuration."""
    id: str
    name: str
    description: str
    variants: List[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    min_sample_size: int = 1000
    significance_level: float = 0.05
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "variants": [
                {
                    "name": v.name,
                    "model_version": v.model_version,
                    "traffic_percent": v.traffic_percent,
                    "sample_size": v.sample_size,
                    "mean_error": round(v.mean_error, 4),
                    "mean_latency": round(v.mean_latency, 4),
                }
                for v in self.variants
            ],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ABTestingService:
    """
    Service for managing A/B tests between model versions.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiment_id: Optional[str] = None
    
    def create_experiment(
        self,
        name: str,
        description: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 50.0,
        min_sample_size: int = 1000,
    ) -> Experiment:
        """
        Create a new A/B experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            control_model: Control (baseline) model version
            treatment_model: Treatment (new) model version
            traffic_split: Percent traffic to treatment (0-100)
            min_sample_size: Minimum samples before analysis
        """
        exp_id = hashlib.md5(f"{name}-{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
        
        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            variants=[
                Variant(
                    name="control",
                    model_version=control_model,
                    traffic_percent=100 - traffic_split,
                ),
                Variant(
                    name="treatment",
                    model_version=treatment_model,
                    traffic_percent=traffic_split,
                ),
            ],
            min_sample_size=min_sample_size,
        )
        
        self.experiments[exp_id] = experiment
        log.info("experiment_created", id=exp_id, name=name)
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.utcnow().isoformat()
        self.active_experiment_id = experiment_id
        
        log.info("experiment_started", id=experiment_id)
        return True
    
    def assign_variant(self, user_id: str) -> Optional[Variant]:
        """
        Assign a user to a variant based on consistent hashing.
        
        Args:
            user_id: User identifier for consistent assignment
            
        Returns:
            Assigned Variant or None if no active experiment
        """
        if not self.active_experiment_id:
            return None
        
        exp = self.experiments.get(self.active_experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None
        
        # Consistent hashing for reproducible assignment
        hash_val = int(hashlib.md5(f"{exp.id}-{user_id}".encode()).hexdigest(), 16)
        bucket = (hash_val % 100) + 1  # 1-100
        
        cumulative = 0
        for variant in exp.variants:
            cumulative += variant.traffic_percent
            if bucket <= cumulative:
                return variant
        
        return exp.variants[-1]  # Fallback to last variant
    
    def record_observation(
        self,
        variant_name: str,
        prediction: float,
        actual: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ):
        """Record an observation for a variant."""
        if not self.active_experiment_id:
            return
        
        exp = self.experiments.get(self.active_experiment_id)
        if not exp:
            return
        
        for variant in exp.variants:
            if variant.name == variant_name:
                variant.predictions.append(prediction)
                if actual is not None:
                    variant.actuals.append(actual)
                if latency_ms is not None:
                    variant.latencies.append(latency_ms)
                break
    
    def analyze_experiment(self, experiment_id: str) -> Dict:
        """
        Analyze experiment results with statistical testing.
        
        Returns:
            Analysis results including significance and winner
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}
        
        if len(exp.variants) != 2:
            return {"error": "Only 2-variant tests supported"}
        
        control = exp.variants[0]
        treatment = exp.variants[1]
        
        result = {
            "experiment_id": experiment_id,
            "experiment_name": exp.name,
            "status": exp.status.value,
            "control": {
                "sample_size": control.sample_size,
                "mean_error": round(control.mean_error, 4),
                "mean_latency": round(control.mean_latency, 4),
            },
            "treatment": {
                "sample_size": treatment.sample_size,
                "mean_error": round(treatment.mean_error, 4),
                "mean_latency": round(treatment.mean_latency, 4),
            },
            "sufficient_sample_size": min(control.sample_size, treatment.sample_size) >= exp.min_sample_size,
        }
        
        # Statistical significance testing (if we have actuals)
        if control.actuals and treatment.actuals:
            # Calculate prediction errors
            control_errors = [abs(p - a) for p, a in zip(control.predictions[:len(control.actuals)], control.actuals)]
            treatment_errors = [abs(p - a) for p, a in zip(treatment.predictions[:len(treatment.actuals)], treatment.actuals)]
            
            if len(control_errors) >= 30 and len(treatment_errors) >= 30:
                # T-test for difference in means
                t_stat, p_value = stats.ttest_ind(control_errors, treatment_errors)
                
                result["statistical_test"] = {
                    "test": "t-test",
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_value, 4),
                    "significant": p_value < exp.significance_level,
                }
                
                # Determine winner
                if p_value < exp.significance_level:
                    if np.mean(treatment_errors) < np.mean(control_errors):
                        result["winner"] = "treatment"
                        result["improvement"] = round(
                            (np.mean(control_errors) - np.mean(treatment_errors)) / np.mean(control_errors) * 100, 2
                        )
                    else:
                        result["winner"] = "control"
                else:
                    result["winner"] = "inconclusive"
        
        return result
    
    def complete_experiment(self, experiment_id: str, winner: str) -> bool:
        """Mark experiment as completed with a winner."""
        if experiment_id not in self.experiments:
            return False
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.utcnow().isoformat()
        
        if self.active_experiment_id == experiment_id:
            self.active_experiment_id = None
        
        log.info("experiment_completed", id=experiment_id, winner=winner)
        return True


# Singleton instance
_ab_service: Optional[ABTestingService] = None


def get_ab_service() -> ABTestingService:
    """Get the A/B testing service singleton."""
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestingService()
    return _ab_service


if __name__ == "__main__":
    # Example usage
    service = get_ab_service()
    
    # Create experiment
    exp = service.create_experiment(
        name="LSTM v2 Test",
        description="Testing new LSTM architecture",
        control_model="lstm_v1.0",
        treatment_model="lstm_v2.0",
        traffic_split=50,
        min_sample_size=100,
    )
    
    # Start experiment
    service.start_experiment(exp.id)
    
    # Simulate observations
    np.random.seed(42)
    for i in range(200):
        user_id = f"user_{i}"
        variant = service.assign_variant(user_id)
        
        if variant:
            # Simulate prediction and actual
            if variant.name == "control":
                pred = np.random.normal(100, 10)
                actual = pred + np.random.normal(0, 5)
            else:
                pred = np.random.normal(100, 8)  # Better model
                actual = pred + np.random.normal(0, 4)
            
            service.record_observation(
                variant.name,
                prediction=pred,
                actual=actual,
                latency_ms=np.random.uniform(10, 50)
            )
    
    # Analyze results
    analysis = service.analyze_experiment(exp.id)
    
    import json
    print(json.dumps(analysis, indent=2))
