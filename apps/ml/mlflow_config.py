"""
MLflow Configuration and Model Registry

Setup for experiment tracking and model versioning.
"""

import os
from typing import Optional, Dict, Any
import structlog

log = structlog.get_logger()

# Check if mlflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed. Run: pip install mlflow")


# ===========================================
# CONFIGURATION
# ===========================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "sentinance-models"


def setup_mlflow():
    """Initialize MLflow tracking."""
    if not MLFLOW_AVAILABLE:
        return False
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        log.info("mlflow_initialized", uri=MLFLOW_TRACKING_URI)
        return True
    except Exception as e:
        log.warning("mlflow_setup_failed", error=str(e))
        return False


# ===========================================
# EXPERIMENT TRACKING
# ===========================================

class ExperimentTracker:
    """Track ML experiments with MLflow."""
    
    def __init__(self):
        self.enabled = setup_mlflow() if MLFLOW_AVAILABLE else False
    
    def start_run(self, run_name: str) -> Optional[str]:
        """Start a new training run."""
        if not self.enabled:
            return None
        
        run = mlflow.start_run(run_name=run_name)
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.enabled:
            return
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics."""
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, model_name: str, framework: str = "pytorch"):
        """Log trained model."""
        if not self.enabled:
            return
        
        if framework == "pytorch":
            mlflow.pytorch.log_model(model, model_name)
        elif framework == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
    
    def end_run(self):
        """End the current run."""
        if self.enabled:
            mlflow.end_run()
    
    def register_model(self, run_id: str, model_name: str, stage: str = "Staging"):
        """Register model in the model registry."""
        if not self.enabled:
            return None
        
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/{model_name}"
        
        # Register
        result = mlflow.register_model(model_uri, model_name)
        
        # Transition to stage
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage
        )
        
        log.info("model_registered", name=model_name, version=result.version, stage=stage)
        return result.version


# ===========================================
# MODEL REGISTRY
# ===========================================

def get_production_model(model_name: str):
    """Load the production version of a model."""
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        log.warning("model_load_failed", model=model_name, error=str(e))
        return None


def list_models():
    """List all registered models."""
    if not MLFLOW_AVAILABLE:
        return []
    
    client = MlflowClient()
    return [m.name for m in client.search_registered_models()]


# ===========================================
# TRAINING EXAMPLE
# ===========================================

def train_with_tracking(
    model,
    train_data,
    val_data,
    model_name: str = "price_predictor",
    epochs: int = 100,
    **params
):
    """
    Example training loop with MLflow tracking.
    
    Usage:
        train_with_tracking(
            model=LSTMModel(),
            train_data=train_loader,
            val_data=val_loader,
            model_name="lstm_btc",
            epochs=50,
            learning_rate=0.001,
            hidden_size=64
        )
    """
    tracker = ExperimentTracker()
    
    run_id = tracker.start_run(f"{model_name}_training")
    tracker.log_params(params)
    
    try:
        for epoch in range(epochs):
            # Training step (placeholder)
            train_loss = 0.1 * (1 - epoch / epochs)  # Simulated
            val_loss = 0.12 * (1 - epoch / epochs)
            
            tracker.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, step=epoch)
        
        # Log final model
        tracker.log_model(model, model_name)
        
        # Register to staging
        if run_id:
            tracker.register_model(run_id, model_name, "Staging")
        
    finally:
        tracker.end_run()
    
    return run_id
