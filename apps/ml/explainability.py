"""
Model Explainability with SHAP and LIME

Provides interpretable explanations for ML model predictions.

Features:
- SHAP values for global and local explanations
- LIME for instance-level interpretability  
- Feature importance visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import structlog

log = structlog.get_logger()

# Check for SHAP/LIME availability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("SHAP not installed. Run: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    log.warning("LIME not installed. Run: pip install lime")


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    method: str  # 'shap' or 'lime'
    instance_explanation: Dict[str, float]  # feature -> contribution
    prediction: float
    base_value: float  # expected value
    feature_importance: Dict[str, float]  # global importance
    
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "instance_explanation": self.instance_explanation,
            "prediction": round(self.prediction, 4),
            "base_value": round(self.base_value, 4),
            "feature_importance": {
                k: round(v, 4) for k, v in self.feature_importance.items()
            },
        }


class ModelExplainer:
    """
    Explains model predictions using SHAP and LIME.
    """
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Args:
            model: Trained model with predict method
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
    
    def init_shap(self, background_data: np.ndarray):
        """Initialize SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            log.warning("SHAP not available")
            return
        
        try:
            # Use KernelExplainer for model-agnostic explanations
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(background_data, 100)  # Sample for efficiency
            )
            log.info("shap_explainer_initialized")
        except Exception as e:
            log.error("shap_init_failed", error=str(e))
    
    def init_lime(self, training_data: np.ndarray):
        """Initialize LIME explainer with training data."""
        if not LIME_AVAILABLE:
            log.warning("LIME not available")
            return
        
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True
            )
            log.info("lime_explainer_initialized")
        except Exception as e:
            log.error("lime_init_failed", error=str(e))
    
    def explain_shap(self, instance: np.ndarray) -> Optional[ExplanationResult]:
        """
        Explain a prediction using SHAP.
        
        Args:
            instance: Single input instance (1D array)
            
        Returns:
            ExplanationResult with SHAP values
        """
        if self.shap_explainer is None:
            log.warning("SHAP explainer not initialized")
            return None
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(instance.reshape(1, -1))
            
            # Create explanation dict
            explanation = dict(zip(
                self.feature_names,
                shap_values[0].tolist()
            ))
            
            # Sort by absolute contribution
            sorted_explanation = dict(sorted(
                explanation.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))
            
            # Get prediction
            prediction = float(self.model.predict(instance.reshape(1, -1))[0])
            
            return ExplanationResult(
                method="shap",
                instance_explanation=sorted_explanation,
                prediction=prediction,
                base_value=float(self.shap_explainer.expected_value),
                feature_importance=self._calculate_global_importance(shap_values[0])
            )
        except Exception as e:
            log.error("shap_explain_failed", error=str(e))
            return None
    
    def explain_lime(
        self, 
        instance: np.ndarray,
        num_features: int = 10
    ) -> Optional[ExplanationResult]:
        """
        Explain a prediction using LIME.
        
        Args:
            instance: Single input instance (1D array)
            num_features: Number of top features to include
            
        Returns:
            ExplanationResult with LIME explanation
        """
        if self.lime_explainer is None:
            log.warning("LIME explainer not initialized")
            return None
        
        try:
            # Get LIME explanation
            exp = self.lime_explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )
            
            # Convert to dict
            explanation = dict(exp.as_list())
            
            # Get prediction
            prediction = float(self.model.predict(instance.reshape(1, -1))[0])
            
            return ExplanationResult(
                method="lime",
                instance_explanation=explanation,
                prediction=prediction,
                base_value=float(exp.intercept[1]) if hasattr(exp, 'intercept') else 0,
                feature_importance={}  # LIME doesn't provide global importance
            )
        except Exception as e:
            log.error("lime_explain_failed", error=str(e))
            return None
    
    def _calculate_global_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate global feature importance from SHAP values."""
        importance = dict(zip(
            self.feature_names,
            np.abs(shap_values).tolist()
        ))
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def explain_prediction(
    model: Any,
    instance: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    method: str = "shap"
) -> Optional[ExplanationResult]:
    """
    Convenience function to explain a single prediction.
    
    Args:
        model: Trained model
        instance: Input instance to explain
        feature_names: List of feature names
        background_data: Background data for SHAP (optional)
        method: 'shap' or 'lime'
        
    Returns:
        ExplanationResult
    """
    explainer = ModelExplainer(model, feature_names)
    
    if method == "shap":
        if background_data is not None:
            explainer.init_shap(background_data)
        return explainer.explain_shap(instance)
    else:
        if background_data is not None:
            explainer.init_lime(background_data)
        return explainer.explain_lime(instance)


if __name__ == "__main__":
    # Example usage with a simple model
    from sklearn.ensemble import RandomForestRegressor
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + np.random.randn(1000) * 0.1
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Explain a prediction
    feature_names = ['price_lag1', 'price_lag2', 'volume', 'rsi', 'macd']
    instance = X[0]
    
    explainer = ModelExplainer(model, feature_names)
    explainer.init_shap(X[:100])
    
    result = explainer.explain_shap(instance)
    if result:
        print("SHAP Explanation:")
        print(f"  Prediction: {result.prediction}")
        print(f"  Base value: {result.base_value}")
        print(f"  Top features:")
        for feat, val in list(result.instance_explanation.items())[:3]:
            print(f"    {feat}: {val:.4f}")
