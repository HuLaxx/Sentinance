"""
Feast Feature Store Configuration

Manages ML features for:
- Consistent feature serving between training and inference
- Point-in-time correct feature retrieval
- Feature versioning and lineage

Setup:
    pip install feast
    feast init feature_repo
    feast apply
"""

from datetime import timedelta
from typing import Dict, List, Optional
import os

# Check if Feast is available
try:
    from feast import Entity, Feature, FeatureView, FileSource, ValueType
    from feast.feature_store import FeatureStore
    from feast.on_demand_feature_view import on_demand_feature_view
    from feast.field import Field
    from feast.types import Float64, Int64, String
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    print("⚠️ Feast not installed. Run: pip install feast")


# ============================================
# FEATURE STORE CONFIGURATION
# ============================================

FEATURE_REPO_PATH = os.path.join(os.path.dirname(__file__), "feature_repo")


# ============================================
# ENTITY DEFINITIONS
# ============================================

if FEAST_AVAILABLE:
    # Crypto asset entity
    crypto_asset = Entity(
        name="crypto_asset",
        join_keys=["symbol"],
        description="Cryptocurrency trading pair (e.g., BTCUSDT)",
    )


# ============================================
# FEATURE DEFINITIONS (as Python config)
# ============================================

FEATURE_DEFINITIONS = {
    "price_features": {
        "description": "Real-time and historical price features",
        "features": [
            {"name": "close_price", "dtype": "float64"},
            {"name": "open_price", "dtype": "float64"},
            {"name": "high_price", "dtype": "float64"},
            {"name": "low_price", "dtype": "float64"},
            {"name": "volume", "dtype": "float64"},
            {"name": "price_change_pct", "dtype": "float64"},
        ],
        "ttl_days": 1,
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "features": [
            {"name": "rsi_14", "dtype": "float64"},
            {"name": "macd", "dtype": "float64"},
            {"name": "macd_signal", "dtype": "float64"},
            {"name": "sma_20", "dtype": "float64"},
            {"name": "sma_50", "dtype": "float64"},
            {"name": "ema_12", "dtype": "float64"},
            {"name": "bollinger_upper", "dtype": "float64"},
            {"name": "bollinger_lower", "dtype": "float64"},
        ],
        "ttl_days": 1,
    },
    "market_features": {
        "description": "Market-wide aggregated features",
        "features": [
            {"name": "btc_dominance", "dtype": "float64"},
            {"name": "total_market_cap", "dtype": "float64"},
            {"name": "fear_greed_index", "dtype": "int64"},
            {"name": "num_gainers", "dtype": "int64"},
            {"name": "num_losers", "dtype": "int64"},
        ],
        "ttl_days": 1,
    },
}


# ============================================
# FEATURE STORE SERVICE
# ============================================

class FeatureStoreService:
    """
    Service for interacting with Feast feature store.
    """
    
    def __init__(self, repo_path: str = FEATURE_REPO_PATH):
        self.repo_path = repo_path
        self.store = None
        
        if FEAST_AVAILABLE and os.path.exists(repo_path):
            try:
                self.store = FeatureStore(repo_path=repo_path)
            except Exception as e:
                print(f"Warning: Could not initialize feature store: {e}")
    
    def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict],
    ) -> Optional[Dict]:
        """
        Get features from online store for real-time inference.
        
        Args:
            feature_refs: List of feature references (e.g., ['price_features:close_price'])
            entity_rows: List of entity key dicts (e.g., [{'symbol': 'BTCUSDT'}])
            
        Returns:
            Dictionary of feature values
        """
        if not self.store:
            return self._get_mock_features(feature_refs, entity_rows)
        
        try:
            response = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            return response.to_dict()
        except Exception as e:
            print(f"Error getting online features: {e}")
            return None
    
    def _get_mock_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict],
    ) -> Dict:
        """Return mock features when Feast is not available."""
        import random
        
        result = {"symbol": [row.get("symbol", "BTCUSDT") for row in entity_rows]}
        
        for ref in feature_refs:
            feature_name = ref.split(":")[-1] if ":" in ref else ref
            
            # Generate mock values based on feature name
            if "price" in feature_name.lower() or "close" in feature_name.lower():
                result[feature_name] = [random.uniform(90000, 100000) for _ in entity_rows]
            elif "rsi" in feature_name.lower():
                result[feature_name] = [random.uniform(30, 70) for _ in entity_rows]
            elif "volume" in feature_name.lower():
                result[feature_name] = [random.uniform(1e9, 5e9) for _ in entity_rows]
            elif "index" in feature_name.lower():
                result[feature_name] = [random.randint(20, 80) for _ in entity_rows]
            else:
                result[feature_name] = [random.uniform(0, 100) for _ in entity_rows]
        
        return result
    
    def materialize_features(
        self,
        start_date: str,
        end_date: str,
    ) -> bool:
        """
        Materialize features to online store.
        
        Args:
            start_date: Start date for materialization (ISO format)
            end_date: End date for materialization (ISO format)
        """
        if not self.store:
            print("Feature store not initialized")
            return False
        
        try:
            from datetime import datetime
            
            self.store.materialize(
                start_date=datetime.fromisoformat(start_date),
                end_date=datetime.fromisoformat(end_date),
            )
            return True
        except Exception as e:
            print(f"Error materializing features: {e}")
            return False
    
    def list_feature_views(self) -> List[str]:
        """List all registered feature views."""
        if not self.store:
            return list(FEATURE_DEFINITIONS.keys())
        
        try:
            return [fv.name for fv in self.store.list_feature_views()]
        except Exception as e:
            print(f"Error listing feature views: {e}")
            return []


def generate_feast_config():
    """Generate Feast feature_store.yaml configuration."""
    config = """
project: sentinance
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
"""
    return config


def generate_feature_definitions():
    """Generate Python feature definitions for Feast."""
    if not FEAST_AVAILABLE:
        return "# Feast not installed - cannot generate definitions"
    
    definitions = '''
"""
Feast Feature Definitions for Sentinance

Apply with: feast apply
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

# Entity
crypto_asset = Entity(
    name="crypto_asset",
    join_keys=["symbol"],
)

# Data source
price_source = FileSource(
    path="data/price_features.parquet",
    timestamp_field="timestamp",
)

# Feature view
price_features = FeatureView(
    name="price_features",
    entities=[crypto_asset],
    ttl=timedelta(days=1),
    schema=[
        Field(name="close_price", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="price_change_pct", dtype=Float64),
    ],
    source=price_source,
)
'''
    return definitions


# Singleton
_feature_service: Optional[FeatureStoreService] = None


def get_feature_service() -> FeatureStoreService:
    """Get feature store service singleton."""
    global _feature_service
    if _feature_service is None:
        _feature_service = FeatureStoreService()
    return _feature_service


if __name__ == "__main__":
    # Test feature store
    service = get_feature_service()
    
    # Get features
    features = service.get_online_features(
        feature_refs=[
            "price_features:close_price",
            "technical_indicators:rsi_14",
        ],
        entity_rows=[
            {"symbol": "BTCUSDT"},
            {"symbol": "ETHUSDT"},
        ]
    )
    
    print("Features retrieved:")
    for k, v in features.items():
        print(f"  {k}: {v}")
