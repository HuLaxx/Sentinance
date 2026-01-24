"""
Intelligent Model Router

Routes queries to the optimal model based on:
1. Query complexity analysis
2. Cost optimization
3. Latency requirements
4. Model specialization

This demonstrates:
- Intelligent routing patterns
- Cost-aware model selection
- Multi-model orchestration
- Query classification
"""

import os
import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import structlog

log = structlog.get_logger()


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"          # Factual, short answers
    MODERATE = "moderate"      # Analysis, comparisons
    COMPLEX = "complex"        # Multi-step reasoning
    EXPERT = "expert"          # Deep domain expertise


class ModelTier(Enum):
    """Model performance tiers."""
    FAST = "fast"              # Low latency, basic tasks
    BALANCED = "balanced"      # Good balance of speed/quality
    POWERFUL = "powerful"      # High quality, slower
    SPECIALIZED = "specialized"  # Domain-specific


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str
    tier: ModelTier
    cost_per_1k_tokens: float
    max_context: int
    specialties: List[str]
    latency_ms: int  # Average latency


# Available models registry
MODELS = {
    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash",
        provider="google",
        tier=ModelTier.BALANCED,
        cost_per_1k_tokens=0.0003,
        max_context=1000000,
        specialties=["general", "code", "analysis"],
        latency_ms=800,
    ),
    "gemini-2.0-pro": ModelConfig(
        name="gemini-2.0-pro",
        provider="google",
        tier=ModelTier.POWERFUL,
        cost_per_1k_tokens=0.005,
        max_context=2000000,
        specialties=["reasoning", "complex", "creative"],
        latency_ms=2000,
    ),
    "llama-3.3-70b-versatile": ModelConfig(
        name="llama-3.3-70b-versatile",
        provider="groq",
        tier=ModelTier.BALANCED,
        cost_per_1k_tokens=0.0006,
        max_context=128000,
        specialties=["general", "conversation"],
        latency_ms=400,
    ),
    "mixtral-8x7b-32768": ModelConfig(
        name="mixtral-8x7b-32768",
        provider="groq",
        tier=ModelTier.FAST,
        cost_per_1k_tokens=0.00024,
        max_context=32768,
        specialties=["fast", "simple"],
        latency_ms=200,
    ),
}


class QueryClassifier:
    """Classify query complexity and type."""
    
    # Keywords indicating complexity
    COMPLEX_KEYWORDS = [
        "analyze", "compare", "contrast", "evaluate", "explain why",
        "step by step", "detailed", "comprehensive", "in-depth",
        "technical analysis", "predict", "forecast", "strategy"
    ]
    
    SIMPLE_KEYWORDS = [
        "what is", "who is", "when", "where", "define", "list",
        "price of", "current", "latest"
    ]
    
    EXPERT_KEYWORDS = [
        "arbitrage", "derivatives", "options", "futures", "yield farming",
        "liquidity pool", "impermanent loss", "tokenomics", "on-chain"
    ]
    
    @classmethod
    def classify(cls, query: str) -> Tuple[QueryComplexity, List[str]]:
        """
        Classify query complexity and detect specialties.
        
        Returns:
            (complexity, detected_specialties)
        """
        query_lower = query.lower()
        detected_specialties = []
        
        # Check for expert-level terms
        expert_matches = sum(1 for kw in cls.EXPERT_KEYWORDS if kw in query_lower)
        if expert_matches >= 2:
            return QueryComplexity.EXPERT, ["crypto", "defi"]
        
        # Check for complex patterns
        complex_matches = sum(1 for kw in cls.COMPLEX_KEYWORDS if kw in query_lower)
        if complex_matches >= 2 or len(query.split()) > 50:
            if "code" in query_lower or "implement" in query_lower:
                detected_specialties.append("code")
            if "analyze" in query_lower or "analysis" in query_lower:
                detected_specialties.append("analysis")
            return QueryComplexity.COMPLEX, detected_specialties
        
        # Check for simple patterns
        simple_matches = sum(1 for kw in cls.SIMPLE_KEYWORDS if kw in query_lower)
        if simple_matches >= 1 and len(query.split()) < 15:
            return QueryComplexity.SIMPLE, ["general"]
        
        return QueryComplexity.MODERATE, ["general"]


class ModelRouter:
    """
    Intelligent router that selects optimal model.
    
    Routing factors:
    1. Query complexity
    2. Cost budget
    3. Latency requirements
    4. Domain specialization
    """
    
    def __init__(
        self,
        max_cost_per_query: float = 0.01,
        max_latency_ms: int = 5000,
        prefer_provider: Optional[str] = None,
    ):
        self.max_cost = max_cost_per_query
        self.max_latency = max_latency_ms
        self.prefer_provider = prefer_provider
        
        # Track model performance
        self.model_stats: Dict[str, Dict] = {
            name: {"calls": 0, "avg_latency": config.latency_ms, "success_rate": 1.0}
            for name, config in MODELS.items()
        }
    
    def route(
        self,
        query: str,
        force_tier: Optional[ModelTier] = None,
        estimated_tokens: int = 1000,
    ) -> Tuple[str, Dict]:
        """
        Route query to optimal model.
        
        Args:
            query: User query
            force_tier: Force specific tier
            estimated_tokens: Estimated token count
            
        Returns:
            (model_name, routing_metadata)
        """
        # Classify query
        complexity, specialties = QueryClassifier.classify(query)
        
        # Determine required tier
        if force_tier:
            required_tier = force_tier
        else:
            tier_map = {
                QueryComplexity.SIMPLE: ModelTier.FAST,
                QueryComplexity.MODERATE: ModelTier.BALANCED,
                QueryComplexity.COMPLEX: ModelTier.POWERFUL,
                QueryComplexity.EXPERT: ModelTier.POWERFUL,
            }
            required_tier = tier_map[complexity]
        
        # Filter eligible models
        eligible = []
        for name, config in MODELS.items():
            # Check provider preference
            if self.prefer_provider and config.provider != self.prefer_provider:
                continue
            
            # Check cost constraint
            estimated_cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
            if estimated_cost > self.max_cost:
                continue
            
            # Check latency constraint
            if config.latency_ms > self.max_latency:
                continue
            
            # Check tier compatibility
            tier_order = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.POWERFUL, ModelTier.SPECIALIZED]
            if tier_order.index(config.tier) >= tier_order.index(required_tier):
                # Check specialty match
                specialty_match = any(s in config.specialties for s in specialties)
                if specialty_match or "general" in config.specialties:
                    eligible.append((name, config, specialty_match))
        
        if not eligible:
            # Fallback to most general available model
            eligible = [(n, c, False) for n, c in MODELS.items()]
        
        # Score and select best model
        def score_model(item):
            name, config, specialty_match = item
            stats = self.model_stats[name]
            
            score = 0
            # Prefer specialty matches
            score += 10 if specialty_match else 0
            # Prefer lower cost
            score -= config.cost_per_1k_tokens * 100
            # Prefer lower latency
            score -= config.latency_ms / 1000
            # Prefer higher success rate
            score += stats["success_rate"] * 5
            
            return score
        
        eligible.sort(key=score_model, reverse=True)
        selected_name, selected_config, _ = eligible[0]
        
        metadata = {
            "model": selected_name,
            "provider": selected_config.provider,
            "complexity": complexity.value,
            "detected_specialties": specialties,
            "estimated_cost": (estimated_tokens / 1000) * selected_config.cost_per_1k_tokens,
            "expected_latency_ms": selected_config.latency_ms,
            "routing_reason": self._explain_routing(
                selected_name, complexity, specialties, required_tier
            ),
        }
        
        log.info("model_routed", **metadata)
        
        return selected_name, metadata
    
    def _explain_routing(
        self,
        model: str,
        complexity: QueryComplexity,
        specialties: List[str],
        required_tier: ModelTier,
    ) -> str:
        """Generate human-readable routing explanation."""
        config = MODELS[model]
        
        reasons = []
        reasons.append(f"Query classified as {complexity.value}")
        reasons.append(f"Required tier: {required_tier.value}")
        reasons.append(f"Selected {model} ({config.provider})")
        
        if specialties:
            reasons.append(f"Detected specialties: {', '.join(specialties)}")
        
        return "; ".join(reasons)
    
    def update_stats(self, model: str, latency_ms: int, success: bool):
        """Update model performance stats."""
        if model not in self.model_stats:
            return
        
        stats = self.model_stats[model]
        stats["calls"] += 1
        
        # Rolling average latency
        stats["avg_latency"] = (
            stats["avg_latency"] * 0.9 + latency_ms * 0.1
        )
        
        # Success rate
        stats["success_rate"] = (
            stats["success_rate"] * 0.95 + (1.0 if success else 0.0) * 0.05
        )


# Singleton router
_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get singleton router."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


def route_query(query: str, **kwargs) -> Tuple[str, Dict]:
    """Convenience function to route a query."""
    return get_router().route(query, **kwargs)


if __name__ == "__main__":
    # Test routing
    test_queries = [
        "What is Bitcoin?",
        "Analyze the technical indicators for BTC and give me a detailed prediction for the next week with confidence levels",
        "Explain the arbitrage opportunities in DeFi yield farming liquidity pools",
        "List the top 5 cryptos by market cap",
    ]
    
    router = get_router()
    
    for query in test_queries:
        model, meta = router.route(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"  Model: {model}")
        print(f"  Complexity: {meta['complexity']}")
        print(f"  Reason: {meta['routing_reason']}")
