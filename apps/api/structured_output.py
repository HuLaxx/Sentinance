"""
Structured Output Extraction with Pydantic

Parses LLM responses into validated, typed data structures.

Features:
1. Pydantic models for LLM outputs
2. Retry logic with format correction
3. JSON mode enforcement
4. Schema generation for prompts

This demonstrates:
- Type-safe LLM integration
- Structured data extraction
- Validation patterns
- Error recovery
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError
import structlog

log = structlog.get_logger()

T = TypeVar('T', bound=BaseModel)


# ============================================
# OUTPUT MODELS
# ============================================

class MarketAnalysis(BaseModel):
    """Structured market analysis output."""
    symbol: str = Field(description="Trading pair symbol")
    sentiment: str = Field(description="Overall sentiment: bullish, bearish, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    key_factors: List[str] = Field(description="Key factors driving the analysis")
    price_prediction: Optional[float] = Field(None, description="Predicted price")
    timeframe: str = Field(description="Prediction timeframe: 1h, 24h, 7d")
    risk_level: str = Field(description="Risk level: low, medium, high")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "sentiment": "bullish",
                "confidence": 0.75,
                "key_factors": ["RSI oversold", "MACD bullish crossover"],
                "price_prediction": 98000,
                "timeframe": "24h",
                "risk_level": "medium"
            }
        }


class TradingRecommendation(BaseModel):
    """Structured trading recommendation."""
    action: str = Field(description="Action: buy, sell, hold")
    symbol: str = Field(description="Trading pair")
    entry_price: float = Field(description="Recommended entry price")
    stop_loss: float = Field(description="Stop loss price")
    take_profit: float = Field(description="Take profit price")
    position_size_percent: float = Field(ge=0, le=100, description="Position size as % of portfolio")
    reasoning: str = Field(description="Explanation for the recommendation")
    confidence: float = Field(ge=0, le=1)


class NewsDigest(BaseModel):
    """Structured news summary."""
    headline: str = Field(description="Main headline")
    summary: str = Field(description="Brief summary")
    sentiment: str = Field(description="Sentiment: positive, negative, neutral")
    impact: str = Field(description="Expected market impact: high, medium, low")
    affected_assets: List[str] = Field(description="Affected crypto assets")
    source: Optional[str] = Field(None, description="News source")


class PortfolioAdvice(BaseModel):
    """Structured portfolio advice."""
    current_allocation: Dict[str, float] = Field(description="Current allocation by asset")
    recommended_changes: List[Dict[str, Any]] = Field(description="Recommended changes")
    risk_score: float = Field(ge=0, le=10, description="Portfolio risk score 0-10")
    diversification_score: float = Field(ge=0, le=10)
    overall_recommendation: str = Field(description="Overall recommendation summary")


# ============================================
# STRUCTURED OUTPUT EXTRACTOR
# ============================================

class StructuredOutputExtractor:
    """
    Extract structured data from LLM responses.
    
    Uses:
    1. JSON mode when available
    2. Schema injection into prompts
    3. Regex extraction as fallback
    4. Validation with Pydantic
    """
    
    def __init__(self):
        self.retry_count = 3
    
    def generate_schema_prompt(self, model: Type[T]) -> str:
        """Generate prompt instructions from Pydantic model."""
        schema = model.model_json_schema()
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        fields_desc = []
        for field_name, field_info in properties.items():
            desc = field_info.get("description", "")
            field_type = field_info.get("type", "string")
            req = "(required)" if field_name in required else "(optional)"
            fields_desc.append(f"  - {field_name}: {field_type} {req} - {desc}")
        
        return f"""
You must respond with a valid JSON object matching this exact schema:

Fields:
{chr(10).join(fields_desc)}

Example:
{json.dumps(schema.get("example", {}), indent=2)}

IMPORTANT: Respond ONLY with the JSON object, no additional text.
"""
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text, handling various formats."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
            r'```\s*([\s\S]*?)\s*```',       # Generic code block
            r'\{[\s\S]*\}',                   # Raw JSON object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        return None
    
    def parse_response(
        self,
        response: str,
        model: Type[T],
    ) -> Optional[T]:
        """
        Parse LLM response into Pydantic model.
        
        Args:
            response: Raw LLM response text
            model: Pydantic model class to parse into
            
        Returns:
            Validated Pydantic model instance or None
        """
        # Extract JSON
        data = self.extract_json(response)
        
        if not data:
            log.warning("json_extraction_failed", response_start=response[:100])
            return None
        
        # Validate with Pydantic
        try:
            return model.model_validate(data)
        except ValidationError as e:
            log.warning("pydantic_validation_failed", errors=str(e))
            return None
    
    async def extract_with_retry(
        self,
        llm_call,
        model: Type[T],
        base_prompt: str,
    ) -> Optional[T]:
        """
        Extract structured output with retry logic.
        
        Args:
            llm_call: Async function that takes prompt and returns response
            model: Pydantic model class
            base_prompt: Base prompt without schema instructions
            
        Returns:
            Validated model instance or None
        """
        schema_prompt = self.generate_schema_prompt(model)
        full_prompt = f"{base_prompt}\n\n{schema_prompt}"
        
        for attempt in range(self.retry_count):
            try:
                response = await llm_call(full_prompt)
                result = self.parse_response(response, model)
                
                if result:
                    log.info("structured_extraction_success", model=model.__name__, attempt=attempt+1)
                    return result
                
                # Add correction hint for retry
                if attempt < self.retry_count - 1:
                    full_prompt += "\n\nPrevious response was not valid JSON. Please try again with ONLY a JSON object."
                    
            except Exception as e:
                log.warning("llm_call_failed", error=str(e), attempt=attempt+1)
        
        log.error("structured_extraction_failed", model=model.__name__)
        return None


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

_extractor = StructuredOutputExtractor()


def parse_market_analysis(response: str) -> Optional[MarketAnalysis]:
    """Parse market analysis from LLM response."""
    return _extractor.parse_response(response, MarketAnalysis)


def parse_trading_recommendation(response: str) -> Optional[TradingRecommendation]:
    """Parse trading recommendation from LLM response."""
    return _extractor.parse_response(response, TradingRecommendation)


def get_schema_prompt(model: Type[BaseModel]) -> str:
    """Get schema prompt for a model."""
    return _extractor.generate_schema_prompt(model)


# ============================================
# PROMPT TEMPLATES WITH VERSIONING
# ============================================

class PromptTemplate:
    """Versioned prompt template with performance tracking."""
    
    def __init__(
        self,
        name: str,
        version: str,
        template: str,
        output_model: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.version = version
        self.template = template
        self.output_model = output_model
        
        # Performance tracking
        self.uses = 0
        self.successes = 0
        self.avg_tokens = 0
    
    def render(self, **kwargs) -> str:
        """Render template with variables."""
        prompt = self.template.format(**kwargs)
        
        if self.output_model:
            prompt += "\n\n" + _extractor.generate_schema_prompt(self.output_model)
        
        self.uses += 1
        return prompt
    
    def record_result(self, success: bool, tokens: int = 0):
        """Record result for performance tracking."""
        if success:
            self.successes += 1
        self.avg_tokens = (self.avg_tokens * (self.uses - 1) + tokens) / self.uses if self.uses > 0 else tokens
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0


# Prompt template registry
PROMPTS = {
    "market_analysis_v1": PromptTemplate(
        name="market_analysis",
        version="1.0",
        template="""Analyze the market conditions for {symbol}.

Current Data:
- Price: ${price:,.2f}
- 24h Change: {change_24h:+.2f}%
- RSI(14): {rsi}
- MACD: {macd}

Provide a comprehensive market analysis.""",
        output_model=MarketAnalysis,
    ),
    
    "trading_recommendation_v1": PromptTemplate(
        name="trading_recommendation",
        version="1.0",
        template="""Based on the following market data for {symbol}:

Price: ${price:,.2f}
RSI: {rsi}
MACD: {macd}
Trend: {trend}

Generate a trading recommendation with specific entry, stop loss, and take profit levels.""",
        output_model=TradingRecommendation,
    ),
}


def get_prompt(name: str) -> Optional[PromptTemplate]:
    """Get prompt template by name."""
    return PROMPTS.get(name)


if __name__ == "__main__":
    # Test structured extraction
    sample_response = """
    Based on my analysis, here's my assessment:
    
    ```json
    {
        "symbol": "BTCUSDT",
        "sentiment": "bullish",
        "confidence": 0.78,
        "key_factors": ["Strong support at $94k", "Increasing volume", "RSI recovering from oversold"],
        "price_prediction": 98500,
        "timeframe": "24h",
        "risk_level": "medium"
    }
    ```
    
    This is based on current market conditions.
    """
    
    result = parse_market_analysis(sample_response)
    
    if result:
        print("Parsed successfully!")
        print(f"Symbol: {result.symbol}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence}")
        print(f"Key factors: {result.key_factors}")
    else:
        print("Parsing failed")
