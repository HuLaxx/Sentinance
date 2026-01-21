"""
Anomaly Detection Service

Detects unusual market activity:
- Price spikes/crashes
- Volume anomalies
- Manipulation patterns
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import structlog

log = structlog.get_logger()


@dataclass
class Anomaly:
    """Detected anomaly."""
    symbol: str
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    value: float
    threshold: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "type": self.anomaly_type,
            "severity": self.severity,
            "description": self.description,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "timestamp": self.timestamp,
        }


def detect_price_spike(
    prices: list[float],
    threshold_percent: float = 5.0
) -> Optional[Anomaly]:
    """Detect sudden price spikes."""
    if len(prices) < 2:
        return None
    
    current = prices[-1]
    previous = prices[-2]
    
    if previous == 0:
        return None
    
    change = ((current - previous) / previous) * 100
    
    if abs(change) > threshold_percent:
        severity = "critical" if abs(change) > 10 else "high" if abs(change) > 7 else "medium"
        direction = "spike" if change > 0 else "crash"
        
        return Anomaly(
            symbol="",
            anomaly_type=f"price_{direction}",
            severity=severity,
            description=f"Price {direction} of {abs(change):.2f}% detected",
            value=change,
            threshold=threshold_percent,
            timestamp=datetime.utcnow().isoformat(),
        )
    
    return None


def detect_volume_anomaly(
    current_volume: float,
    avg_volume: float,
    multiplier: float = 3.0
) -> Optional[Anomaly]:
    """Detect unusual volume."""
    if avg_volume == 0:
        return None
    
    ratio = current_volume / avg_volume
    
    if ratio > multiplier:
        severity = "critical" if ratio > 10 else "high" if ratio > 5 else "medium"
        
        return Anomaly(
            symbol="",
            anomaly_type="volume_spike",
            severity=severity,
            description=f"Volume is {ratio:.1f}x higher than average",
            value=ratio,
            threshold=multiplier,
            timestamp=datetime.utcnow().isoformat(),
        )
    
    return None


def detect_spread_anomaly(
    bid: float,
    ask: float,
    normal_spread_percent: float = 0.1
) -> Optional[Anomaly]:
    """Detect abnormal bid-ask spread."""
    if bid == 0:
        return None
    
    spread_percent = ((ask - bid) / bid) * 100
    
    if spread_percent > normal_spread_percent * 5:  # 5x normal spread
        severity = "high" if spread_percent > normal_spread_percent * 10 else "medium"
        
        return Anomaly(
            symbol="",
            anomaly_type="spread_anomaly",
            severity=severity,
            description=f"Bid-ask spread of {spread_percent:.3f}% is abnormally high",
            value=spread_percent,
            threshold=normal_spread_percent,
            timestamp=datetime.utcnow().isoformat(),
        )
    
    return None


def detect_manipulation_pattern(
    prices: list[float],
    volumes: list[float],
) -> list[Anomaly]:
    """
    Detect potential manipulation patterns:
    - Pump and dump: Volume spike + price spike then crash
    - Wash trading: High volume with minimal price movement
    """
    anomalies = []
    
    if len(prices) < 10 or len(volumes) < 10:
        return anomalies
    
    # Check for pump and dump (simplified)
    max_price = max(prices[-10:])
    current = prices[-1]
    avg_vol = sum(volumes[-10:]) / 10
    recent_vol = volumes[-1]
    
    # Price dropped significantly from recent high with volume spike
    if current < max_price * 0.9 and recent_vol > avg_vol * 2:
        price_drop = ((max_price - current) / max_price) * 100
        anomalies.append(Anomaly(
            symbol="",
            anomaly_type="potential_pump_dump",
            severity="high",
            description=f"Price dropped {price_drop:.1f}% from recent high with elevated volume",
            value=price_drop,
            threshold=10.0,
            timestamp=datetime.utcnow().isoformat(),
        ))
    
    # Check for wash trading (high volume, low volatility)
    price_range = (max(prices[-10:]) - min(prices[-10:])) / min(prices[-10:]) * 100
    if price_range < 0.5 and recent_vol > avg_vol * 3:
        anomalies.append(Anomaly(
            symbol="",
            anomaly_type="potential_wash_trading",
            severity="medium",
            description=f"High volume ({recent_vol/avg_vol:.1f}x avg) with minimal price movement ({price_range:.2f}%)",
            value=recent_vol / avg_vol,
            threshold=3.0,
            timestamp=datetime.utcnow().isoformat(),
        ))
    
    return anomalies


def run_anomaly_detection(
    symbol: str,
    prices: list[float],
    volumes: Optional[list[float]] = None,
) -> list[Anomaly]:
    """Run all anomaly detection checks for a symbol."""
    anomalies = []
    
    # Price spike detection
    price_anomaly = detect_price_spike(prices)
    if price_anomaly:
        price_anomaly.symbol = symbol
        anomalies.append(price_anomaly)
    
    # Volume anomaly
    if volumes and len(volumes) >= 2:
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        vol_anomaly = detect_volume_anomaly(volumes[-1], avg_vol)
        if vol_anomaly:
            vol_anomaly.symbol = symbol
            anomalies.append(vol_anomaly)
    
    # Manipulation patterns
    if volumes:
        manipulation = detect_manipulation_pattern(prices, volumes)
        for m in manipulation:
            m.symbol = symbol
            anomalies.append(m)
    
    return anomalies
