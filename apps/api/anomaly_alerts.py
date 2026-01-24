"""
Real-Time Anomaly Alert System

Push notifications for market anomalies via WebSocket.

Features:
1. Real-time anomaly detection
2. Severity classification
3. WebSocket push notifications
4. Alert aggregation & deduplication
5. Cooldown periods to prevent spam

This demonstrates:
- Event-driven architecture
- Real-time notification systems
- Anomaly detection patterns
- WebSocket broadcasting
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog

log = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of market alerts."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    WHALE_MOVEMENT = "whale_movement"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    EXCHANGE_ISSUE = "exchange_issue"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_EXPLOSION = "volatility_explosion"


@dataclass
class MarketAlert:
    """A market anomaly alert."""
    id: str
    type: AlertType
    severity: AlertSeverity
    symbol: str
    title: str
    message: str
    data: Dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    acknowledged: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }
    
    def to_sse(self) -> str:
        """Format for Server-Sent Events."""
        return f"event: alert\ndata: {json.dumps(self.to_dict())}\n\n"


class AlertManager:
    """
    Manages real-time alert generation and distribution.
    
    Features:
    - Anomaly detection triggers
    - Cooldown periods (prevent alert spam)
    - Severity escalation
    - WebSocket broadcast
    """
    
    def __init__(self):
        self.alerts: List[MarketAlert] = []
        self.subscribers: Set[Callable] = set()
        self.cooldowns: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = {}  # For escalation
        
        # Cooldown periods by alert type (seconds)
        self.cooldown_periods = {
            AlertType.PRICE_SPIKE: 300,      # 5 minutes
            AlertType.VOLUME_SURGE: 600,     # 10 minutes
            AlertType.WHALE_MOVEMENT: 300,
            AlertType.LIQUIDATION_CASCADE: 60,  # 1 minute (urgent)
            AlertType.EXCHANGE_ISSUE: 1800,   # 30 minutes
            AlertType.CORRELATION_BREAK: 900,
            AlertType.VOLATILITY_EXPLOSION: 300,
        }
    
    def subscribe(self, callback: Callable):
        """Subscribe to alerts."""
        self.subscribers.add(callback)
        log.info("alert_subscriber_added", total=len(self.subscribers))
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from alerts."""
        self.subscribers.discard(callback)
    
    def _check_cooldown(self, alert_type: AlertType, symbol: str) -> bool:
        """Check if alert is in cooldown period."""
        key = f"{alert_type.value}:{symbol}"
        
        if key in self.cooldowns:
            if datetime.utcnow() < self.cooldowns[key]:
                return True  # Still in cooldown
        
        return False
    
    def _set_cooldown(self, alert_type: AlertType, symbol: str):
        """Set cooldown for alert type."""
        key = f"{alert_type.value}:{symbol}"
        period = self.cooldown_periods.get(alert_type, 300)
        self.cooldowns[key] = datetime.utcnow() + timedelta(seconds=period)
    
    def _escalate_severity(self, alert_type: AlertType, symbol: str, base_severity: AlertSeverity) -> AlertSeverity:
        """Escalate severity based on frequency."""
        key = f"{alert_type.value}:{symbol}"
        
        # Increment count
        self.alert_counts[key] = self.alert_counts.get(key, 0) + 1
        count = self.alert_counts[key]
        
        # Escalate based on frequency
        if count >= 10 and base_severity != AlertSeverity.EMERGENCY:
            return AlertSeverity.EMERGENCY
        elif count >= 5 and base_severity == AlertSeverity.INFO:
            return AlertSeverity.WARNING
        elif count >= 5 and base_severity == AlertSeverity.WARNING:
            return AlertSeverity.CRITICAL
        
        return base_severity
    
    async def create_alert(
        self,
        alert_type: AlertType,
        symbol: str,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        data: Optional[Dict] = None,
        ignore_cooldown: bool = False,
    ) -> Optional[MarketAlert]:
        """
        Create and broadcast an alert.
        
        Args:
            alert_type: Type of alert
            symbol: Affected symbol
            title: Alert title
            message: Detailed message
            severity: Base severity level
            data: Additional data
            ignore_cooldown: Force alert even in cooldown
            
        Returns:
            Created alert or None if in cooldown
        """
        # Check cooldown
        if not ignore_cooldown and self._check_cooldown(alert_type, symbol):
            log.debug("alert_in_cooldown", type=alert_type.value, symbol=symbol)
            return None
        
        # Escalate severity if needed
        final_severity = self._escalate_severity(alert_type, symbol, severity)
        
        # Create alert
        alert = MarketAlert(
            id=f"{symbol}-{alert_type.value}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            type=alert_type,
            severity=final_severity,
            symbol=symbol,
            title=title,
            message=message,
            data=data or {},
        )
        
        self.alerts.append(alert)
        self._set_cooldown(alert_type, symbol)
        
        # Broadcast to subscribers
        await self._broadcast(alert)
        
        log.info(
            "alert_created",
            type=alert_type.value,
            symbol=symbol,
            severity=final_severity.value,
        )
        
        return alert
    
    async def _broadcast(self, alert: MarketAlert):
        """Broadcast alert to all subscribers."""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                log.error("alert_broadcast_failed", error=str(e))
    
    def get_recent_alerts(
        self,
        symbol: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        hours: int = 24,
    ) -> List[MarketAlert]:
        """Get recent alerts with optional filters."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a.timestamp) > cutoff
        ]
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)


# ============================================
# ANOMALY DETECTORS
# ============================================

class AnomalyDetector:
    """Detect market anomalies and trigger alerts."""
    
    def __init__(self, alert_manager: AlertManager):
        self.manager = alert_manager
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
    
    async def check_price_spike(
        self,
        symbol: str,
        current_price: float,
        previous_price: float,
    ):
        """Detect sudden price spikes."""
        if previous_price == 0:
            return
        
        change_pct = ((current_price - previous_price) / previous_price) * 100
        
        # Track history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(change_pct)
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol].pop(0)
        
        # Detect anomaly
        if abs(change_pct) > 5:  # 5% spike
            severity = AlertSeverity.CRITICAL if abs(change_pct) > 10 else AlertSeverity.WARNING
            direction = "⬆️ UP" if change_pct > 0 else "⬇️ DOWN"
            
            await self.manager.create_alert(
                alert_type=AlertType.PRICE_SPIKE,
                symbol=symbol,
                title=f"{symbol} Price Spike {direction}",
                message=f"{symbol} moved {change_pct:+.2f}% in a short period. Current: ${current_price:,.2f}",
                severity=severity,
                data={
                    "change_percent": round(change_pct, 2),
                    "current_price": current_price,
                    "previous_price": previous_price,
                },
            )
    
    async def check_volume_surge(
        self,
        symbol: str,
        current_volume: float,
        avg_volume: float,
    ):
        """Detect unusual volume."""
        if avg_volume == 0:
            return
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 3:  # 3x average volume
            severity = AlertSeverity.CRITICAL if volume_ratio > 5 else AlertSeverity.WARNING
            
            await self.manager.create_alert(
                alert_type=AlertType.VOLUME_SURGE,
                symbol=symbol,
                title=f"{symbol} Volume Surge",
                message=f"Trading volume is {volume_ratio:.1f}x the average. Unusual activity detected.",
                severity=severity,
                data={
                    "volume_ratio": round(volume_ratio, 2),
                    "current_volume": current_volume,
                    "avg_volume": avg_volume,
                },
            )
    
    async def check_volatility(
        self,
        symbol: str,
        current_volatility: float,
        avg_volatility: float,
    ):
        """Detect volatility explosions."""
        if avg_volatility == 0:
            return
        
        vol_ratio = current_volatility / avg_volatility
        
        if vol_ratio > 2:  # 2x average volatility
            severity = AlertSeverity.CRITICAL if vol_ratio > 3 else AlertSeverity.WARNING
            
            await self.manager.create_alert(
                alert_type=AlertType.VOLATILITY_EXPLOSION,
                symbol=symbol,
                title=f"{symbol} Volatility Explosion",
                message=f"Volatility is {vol_ratio:.1f}x normal levels. High risk environment.",
                severity=severity,
                data={
                    "volatility_ratio": round(vol_ratio, 2),
                    "current_volatility": current_volatility,
                },
            )


# ============================================
# SINGLETON & API
# ============================================

_alert_manager: Optional[AlertManager] = None
_anomaly_detector: Optional[AnomalyDetector] = None


def get_alert_manager() -> AlertManager:
    """Get singleton alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def get_anomaly_detector() -> AnomalyDetector:
    """Get singleton anomaly detector."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(get_alert_manager())
    return _anomaly_detector


async def check_for_anomalies(
    symbol: str,
    current_price: float,
    previous_price: float,
    current_volume: float,
    avg_volume: float,
):
    """Main entry point for anomaly checking."""
    detector = get_anomaly_detector()
    
    await detector.check_price_spike(symbol, current_price, previous_price)
    await detector.check_volume_surge(symbol, current_volume, avg_volume)


if __name__ == "__main__":
    async def main():
        manager = get_alert_manager()
        
        # Subscribe to alerts
        async def print_alert(alert: MarketAlert):
            print(f"🚨 ALERT: {alert.title}")
            print(f"   Severity: {alert.severity.value}")
            print(f"   Message: {alert.message}")
        
        manager.subscribe(print_alert)
        
        # Simulate anomalies
        await check_for_anomalies(
            symbol="BTCUSDT",
            current_price=100000,
            previous_price=92000,  # ~8.7% spike
            current_volume=5000000000,
            avg_volume=1000000000,  # 5x volume
        )
    
    asyncio.run(main())
