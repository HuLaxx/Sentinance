"""
Alerts Service

Manages price alerts - triggers notifications when conditions are met.
Stores alerts in memory for now (will migrate to PostgreSQL).
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import structlog

log = structlog.get_logger()


class AlertType(str, Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PERCENT_CHANGE = "percent_change"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"


class Alert(BaseModel):
    """Alert model for price notifications."""
    id: str
    user_id: str  # For future auth integration
    symbol: str
    alert_type: AlertType
    target_value: float
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None
    message: Optional[str] = None


class CreateAlertRequest(BaseModel):
    """Request to create a new alert."""
    symbol: str
    alert_type: AlertType
    target_value: float
    message: Optional[str] = None


class AlertsService:
    """
    Service to manage price alerts.
    
    Currently stores alerts in memory - will migrate to PostgreSQL.
    
    Usage:
        service = AlertsService()
        alert = service.create_alert("user123", ...)
        triggered = service.check_alerts(current_prices)
    """
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        self._counter += 1
        return f"alert_{self._counter:04d}"
    
    def create_alert(
        self,
        user_id: str,
        symbol: str,
        alert_type: AlertType,
        target_value: float,
        message: Optional[str] = None
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            id=self._generate_id(),
            user_id=user_id,
            symbol=symbol.upper(),
            alert_type=alert_type,
            target_value=target_value,
            message=message
        )
        self._alerts[alert.id] = alert
        
        log.info("alert_created", 
                 alert_id=alert.id, 
                 symbol=alert.symbol, 
                 type=alert.alert_type,
                 target=alert.target_value)
        
        return alert
    
    def get_alerts(self, user_id: Optional[str] = None) -> List[Alert]:
        """Get all alerts, optionally filtered by user."""
        alerts = list(self._alerts.values())
        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        return alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (non-triggered) alerts."""
        return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert by ID."""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            log.info("alert_deleted", alert_id=alert_id)
            return True
        return False
    
    def check_alerts(self, prices: Dict[str, dict]) -> List[Alert]:
        """
        Check all active alerts against current prices.
        Returns list of newly triggered alerts.
        """
        triggered = []
        
        for alert in self.get_active_alerts():
            price_data = prices.get(alert.symbol)
            if not price_data:
                continue
            
            current_price = price_data.get("price", 0)
            current_change = price_data.get("change_24h", 0)
            
            should_trigger = False
            
            if alert.alert_type == AlertType.PRICE_ABOVE:
                should_trigger = current_price >= alert.target_value
            elif alert.alert_type == AlertType.PRICE_BELOW:
                should_trigger = current_price <= alert.target_value
            elif alert.alert_type == AlertType.PERCENT_CHANGE:
                should_trigger = abs(current_change) >= alert.target_value
            
            if should_trigger:
                alert.status = AlertStatus.TRIGGERED
                alert.triggered_at = datetime.utcnow()
                triggered.append(alert)
                
                log.info("alert_triggered",
                         alert_id=alert.id,
                         symbol=alert.symbol,
                         type=alert.alert_type,
                         price=current_price)
        
        return triggered


# Global service instance
_alerts_service: Optional[AlertsService] = None


def get_alerts_service() -> AlertsService:
    """Get or create the global alerts service."""
    global _alerts_service
    if _alerts_service is None:
        _alerts_service = AlertsService()
    return _alerts_service
