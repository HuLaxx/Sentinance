"""
Repository Layer

Database CRUD operations using SQLAlchemy async.
Implements Repository pattern for clean data access.
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from db_models import User, Alert, Price, Prediction, AlertStatus, AlertType
import uuid
import structlog

log = structlog.get_logger()


# ============================================
# USER REPOSITORY
# ============================================

class UserRepository:
    """Repository for User CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, email: str, hashed_password: str, full_name: Optional[str] = None) -> User:
        """Create a new user."""
        user = User(
            id=str(uuid.uuid4()),
            email=email.lower(),
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        log.info("user_created", user_id=user.id)
        return user
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    async def update(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user fields."""
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(updated_at=datetime.utcnow(), **kwargs)
        )
        await self.session.commit()
        return await self.get_by_id(user_id)
    
    async def delete(self, user_id: str) -> bool:
        """Delete user by ID."""
        result = await self.session.execute(
            delete(User).where(User.id == user_id)
        )
        await self.session.commit()
        return result.rowcount > 0


# ============================================
# ALERT REPOSITORY
# ============================================

class AlertRepository:
    """Repository for Alert CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        user_id: str,
        symbol: str,
        alert_type: AlertType,
        target_value: float,
        message: Optional[str] = None
    ) -> Alert:
        """Create a new price alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol,
            alert_type=alert_type,
            target_value=target_value,
            status=AlertStatus.ACTIVE,
            message=message,
            created_at=datetime.utcnow(),
        )
        self.session.add(alert)
        await self.session.commit()
        await self.session.refresh(alert)
        log.info("alert_created", alert_id=alert.id, symbol=symbol)
        return alert
    
    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        result = await self.session.execute(
            select(Alert).where(Alert.id == alert_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_user(self, user_id: str, status: Optional[AlertStatus] = None) -> List[Alert]:
        """Get all alerts for a user, optionally filtered by status."""
        query = select(Alert).where(Alert.user_id == user_id)
        if status:
            query = query.where(Alert.status == status)
        query = query.order_by(Alert.created_at.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_active(self) -> List[Alert]:
        """Get all active alerts (for checking against prices)."""
        result = await self.session.execute(
            select(Alert).where(Alert.status == AlertStatus.ACTIVE)
        )
        return list(result.scalars().all())
    
    async def trigger(self, alert_id: str) -> Optional[Alert]:
        """Mark alert as triggered."""
        await self.session.execute(
            update(Alert)
            .where(Alert.id == alert_id)
            .values(status=AlertStatus.TRIGGERED, triggered_at=datetime.utcnow())
        )
        await self.session.commit()
        log.info("alert_triggered", alert_id=alert_id)
        return await self.get_by_id(alert_id)
    
    async def delete(self, alert_id: str) -> bool:
        """Delete alert by ID."""
        result = await self.session.execute(
            delete(Alert).where(Alert.id == alert_id)
        )
        await self.session.commit()
        return result.rowcount > 0


# ============================================
# PRICE REPOSITORY
# ============================================

class PriceRepository:
    """Repository for Price history CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, prices: List[dict]) -> int:
        """Save multiple price records."""
        count = 0
        for p in prices:
            price = Price(
                symbol=p["symbol"],
                price=p["price"],
                volume=p.get("volume", 0),
                high=p.get("high", p["price"]),
                low=p.get("low", p["price"]),
                change_24h=p.get("change_24h", 0),
                timestamp=datetime.utcnow(),
            )
            self.session.add(price)
            count += 1
        await self.session.commit()
        return count
    
    async def get_latest(self, symbol: str) -> Optional[Price]:
        """Get the most recent price for a symbol."""
        result = await self.session.execute(
            select(Price)
            .where(Price.symbol == symbol)
            .order_by(Price.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_history(self, symbol: str, limit: int = 100) -> List[Price]:
        """Get price history for a symbol."""
        result = await self.session.execute(
            select(Price)
            .where(Price.symbol == symbol)
            .order_by(Price.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def cleanup_old(self, days: int = 7) -> int:
        """Delete prices older than N days."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            delete(Price).where(Price.timestamp < cutoff)
        )
        await self.session.commit()
        return result.rowcount


# ============================================
# PREDICTION REPOSITORY
# ============================================

class PredictionRepository:
    """Repository for ML Prediction CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(
        self,
        symbol: str,
        model_name: str,
        prediction_type: str,
        prediction_value: float,
        confidence: Optional[float] = None,
        horizon: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Prediction:
        """Save a new prediction."""
        prediction = Prediction(
            symbol=symbol,
            model_name=model_name,
            model_version=model_version,
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            confidence=confidence,
            horizon=horizon,
            created_at=datetime.utcnow(),
        )
        self.session.add(prediction)
        await self.session.commit()
        await self.session.refresh(prediction)
        return prediction
    
    async def get_latest(self, symbol: str, prediction_type: str) -> Optional[Prediction]:
        """Get the most recent prediction for a symbol and type."""
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.symbol == symbol)
            .where(Prediction.prediction_type == prediction_type)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
