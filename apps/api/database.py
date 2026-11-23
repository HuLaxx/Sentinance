"""
Database Configuration

SQLAlchemy async database setup with connection pooling.
"""

import os
from collections.abc import AsyncGenerator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


def normalize_async_database_url(raw_url: str) -> str:
    """Use the asyncpg dialect for configured PostgreSQL URLs.

    CI and hosted providers commonly supply `postgresql://` or `postgres://`
    even though SQLAlchemy's async engine requires an explicit async driver.
    Reject unsupported drivers rather than accidentally loading psycopg2.
    """
    url = make_url(raw_url)
    if url.drivername in ("postgresql", "postgres", "postgresql+asyncpg"):
        return url.set(drivername="postgresql+asyncpg").render_as_string(
            hide_password=False
        )
    raise ValueError("Sentinance requires a PostgreSQL URL with the asyncpg driver")


# Database URL from environment, with explicit async dialect normalization.
DATABASE_URL = normalize_async_database_url(
    os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://sentinance:password@localhost:5432/sentinance",
    )
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set True for SQL logging
    pool_pre_ping=True,  # Verify connections before use
    pool_size=10,
    max_overflow=20,
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    await engine.dispose()
