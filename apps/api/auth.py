"""
Authentication Service

JWT-based authentication with secure password hashing.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import structlog

log = structlog.get_logger()

# ============================================
# CONFIGURATION
# ============================================

# JWT settings from environment
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRATION_HOURS", "24")) * 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

if not SECRET_KEY:
    import sys
    log.error("jwt_secret_missing", message="JWT_SECRET environment variable is required")
    # In production, fail fast. In tests, allow mock
    if "pytest" not in sys.modules:
        raise ValueError("FATAL: JWT_SECRET environment variable must be set")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================
# MODELS
# ============================================

class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    email: str
    exp: datetime
    type: str  # "access" or "refresh"


class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response (without password)."""
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime


# ============================================
# PASSWORD UTILITIES
# ============================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Bcrypt has a 72-byte limit. We truncate passwords to 72 bytes
    to prevent errors while maintaining security.
    """
    # Truncate to 72 bytes (bcrypt limit)
    password_bytes = password.encode('utf-8')[:72]
    password_truncated = password_bytes.decode('utf-8', errors='ignore')
    return pwd_context.hash(password_truncated)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Uses same truncation as hash_password for consistency.
    """
    # Truncate to 72 bytes (bcrypt limit)
    password_bytes = plain_password.encode('utf-8')[:72]
    password_truncated = password_bytes.decode('utf-8', errors='ignore')
    return pwd_context.verify(password_truncated, hashed_password)


# ============================================
# JWT UTILITIES
# ============================================

def create_access_token(user_id: str, email: str) -> str:
    """Create a JWT access token."""
    if not SECRET_KEY:
        raise RuntimeError("JWT_SECRET is not set")
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str, email: str) -> str:
    """Create a JWT refresh token."""
    if not SECRET_KEY:
        raise RuntimeError("JWT_SECRET is not set")
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_tokens(user_id: str, email: str) -> Token:
    """Create both access and refresh tokens."""
    return Token(
        access_token=create_access_token(user_id, email),
        refresh_token=create_refresh_token(user_id, email),
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate a JWT token."""
    try:
        if not SECRET_KEY:
            raise RuntimeError("JWT_SECRET is not set")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenPayload(
            sub=payload["sub"],
            email=payload["email"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            type=payload["type"]
        )
    except JWTError as e:
        log.warning("token_decode_failed", error=str(e))
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[TokenPayload]:
    """Verify a token is valid and of the correct type."""
    payload = decode_token(token)
    
    if not payload:
        return None
    
    if payload.type != token_type:
        log.warning("token_type_mismatch", expected=token_type, got=payload.type)
        return None
    
    if payload.exp < datetime.now(timezone.utc):
        log.warning("token_expired", email=payload.email)
        return None
    
    return payload
