"""
Authentication Router

API endpoints for user registration, login, and token refresh.
"""

import uuid
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from auth import (
    UserCreate, 
    UserLogin, 
    UserResponse, 
    Token,
    hash_password, 
    verify_password, 
    create_tokens,
    verify_token
)
from dependencies import get_current_user
from auth import TokenPayload
import structlog

log = structlog.get_logger()

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# In-memory user store (will be replaced with database)
USERS_DB: dict = {}


class RefreshTokenRequest(BaseModel):
    """Refresh token request body."""
    refresh_token: str


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    Request:
        {
            "email": "user@example.com",
            "password": "securepassword123",
            "full_name": "John Doe"
        }
    
    Returns:
        JWT tokens (access + refresh)
    """
    # Check if user exists
    if user_data.email.lower() in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_id = str(uuid.uuid4())
    USERS_DB[user_data.email.lower()] = {
        "id": user_id,
        "email": user_data.email.lower(),
        "hashed_password": hash_password(user_data.password),
        "full_name": user_data.full_name,
        "is_active": True,
        "is_verified": False,
    }
    
    log.info("user_registered", email=user_data.email)
    
    # Return tokens
    return create_tokens(user_id, user_data.email.lower())


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """
    Login with email and password.
    
    Request:
        {
            "email": "user@example.com",
            "password": "securepassword123"
        }
    
    Returns:
        JWT tokens (access + refresh)
    """
    # Find user
    user = USERS_DB.get(credentials.email.lower())
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(credentials.password, user["hashed_password"]):
        log.warning("login_failed", email=credentials.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if active
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    log.info("user_logged_in", email=credentials.email)
    
    return create_tokens(user["id"], user["email"])


@router.post("/refresh", response_model=Token)
async def refresh_token(request: RefreshTokenRequest):
    """
    Get new access token using refresh token.
    
    Request body:
        {
            "refresh_token": "..."
        }
    
    Returns:
        New JWT tokens
    """
    payload = verify_token(request.refresh_token, token_type="refresh")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Verify user still exists
    user = None
    for u in USERS_DB.values():
        if u["id"] == payload.sub:
            user = u
            break
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    log.info("token_refreshed", user_id=payload.sub)
    
    return create_tokens(user["id"], user["email"])


@router.get("/me", response_model=dict)
async def get_current_user_info(current_user: TokenPayload = Depends(get_current_user)):
    """
    Get current authenticated user info.
    
    Requires: Bearer token in Authorization header
    
    Returns:
        User information
    """
    user = None
    for u in USERS_DB.values():
        if u["id"] == current_user.sub:
            user = u
            break
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "is_active": user["is_active"],
        "is_verified": user["is_verified"],
    }


@router.post("/logout")
async def logout(current_user: TokenPayload = Depends(get_current_user)):
    """
    Logout current user.
    
    Note: With JWT, logout is handled client-side by deleting the token.
    This endpoint is for logging/audit purposes.
    """
    log.info("user_logged_out", user_id=current_user.sub)
    return {"message": "Logged out successfully"}
