"""
Authentication Dependencies

FastAPI dependencies for protected routes.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth import verify_token, TokenPayload
import structlog

log = structlog.get_logger()

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenPayload:
    """
    Dependency to get the current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: TokenPayload = Depends(get_current_user)):
            return {"user_id": user.sub}
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_token(credentials.credentials, token_type="access")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenPayload]:
    """
    Dependency to optionally get the current user.
    Returns None if not authenticated (doesn't raise error).
    
    Usage:
        @app.get("/public")
        async def public_route(user: Optional[TokenPayload] = Depends(get_current_user_optional)):
            if user:
                return {"message": f"Hello, {user.email}"}
            return {"message": "Hello, anonymous"}
    """
    if not credentials:
        return None
    
    return verify_token(credentials.credentials, token_type="access")
