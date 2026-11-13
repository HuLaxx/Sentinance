"""Fail-closed, Redis-backed rate limit for sensitive authentication endpoints.

The source peer is the ASGI socket address, not an untrusted X-Forwarded-For
header. Behind a reverse proxy, configure a trusted proxy chain separately
so the effective client address is accurate.
"""

from hashlib import sha256

from fastapi import HTTPException, Request, status

from cache import evaluate_rate_limit

AUTH_MAX_ATTEMPTS = 20
AUTH_WINDOW_SECONDS = 60


async def enforce_auth_rate_limit(request: Request) -> None:
    """Enforce a shared fixed-window limit before token-issuing operations."""
    peer = request.client.host if request.client else "unknown"
    raw_identifier = f"{request.url.path}|{peer}"
    key = sha256(raw_identifier.encode("utf-8")).hexdigest()
    decision = await evaluate_rate_limit(
        key, max_requests=AUTH_MAX_ATTEMPTS, window_seconds=AUTH_WINDOW_SECONDS
    )

    if not decision.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable",
            headers={"Retry-After": "5"},
        )

    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts",
            headers={"Retry-After": str(AUTH_WINDOW_SECONDS)},
        )
