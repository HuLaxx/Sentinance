"""
Authentication Tests

Tests for JWT authentication, user registration, and login.
"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_login_invalid_credentials(client: TestClient):
    """Test login with invalid credentials returns 401."""
    response = client.post(
        "/api/auth/login",
        json={"email": "nonexistent@test.com", "password": "wrongpassword"},
    )
    assert response.status_code in [401, 404, 422]


def test_register_user_validation(client: TestClient):
    """Test registration with invalid data returns validation error."""
    response = client.post(
        "/api/auth/register", json={"email": "invalid-email", "password": "short"}
    )
    # Should fail validation
    assert response.status_code in [400, 422]


def test_protected_endpoint_without_token(client: TestClient):
    """Test accessing protected endpoint without token returns 401."""
    response = client.get("/api/alerts")
    assert response.status_code in [401, 403]


def test_protected_endpoint_with_invalid_token(client: TestClient):
    """Test accessing protected endpoint with invalid token."""
    response = client.get(
        "/api/alerts", headers={"Authorization": "Bearer invalid_token_here"}
    )
    assert response.status_code in [401, 403]


def test_jwt_token_format():
    """Test JWT token generation format."""
    from auth import create_access_token

    token = create_access_token(user_id="test-user-id", email="test@example.com")

    assert token is not None
    assert isinstance(token, str)
    # JWT tokens have 3 parts separated by dots
    parts = token.split(".")
    assert len(parts) == 3


def test_jwt_token_decode():
    """Test JWT token decoding."""
    from auth import create_access_token, verify_token

    test_email = "test@example.com"
    token = create_access_token(user_id="test-user-id", email=test_email)

    payload = verify_token(token)

    assert payload is not None
    assert payload.email == test_email


def test_password_hashing():
    """Test password hashing and verification."""
    from auth import hash_password, verify_password

    password = "SecurePassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("WrongPassword", hashed)


def test_rejects_invalid_jwt_signature():
    """A correctly shaped HS256 token signed by another key must not authenticate."""
    import jwt
    from datetime import UTC, datetime, timedelta
    from auth import decode_token

    forged = jwt.encode(
        {"sub": "attacker", "email": "attacker@example.test",
         "type": "access", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        "untrusted-different-signing-key",
        algorithm="HS256",
    )
    assert decode_token(forged) is None


def test_rejects_unsigned_jwt():
    """The decoder's explicit HS256 allow-list must never accept alg=none."""
    import jwt
    from datetime import UTC, datetime, timedelta
    from auth import decode_token

    unsigned = jwt.encode(
        {"sub": "attacker", "email": "attacker@example.test",
         "type": "access", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        key="",
        algorithm="none",
    )
    assert decode_token(unsigned) is None


def test_rejects_expired_jwt():
    import jwt
    from datetime import UTC, datetime, timedelta
    from auth import ALGORITHM, SECRET_KEY, decode_token

    assert SECRET_KEY is not None
    expired = jwt.encode(
        {"sub": "test", "email": "test@example.test",
         "type": "access", "exp": datetime.now(UTC) - timedelta(minutes=5)},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    assert decode_token(expired) is None


def test_refresh_token_cannot_be_used_as_access_token():
    from auth import create_refresh_token, verify_token
    refresh = create_refresh_token("test-user", "test@example.test")
    assert verify_token(refresh, token_type="access") is None
    assert verify_token(refresh, token_type="refresh") is not None
