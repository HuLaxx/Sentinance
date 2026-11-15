"""Fail-closed browser Origin, HTTP Host and reverse-proxy trust policies.

Origin is a browser cross-site control, not user authentication. Forwarded
headers are trusted only when the network peer is a configured proxy IP.
"""

import ipaddress
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")
VALID_HOST = re.compile(r"(?:[a-z0-9-]+\.)*[a-z0-9-]+", re.IGNORECASE)


def is_production(settings: Mapping[str, str]) -> bool:
    return settings.get("SENTINANCE_ENV", "development").strip().lower() in {
        "production",
        "prod",
    }


def canonical_origin(value: str, *, production: bool) -> str:
    """Accept a complete web origin, never a path, wildcard or credential URL."""
    raw = value.strip()
    parts = urlsplit(raw)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
        or "*" in parts.netloc
        or any(c.isspace() for c in raw)
    ):
        raise ValueError(f"Invalid CORS/WebSocket origin: {value!r}")
    # Accessing .port also validates invalid port literals.
    port = parts.port
    host = parts.hostname
    if not host or production and parts.scheme != "https":
        raise ValueError(f"Production origins require valid HTTPS hosts: {value!r}")
    if production and (host in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("Loopback browser origins are not allowed in production")
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return f"{parts.scheme}://{authority}"


def allowed_browser_origins(settings: Mapping[str, str]) -> tuple[str, ...]:
    prod = is_production(settings)
    sources = [] if prod else list(DEV_ORIGINS)
    frontend = settings.get("FRONTEND_URL", "").strip()
    if frontend:
        sources.append(frontend)
    sources.extend(
        value.strip()
        for value in settings.get("CORS_ORIGINS", "").split(",")
        if value.strip()
    )
    if prod and not sources:
        raise ValueError("Production requires FRONTEND_URL or explicit CORS_ORIGINS")
    return tuple(dict.fromkeys(canonical_origin(value, production=prod) for value in sources))


def websocket_origin_allowed(
    incoming: str | None, allowed: tuple[str, ...]
) -> bool:
    # Browsers send Origin during WebSocket handshakes. Missing/null, forged
    # wildcard and suffix-matching origins must never silently be trusted.
    return incoming is not None and incoming in allowed


def trusted_http_hosts(settings: Mapping[str, str]) -> tuple[str, ...] | None:
    raw = settings.get("SENTINANCE_ALLOWED_HOSTS", "")
    if not raw.strip():
        if is_production(settings):
            raise ValueError("Production requires SENTINANCE_ALLOWED_HOSTS")
        return None  # Development-only: allow local TestClient/preview hosts.
    hosts = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",")))
    if any(
        not host
        or host == "*"
        or not VALID_HOST.fullmatch(host)
        for host in hosts
    ):
        raise ValueError("Allowed HTTP hosts must be exact names, not wildcards/URLs")
    return hosts


def trusted_proxy_ips(value: str | None) -> str:
    """Uvicorn 0.24-compatible comma-separated *literal* proxy IP addresses."""
    raw = value if value is not None and value.strip() else "127.0.0.1"
    result = []
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate or candidate == "*":
            raise ValueError("Wildcard/empty forwarded-header proxy trust is forbidden")
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError as exc:
            raise ValueError(f"Trusted proxy must be a literal IP: {candidate!r}") from exc
        if address.is_unspecified:
            raise ValueError("Unspecified bind addresses cannot be trusted proxies")
        result.append(str(address))
    return ",".join(dict.fromkeys(result))
