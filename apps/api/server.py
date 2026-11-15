"""Uvicorn runtime entry point with explicitly trusted reverse-proxy peers.

A browser-supplied X-Forwarded-For is untrusted unless the network peer
matches a configured literal IP. Wildcards, hostnames and broad CIDRs fail.
"""

import os
from collections.abc import Mapping
from typing import Any

import uvicorn

from security_policy import (
    allowed_browser_origins,
    is_production,
    trusted_http_hosts,
    trusted_proxy_ips,
)


def server_options(settings: Mapping[str, str]) -> dict[str, Any]:
    """Validate deployment settings before binding a listening socket."""
    if is_production(settings):
        # Fail startup rather than silently running production with dev CORS
        # or no host allowlist.
        allowed_browser_origins(settings)
        trusted_http_hosts(settings)
    raw_port = settings.get("PORT", settings.get("API_PORT", "8000"))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("API listen port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("API listen port must be between 1 and 65535")

    return {
        "host": "0.0.0.0",
        "port": port,
        "proxy_headers": True,
        "forwarded_allow_ips": trusted_proxy_ips(
            settings.get("SENTINANCE_TRUSTED_PROXY_IPS")
        ),
    }


if __name__ == "__main__":
    # The container entry point is production-safe by default. Developers
    # must explicitly choose SENTINANCE_ENV=development for local use.
    os.environ.setdefault("SENTINANCE_ENV", "production")
    uvicorn.run("main:app", **server_options(os.environ))
