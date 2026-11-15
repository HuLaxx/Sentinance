"""Offline trusted-host, browser WebSocket and proxy-peer security contracts.

Origin restricts browser handshake sites, not authentication of individual
clients. Proxy forwarding is independently gated by the actual network peer.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import main
from security_policy import (
    allowed_browser_origins,
    canonical_origin,
    trusted_http_hosts,
    trusted_proxy_ips,
    websocket_origin_allowed,
)
from server import server_options


def production(**settings):
    return {"SENTINANCE_ENV": "production", **settings}


def test_browser_origins_default_to_local_development_only():
    origins = allowed_browser_origins({})
    assert origins == ("http://localhost:3000", "http://127.0.0.1:3000")
    assert not websocket_origin_allowed(None, origins)
    assert not websocket_origin_allowed("null", origins)
    assert not websocket_origin_allowed("https://localhost:3000.evil.test", origins)
    assert websocket_origin_allowed("http://localhost:3000", origins)


def test_production_frontend_origin_requires_explicit_https():
    origins = allowed_browser_origins(
        production(FRONTEND_URL="https://sentinance.example.org/")
    )
    assert origins == ("https://sentinance.example.org",)
    assert not websocket_origin_allowed("http://sentinance.example.org", origins)
    assert not websocket_origin_allowed("https://evil.example.org", origins)


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "https://*.example.org",
        "https://example.org.evil.test/path",
        "https://example.org?next=evil",
        "https://user:pass@example.org",
        "https://example.org#fragment",
        "ftp://example.org",
        "http://example.org",
        "http://localhost:3000",
    ],
)
def test_production_rejects_unsafe_origin_values(origin):
    with pytest.raises(ValueError):
        canonical_origin(origin, production=True)


def test_production_fails_closed_without_frontend_or_exact_host():
    with pytest.raises(ValueError, match="requires FRONTEND_URL"):
        server_options(production(SENTINANCE_ALLOWED_HOSTS="api.example.org"))
    with pytest.raises(ValueError, match="requires SENTINANCE_ALLOWED_HOSTS"):
        server_options(production(FRONTEND_URL="https://example.org"))


def test_production_host_allowlist_is_exact():
    assert trusted_http_hosts(
        production(SENTINANCE_ALLOWED_HOSTS="api.example.org,api2.example.org")
    ) == ("api.example.org", "api2.example.org")
    with pytest.raises(ValueError):
        trusted_http_hosts(production(SENTINANCE_ALLOWED_HOSTS="*"))
    with pytest.raises(ValueError):
        trusted_http_hosts(production(SENTINANCE_ALLOWED_HOSTS="https://api.example.org"))
    assert trusted_http_hosts({}) is None


def test_host_middleware_blocks_unknown_host_before_route():
    mini_app = FastAPI()
    mini_app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(
            trusted_http_hosts(
                production(SENTINANCE_ALLOWED_HOSTS="api.example.org")
            )
        ),
    )

    @mini_app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(mini_app)
    assert client.get("/ping", headers={"host": "api.example.org"}).status_code == 200
    assert client.get("/ping", headers={"host": "evil.example.org"}).status_code == 400


@pytest.mark.parametrize(
    "proxy",
    ["*", "0.0.0.0", "0.0.0.0/0", "::", "my-ingress", "127.0.0.1, *", ""],
)
def test_trusted_proxy_parser_rejects_unbounded_or_invalid_explicit_values(proxy):
    if proxy == "":
        assert trusted_proxy_ips(proxy) == "127.0.0.1"
    else:
        with pytest.raises(ValueError):
            trusted_proxy_ips(proxy)


def test_uvicorn_options_never_trust_arbitrary_forwarded_hosts():
    cfg = server_options(
        production(
            FRONTEND_URL="https://app.example.org",
            SENTINANCE_ALLOWED_HOSTS="api.example.org",
            SENTINANCE_TRUSTED_PROXY_IPS="127.0.0.1,192.0.2.10",
            PORT="9000",
        )
    )
    assert cfg["port"] == 9000
    assert cfg["proxy_headers"] is True
    assert cfg["forwarded_allow_ips"] == "127.0.0.1,192.0.2.10"


@pytest.mark.parametrize("port", ["0", "70000", "not-a-port"])
def test_invalid_listen_port_rejected(port):
    with pytest.raises(ValueError):
        server_options({"PORT": port})


@pytest.mark.asyncio
async def test_untrusted_x_forwarded_headers_do_not_replace_remote_address():
    captured = []

    async def app(scope, _receive, _send):
        captured.append(scope.copy())

    asgi = ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1")
    headers = [
        (b"x-forwarded-for", b"198.51.100.77"),
        (b"x-forwarded-proto", b"https"),
    ]
    untrusted = {
        "type": "http",
        "headers": headers,
        "client": ("203.0.113.21", 4000),
        "scheme": "http",
    }
    await asgi(untrusted, None, None)
    assert captured[-1]["client"] == ("203.0.113.21", 4000)
    assert captured[-1]["scheme"] == "http"

    trusted = {
        "type": "http",
        "headers": headers,
        "client": ("127.0.0.1", 4000),
        "scheme": "http",
    }
    await asgi(trusted, None, None)
    assert captured[-1]["client"][0] == "198.51.100.77"
    assert captured[-1]["scheme"] == "https"


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [None, "null", "https://attacker.invalid"])
async def test_websocket_handshake_rejects_untrusted_or_missing_origin(
    monkeypatch, origin
):
    connect = AsyncMock()
    disconnect = Mock()
    monkeypatch.setattr(main.manager, "connect", connect)
    monkeypatch.setattr(main.manager, "disconnect", disconnect)
    headers = {"origin": origin} if origin is not None else {}
    ws = SimpleNamespace(headers=headers, close=AsyncMock())

    await main.websocket_prices(ws)
    ws.close.assert_awaited_once_with(code=1008, reason="Untrusted WebSocket origin")
    connect.assert_not_awaited()
    disconnect.assert_not_called()


@pytest.mark.asyncio
async def test_allowed_websocket_ping_and_connection_cleanup(monkeypatch):
    monkeypatch.setattr(
        main, "CURRENT_PRICES", {"BTCUSDT": {"symbol": "BTCUSDT", "price": 1}}
    )
    connect = AsyncMock()
    disconnect = Mock()
    monkeypatch.setattr(main.manager, "connect", connect)
    monkeypatch.setattr(main.manager, "disconnect", disconnect)
    ws = SimpleNamespace(
        headers={"origin": "http://localhost:3000"},
        close=AsyncMock(),
        send_json=AsyncMock(),
        receive_text=AsyncMock(side_effect=["ping", WebSocketDisconnect(code=1000)]),
    )

    await main.websocket_prices(ws)
    connect.assert_awaited_once_with(ws)
    disconnect.assert_called_once_with(ws)
    assert ws.send_json.await_count == 2
    assert ws.send_json.await_args_list[-1].args[0] == {"type": "pong"}
    ws.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_price_send_failure_also_cleans_up_socket(monkeypatch):
    monkeypatch.setattr(
        main, "CURRENT_PRICES", {"BTCUSDT": {"symbol": "BTCUSDT", "price": 1}}
    )
    connect = AsyncMock()
    disconnect = Mock()
    monkeypatch.setattr(main.manager, "connect", connect)
    monkeypatch.setattr(main.manager, "disconnect", disconnect)
    ws = SimpleNamespace(
        headers={"origin": "http://localhost:3000"},
        close=AsyncMock(),
        send_json=AsyncMock(side_effect=OSError("disconnected before initial send")),
    )
    with pytest.raises(OSError, match="disconnected"):
        await main.websocket_prices(ws)
    connect.assert_awaited_once_with(ws)
    disconnect.assert_called_once_with(ws)
