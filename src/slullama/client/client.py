"""SlulamaClient — Ollama-compatible client that tunnels through slullama."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

from slullama.client.tunnel import ClientTunnel
from slullama.config import ClientConfig, Config

log = logging.getLogger("slullama.client")


class SlulamaClient:
    """High-level client for slullama.

    Manages an SSH tunnel to the head node and provides an
    Ollama-compatible API. Works as both a context manager and
    a lazy singleton for litellm integration.

    Usage (context manager)::

        async with SlulamaClient(host="user@headnode") as client:
            resp = await client.chat("qwen3.5:9b", messages=[...])

    Usage (singleton for litellm)::

        client = SlulamaClient.get_default()
        url = client.ollama_url  # "http://localhost:11434"
    """

    _default: SlulamaClient | None = None

    def __init__(
        self,
        host: str | None = None,
        server_port: int | None = None,
        token: str | None = None,
        local_port: int | None = None,
        config: ClientConfig | None = None,
    ) -> None:
        if config is None:
            cfg = Config.load()
            config = cfg.client

        self._config = config
        if host is not None:
            self._config.host = host
        if server_port is not None:
            self._config.server_port = server_port
        if token is not None:
            self._config.token = token
        if local_port is not None:
            self._config.local_port = local_port

        if not self._config.host:
            raise ValueError(
                "No host configured. Set SLULLAMA_HOST or configure client.host "
                "in ~/.config/slullama/config.toml"
            )

        self._tunnel = ClientTunnel(
            host=self._config.host,
            server_port=self._config.server_port,
            local_port=self._config.local_port,
        )
        self._http: httpx.AsyncClient | None = None

    # ── Singleton ──────────────────────────────────────────────

    @classmethod
    def get_default(cls) -> SlulamaClient:
        """Return (or create) the singleton instance."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def reset_default(cls) -> None:
        cls._default = None

    # ── Context manager ────────────────────────────────────────

    async def __aenter__(self) -> SlulamaClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── Connection lifecycle ───────────────────────────────────

    async def connect(self) -> None:
        """Open the SSH tunnel."""
        await self._tunnel.open()

    def connect_sync(self) -> None:
        """Open the SSH tunnel (synchronous, for litellm)."""
        self._tunnel.open_sync()

    async def close(self) -> None:
        """Close tunnel and HTTP client."""
        await self._tunnel.close()
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    @property
    def is_connected(self) -> bool:
        return self._tunnel.is_alive

    @property
    def ollama_url(self) -> str:
        """Local URL that speaks Ollama API (through the tunnel)."""
        return self._tunnel.url

    # ── HTTP helpers ───────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if self._config.token:
            return {"Authorization": f"Bearer {self._config.token}"}
        return {}

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        return self._http

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            # Try sync connect for convenience
            self.connect_sync()

    # ── Ollama-compatible API ──────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Ollama /api/chat endpoint."""
        self._ensure_connected()
        http = await self._get_http()

        payload = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if stream:
            return self._stream_chat(http, payload)

        resp = await http.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def _stream_chat(
        self, http: httpx.AsyncClient, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream /api/chat responses."""
        import json

        async with http.stream(
            "POST",
            f"{self.ollama_url}/api/chat",
            json=payload,
            headers=self._auth_headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def tags(self) -> dict[str, Any]:
        """Ollama /api/tags endpoint — list models."""
        self._ensure_connected()
        http = await self._get_http()
        resp = await http.get(
            f"{self.ollama_url}/api/tags",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict[str, Any]:
        """slullama /slullama/status endpoint."""
        self._ensure_connected()
        http = await self._get_http()
        resp = await http.get(
            f"{self.ollama_url}/slullama/status",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()
