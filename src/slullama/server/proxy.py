"""HTTP reverse proxy — Ollama-compatible API on the head node.

Handles:
- Transparent proxying of all Ollama API endpoints (streaming included)
- Bearer token authentication
- On-demand Slurm job submission + SSH tunnel setup
- Keep-alive: extends Slurm job timeout on every request
- Idle watchdog: tears down job + tunnel after inactivity
- /slullama/status endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from aiohttp import ClientSession, ClientTimeout, web

from slullama.config import ServerConfig
from slullama.server.slurm import JobState, SlurmManager
from slullama.server.tunnel import SshTunnel

log = logging.getLogger("slullama.proxy")

# Port on the head node used to reach ollama through the SSH tunnel.
# This is distinct from the server's own listening port.
_TUNNEL_LOCAL_PORT = 19434


class OllamaProxy:
    """Reverse proxy that fronts a Slurm-managed Ollama instance."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.slurm = SlurmManager(config)
        self.tunnel: SshTunnel | None = None
        self.last_activity: float = 0.0
        self._boot_lock = asyncio.Lock()
        self._http: ClientSession | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._started_at: float = 0.0
        self._request_count: int = 0

    # ── Backend lifecycle ──────────────────────────────────────

    async def _ensure_backend(self) -> str:
        """Ensure Slurm job + SSH tunnel are up. Returns ollama base URL."""
        async with self._boot_lock:
            # Already running?
            if self.tunnel and self.tunnel.is_alive:
                if await self._health_check():
                    return self.tunnel.url
                # Tunnel alive but ollama not responding — tear down and retry
                log.warning("Tunnel alive but ollama unhealthy, recycling")
                await self._teardown()

            # Submit job if needed
            if not self.slurm.is_active:
                await self.slurm.submit()

            # Wait for RUNNING
            info = await self.slurm.wait_for_running(timeout=300)

            # Open SSH tunnel to compute node
            self.tunnel = SshTunnel(
                node=info.node,
                remote_port=self.config.ollama.port,
                local_port=_TUNNEL_LOCAL_PORT,
            )
            await self.tunnel.open()

            # Wait for ollama to be healthy through the tunnel
            await self._wait_healthy(timeout=120)

            log.info("Backend ready: %s via %s", self.tunnel.url, info.node)
            return self.tunnel.url

    async def _health_check(self) -> bool:
        """Quick health check against ollama through the tunnel."""
        if not self.tunnel:
            return False
        try:
            http = await self._get_http()
            resp = await http.get(
                f"{self.tunnel.url}/api/tags",
                timeout=ClientTimeout(total=5),
            )
            return resp.status == 200
        except Exception:
            return False

    async def _wait_healthy(self, timeout: int = 120) -> None:
        """Poll ollama health until it responds."""
        elapsed = 0
        while elapsed < timeout:
            if await self._health_check():
                log.info("Ollama healthy after %ds", elapsed)
                return
            await asyncio.sleep(2)
            elapsed += 2
        raise TimeoutError(f"Ollama not healthy after {timeout}s")

    async def _teardown(self) -> None:
        """Tear down tunnel and cancel Slurm job."""
        if self.tunnel:
            await self.tunnel.close()
            self.tunnel = None
        if self.slurm.is_active:
            await self.slurm.cancel()

    async def _get_http(self) -> ClientSession:
        if self._http is None or self._http.closed:
            self._http = ClientSession(timeout=ClientTimeout(total=600))
        return self._http

    # ── Auth ───────────────────────────────────────────────────

    def _check_auth(self, request: web.Request) -> bool:
        """Validate bearer token if configured."""
        if not self.config.token:
            return True
        auth = request.headers.get("Authorization", "")
        return auth == f"Bearer {self.config.token}"

    # ── Request handlers ───────────────────────────────────────

    async def handle_proxy(self, request: web.Request) -> web.StreamResponse:
        """Proxy any Ollama API request, streaming the response."""
        if not self._check_auth(request):
            return web.json_response(
                {"error": "unauthorized"}, status=401
            )

        try:
            backend_url = await self._ensure_backend()
        except (RuntimeError, TimeoutError) as exc:
            log.error("Backend not available: %s", exc)
            return web.json_response(
                {"error": f"backend unavailable: {exc}"},
                status=503,
            )

        # Track activity
        self.last_activity = time.monotonic()
        self._request_count += 1

        # Extend Slurm job timeout
        if self.config.slurm.keep_alive == "extend":
            asyncio.create_task(self.slurm.extend_time())

        # Forward the request
        target = f"{backend_url}{request.path_qs}"
        body = await request.read()
        http = await self._get_http()

        try:
            resp = await http.request(
                method=request.method,
                url=target,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in ("host", "authorization")
                },
                data=body,
                timeout=ClientTimeout(total=600),
            )
        except Exception as exc:
            log.error("Proxy request failed: %s", exc)
            return web.json_response(
                {"error": f"proxy error: {exc}"}, status=502
            )

        # Stream the response back
        proxy_resp = web.StreamResponse(
            status=resp.status,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding")
            },
        )
        await proxy_resp.prepare(request)

        async for chunk in resp.content.iter_any():
            await proxy_resp.write(chunk)

        await proxy_resp.write_eof()
        return proxy_resp

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return server status as JSON."""
        if not self._check_auth(request):
            return web.json_response({"error": "unauthorized"}, status=401)

        job_info = await self.slurm.query() if self.slurm.is_active else None
        now = time.monotonic()
        idle_seconds = int(now - self.last_activity) if self.last_activity else -1

        status = {
            "version": "0.1.0",
            "uptime_seconds": int(now - self._started_at) if self._started_at else 0,
            "request_count": self._request_count,
            "idle_seconds": idle_seconds,
            "idle_timeout_minutes": self.config.slurm.idle_timeout,
            "tunnel_alive": self.tunnel.is_alive if self.tunnel else False,
            "job": {
                "id": job_info.job_id if job_info else None,
                "state": job_info.state.value if job_info else "none",
                "node": job_info.node if job_info else None,
                "time_left": job_info.time_left if job_info else None,
            },
            "config": {
                "partition": self.config.slurm.partition,
                "gres": self.config.slurm.gres,
                "ollama_port": self.config.ollama.port,
                "keep_alive": self.config.slurm.keep_alive,
            },
        }
        return web.json_response(status)

    # ── Idle watchdog ──────────────────────────────────────────

    async def _idle_watchdog(self) -> None:
        """Background task: tear down backend after idle timeout."""
        interval = 30  # check every 30s
        while True:
            await asyncio.sleep(interval)

            if not self.slurm.is_active:
                continue

            if not self.last_activity:
                continue

            idle = time.monotonic() - self.last_activity
            timeout = self.config.slurm.idle_timeout * 60

            if idle >= timeout:
                log.info(
                    "Idle for %.0fs (timeout=%ds), tearing down backend",
                    idle, timeout,
                )
                await self._teardown()

    # ── Main entry point ───────────────────────────────────────

    async def serve(self) -> None:
        """Start the proxy server (blocking)."""
        self._started_at = time.monotonic()

        app = web.Application()
        app.router.add_route("GET", "/slullama/status", self.handle_status)
        # Catch-all: proxy everything else to ollama
        app.router.add_route("*", "/{path:.*}", self.handle_proxy)

        app.on_shutdown.append(self._on_shutdown)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "0.0.0.0", self.config.port)
        await site.start()

        log.info("slullama proxy listening on 0.0.0.0:%d", self.config.port)

        # Start idle watchdog
        self._watchdog_task = asyncio.create_task(self._idle_watchdog())

        # Run forever
        try:
            await asyncio.Event().wait()
        finally:
            self._watchdog_task.cancel()
            await runner.cleanup()

    async def _on_shutdown(self, _app: web.Application) -> None:
        """Clean up on server shutdown."""
        log.info("Shutting down slullama proxy")
        await self._teardown()
        if self._http and not self._http.closed:
            await self._http.close()
