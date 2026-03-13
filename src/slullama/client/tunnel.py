"""SSH tunnel from client machine to head node."""

from __future__ import annotations

import asyncio
import atexit
import logging
import subprocess

log = logging.getLogger("slullama.client.tunnel")


class ClientTunnel:
    """Manage an SSH port-forward from the local machine to the head node.

    Forwards local_port on this machine to server_port on the head node,
    so local Ollama-speaking tools hit the slullama proxy transparently.
    """

    def __init__(
        self,
        host: str,
        server_port: int,
        local_port: int = 11434,
    ) -> None:
        self.host = host
        self.server_port = server_port
        self.local_port = local_port
        self._proc: subprocess.Popen | None = None
        self._async_proc: asyncio.subprocess.Process | None = None

    # ── Synchronous API (for litellm / atexit) ─────────────────

    def open_sync(self) -> None:
        """Open the SSH tunnel (blocking, subprocess)."""
        if self._proc and self._proc.poll() is None:
            return

        cmd = [
            "ssh",
            "-N",
            "-L", f"{self.local_port}:localhost:{self.server_port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            self.host,
        ]
        log.info(
            "Opening SSH tunnel: localhost:%d → %s:%d",
            self.local_port, self.host, self.server_port,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Register cleanup
        atexit.register(self.close_sync)
        # Give SSH time to establish
        import time
        time.sleep(1.5)

        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode() if self._proc.stderr else ""
            raise RuntimeError(
                f"SSH tunnel to {self.host} failed (rc={self._proc.returncode}): {stderr}"
            )
        log.info("SSH tunnel established (sync)")

    def close_sync(self) -> None:
        """Close the SSH tunnel (synchronous)."""
        if self._proc and self._proc.poll() is None:
            log.info("Closing SSH tunnel to %s (sync)", self.host)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    # ── Async API ──────────────────────────────────────────────

    async def open(self) -> None:
        """Open the SSH tunnel (async)."""
        if self._async_proc and self._async_proc.returncode is None:
            return

        cmd = [
            "ssh",
            "-N",
            "-L", f"{self.local_port}:localhost:{self.server_port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            self.host,
        ]
        log.info(
            "Opening SSH tunnel: localhost:%d → %s:%d",
            self.local_port, self.host, self.server_port,
        )
        self._async_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.sleep(1.5)

        if self._async_proc.returncode is not None:
            stderr = ""
            if self._async_proc.stderr:
                stderr = (await self._async_proc.stderr.read()).decode()
            raise RuntimeError(
                f"SSH tunnel to {self.host} failed "
                f"(rc={self._async_proc.returncode}): {stderr}"
            )
        log.info("SSH tunnel established (async)")

    async def close(self) -> None:
        """Close the SSH tunnel (async)."""
        if self._async_proc and self._async_proc.returncode is None:
            log.info("Closing SSH tunnel to %s (async)", self.host)
            self._async_proc.terminate()
            try:
                await asyncio.wait_for(self._async_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._async_proc.kill()
                await self._async_proc.wait()
        self._async_proc = None

    @property
    def is_alive(self) -> bool:
        if self._async_proc and self._async_proc.returncode is None:
            return True
        if self._proc and self._proc.poll() is None:
            return True
        return False

    @property
    def url(self) -> str:
        return f"http://localhost:{self.local_port}"
