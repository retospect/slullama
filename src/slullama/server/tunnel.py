"""SSH tunnel from head node to compute node."""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("slullama.tunnel")


class SshTunnel:
    """Manage an SSH port-forward to a compute node.

    Forwards local_port on the head node to remote_port on the compute node.
    """

    def __init__(
        self,
        node: str,
        remote_port: int,
        local_port: int = 0,
    ) -> None:
        self.node = node
        self.remote_port = remote_port
        self.local_port = local_port or remote_port
        self._proc: asyncio.subprocess.Process | None = None

    async def open(self) -> None:
        """Open the SSH tunnel."""
        if self._proc and self._proc.returncode is None:
            return

        cmd = [
            "ssh",
            "-N",  # no remote command
            "-L", f"{self.local_port}:localhost:{self.remote_port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            self.node,
        ]
        log.info(
            "Opening SSH tunnel: localhost:%d → %s:%d",
            self.local_port, self.node, self.remote_port,
        )
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Give SSH a moment to establish the tunnel
        await asyncio.sleep(1.5)

        if self._proc.returncode is not None:
            stderr = (await self._proc.stderr.read()).decode() if self._proc.stderr else ""
            raise RuntimeError(
                f"SSH tunnel to {self.node} failed (rc={self._proc.returncode}): {stderr}"
            )
        log.info("SSH tunnel established")

    async def close(self) -> None:
        """Close the SSH tunnel."""
        if self._proc and self._proc.returncode is None:
            log.info("Closing SSH tunnel to %s", self.node)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.local_port}"
