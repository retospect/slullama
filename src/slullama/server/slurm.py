"""Slurm job lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from slullama.config import ServerConfig
from slullama.template import render_template

log = logging.getLogger("slullama.slurm")


class JobState(Enum):
    NONE = "none"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, s: str) -> JobState:
        s = s.strip().upper()
        try:
            return cls(s.lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class JobInfo:
    job_id: str
    state: JobState
    node: str = ""
    time_left: str = ""


class SlurmManager:
    """Submit, monitor, extend, and cancel Slurm jobs running ollama."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.job_id: str | None = None
        self.node: str | None = None
        self._script_path: str | None = None

    async def submit(self) -> str:
        """Render template, write script, sbatch it. Returns job ID."""
        script = render_template(self.config)

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        fd, path = tempfile.mkstemp(
            suffix=".sh", prefix="slullama_job_", dir=str(log_dir)
        )
        with open(fd, "w") as f:
            f.write(script)
        self._script_path = path

        log.info("Submitting sbatch script: %s", path)
        proc = await asyncio.create_subprocess_exec(
            "sbatch",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

        # Parse "Submitted batch job 12345"
        m = re.search(r"(\d+)", stdout.decode())
        if not m:
            raise RuntimeError(
                f"Could not parse job ID from: {stdout.decode().strip()}"
            )

        self.job_id = m.group(1)
        log.info("Submitted Slurm job %s", self.job_id)
        return self.job_id

    async def query(self) -> JobInfo:
        """Query current job state via scontrol."""
        if not self.job_id:
            return JobInfo(job_id="", state=JobState.NONE)

        proc = await asyncio.create_subprocess_exec(
            "scontrol",
            "show",
            "job",
            self.job_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        text = stdout.decode()

        state = JobState.UNKNOWN
        node = ""
        time_left = ""

        if m := re.search(r"JobState=(\S+)", text):
            state = JobState.from_str(m.group(1))
        if (m := re.search(r"BatchHost=(\S+)", text)) or (
            m := re.search(r"NodeList=(\S+)", text)
        ):
            node = m.group(1)
        if m := re.search(r"TimeLeft=(\S+)", text):
            time_left = m.group(1)

        self.node = node if state == JobState.RUNNING else None
        return JobInfo(
            job_id=self.job_id,
            state=state,
            node=node,
            time_left=time_left,
        )

    async def extend_time(self, minutes: int = 0) -> bool:
        """Extend job time limit. Uses idle_timeout from config if minutes=0."""
        if not self.job_id:
            return False

        mins = minutes or self.config.slurm.idle_timeout
        proc = await asyncio.create_subprocess_exec(
            "scontrol",
            "update",
            f"JobId={self.job_id}",
            f"TimeLimit=+{mins}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            log.warning(
                "scontrol update failed (rc=%d): %s — "
                "cluster may restrict time extensions",
                proc.returncode,
                stderr.decode().strip(),
            )
            return False

        log.debug("Extended job %s by %d minutes", self.job_id, mins)
        return True

    async def cancel(self) -> None:
        """Cancel the running job."""
        if not self.job_id:
            return

        log.info("Cancelling job %s", self.job_id)
        proc = await asyncio.create_subprocess_exec(
            "scancel",
            self.job_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        self.job_id = None
        self.node = None

    async def wait_for_running(self, timeout: int = 300) -> JobInfo:
        """Poll until job reaches RUNNING state or timeout."""
        elapsed = 0
        interval = 2
        while elapsed < timeout:
            info = await self.query()
            if info.state == JobState.RUNNING:
                log.info("Job %s running on %s", info.job_id, info.node)
                return info
            if info.state in (
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.COMPLETED,
                JobState.TIMEOUT,
            ):
                raise RuntimeError(
                    f"Job {info.job_id} entered terminal state: {info.state.value}"
                )
            log.debug(
                "Job %s state=%s, waiting... (%ds/%ds)",
                info.job_id,
                info.state.value,
                elapsed,
                timeout,
            )
            await asyncio.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Job {self.job_id} did not start within {timeout}s")

    @property
    def is_active(self) -> bool:
        return self.job_id is not None
