"""Configuration for slullama — TOML + env var overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlurmConfig:
    """Slurm job parameters."""

    partition: str = "gpu"
    gres: str = "gpu:1"
    mem: str = ""
    time: str = "4:00:00"
    idle_timeout: int = 30  # minutes before idle teardown
    keep_alive: str = "extend"  # "extend" (scontrol update) or "cancel" (resubmit)
    extra_args: list[str] = field(default_factory=list)


@dataclass
class OllamaConfig:
    """Ollama binary and model configuration."""

    port: int = 11434
    binary: str = "ollama"
    models_dir: str = ""
    copy_binary: bool = False
    copy_source: str = ""
    cleanup_binary: bool = False
    pre_pull: list[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    """Server-side (head node) configuration."""

    port: int = 11435
    token: str = ""
    log_dir: str = "/tmp/slullama"
    job_template: str = ""  # path to custom sbatch template; empty = built-in
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class ClientConfig:
    """Client-side (laptop) configuration."""

    host: str = ""  # user@headnode
    server_port: int = 11435
    token: str = ""
    local_port: int = 11434


@dataclass
class Config:
    """Top-level configuration container."""

    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)

    @classmethod
    def default_path(cls) -> Path:
        return Path(os.environ.get("SLULLAMA_CONFIG", "")) or (
            Path.home() / ".config" / "slullama" / "config.toml"
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Load configuration from TOML file with env var overrides."""
        if path is None:
            path = cls.default_path()
        path = Path(path)

        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            data = {}

        server_data = data.get("server", {})
        slurm_data = server_data.pop("slurm", data.get("slurm", {}))
        ollama_data = server_data.pop("ollama", data.get("ollama", {}))
        client_data = data.get("client", {})

        slurm = _from_dict(SlurmConfig, slurm_data)
        ollama = _from_dict(OllamaConfig, ollama_data)
        server = _from_dict(ServerConfig, server_data, slurm=slurm, ollama=ollama)
        client = _from_dict(ClientConfig, client_data)

        cfg = cls(server=server, client=client)
        _apply_env_overrides(cfg)
        return cfg


def _from_dict(cls: type, data: dict[str, Any], **extra: Any) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid}
    filtered.update(extra)
    return cls(**filtered)


def _apply_env_overrides(cfg: Config) -> None:
    """Override config values from SLULLAMA_* environment variables."""
    _env = os.environ.get

    if v := _env("SLULLAMA_TOKEN"):
        cfg.server.token = v
        cfg.client.token = v
    if v := _env("SLULLAMA_SERVER_PORT"):
        cfg.server.port = int(v)
        cfg.client.server_port = int(v)
    if v := _env("SLULLAMA_HOST"):
        cfg.client.host = v
    if v := _env("SLULLAMA_LOCAL_PORT"):
        cfg.client.local_port = int(v)
    if v := _env("SLULLAMA_PARTITION"):
        cfg.server.slurm.partition = v
    if v := _env("SLULLAMA_GRES"):
        cfg.server.slurm.gres = v
    if v := _env("SLULLAMA_IDLE_TIMEOUT"):
        cfg.server.slurm.idle_timeout = int(v)
    if v := _env("SLULLAMA_OLLAMA_BINARY"):
        cfg.server.ollama.binary = v
    if v := _env("SLULLAMA_OLLAMA_PORT"):
        cfg.server.ollama.port = int(v)
