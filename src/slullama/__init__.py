"""slullama — Shared Ollama gateway over Slurm."""

from importlib.metadata import version

__version__ = version("slullama")

from slullama.client.client import SlulamaClient
from slullama.config import (
    ClientConfig,
    Config,
    OllamaConfig,
    ServerConfig,
    SlurmConfig,
)

__all__ = [
    "ClientConfig",
    "Config",
    "OllamaConfig",
    "ServerConfig",
    "SlulamaClient",
    "SlurmConfig",
]


def register_litellm() -> None:
    """Register slullama as a litellm custom provider."""
    from slullama.litellm_provider import register

    register()


# Auto-register if litellm is available
try:
    register_litellm()
except Exception:
    pass
