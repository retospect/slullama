"""slullama — Shared Ollama gateway over Slurm."""

__version__ = "0.1.0"

from slullama.client.client import SlulamaClient
from slullama.config import ClientConfig, Config, OllamaConfig, ServerConfig, SlurmConfig

__all__ = [
    "SlulamaClient",
    "Config",
    "ServerConfig",
    "ClientConfig",
    "SlurmConfig",
    "OllamaConfig",
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
