"""Tests for slullama configuration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from slullama.config import (
    Config,
)


def test_default_config():
    """Default config has sensible values."""
    cfg = Config()
    assert cfg.server.port == 11435
    assert cfg.server.slurm.partition == "gpu"
    assert cfg.server.slurm.gres == "gpu:1"
    assert cfg.server.slurm.idle_timeout == 30
    assert cfg.server.ollama.port == 11434
    assert cfg.server.ollama.binary == "ollama"
    assert cfg.client.local_port == 11434


def test_load_from_toml(tmp_path: Path):
    """Config loads from a TOML file."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        textwrap.dedent("""\
        [server]
        port = 9999
        token = "secret"

        [slurm]
        partition = "gpu-large"
        gres = "gpu:2"
        idle_timeout = 60

        [ollama]
        binary = "/opt/ollama"
        models_dir = "/data/models"
        copy_binary = true
        copy_source = "/shared/ollama"
        cleanup_binary = true

        [client]
        host = "alice@cluster"
        server_port = 9999
        token = "secret"
        local_port = 8888
    """)
    )

    cfg = Config.load(toml)

    assert cfg.server.port == 9999
    assert cfg.server.token == "secret"
    assert cfg.server.slurm.partition == "gpu-large"
    assert cfg.server.slurm.gres == "gpu:2"
    assert cfg.server.slurm.idle_timeout == 60
    assert cfg.server.ollama.binary == "/opt/ollama"
    assert cfg.server.ollama.models_dir == "/data/models"
    assert cfg.server.ollama.copy_binary is True
    assert cfg.server.ollama.copy_source == "/shared/ollama"
    assert cfg.server.ollama.cleanup_binary is True
    assert cfg.client.host == "alice@cluster"
    assert cfg.client.local_port == 8888


def test_load_missing_file(tmp_path: Path):
    """Loading a non-existent config returns defaults."""
    cfg = Config.load(tmp_path / "nope.toml")
    assert cfg.server.port == 11435
    assert cfg.server.slurm.partition == "gpu"


def test_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Environment variables override config values."""
    toml = tmp_path / "config.toml"
    toml.write_text("[server]\nport = 1111\n")

    monkeypatch.setenv("SLULLAMA_TOKEN", "env-token")
    monkeypatch.setenv("SLULLAMA_HOST", "bob@node")
    monkeypatch.setenv("SLULLAMA_SERVER_PORT", "2222")
    monkeypatch.setenv("SLULLAMA_PARTITION", "debug")

    cfg = Config.load(toml)

    assert cfg.server.token == "env-token"
    assert cfg.client.token == "env-token"
    assert cfg.client.host == "bob@node"
    assert cfg.server.port == 2222
    assert cfg.client.server_port == 2222
    assert cfg.server.slurm.partition == "debug"


def test_unknown_keys_ignored(tmp_path: Path):
    """Unknown keys in TOML don't cause errors."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        textwrap.dedent("""\
        [server]
        port = 5555
        unknown_key = "ignored"

        [slurm]
        partition = "test"
        future_option = true
    """)
    )

    cfg = Config.load(toml)
    assert cfg.server.port == 5555
    assert cfg.server.slurm.partition == "test"
