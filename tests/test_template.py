"""Tests for sbatch template rendering."""

from __future__ import annotations

from slullama.config import OllamaConfig, ServerConfig, SlurmConfig
from slullama.template import render_template


def test_default_template():
    """Default template renders without errors."""
    config = ServerConfig()
    script = render_template(config)
    assert "#!/bin/bash" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "ollama serve" in script
    assert "SLULLAMA_READY" in script


def test_copy_binary():
    """Template includes copy/cleanup commands when configured."""
    config = ServerConfig(
        ollama=OllamaConfig(
            copy_binary=True,
            copy_source="/shared/ollama-bin",
            cleanup_binary=True,
        ),
    )
    script = render_template(config)
    assert "cp" in script
    assert "/shared/ollama-bin" in script
    assert "rm -f" in script


def test_pre_pull():
    """Template includes model pull commands."""
    config = ServerConfig(
        ollama=OllamaConfig(pre_pull=["qwen3.5:9b", "llama3:8b"]),
    )
    script = render_template(config)
    assert 'pull "qwen3.5:9b"' in script
    assert 'pull "llama3:8b"' in script


def test_models_dir():
    """Template exports OLLAMA_MODELS when configured."""
    config = ServerConfig(
        ollama=OllamaConfig(models_dir="/data/models"),
    )
    script = render_template(config)
    assert 'OLLAMA_MODELS="/data/models"' in script


def test_extra_sbatch_args():
    """Extra sbatch args appear in the script."""
    config = ServerConfig(
        slurm=SlurmConfig(
            mem="64G",
            extra_args=["--constraint=a100", "--exclusive"],
        ),
    )
    script = render_template(config)
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --constraint=a100" in script
    assert "#SBATCH --exclusive" in script
