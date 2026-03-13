"""Default sbatch job template for slullama."""

from __future__ import annotations

from pathlib import Path
from string import Template

from slullama.config import ServerConfig

DEFAULT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=slullama
#SBATCH --partition=${partition}
#SBATCH --gres=${gres}
#SBATCH --time=${time}
#SBATCH --output=${log_dir}/slullama-%j.out
#SBATCH --error=${log_dir}/slullama-%j.err
${extra_sbatch}
set -euo pipefail

# ── Copy binary if configured ──────────────────────────────────
${copy_commands}

# ── Environment ────────────────────────────────────────────────
export OLLAMA_HOST="0.0.0.0:${ollama_port}"
${models_env}

# ── Cleanup on exit ────────────────────────────────────────────
cleanup() {
    echo "slullama: shutting down ollama (pid $$OLLAMA_PID)"
    kill $$OLLAMA_PID 2>/dev/null || true
    wait $$OLLAMA_PID 2>/dev/null || true
    ${cleanup_commands}
    echo "slullama: cleanup done"
}
trap cleanup EXIT INT TERM

# ── Start ollama ───────────────────────────────────────────────
echo "slullama: starting ollama on $$(hostname):${ollama_port}"
${ollama_binary} serve &
OLLAMA_PID=$$!

# ── Wait for healthy ──────────────────────────────────────────
for i in $$(seq 1 120); do
    if curl -sf http://localhost:${ollama_port}/api/tags > /dev/null 2>&1; then
        echo "SLULLAMA_READY hostname=$$(hostname) port=${ollama_port}"
        break
    fi
    sleep 1
done

# ── Pre-pull models ────────────────────────────────────────────
${pull_commands}

# ── Keep running ───────────────────────────────────────────────
echo "slullama: ollama running, waiting for termination"
wait $$OLLAMA_PID
"""


def render_template(config: ServerConfig) -> str:
    """Render the sbatch job script from config."""
    if config.job_template:
        path = Path(config.job_template)
        tpl_str = path.read_text()
    else:
        tpl_str = DEFAULT_TEMPLATE

    oc = config.ollama
    sc = config.slurm

    # Build conditional blocks
    copy_commands = ""
    cleanup_commands = ""
    if oc.copy_binary and oc.copy_source:
        dest = f"/tmp/slullama_ollama_$$SLURM_JOB_ID"
        copy_commands = f'cp "{oc.copy_source}" "{dest}" && chmod +x "{dest}"'
        ollama_binary = dest
        if oc.cleanup_binary:
            cleanup_commands = f'rm -f "{dest}"'
    else:
        ollama_binary = oc.binary

    models_env = ""
    if oc.models_dir:
        models_env = f'export OLLAMA_MODELS="{oc.models_dir}"'

    extra_sbatch = ""
    if sc.mem:
        extra_sbatch += f"#SBATCH --mem={sc.mem}\n"
    for arg in sc.extra_args:
        extra_sbatch += f"#SBATCH {arg}\n"

    pull_commands = ""
    for model in oc.pre_pull:
        pull_commands += f'{ollama_binary} pull "{model}"\n'

    mapping = {
        "partition": sc.partition,
        "gres": sc.gres,
        "time": sc.time,
        "log_dir": config.log_dir,
        "extra_sbatch": extra_sbatch.rstrip(),
        "copy_commands": copy_commands or "# (no copy configured)",
        "ollama_port": str(oc.port),
        "models_env": models_env or "# (using default model dir)",
        "cleanup_commands": cleanup_commands or "# (no cleanup configured)",
        "ollama_binary": ollama_binary,
        "pull_commands": pull_commands or "# (no pre-pull configured)",
    }

    return Template(tpl_str).safe_substitute(mapping)
