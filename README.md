# slullama

**Shared Ollama gateway over Slurm** — wake a GPU node on demand.

A slumbering llama that wakes on demand. One package, two roles: a **server
daemon** on the Slurm head node and a **client library** on team members'
laptops. The first API request submits a Slurm job, opens an SSH tunnel to
the compute node, and proxies standard Ollama API traffic. When nobody has
made a request for a configurable idle period the job is torn down
automatically. The next request wakes it all back up.

Multiuser by design — the whole team shares one GPU node and one Ollama
process. Authentication is via a shared bearer token.

---

## Architecture

```
laptop (any team member)       head node                  compute node (GPU)
  │                              │                              │
  │──SSH tunnel (port 11434)───▶│                              │
  │                              │──SSH tunnel (port 19434)───▶│
  │                              │   aiohttp reverse proxy     │   ollama serve
  │                              │   (listens 0.0.0.0:11435)   │   (0.0.0.0:11434)
  │                              │   + SlurmManager             │
  │                              │   + idle watchdog            │
  │                              │                              │
```

### Request flow

1. Client code (litellm, `SlulamaClient`, or raw curl) hits
   `localhost:11434` on the laptop.
2. The SSH tunnel forwards the request to the head node proxy on port 11435.
3. The proxy checks the bearer token (`Authorization: Bearer <token>`).
4. If no Slurm job is running, the proxy:
   a. Renders the sbatch template and submits it (`sbatch`).
   b. Polls `scontrol show job` until the job reaches `RUNNING`.
   c. Opens an SSH tunnel from the head node to the compute node's Ollama
      port.
   d. Polls Ollama's `/api/tags` health endpoint through the tunnel until
      healthy.
5. The proxy forwards the request to Ollama (streaming supported).
6. On every proxied request the Slurm job timeout is extended (or the
   last-activity timestamp is recorded, depending on keep-alive strategy).
7. A background watchdog checks idle time every 30 s. If idle ≥
   `idle_timeout` minutes it cancels the job and closes tunnels.
8. The next request goes back to step 4.

### Two SSH hops

Both hops are managed automatically:

- **Laptop → head node**: managed by `ClientTunnel` (client side). Sync or
  async. Opened lazily on first request, closed on process exit / context
  manager exit.
- **Head node → compute node**: managed by `SshTunnel` (server side). Opened
  after the Slurm job reaches RUNNING, closed on idle teardown.

Compute nodes are typically not reachable from outside the cluster, which is
why both hops are necessary.

---

## Quick start

### 1. Head node (server)

```bash
pip install slullama          # or: uv pip install slullama

mkdir -p ~/.config/slullama
```

Create `~/.config/slullama/config.toml`:

```toml
[server]
port = 11435                  # proxy listens here
token = "your-shared-secret"  # bearer token for auth
log_dir = "/tmp/slullama"     # sbatch scripts + job logs

[slurm]
partition = "gpu"
gres = "gpu:1"
mem = ""                      # optional, e.g. "64G"
time = "4:00:00"              # initial job time limit
idle_timeout = 30             # minutes of inactivity before teardown
keep_alive = "extend"         # "extend" or "cancel" (see below)
extra_args = []               # extra #SBATCH lines, e.g. ["--exclusive"]

[ollama]
port = 11434                  # port ollama listens on inside the compute node
binary = "ollama"             # path to ollama binary (must be on compute node)
models_dir = ""               # OLLAMA_MODELS dir; empty = ollama default
pre_pull = []                 # models to pull on cold start, e.g. ["qwen3.5:9b"]

# If the binary is NOT on a shared filesystem, copy it per job:
# copy_binary = true
# copy_source = "/shared/bin/ollama"
# cleanup_binary = true       # rm the copy when the job ends
```

Start the daemon (foreground; use tmux/screen/systemd for persistence):

```bash
slullama serve                       # uses ~/.config/slullama/config.toml
slullama serve --config /etc/slullama.toml   # explicit path
slullama serve --port 9999 --partition gpu-large  # CLI overrides
slullama serve -v                    # debug logging
```

### 2. Laptop (client)

```bash
pip install slullama                 # base client
pip install "slullama[litellm]"      # with litellm integration
```

Create `~/.config/slullama/config.toml` (client section only is fine):

```toml
[client]
host = "youruser@headnode"    # SSH destination
server_port = 11435           # must match server's port
token = "your-shared-secret"  # must match server's token
local_port = 11434            # local Ollama-compatible endpoint
```

#### Option A — litellm (recommended)

```python
import slullama   # auto-registers the "slullama/" provider on import
import litellm

# Synchronous
resp = litellm.completion(
    model="slullama/qwen3.5:9b",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)

# Streaming
for chunk in litellm.completion(
    model="slullama/qwen3.5:9b",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content, end="")
```

Under the hood this opens the SSH tunnel on the first call (lazily), proxies
through the head node, and returns standard litellm `ModelResponse` objects.

#### Option B — Python client (async)

```python
from slullama import SlulamaClient

async with SlulamaClient(host="user@headnode") as client:
    # Ollama-compatible chat
    resp = await client.chat("qwen3.5:9b", messages=[
        {"role": "user", "content": "What is electrochemistry?"},
    ])
    print(resp["message"]["content"])

    # List models
    tags = await client.tags()
    print(tags)

    # Server status (job state, idle time, etc.)
    status = await client.status()
    print(status)

    # Raw Ollama URL for any tool that speaks Ollama
    print(client.ollama_url)   # http://localhost:11434
```

#### Option C — Python client (singleton, no context manager)

```python
from slullama import SlulamaClient

client = SlulamaClient.get_default()   # lazy singleton
client.connect_sync()                  # opens SSH tunnel (blocking)
print(client.ollama_url)               # http://localhost:11434
# Tunnel stays open until process exits (atexit hook).
```

#### Option D — CLI tunnel (foreground)

```bash
slullama connect user@headnode
# Tunnel open: localhost:11434 → headnode:11435
# Press Ctrl+C to close.

# Now any Ollama-speaking tool works:
curl http://localhost:11434/api/tags
```

### 3. Check status

```bash
# From the head node (no tunnel needed):
slullama status

# From a laptop (uses config or args):
slullama status headnode --port 11435 --token your-shared-secret
```

Returns JSON with job state, node, idle time, request count, uptime, etc.

---

## Full configuration reference

Config file: `~/.config/slullama/config.toml`
Override path: `--config <path>` or `SLULLAMA_CONFIG=<path>`

### `[server]` — head node daemon

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | `11435` | Port the proxy listens on |
| `token` | str | `""` | Bearer token for auth (empty = no auth) |
| `log_dir` | str | `"/tmp/slullama"` | Directory for sbatch scripts and job logs |
| `job_template` | str | `""` | Path to custom sbatch template (empty = built-in) |

### `[slurm]` — Slurm job parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `partition` | str | `"gpu"` | Slurm partition |
| `gres` | str | `"gpu:1"` | Generic resources |
| `mem` | str | `""` | Memory (e.g. `"64G"`); empty = cluster default |
| `time` | str | `"4:00:00"` | Initial job time limit |
| `idle_timeout` | int | `30` | Minutes of inactivity before teardown |
| `keep_alive` | str | `"extend"` | `"extend"` or `"cancel"` (see below) |
| `extra_args` | list | `[]` | Extra `#SBATCH` directives (e.g. `["--exclusive"]`) |

### `[ollama]` — Ollama on the compute node

| Key | Type | Default | Description |
|---|---|---|---|
| `port` | int | `11434` | Port ollama listens on |
| `binary` | str | `"ollama"` | Path to ollama binary |
| `models_dir` | str | `""` | `OLLAMA_MODELS` dir; empty = ollama default |
| `copy_binary` | bool | `false` | Copy binary to compute node per job |
| `copy_source` | str | `""` | Source path for copy |
| `cleanup_binary` | bool | `false` | Delete the copy when the job ends |
| `pre_pull` | list | `[]` | Models to pull after ollama starts |

### `[client]` — laptop / workstation

| Key | Type | Default | Description |
|---|---|---|---|
| `host` | str | `""` | SSH destination (`user@headnode`) |
| `server_port` | int | `11435` | Proxy port on head node |
| `token` | str | `""` | Bearer token |
| `local_port` | int | `11434` | Local port (appears as Ollama to tools) |

### Environment variable overrides

Environment variables override the config file. Useful for CI, containers,
or quick one-offs.

| Variable | Overrides |
|---|---|
| `SLULLAMA_CONFIG` | Config file path |
| `SLULLAMA_TOKEN` | `server.token` and `client.token` |
| `SLULLAMA_HOST` | `client.host` |
| `SLULLAMA_SERVER_PORT` | `server.port` and `client.server_port` |
| `SLULLAMA_LOCAL_PORT` | `client.local_port` |
| `SLULLAMA_PARTITION` | `slurm.partition` |
| `SLULLAMA_GRES` | `slurm.gres` |
| `SLULLAMA_IDLE_TIMEOUT` | `slurm.idle_timeout` |
| `SLULLAMA_OLLAMA_BINARY` | `ollama.binary` |
| `SLULLAMA_OLLAMA_PORT` | `ollama.port` |

---

## Keep-alive strategies

| Strategy | How it works | When to use |
|---|---|---|
| `extend` (default) | On each request, runs `scontrol update JobId=X TimeLimit=+Nmin` to push the deadline forward. | Clusters that allow job time extensions. |
| `cancel` | Submits the job with the full `time` limit. On idle timeout, runs `scancel`. Next request resubmits. | Clusters that restrict `scontrol update`. |

Set via `keep_alive` in `[slurm]` config or try both on your cluster — the
server logs a warning if `scontrol update` fails.

---

## Custom job template

The default sbatch template is in `slullama/template.py`. To use your own:

```toml
[server]
job_template = "/path/to/my_template.sh"
```

Templates use Python `string.Template` syntax (`${variable_name}`).
Available variables:

| Variable | Value |
|---|---|
| `${partition}` | Slurm partition |
| `${gres}` | GRES string |
| `${time}` | Time limit |
| `${log_dir}` | Log directory |
| `${extra_sbatch}` | Extra `#SBATCH` lines (rendered from `mem` + `extra_args`) |
| `${copy_commands}` | Binary copy commands (or comment if disabled) |
| `${ollama_port}` | Ollama listen port |
| `${models_env}` | `export OLLAMA_MODELS=...` (or comment if not set) |
| `${cleanup_commands}` | Binary cleanup commands (or comment if disabled) |
| `${ollama_binary}` | Path to ollama binary (may be the copied path) |
| `${pull_commands}` | `ollama pull` commands (or comment if empty) |

Use `$$` for literal `$` in bash (e.g. `$$SLURM_JOB_ID`, `$$!`).

---

## Module map

```
src/slullama/
├── __init__.py              # Public API: SlulamaClient, Config, auto-registers litellm
├── config.py                # Dataclasses: Config, ServerConfig, ClientConfig, SlurmConfig, OllamaConfig
│                            #   Config.load() reads TOML + env overrides
├── template.py              # DEFAULT_TEMPLATE + render_template(ServerConfig) → str
├── cli.py                   # CLI: slullama serve | connect | status
├── litellm_provider.py      # litellm CustomLLM subclass, registers "slullama/" prefix
├── server/
│   ├── slurm.py             # SlurmManager: submit(), query(), extend_time(), cancel(), wait_for_running()
│   ├── tunnel.py            # SshTunnel: head node → compute node SSH port-forward
│   └── proxy.py             # OllamaProxy: aiohttp reverse proxy, auth, idle watchdog, /slullama/status
└── client/
    ├── tunnel.py            # ClientTunnel: laptop → head node SSH port-forward (sync + async)
    └── client.py            # SlulamaClient: high-level client, context manager, singleton
```

### Key classes

- **`SlurmManager`** (`server/slurm.py`): Wraps `sbatch`, `scontrol`,
  `scancel` as async subprocess calls. Tracks job ID and node. Parses
  `scontrol show job` output for state, node, time left.

- **`SshTunnel`** (`server/tunnel.py`): Async SSH port-forward via
  `ssh -N -L`. Used on the head node to reach the compute node.

- **`OllamaProxy`** (`server/proxy.py`): aiohttp web app that:
  - Serves `/slullama/status` (GET) — JSON status endpoint.
  - Proxies everything else (`/{path:.*}`) to Ollama on the compute node.
  - Manages the full lifecycle: job submission → tunnel → health poll →
    proxy → idle watchdog → teardown.
  - `_boot_lock` ensures only one concurrent boot sequence.
  - Idle watchdog runs every 30 s, tears down if idle ≥ timeout.

- **`ClientTunnel`** (`client/tunnel.py`): SSH port-forward from laptop to
  head node. Has both sync (`open_sync`/`close_sync` for litellm/atexit)
  and async (`open`/`close`) APIs.

- **`SlulamaClient`** (`client/client.py`): High-level client.
  - Context manager: `async with SlulamaClient(...) as c:`
  - Lazy singleton: `SlulamaClient.get_default()` for litellm provider use.
  - Methods: `chat()`, `tags()`, `status()`.
  - Property: `ollama_url` — the local URL that speaks Ollama.

- **`SlulamaLLM`** (`litellm_provider.py`): litellm `CustomLLM` subclass
  with `completion()` and `streaming()`. Registered into
  `litellm.custom_provider_map` on `import slullama`.

---

## Integration with acatome-lambic

The existing `LlmClient` in acatome-lambic supports `provider="ollama"` with
a configurable `ollama_url`. Two integration paths:

1. **Via litellm**: Set `provider="slullama"` and `model="qwen3.5:9b"` in
   `LlmConfig`. Since acatome-lambic already uses litellm as a backend, this
   works if slullama is installed with `[litellm]` extra.

2. **Direct**: Run `slullama connect` in a terminal to create a local tunnel,
   then use `provider="ollama"` with `ollama_url="http://localhost:11434"`.
   No code changes needed in acatome-lambic.

---

## Known limitations / future work

- **Cold start latency**: First request after idle waits for Slurm job
  allocation + Ollama startup (30–120 s depending on cluster). The proxy
  blocks the request until ready; it does not return 503.
- **Single GPU node**: Currently manages one Slurm job. Multiple concurrent
  jobs / load balancing is out of scope for v1.
- **No model routing**: All requests go to the same Ollama instance. Model
  loading/unloading is handled by Ollama natively.
- **`scontrol update` permissions**: Some clusters don't allow users to
  extend job time. Use `keep_alive = "cancel"` as a workaround.
- **No HTTPS**: Tunnel traffic is encrypted by SSH; the HTTP proxy itself is
  plaintext. Fine for a cluster environment.
- **Ollama binary distribution**: If no shared filesystem, the binary must
  be copied per job (`copy_binary = true`). This adds cold start time.

---

## CLI reference

```
slullama serve [--port N] [--token T] [--partition P] [--gres G]
               [--idle-timeout M] [--config PATH] [-v]

    Start the proxy daemon on the head node.

slullama connect [user@headnode] [--port N] [--token T]
                 [--local-port N] [--config PATH] [-v]

    Open an SSH tunnel to the head node (foreground, Ctrl+C to close).

slullama status [headnode] [--port N] [--token T] [--config PATH]

    Query the server's /slullama/status endpoint and print JSON.
```

---

## License

MIT
