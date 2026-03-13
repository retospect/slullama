"""CLI entry points for slullama."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from slullama.config import Config


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the proxy daemon on the head node."""
    config = Config.load(args.config)

    if args.port:
        config.server.port = args.port
    if args.token:
        config.server.token = args.token
    if args.partition:
        config.server.slurm.partition = args.partition
    if args.gres:
        config.server.slurm.gres = args.gres
    if args.idle_timeout is not None:
        config.server.slurm.idle_timeout = args.idle_timeout

    from slullama.server.proxy import OllamaProxy

    proxy = OllamaProxy(config.server)
    asyncio.run(proxy.serve())


def cmd_connect(args: argparse.Namespace) -> None:
    """Open an SSH tunnel to the head node (foreground)."""
    config = Config.load(args.config)

    host = args.host or config.client.host
    if not host:
        print("Error: no host specified. Use: slullama connect user@headnode", file=sys.stderr)
        sys.exit(1)

    config.client.host = host
    if args.port:
        config.client.server_port = args.port
    if args.token:
        config.client.token = args.token
    if args.local_port:
        config.client.local_port = args.local_port

    from slullama.client.tunnel import ClientTunnel

    tunnel = ClientTunnel(
        host=config.client.host,
        server_port=config.client.server_port,
        local_port=config.client.local_port,
    )

    try:
        tunnel.open_sync()
        print(
            f"slullama tunnel open: localhost:{tunnel.local_port} → "
            f"{tunnel.host}:{tunnel.server_port}"
        )
        print("Press Ctrl+C to close.")
        # Block until interrupted
        import signal
        signal.pause()
    except KeyboardInterrupt:
        print("\nClosing tunnel...")
    finally:
        tunnel.close_sync()


def cmd_status(args: argparse.Namespace) -> None:
    """Query the server status."""
    config = Config.load(args.config)

    host = args.host or config.client.host
    port = args.port or config.client.server_port
    token = args.token or config.client.token

    if not host:
        # Try localhost (maybe we're on the head node)
        host = "localhost"

    import httpx

    url = f"http://{host}:{port}/slullama/status"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(json.dumps(data, indent=2))
    except httpx.ConnectError:
        print(f"Error: cannot connect to {url}", file=sys.stderr)
        print("Is the slullama server running?", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"Error: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="slullama",
        description="Shared Ollama gateway over Slurm",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    sub = parser.add_subparsers(dest="command")

    # ── serve ──────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the proxy daemon (head node)")
    serve_p.add_argument("--port", type=int, help="Proxy listen port")
    serve_p.add_argument("--token", type=str, help="Auth token")
    serve_p.add_argument("--partition", type=str, help="Slurm partition")
    serve_p.add_argument("--gres", type=str, help="Slurm GRES (e.g. gpu:1)")
    serve_p.add_argument(
        "--idle-timeout", type=int, help="Minutes before idle teardown"
    )

    # ── connect ────────────────────────────────────────────────
    conn_p = sub.add_parser("connect", help="Open SSH tunnel to head node")
    conn_p.add_argument("host", nargs="?", help="user@headnode")
    conn_p.add_argument("--port", type=int, help="Server port on head node")
    conn_p.add_argument("--token", type=str, help="Auth token")
    conn_p.add_argument("--local-port", type=int, help="Local port (default 11434)")

    # ── status ─────────────────────────────────────────────────
    stat_p = sub.add_parser("status", help="Query server status")
    stat_p.add_argument("host", nargs="?", help="Head node hostname")
    stat_p.add_argument("--port", type=int, help="Server port")
    stat_p.add_argument("--token", type=str, help="Auth token")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "connect":
        cmd_connect(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)
