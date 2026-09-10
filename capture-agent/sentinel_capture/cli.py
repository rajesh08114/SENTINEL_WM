"""`sentinel-capture` CLI."""
from __future__ import annotations

import asyncio
import socket

import typer

from .agent import run_agent
from .interfaces import list_interfaces

app = typer.Typer(add_completion=False, help="SENTINEL-WM live-capture agent")


@app.command()
def interfaces() -> None:
    """List local network interfaces."""
    rows = list_interfaces()
    w = max((len(i.name) for i in rows), default=4)
    typer.echo(f"{'NAME':<{w}}  {'IPv4':<15}  UP   DESCRIPTION")
    for i in rows:
        typer.echo(f"{i.name:<{w}}  {i.ipv4 or '-':<15}  "
                   f"{'yes' if i.is_up else 'no ':<3}  {i.description}")


@app.command()
def run(backend: str = typer.Option("ws://localhost:8000",
                                    help="backend base ws URL (no /agent suffix)"),
        name: str = typer.Option(socket.gethostname(), help="agent display name")) -> None:
    """Connect to the backend and wait for a capture command from the console."""
    typer.echo(f"[agent] {name} -> {backend}/agent  (Ctrl+C to quit)")
    try:
        asyncio.run(run_agent(backend, name))
    except KeyboardInterrupt:
        typer.echo("\n[agent] bye")


if __name__ == "__main__":
    app()
