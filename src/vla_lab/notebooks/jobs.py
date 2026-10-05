"""Reconnect notebook cells to detached local jobs."""

from __future__ import annotations

from pathlib import Path

from vla_lab.jobs.manager import JobManager
from vla_lab.types import JSONValue


def manager(workspace: str | Path = ".") -> JobManager:
    """Open the stable local job store for this course workspace."""
    root = Path(workspace).expanduser().resolve()
    return JobManager(root / ".local-data" / "jobs")


def submit(
    *,
    config: dict[str, JSONValue],
    purpose: str,
    seed: int,
    command: list[str],
    workspace: str | Path = ".",
    repeat: bool = False,
) -> dict[str, JSONValue]:
    """Submit once; a repeated cell returns its prior run."""
    return manager(workspace).submit(
        config=config, purpose=purpose, seed=seed, command=command, repeat=repeat
    )


def status(run_id: str, workspace: str | Path = ".") -> dict[str, JSONValue]:
    """Read one run after the notebook kernel restarts."""
    return manager(workspace).status(run_id)


def logs(run_id: str, workspace: str | Path = ".", *, lines: int = 40) -> str:
    """Read saved output from one run."""
    return manager(workspace).log_tail(run_id, lines=lines)


def cancel(run_id: str, workspace: str | Path = ".") -> dict[str, JSONValue]:
    """Stop the verified worker process group."""
    return manager(workspace).cancel(run_id)


def retrieve(
    run_id: str, destination: str | Path, workspace: str | Path = ".", *, max_bytes: int = 2**30
) -> list[str]:
    """Copy bounded saved artifacts after a job completes."""
    return manager(workspace).retrieve(run_id, destination, max_bytes=max_bytes)
