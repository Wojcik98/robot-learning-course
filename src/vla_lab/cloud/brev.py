"""Use the installed Brev CLI for one named course instance."""

import json
import shlex
import subprocess
import time
from collections.abc import Sequence
from typing import Protocol

from vla_lab.cloud.budget import CloudQuote
from vla_lab.types import JSONValue


class CommandRunner(Protocol):
    def __call__(
        self, args: Sequence[str], *, capture_output: bool, text: bool, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]: ...


class BrevCLI:
    """Run bounded Brev commands without a shell."""

    def __init__(self, runner: CommandRunner = subprocess.run) -> None:
        self.runner = runner

    def _run(self, *args: str, timeout_s: int = 60) -> str:
        result = self.runner(
            ["brev", *args], text=True, capture_output=True, check=True, timeout=timeout_s
        )
        return result.stdout

    def quote(self, instance_type: str) -> CloudQuote:
        """Read a fresh stoppable 24 GB GPU quote by its exact type."""
        rows = json.loads(self._run("search", "gpu", "--min-vram", "24", "--stoppable", "--json"))
        chosen = next((row for row in rows if row.get("type") == instance_type), None)
        if chosen is None:
            raise ValueError(f"Instance type has no current stoppable quote: {instance_type}")
        disk = chosen.get("target_disk_gb", chosen.get("disk_min_gb", 0))
        disk_monthly = chosen.get("disk_price_per_gb_mo", 0)
        return CloudQuote(
            instance_type=instance_type,
            hourly_usd=float(chosen["price_per_hour"]),
            disk_hourly_usd=float(disk) * float(disk_monthly) / (30 * 24),
            vram_gb=float(chosen["vram_per_gpu_gb"]),
            stoppable=bool(chosen["stoppable"]),
            checked_at=time.time(),
        )

    def instances(self) -> list[dict[str, JSONValue]]:
        """Read active organization instances; an empty list is valid."""
        content = json.loads(self._run("ls", "instances", "--json"))
        rows = content.get("workspaces")
        if rows is None:
            return []
        if not isinstance(rows, list):
            raise TypeError("Brev instance response is not a list")
        return rows

    def state(self, name: str) -> str:
        """Read the named instance's provider state without guessing aliases."""
        matches = [
            row
            for row in self.instances()
            if name in (row.get("name"), row.get("id"), row.get("workspace_name"))
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one Brev instance named {name}, found {len(matches)}")
        state = matches[0].get("status", matches[0].get("state"))
        if isinstance(state, dict):
            state = state.get("status", state.get("state"))
        if not isinstance(state, str):
            raise TypeError("Brev did not return a readable provider state")
        return state.lower()

    def stop(self, name: str) -> None:
        """Request a provider stop for one named course instance through a PTY."""
        if not name or name.startswith("-"):
            raise ValueError("Brev stop needs one exact instance name")
        command = shlex.join(["brev", "stop", name])
        self.runner(
            ["script", "--quiet", "--return", "--command", command, "/dev/null"],
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
        )
