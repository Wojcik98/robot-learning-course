"""Keep a separate local deadline process for a supervised Brev session."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from vla_lab.cloud.brev import BrevCLI
from vla_lab.cloud.lifecycle import STOPPED_STATES, stop_with_retries
from vla_lab.types import JSONValue


def _save(path: Path, data: dict[str, JSONValue]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def watchdog_once(
    provider: BrevCLI, lease: dict[str, JSONValue], *, now: float
) -> dict[str, JSONValue]:
    """Check one lease and stop its named VM at the saved deadline."""
    name = lease["instance_name"]
    if now >= lease["hard_deadline_at"]:
        states = stop_with_retries(provider, name)
        return {"phase": "stopped", "reason": "hard_deadline", "provider_states": states}
    try:
        state = provider.state(name)
    except ValueError:
        return {"phase": "waiting_for_instance"}
    if state in STOPPED_STATES:
        return {"phase": "stopped", "reason": "provider_state", "provider_states": [state]}
    return {"phase": "watching", "provider_state": state}


def run_watchdog(lease_path: Path, *, poll_s: int = 60) -> None:
    """Poll one saved lease until the provider confirms a stop."""
    provider = BrevCLI()
    status_path = lease_path.with_suffix(".status.json")
    while True:
        lease = json.loads(lease_path.read_text())
        try:
            result = watchdog_once(provider, lease, now=time.time())
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            result = {"phase": "stop_unverified", "error": f"{type(error).__name__}: {error}"}
        _save(status_path, {"updated_at": time.time(), **result})
        if result["phase"] == "stopped":
            return
        time.sleep(poll_s)


def main() -> None:
    """Read one lease path for the detached watchdog process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("lease", type=Path)
    args = parser.parse_args()
    run_watchdog(args.lease)


if __name__ == "__main__":
    main()
