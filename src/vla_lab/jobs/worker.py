"""Run one detached command and keep an atomic heartbeat."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from vla_lab.types import JSONValue


def _status(path: Path, **fields: JSONValue) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"updated_at": time.time(), **fields}, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_worker(run_dir: Path) -> int:
    """Run the saved argument list and write terminal status on completion."""
    spec = json.loads((run_dir / "spec.json").read_text())
    status_path = run_dir / "status.json"
    stopped = threading.Event()
    _status(status_path, phase="running")

    def heartbeat() -> None:
        while not stopped.wait(2):
            _status(status_path, phase="running")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    command = [value.replace("{run_dir}", str(run_dir)) for value in spec["command"]]
    environment = {**os.environ, "VLA_JOB_RUN_DIR": str(run_dir)}
    code = 127
    try:
        with (run_dir / "job.log").open("ab") as log:
            try:
                process = subprocess.run(
                    command,
                    cwd=spec["cwd"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                code = process.returncode
            except OSError as error:
                log.write(f"Job could not start: {error}\n".encode())
        return code
    finally:
        stopped.set()
        thread.join(timeout=3)
        _status(status_path, phase="succeeded" if code == 0 else "failed", exit_code=code)


if __name__ == "__main__":
    sys.exit(run_worker(Path(sys.argv[1])))
