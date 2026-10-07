"""Submit, reconnect to, and stop detached local jobs."""

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vla_lab.types import JSONValue


def _atomic_json(path: Path, data: dict[str, JSONValue]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, JSONValue]:
    return json.loads(path.read_text())


def process_start_ticks(pid: int) -> int | None:
    """Read the Linux process start time to reject a reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def same_process(pid: int, start_ticks: int | None) -> bool:
    """Confirm that a PID still belongs to the recorded process."""
    return start_ticks is not None and process_start_ticks(pid) == start_ticks


@contextmanager
def _index_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".index.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _job_key(config: dict[str, JSONValue], purpose: str, seed: int, command: list[str]) -> str:
    content = json.dumps(
        {"config": config, "purpose": purpose, "seed": seed, "command": command},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


class JobManager:
    """Keep local job records under one durable data directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.runs = self.root / "runs"
        self.keys = self.root / "keys"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.keys.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
            raise ValueError("Run ID must be a 32-character hexadecimal value")
        return self.runs / run_id

    def submit(
        self,
        *,
        config: dict[str, JSONValue],
        purpose: str,
        seed: int,
        command: list[str],
        repeat: bool = False,
    ) -> dict[str, JSONValue]:
        """Start one detached run or return the matching prior run."""
        if not purpose.strip() or not command or not all(command):
            raise ValueError("Job needs a purpose and a command argument list")
        key = _job_key(config, purpose, seed, command)
        key_file = self.keys / f"{key}.json"
        with _index_lock(self.root):
            if key_file.is_file():
                prior = _read_json(key_file)
                state = self.status(prior["run_id"])
                if state["phase"] in ("starting", "running") and state["process_alive"]:
                    return state
                if state["phase"] in ("succeeded", "failed", "cancelled") and not repeat:
                    return state
            run_id = uuid.uuid4().hex
            run_dir = self.runs / run_id
            run_dir.mkdir()
            spec = {
                "schema_version": 1,
                "run_id": run_id,
                "key": key,
                "config": config,
                "purpose": purpose,
                "seed": seed,
                "command": command,
                "cwd": str(Path.cwd().resolve()),
                "created_at": time.time(),
            }
            _atomic_json(run_dir / "spec.json", spec)
            command_line = [sys.executable, "-m", "vla_lab.jobs.worker", str(run_dir)]
            process = subprocess.Popen(
                command_line,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=Path.cwd(),
            )
            start_ticks = process_start_ticks(process.pid)
            identity = {"run_id": run_id, "pid": process.pid, "start_ticks": start_ticks}
            _atomic_json(key_file, identity)
            _atomic_json(run_dir / "identity.json", identity)
            return self.status(run_id)

    def status(self, run_id: str) -> dict[str, JSONValue]:
        """Read the latest status and verify the recorded worker identity."""
        run_dir = self._run_dir(run_id)
        identity = _read_json(run_dir / "identity.json")
        status_path = run_dir / "status.json"
        status = _read_json(status_path) if status_path.is_file() else {"phase": "starting"}
        alive = same_process(identity["pid"], identity["start_ticks"])
        if status["phase"] in ("starting", "running") and not alive:
            status = {**status, "phase": "interrupted"}
        return {
            **status,
            "run_id": run_id,
            "process_alive": alive,
            "run_dir": str(run_dir),
            "log": str(run_dir / "job.log"),
        }

    def cancel(self, run_id: str, *, timeout_s: float = 5.0) -> dict[str, JSONValue]:
        """Stop a verified worker process group and mark its run cancelled."""
        run_dir = self._run_dir(run_id)
        identity = _read_json(run_dir / "identity.json")
        pid, ticks = identity["pid"], identity["start_ticks"]
        if same_process(pid, ticks):
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + timeout_s
            while same_process(pid, ticks) and time.monotonic() < deadline:
                time.sleep(0.05)
            if same_process(pid, ticks):
                os.killpg(pid, signal.SIGKILL)
        if same_process(pid, ticks):
            raise RuntimeError("Job process did not stop")
        current = self.status(run_id)
        if current["phase"] in ("succeeded", "failed"):
            return current
        record = {"phase": "cancelled", "updated_at": time.time()}
        _atomic_json(run_dir / "status.json", record)
        return self.status(run_id)

    def artifacts(self, run_id: str) -> list[str]:
        """List files that a job saved outside its control records."""
        run_dir = self._run_dir(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        controls = {"spec.json", "identity.json", "status.json", "job.log"}
        return sorted(
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*")
            if path.is_file() and not path.is_symlink() and path.name not in controls
        )

    def retrieve(
        self, run_id: str, destination: str | Path, *, max_bytes: int = 2**30
    ) -> list[str]:
        """Copy bounded run artifacts to a new local directory."""
        source = self._run_dir(run_id)
        files = self.artifacts(run_id)
        total = sum((source / name).stat().st_size for name in files)
        if total > max_bytes:
            raise ValueError(f"Artifact total {total} bytes exceeds the {max_bytes}-byte limit")
        target = Path(destination).expanduser().resolve()
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Artifact destination is not empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, output)
        return files

    def log_tail(self, run_id: str, *, lines: int = 40) -> str:
        """Read the end of a saved job log after a notebook reconnects."""
        path = self._run_dir(run_id) / "job.log"
        return "\n".join(path.read_text().splitlines()[-lines:]) if path.is_file() else ""
