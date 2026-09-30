"""Check detached jobs, reconnect, duplicate runs, and cancellation."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vla_lab.jobs.manager import JobManager


def _wait(manager: JobManager, run_id: str, phase: str, timeout_s: float = 10) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = manager.status(run_id)
        if status["phase"] == phase:
            return status
        time.sleep(0.05)
    raise AssertionError(f"Job did not reach {phase}: {manager.status(run_id)}")


def test_job_survives_manager_reconnect_and_avoids_duplicate(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('{run_dir}/artifact.txt').write_text('ok'); print('finished')",
    ]
    first = manager.submit(config={"steps": 1}, purpose="probe", seed=7, command=command)
    reconnected = JobManager(tmp_path / "jobs")
    second = reconnected.submit(config={"steps": 1}, purpose="probe", seed=7, command=command)
    assert first["run_id"] == second["run_id"]
    done = _wait(reconnected, first["run_id"], "succeeded")
    assert done["phase"] == "succeeded"
    assert "finished" in reconnected.log_tail(first["run_id"])
    assert reconnected.artifacts(first["run_id"]) == ["artifact.txt"]
    with pytest.raises(ValueError, match="exceeds"):
        reconnected.retrieve(first["run_id"], tmp_path / "too-small", max_bytes=1)
    output = tmp_path / "copy"
    assert reconnected.retrieve(first["run_id"], output, max_bytes=2) == ["artifact.txt"]
    assert (output / "artifact.txt").read_text() == "ok"
    with pytest.raises(FileExistsError, match="not empty"):
        reconnected.retrieve(first["run_id"], output, max_bytes=2)
    same = reconnected.submit(config={"steps": 1}, purpose="probe", seed=7, command=command)
    assert same["run_id"] == first["run_id"]
    repeated = reconnected.submit(
        config={"steps": 1}, purpose="probe", seed=7, command=command, repeat=True
    )
    assert repeated["run_id"] != first["run_id"]
    _wait(reconnected, repeated["run_id"], "succeeded")


def test_cancel_stops_verified_process_group(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    started = manager.submit(config={"wait": 30}, purpose="cancel-probe", seed=0, command=command)
    try:
        _wait(manager, started["run_id"], "running")
        stopped = manager.cancel(started["run_id"])
        assert stopped["phase"] == "cancelled"
        assert not stopped["process_alive"]
    finally:
        if manager.status(started["run_id"])["process_alive"]:
            manager.cancel(started["run_id"])


def test_stale_identity_starts_a_new_run(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    command = [sys.executable, "-c", "print('done')"]
    first = manager.submit(config={"a": 1}, purpose="stale", seed=0, command=command)
    _wait(manager, first["run_id"], "succeeded")
    run_dir = Path(first["run_dir"])
    (run_dir / "status.json").write_text(json.dumps({"phase": "running"}))
    identity = json.loads((run_dir / "identity.json").read_text())
    identity["start_ticks"] = -1
    (run_dir / "identity.json").write_text(json.dumps(identity))
    next_run = manager.submit(config={"a": 1}, purpose="stale", seed=0, command=command)
    assert next_run["run_id"] != first["run_id"]
    _wait(manager, next_run["run_id"], "succeeded")


def test_simultaneous_submit_uses_one_run(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    command = [sys.executable, "-c", "import time; time.sleep(1)"]

    def start() -> dict:
        return manager.submit(config={"steps": 2}, purpose="race", seed=4, command=command)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: start(), range(2)))
    assert first["run_id"] == second["run_id"]
    manager.cancel(first["run_id"])


def test_invalid_run_id_cannot_read_outside_job_store(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    with pytest.raises(ValueError, match="Run ID"):
        manager.status("../outside")
