"""Check exact Brev quote and stop commands without a paid instance."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from vla_lab.cloud.brev import BrevCLI


def test_quote_uses_exact_stoppable_type_and_disk_charge() -> None:
    calls = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [
                    {
                        "type": "g2-standard-4:nvidia-l4:1",
                        "price_per_hour": 0.845421,
                        "vram_per_gpu_gb": 24,
                        "stoppable": True,
                        "target_disk_gb": 10,
                        "disk_price_per_gb_mo": 0.14965,
                    }
                ]
            ),
            "",
        )

    quote = BrevCLI(runner).quote("g2-standard-4:nvidia-l4:1")
    assert quote.disk_hourly_usd > 0
    assert calls == [["brev", "search", "gpu", "--min-vram", "24", "--stoppable", "--json"]]


def test_list_empty_and_stop_one_named_instance() -> None:
    calls = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output = '{"workspaces": null}' if command[1] == "ls" else "stopping"
        return subprocess.CompletedProcess(command, 0, output, "")

    provider = BrevCLI(runner)
    assert provider.instances() == []
    provider.stop("course-probe")
    assert calls[-1] == [
        "script",
        "--quiet",
        "--return",
        "--command",
        "brev stop course-probe",
        "/dev/null",
    ]
    with pytest.raises(ValueError, match="exact instance"):
        provider.stop("--all")


def test_stop_uses_a_tty_and_passes_a_metacharacter_name_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_path = tmp_path / "brev-call.json"
    sentinel = tmp_path / "injected"
    fake_brev = bin_dir / "brev"
    fake_brev.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['BREV_CALL_RECORD'], 'w') as stream:\n"
        "    json.dump({'stdout_isatty': os.isatty(1), 'args': sys.argv[1:]}, stream)\n"
        "print('stopping')\n"
    )
    fake_brev.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BREV_CALL_RECORD", str(record_path))

    exact_name = f"course; touch {sentinel}"
    BrevCLI().stop(exact_name)

    call = json.loads(record_path.read_text())
    assert call["stdout_isatty"] is True
    assert call["args"] == ["stop", exact_name]
    assert not sentinel.exists()


def test_state_reads_only_the_exact_named_instance() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "workspaces": [
                        {"name": "course-probe", "status": "RUNNING"},
                        {"name": "other", "status": "STOPPED"},
                    ]
                }
            ),
            "",
        )

    provider = BrevCLI(runner)
    assert provider.state("course-probe") == "running"
    with pytest.raises(ValueError, match="found 0"):
        provider.state("missing")
