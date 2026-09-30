"""Check exact Brev quote and stop commands without a paid instance."""

import json
import subprocess

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
    assert calls[-1] == ["brev", "stop", "course-probe"]
    with pytest.raises(ValueError, match="exact instance"):
        provider.stop("--all")


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
