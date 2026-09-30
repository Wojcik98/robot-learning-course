"""Check the actual offline Bash completion command."""

import os
import subprocess
import sys
from pathlib import Path


def _complete(words: str, index: int) -> list[str]:
    environment = {
        **os.environ,
        "_ROBOT_LAB_COMPLETE": "complete_bash",
        "COMP_WORDS": words,
        "COMP_CWORD": str(index),
    }
    process = subprocess.run(
        [str(Path(sys.executable).parent / "robot-lab")],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.splitlines()


def test_demo_values_complete_without_network() -> None:
    assert "--state-id" in _complete("robot-lab demo --sta", 2)
    assert _complete("robot-lab demo --task bo", 3) == ["bowl_plate", "bowl_stove"]
    assert _complete("robot-lab demo --profile lo", 3) == ["local"]
    assert "19" in _complete("robot-lab demo --state-id 1", 3)
    assert _complete("robot-lab eval --policy di", 3) == ["diffusion"]
