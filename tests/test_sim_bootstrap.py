"""Check simulator path setup in a new process."""

import os
import subprocess
import sys
from pathlib import Path


def test_simulator_import_uses_workspace_config_without_user_prompt(tmp_path):
    env = os.environ.copy()
    env.pop("LIBERO_CONFIG_PATH", None)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vla_lab.sim.episode; import os; print(os.environ['LIBERO_CONFIG_PATH'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=45,
    )
    assert process.returncode == 0, process.stderr
    assert str(Path.cwd() / ".local-data/libero-config") in process.stdout
