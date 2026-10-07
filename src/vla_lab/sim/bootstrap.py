"""Set local simulator paths before the simulator modules load."""

import importlib.util
import os
from pathlib import Path

import yaml


def configure_process() -> Path:
    """Select the workspace config and EGL without loading robot assets."""
    root = Path(__file__).resolve().parents[3]
    data = root / ".local-data"
    config = Path(os.environ.get("LIBERO_CONFIG_PATH", data / "libero-config"))
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install the locked environment with just setup")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config.mkdir(parents=True, exist_ok=True)
    target = config / "config.yaml"
    if not target.exists():
        target.write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(package),
                    "bddl_files": str(package / "bddl_files"),
                    "init_states": str(package / "init_files"),
                    "datasets": str(data / "hub" / "lerobot-libero"),
                    "assets": str(data / "libero-assets"),
                }
            )
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", str(data / "mpl"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(data / "hf" / "datasets"))
    return config
