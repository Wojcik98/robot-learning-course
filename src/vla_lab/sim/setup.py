"""Prepare LIBERO paths before importing the simulator."""

import importlib.util
import os
from pathlib import Path

import libero.libero as libero_runtime
import yaml


def _installed_libero_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install the locked LIBERO package before simulator setup")
    return Path(next(iter(spec.submodule_search_locations))) / "libero"


def prepare_local_libero(data_root: str | Path, *, package_root: str | Path | None = None) -> Path:
    """Write the local paths and select EGL before the first LIBERO import."""
    root = Path(data_root).expanduser().resolve()
    installed = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else _installed_libero_root().resolve()
    )
    for required in (installed / "bddl_files", installed / "init_files", root / "libero-assets"):
        if not required.is_dir():
            raise FileNotFoundError(f"Required LIBERO directory is missing: {required}")
    if not (root / "libero-assets" / "wall.xml").is_file():
        raise FileNotFoundError(f"Required LIBERO asset is missing: {root / 'libero-assets'}")

    config_dir = root / "libero-config"
    config_path = config_dir / "config.yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(
        {
            "benchmark_root": str(installed),
            "bddl_files": str(installed / "bddl_files"),
            "init_states": str(installed / "init_files"),
            "datasets": str(root / "hub" / "lerobot-libero"),
            "assets": str(root / "libero-assets"),
        },
        sort_keys=False,
    )
    if not config_path.exists() or config_path.read_text() != content:
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(content)
        temporary.replace(config_path)

    libero_runtime.libero_config_path = str(config_dir)
    libero_runtime.config_file = str(config_path)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = "egl"
    (root / "mpl").mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(root / "mpl")
    return config_path


def select_local_assets(data_root: str | Path) -> Path:
    """Select pinned assets in the LIBERO package's asset path cache."""
    assets = Path(data_root).expanduser().resolve() / "libero-assets"
    if not (assets / "wall.xml").is_file():
        raise FileNotFoundError(f"Required LIBERO asset is missing: {assets}")

    # hf-libero 0.1.4 ignores config.yaml's assets key in get_assets_path().
    libero_runtime._assets_path_cache = str(assets)
    return assets
