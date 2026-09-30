"""Check the local LIBERO setup before the package is imported."""

from pathlib import Path

import pytest
import yaml

from vla_lab.sim.setup import prepare_local_libero, select_local_assets


def test_prepare_local_libero_writes_paths_before_import(tmp_path: Path, monkeypatch) -> None:
    package_root = tmp_path / "package"
    (package_root / "bddl_files").mkdir(parents=True)
    (package_root / "init_files").mkdir()
    data_root = tmp_path / "data"
    (data_root / "libero-assets").mkdir(parents=True)
    (data_root / "libero-assets" / "wall.xml").write_text("<mujoco/>")
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    config_path = prepare_local_libero(data_root, package_root=package_root)

    assert config_path == data_root / "libero-config" / "config.yaml"
    assert yaml.safe_load(config_path.read_text()) == {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(data_root / "hub" / "lerobot-libero"),
        "assets": str(data_root / "libero-assets"),
    }
    assert __import__("os").environ["LIBERO_CONFIG_PATH"] == str(config_path.parent)
    assert __import__("os").environ["MUJOCO_GL"] == "egl"


def test_prepare_local_libero_rejects_missing_assets(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    (package_root / "bddl_files").mkdir(parents=True)
    (package_root / "init_files").mkdir()

    with pytest.raises(FileNotFoundError, match="libero-assets"):
        prepare_local_libero(tmp_path / "data", package_root=package_root)


def test_select_local_assets_overrides_libero_default_cache(tmp_path: Path, monkeypatch) -> None:
    package_root = tmp_path / "package"
    (package_root / "bddl_files").mkdir(parents=True)
    (package_root / "init_files").mkdir()
    data_root = tmp_path / "data"
    (data_root / "libero-assets").mkdir(parents=True)
    (data_root / "libero-assets" / "wall.xml").write_text("<mujoco/>")
    prepare_local_libero(data_root, package_root=package_root)

    import libero.libero as libero_runtime

    monkeypatch.setattr(libero_runtime, "_assets_path_cache", None)
    select_local_assets(data_root)
    assert Path(libero_runtime.get_assets_path()) == data_root / "libero-assets"
