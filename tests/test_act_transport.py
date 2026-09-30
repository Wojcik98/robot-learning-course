"""Verify complete ACT bundle transfer and reject changed archive data."""

import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

from vla_lab.policy.transport import install_act_references, pack_act_references


def _workspace(root: Path) -> Path:
    (root / "manifests").mkdir(parents=True)
    bundle = root / ".local-data" / "act" / "checkpoint"
    bundle.mkdir(parents=True)
    (bundle / "model.safetensors").write_bytes(b"saved policy")
    digest = hashlib.sha256(b"saved policy").hexdigest()
    (root / "manifests" / "act_references.yaml").write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "bowl_plate": {
                        "checkpoint_path": ".local-data/act/checkpoint",
                        "checkpoint_files_sha256": {"model.safetensors": digest},
                    }
                }
            }
        )
    )
    return root


def test_pack_and_install_restores_verified_bundle(tmp_path: Path) -> None:
    source = _workspace(tmp_path / "source")
    archive = tmp_path / "reference.zip"
    packed = pack_act_references(source, archive)
    target = tmp_path / "target"
    (target / "manifests").mkdir(parents=True)
    (target / "manifests" / "act_references.yaml").write_bytes(
        (source / "manifests" / "act_references.yaml").read_bytes()
    )
    installed = install_act_references(target, archive)
    assert packed["sha256"] == installed["sha256"]
    assert (target / ".local-data/act/checkpoint/model.safetensors").read_bytes() == b"saved policy"


def test_restore_archive_outside_target_workspace(tmp_path: Path) -> None:
    source = _workspace(tmp_path / "source")
    archive = source / "reference.zip"
    pack_act_references(source, archive)
    target = tmp_path / "target"
    (target / "manifests").mkdir(parents=True)
    (target / "manifests" / "act_references.yaml").write_bytes(
        (source / "manifests" / "act_references.yaml").read_bytes()
    )
    install_act_references(target, archive)
    assert (target / ".local-data/act/checkpoint/model.safetensors").is_file()


def test_changed_archive_cannot_install_any_file(tmp_path: Path) -> None:
    source = _workspace(tmp_path / "source")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("bowl_plate/model.safetensors", b"changed policy")
    with pytest.raises(ValueError, match="hash differs"):
        install_act_references(source, archive)
