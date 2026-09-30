"""Check that the doctor reports a missing or changed ACT bundle."""

from pathlib import Path

from vla_lab.doctor import check_act_references, check_data_files


def test_reference_check_reports_missing_checkpoint(tmp_path: Path) -> None:
    manifest = {
        "tasks": {
            "bowl_plate": {
                "checkpoint_path": "missing",
                "checkpoint_files_sha256": {"model.safetensors": "0" * 64},
            }
        }
    }
    result = check_act_references(tmp_path, manifest)
    assert not result[0]["ok"]
    assert "missing" in result[0]["detail"]


def test_reference_check_rejects_changed_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    manifest = {
        "tasks": {
            "bowl_plate": {
                "checkpoint_path": "checkpoint",
                "checkpoint_files_sha256": {"model.safetensors": "0" * 64},
            }
        }
    }
    result = check_act_references(tmp_path, manifest)
    assert not result[0]["ok"]
    assert "hash" in result[0]["detail"]


def test_data_check_reports_missing_assets(tmp_path: Path) -> None:
    result = check_data_files(tmp_path)
    assert not result[0]["ok"]
