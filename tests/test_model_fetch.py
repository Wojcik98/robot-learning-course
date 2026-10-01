"""Check exact public model revisions and local file hashes."""

import hashlib
from pathlib import Path

import pytest
import yaml

from vla_lab.data import models


def _files(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "course"
    data = tmp_path / "data"
    (workspace / "manifests").mkdir(parents=True)
    manifest = {}
    for key, (name, required) in models.MODEL_FILES.items():
        directory = data / "models" / name
        directory.mkdir(parents=True)
        for filename in required:
            (directory / filename).write_bytes(
                b"source" if filename == "model.safetensors" else b"file"
            )
        manifest[key] = {
            "repo": f"source/{name}",
            "revision": "a" * 40,
            "model_sha256": hashlib.sha256(b"source").hexdigest(),
        }
    (workspace / "manifests" / "smolvla_reference.yaml").write_text(yaml.safe_dump(manifest))
    return workspace, data


def test_offline_model_fetch_verifies_both_sources_without_network(tmp_path: Path) -> None:
    workspace, data = _files(tmp_path)
    report = models.fetch_pinned_models(workspace, data, offline=True)
    assert report["offline"] is True
    assert set(report["models"]) == {"published_reference", "learner_start"}
    assert not (data / "hf").exists()


def test_model_fetch_uses_exact_commits_and_rejects_changed_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, data = _files(tmp_path)
    calls = []
    monkeypatch.setattr(models, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    models.fetch_pinned_models(workspace, data)
    assert len(calls) == 2
    assert all(call["revision"] == "a" * 40 for call in calls)
    assert all(call["max_workers"] == 4 for call in calls)
    (data / "models" / "smolvla-libero" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model weights"):
        models.fetch_pinned_models(workspace, data, offline=True)


def test_model_fetch_rejects_missing_processor_file(tmp_path: Path) -> None:
    workspace, data = _files(tmp_path)
    (data / "models" / "smolvla-libero" / "policy_preprocessor.json").unlink()
    with pytest.raises(FileNotFoundError, match="policy_preprocessor.json"):
        models.fetch_pinned_models(workspace, data, offline=True)
