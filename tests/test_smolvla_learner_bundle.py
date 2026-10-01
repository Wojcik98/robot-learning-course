"""Check learner checkpoint provenance and the selected training split."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_lab.policy.smolvla_learner import load_learner_bundle, save_learner_bundle

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class FakeSaver:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def save_pretrained(self, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / self.filename).write_bytes(b"saved")
        if self.filename == "model.safetensors":
            (target / "config.json").write_text('{"chunk_size": 50, "n_action_steps": 10}')
        if self.filename == "policy_preprocessor.json":
            (target / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(
                b"stats"
            )


def test_learner_bundle_records_vlm_and_training_stats(tmp_path: Path) -> None:
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    (vlm / "model.safetensors").write_bytes(b"source")
    digest = hashlib.sha256(b"source").hexdigest()
    policy = FakeSaver("model.safetensors")
    policy.config = SimpleNamespace(chunk_size=50, n_action_steps=10)
    destination = tmp_path / "checkpoint"

    save_learner_bundle(
        policy=policy,
        preprocessor=FakeSaver("policy_preprocessor.json"),
        postprocessor=FakeSaver("policy_postprocessor.json"),
        destination=destination,
        task_alias="bowl_plate",
        configs_root=CONFIGS,
        vlm_dir=vlm,
        vlm_revision="pinned-revision",
        vlm_sha256=digest,
        stage="c1",
        step=100,
        optimizer_state={"step": 100},
    )

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["stage"] == "c1"
    assert manifest["step"] == 100
    assert manifest["vlm_sha256"] == digest
    assert (
        manifest["config_sha256"]
        == hashlib.sha256((destination / "config.json").read_bytes()).hexdigest()
    )
    assert (
        manifest["optimizer_sha256"]
        == hashlib.sha256((destination / "optimizer.pt").read_bytes()).hexdigest()
    )
    assert (
        manifest["split_sha256"]
        == json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())["split_sha256"]
    )
    assert set(manifest["training_stats_sha256"]) == {"observation.state", "action"}
    assert set(manifest["processor_files_sha256"]) == {
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    }
    (destination / "policy_preprocessor_step_5_normalizer_processor.safetensors").write_bytes(
        b"other stats"
    )
    with pytest.raises(ValueError, match="normalizer_processor.safetensors"):
        load_learner_bundle(destination, vlm, device="cpu")


def test_learner_bundle_rejects_a_changed_config(tmp_path: Path) -> None:
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    (vlm / "model.safetensors").write_bytes(b"source")
    policy = FakeSaver("model.safetensors")
    policy.config = SimpleNamespace(chunk_size=50, n_action_steps=10)
    destination = tmp_path / "checkpoint"
    save_learner_bundle(
        policy=policy,
        preprocessor=FakeSaver("policy_preprocessor.json"),
        postprocessor=FakeSaver("policy_postprocessor.json"),
        destination=destination,
        task_alias="bowl_plate",
        configs_root=CONFIGS,
        vlm_dir=vlm,
        vlm_revision="pinned-revision",
        vlm_sha256=hashlib.sha256(b"source").hexdigest(),
        stage="c1",
        step=100,
    )
    (destination / "config.json").write_text('{"chunk_size": 50, "n_action_steps": 1}')
    with pytest.raises(ValueError, match="config hash"):
        load_learner_bundle(destination, vlm, device="cpu")
