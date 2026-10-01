"""Reject an invalid learner continuation before GPU work starts."""

from pathlib import Path

import pytest

from vla_lab.training.smolvla import _validate_resume_metadata, train_smolvla_stage

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train" / "smolvla_bowl_plate.yaml"


def test_c2_needs_a_c1_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C1 checkpoint"):
        train_smolvla_stage(
            config_path=CONFIG,
            configs_root=ROOT / "configs",
            data_root=tmp_path,
            vlm_dir=tmp_path / "vlm",
            output_dir=tmp_path / "run",
            stage="c2",
        )


def test_c1_rejects_a_resume_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C1 starts"):
        train_smolvla_stage(
            config_path=CONFIG,
            configs_root=ROOT / "configs",
            data_root=tmp_path,
            vlm_dir=tmp_path / "vlm",
            output_dir=tmp_path / "run",
            stage="c1",
            resume_from=tmp_path / "checkpoint",
        )


def test_c2_can_extend_the_same_run_after_a_development_check() -> None:
    expected = {"task_alias": "bowl_plate", "seed": 12, "config_sha256": "new"}
    source_run = {
        "task_alias": "bowl_plate",
        "seed": 12,
        "config_sha256": "old",
        "stage": "c2",
        "steps": 500,
        "resumed_from_step": 100,
    }
    source_manifest = {
        "stage": "c2",
        "step": 500,
        "task_alias": "bowl_plate",
        "split_sha256": "split",
        "vlm_revision": "revision",
        "vlm_sha256": "vlm",
        "chunk_size": 50,
        "n_action_steps": 10,
    }
    assert (
        _validate_resume_metadata(
            source_run,
            source_manifest,
            expected,
            c1_steps=100,
            target_steps=2000,
            split_sha256="split",
            vlm_revision="revision",
            vlm_sha256="vlm",
        )
        == 500
    )
    with pytest.raises(ValueError, match="exceed"):
        _validate_resume_metadata(
            source_run,
            source_manifest,
            expected,
            c1_steps=100,
            target_steps=500,
            split_sha256="split",
            vlm_revision="revision",
            vlm_sha256="vlm",
        )
    source_run["resumed_from_step"] = 99
    with pytest.raises(ValueError, match="C1 step"):
        _validate_resume_metadata(
            source_run,
            source_manifest,
            expected,
            c1_steps=100,
            target_steps=2000,
            split_sha256="split",
            vlm_revision="revision",
            vlm_sha256="vlm",
        )
