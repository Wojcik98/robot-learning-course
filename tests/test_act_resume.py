"""Check that ACT continuation keeps the model and optimizer state."""

import json
from pathlib import Path

import pytest
import torch
from lerobot.policies.act import ACTPolicy
from torch.utils.data import Dataset

from vla_lab.training.act import train_act_reference

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class FixedFrames(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation.images.image": torch.full((3, 256, 256), index / 4),
            "observation.images.image2": torch.full((3, 256, 256), index / 4),
            "observation.state": torch.full((8,), index / 10),
            "action": torch.full((20, 7), index / 10),
            "action_is_pad": torch.zeros(20, dtype=torch.bool),
        }


def _weights(path: Path) -> dict[str, torch.Tensor]:
    return ACTPolicy.from_pretrained(path, local_files_only=True).state_dict()


def test_resume_matches_two_uninterrupted_steps(tmp_path: Path) -> None:
    shared = {
        "task_alias": "bowl_plate",
        "configs_root": CONFIGS,
        "data_root": tmp_path / "data",
        "batch_size": 2,
        "device": "cpu",
        "seed": 31,
        "checkpoint_every": 1,
        "dataset": FixedFrames(),
    }
    train_act_reference(output_dir=tmp_path / "full", steps=2, **shared)
    train_act_reference(output_dir=tmp_path / "first", steps=1, **shared)
    source = tmp_path / "first" / "checkpoint-000001"
    resumed = train_act_reference(
        output_dir=tmp_path / "resumed", steps=2, resume_from=source, **shared
    )

    assert resumed.steps == 2
    assert [
        json.loads(line)["step"]
        for line in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
    ] == [2]
    assert json.loads((tmp_path / "resumed" / "run.json").read_text())["resumed_from_step"] == 1
    full = _weights(tmp_path / "full" / "checkpoint-000002")
    continued = _weights(tmp_path / "resumed" / "checkpoint-000002")
    assert all(torch.equal(full[key], continued[key]) for key in full)


def test_resume_rejects_changed_seed(tmp_path: Path) -> None:
    shared = {
        "task_alias": "bowl_plate",
        "configs_root": CONFIGS,
        "data_root": tmp_path / "data",
        "batch_size": 2,
        "device": "cpu",
        "checkpoint_every": 1,
        "dataset": FixedFrames(),
    }
    train_act_reference(output_dir=tmp_path / "first", steps=1, seed=31, **shared)
    with pytest.raises(ValueError, match="seed"):
        train_act_reference(
            output_dir=tmp_path / "wrong",
            steps=2,
            seed=32,
            resume_from=tmp_path / "first" / "checkpoint-000001",
            **shared,
        )
