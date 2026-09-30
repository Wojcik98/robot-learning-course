"""Check one complete ACT update and checkpoint."""

from pathlib import Path

import torch
from lerobot.policies.act import ACTPolicy
from torch.utils.data import Dataset

from vla_lab.policy.act import make_act_config
from vla_lab.training.act import train_act_reference

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class SyntheticFrames(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation.images.image": torch.full((3, 256, 256), float(index) / 2),
            "observation.images.image2": torch.full((3, 256, 256), float(index) / 2),
            "observation.state": torch.tensor([0.0] * 8),
            "action": torch.full((20, 7), 0.1 * (index + 1)),
            "action_is_pad": torch.zeros(20, dtype=torch.bool),
        }


def test_one_act_training_step_changes_weights_and_saves_bundle(tmp_path: Path) -> None:
    policy = ACTPolicy(make_act_config(device="cpu"))
    original = {name: value.detach().clone() for name, value in policy.named_parameters()}

    result = train_act_reference(
        task_alias="bowl_plate",
        configs_root=CONFIGS,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "run",
        steps=1,
        batch_size=1,
        device="cpu",
        dataset=SyntheticFrames(),
        policy=policy,
    )

    assert result.steps == 1
    assert result.last_loss > 0
    assert any(
        not torch.equal(original[name], value.detach()) for name, value in policy.named_parameters()
    )
    assert (tmp_path / "run" / "checkpoint-000001" / "manifest.json").is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
