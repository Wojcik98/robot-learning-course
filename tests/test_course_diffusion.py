"""Check native checkpoint export for a learner Diffusion U-Net."""

import pytest
import torch

from vla_lab.teaching.course_diffusion import learner_unet_state_for_lerobot


def test_learner_unet_state_uses_native_names_and_copies_values() -> None:
    source = {
        "time_mlp.0.weight": torch.arange(4, dtype=torch.float32),
        "down_blocks.0.1.bias": torch.tensor([2.0]),
        "mid_blocks.1.weight": torch.tensor([[3.0]]),
        "up_blocks.0.0.weight": torch.tensor([[4.0]]),
        "head.1.bias": torch.tensor([5.0]),
    }

    converted = learner_unet_state_for_lerobot(source)

    assert set(converted) == {
        "diffusion_step_encoder.0.weight",
        "down_modules.0.1.bias",
        "mid_modules.1.weight",
        "up_modules.0.0.weight",
        "final_conv.1.bias",
    }
    for target, value in zip(converted.values(), source.values(), strict=True):
        assert torch.equal(target, value)
        assert target.data_ptr() != value.data_ptr()


def test_learner_unet_export_rejects_unknown_root_name() -> None:
    with pytest.raises(KeyError, match="unknown learner U-Net key"):
        learner_unet_state_for_lerobot({"new_stage.weight": torch.ones(1)})
