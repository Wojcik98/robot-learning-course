"""Check Diffusion Policy shapes and training-only statistics."""

import json
from pathlib import Path

import pytest
import torch

from vla_lab.policy.diffusion import make_diffusion_config, make_diffusion_processors

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_diffusion_processors_use_frozen_min_max_and_imagenet_values() -> None:
    config = make_diffusion_config(device="cpu")
    preprocessor, postprocessor = make_diffusion_processors(config, "bowl_plate", CONFIGS)
    stats = json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())
    state_min = torch.tensor(stats["modalities"]["observation.state"]["min"])
    state_max = torch.tensor(stats["modalities"]["observation.state"]["max"])
    action_min = torch.tensor(stats["modalities"]["action"]["min"])
    action_max = torch.tensor(stats["modalities"]["action"]["max"])
    midpoint = (state_min + state_max) / 2
    action_midpoint = (action_min + action_max) / 2
    image = torch.full((1, 2, 3, 256, 256), 0.485)

    batch = preprocessor(
        {
            "observation.state": midpoint.repeat(1, 2, 1),
            "observation.images.image": image,
            "observation.images.image2": image,
            "action": action_midpoint.repeat(1, config.horizon, 1),
        }
    )

    assert config.n_obs_steps == 2
    assert config.horizon == 16
    assert config.n_action_steps == 8
    assert batch["observation.state"].shape == (1, 2, 8)
    assert torch.allclose(batch["observation.state"], torch.zeros((1, 2, 8)), atol=1e-5)
    assert torch.allclose(batch["action"], torch.zeros((1, 16, 7)), atol=1e-5)
    assert batch["observation.images.image"][0, 0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(postprocessor(torch.zeros((1, 7))), action_midpoint.unsqueeze(0))


def test_diffusion_processors_reject_stats_for_another_split(tmp_path: Path) -> None:
    target = tmp_path / "data" / "stats" / "bowl_plate.json"
    target.parent.mkdir(parents=True)
    stats = json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())
    stats["split_sha256"] = "0" * 64
    target.write_text(json.dumps(stats))
    (tmp_path / "data" / "episodes.yaml").write_text(
        (CONFIGS / "data" / "episodes.yaml").read_text()
    )

    with pytest.raises(ValueError, match="split hash"):
        make_diffusion_processors(make_diffusion_config(device="cpu"), "bowl_plate", tmp_path)
