"""Check that ACT uses only the frozen training statistics."""

import json
from pathlib import Path

import pytest
import torch

from vla_lab.policy.act import make_act_config, make_act_processors

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_act_processors_apply_frozen_state_and_action_statistics() -> None:
    config = make_act_config(device="cpu")
    preprocessor, postprocessor = make_act_processors(config, "bowl_plate", CONFIGS)
    stats = json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())
    state = torch.tensor(stats["modalities"]["observation.state"]["mean"], dtype=torch.float32)
    state += torch.tensor(stats["modalities"]["observation.state"]["std"])
    action = torch.tensor(stats["modalities"]["action"]["mean"], dtype=torch.float32)
    action += torch.tensor(stats["modalities"]["action"]["std"])
    image = torch.full((3, 256, 256), 0.25)

    batch = preprocessor(
        {
            "observation.state": state,
            "observation.images.image": image,
            "observation.images.image2": image,
            "action": action.repeat(config.chunk_size, 1),
        }
    )

    assert torch.allclose(batch["observation.state"], torch.ones((1, 8)), atol=1e-5)
    assert torch.allclose(batch["action"], torch.ones((1, config.chunk_size, 7)), atol=1e-5)
    assert torch.all(batch["observation.images.image"] == 0.25)
    physical_action = postprocessor(torch.zeros((1, 7)))
    assert physical_action.shape == (1, 7)
    assert physical_action[0, 0].item() == pytest.approx(0.1972228995947525, abs=1e-6)


def test_act_processors_reject_stats_for_another_split(tmp_path: Path) -> None:
    source = CONFIGS / "data" / "stats" / "bowl_plate.json"
    target = tmp_path / "data" / "stats" / "bowl_plate.json"
    target.parent.mkdir(parents=True)
    payload = json.loads(source.read_text())
    payload["split_sha256"] = "0" * 64
    target.write_text(json.dumps(payload))
    (tmp_path / "data" / "episodes.yaml").write_text(
        (CONFIGS / "data" / "episodes.yaml").read_text()
    )

    with pytest.raises(ValueError, match="split hash"):
        make_act_processors(make_act_config(device="cpu"), "bowl_plate", tmp_path)
