"""Check that an ACT checkpoint includes its processors and split identity."""

import json
from pathlib import Path

import pytest
import torch
from lerobot.policies.act import ACTPolicy

from vla_lab.policy.act import make_act_config, make_act_processors
from vla_lab.policy.bundle import load_act_bundle, save_act_bundle

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_saved_act_bundle_reloads_weights_and_action_scale(tmp_path: Path) -> None:
    config = make_act_config(device="cpu")
    policy = ACTPolicy(config)
    preprocessor, postprocessor = make_act_processors(config, "bowl_plate", CONFIGS)
    policy.eval()

    save_act_bundle(policy, preprocessor, postprocessor, tmp_path, "bowl_plate", CONFIGS)
    reloaded, new_preprocessor, new_postprocessor, manifest = load_act_bundle(tmp_path)

    assert manifest["task_alias"] == "bowl_plate"
    assert (
        manifest["split_sha256"]
        == json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())["split_sha256"]
    )
    first_key = next(iter(policy.state_dict()))
    assert torch.equal(policy.state_dict()[first_key], reloaded.state_dict()[first_key])
    state = torch.tensor(
        json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())["modalities"][
            "observation.state"
        ]["mean"],
        dtype=torch.float32,
    )
    image = torch.zeros((3, 256, 256))
    observation = {
        "observation.state": state,
        "observation.images.image": image,
        "observation.images.image2": image,
    }
    assert torch.allclose(
        preprocessor(observation)["observation.state"],
        new_preprocessor(observation)["observation.state"],
    )
    assert torch.allclose(
        postprocessor(torch.zeros((1, 7))), new_postprocessor(torch.zeros((1, 7)))
    )


def test_bundle_load_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_act_bundle(tmp_path)
