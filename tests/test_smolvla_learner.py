"""Check the learner model configuration and source-weight audit."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vla_lab.policy import smolvla_learner

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_learner_config_loads_vlm_and_trains_action_expert(tmp_path: Path) -> None:
    config = smolvla_learner.make_learner_config(tmp_path, device="cpu")

    assert config.load_vlm_weights is True
    assert config.vlm_model_name == str(tmp_path.resolve())
    assert config.num_vlm_layers == 16
    assert config.expert_width_multiplier == 0.75
    assert config.freeze_vision_encoder is True
    assert config.train_expert_only is True
    assert config.train_state_proj is True
    assert config.n_action_steps == 10
    assert config.action_feature.shape == (7,)


def test_learner_processors_use_verified_training_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = smolvla_learner.make_learner_config(tmp_path, device="cpu")
    captured = {}

    def make_processors(policy_config, dataset_stats):
        captured.update(dataset_stats)
        return "pre", "post"

    monkeypatch.setattr(smolvla_learner, "make_smolvla_pre_post_processors", make_processors)
    assert smolvla_learner.make_learner_processors(config, "bowl_plate", CONFIGS) == (
        "pre",
        "post",
    )
    assert captured["observation.state"]["mean"].shape == (8,)
    assert captured["action"]["std"].shape == (7,)


def test_loaded_vlm_audit_rejects_changed_weight(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    save_file({"part.weight": torch.tensor([1.0, 2.0])}, weights)
    model = SimpleNamespace(state_dict=lambda: {"part.weight": torch.tensor([1.0, 3.0])})

    with pytest.raises(ValueError, match="differs"):
        smolvla_learner.audit_loaded_vlm(model, weights)


def test_parameter_audit_rejects_trainable_backbone() -> None:
    policy = SimpleNamespace(
        named_parameters=lambda: iter(
            [("model.vlm_with_expert.vlm.model.weight", torch.nn.Parameter(torch.ones(2)))]
        )
    )
    with pytest.raises(ValueError, match="backbone"):
        smolvla_learner.audit_learner_parameter_groups(policy)
