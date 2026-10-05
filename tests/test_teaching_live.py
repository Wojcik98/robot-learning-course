from __future__ import annotations

import pytest
import torch
from torch import nn

from vla_lab.teaching.live import find_workspace, load_act_batches, make_act_raw_model


def test_find_workspace_walks_up_to_course_manifest(tmp_path):
    root = tmp_path / "course"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "act_references.yaml").write_text("schema_version: 1\n")
    nested = root / "tutorials" / "unit"
    nested.mkdir(parents=True)
    assert find_workspace(nested) == root


def test_find_workspace_reports_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="workspace manifest"):
        find_workspace(tmp_path)


def test_batch_loader_fails_closed_when_local_dataset_is_missing(tmp_path):
    (tmp_path / "configs" / "data").mkdir(parents=True)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "act_references.yaml").write_text("schema_version: 1\n")
    (tmp_path / "configs" / "data" / "episodes.yaml").write_text(
        "dataset_revision: pinned\ntasks:\n  bowl_plate:\n"
        "    train_episode_ids: [1]\n    validation_episode_ids: [2]\n    split_sha256: abc\n"
    )
    with pytest.raises(FileNotFoundError, match="never downloads"):
        load_act_batches("bowl_plate", tmp_path, batch_size=1)


def test_raw_act_bridge_calls_model_without_policy_loss_and_keeps_gradients():
    from lerobot.utils.constants import OBS_IMAGES

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, batch):
            image = batch[OBS_IMAGES][0]
            return image * self.weight, (None, None)

    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"image_features": ("camera",)})()
            self.model = FakeModel()

        def forward(self, _batch):
            raise AssertionError("The helper must call model, not the built-in loss")

    policy = FakePolicy()
    raw_model = make_act_raw_model(policy)
    image = torch.ones(1, 3, 2, 2)
    actions, latent = raw_model({"camera": image})
    assert torch.equal(actions, image * 2)
    assert latent == (None, None)
    actions.sum().backward()
    assert policy.model.weight.grad.item() == image.numel()


def test_published_smolvla_manifest_has_every_loader_field():
    import yaml

    root = find_workspace()
    manifest = yaml.safe_load((root / "manifests" / "smolvla_reference.yaml").read_text())
    assert {
        "revision",
        "local_dir",
        "model_sha256",
        "training_data",
        "selected_n_action_steps",
    } <= manifest["published_reference"].keys()
    assert {"local_dir", "model_sha256"} <= manifest["learner_start"].keys()


def test_published_smolvla_loader_reads_both_local_paths_from_reference_manifest(
    tmp_path, monkeypatch
):
    from pathlib import Path

    import yaml

    from vla_lab.policy import smolvla
    from vla_lab.teaching import live

    (tmp_path / "manifests").mkdir()
    reference = {
        "revision": "pinned-published-revision",
        "local_dir": ".local-data/models/published",
        "model_sha256": "published-hash",
        "training_data": "unknown",
        "selected_n_action_steps": 10,
    }
    learner_start = {
        "revision": "pinned-learner-revision",
        "local_dir": ".local-data/models/learner-vlm",
        "model_sha256": "learner-hash",
    }
    (tmp_path / "manifests" / "smolvla_reference.yaml").write_text(
        yaml.safe_dump({"published_reference": reference, "learner_start": learner_start})
    )
    # The core manifest has repository metadata, but no local path or weight hash.
    (tmp_path / "manifests" / "core.yaml").write_text(
        "published_smolvla:\n  repository: example/model\n  revision: pinned-revision\n"
    )
    expected_paths = {
        (tmp_path / reference["local_dir"] / "model.safetensors").resolve(): "published-hash",
        (tmp_path / learner_start["local_dir"] / "model.safetensors").resolve(): "learner-hash",
    }
    for model_path in expected_paths:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
    seen_hash_paths = []

    def fake_hash(path):
        resolved = path.resolve()
        seen_hash_paths.append(resolved)
        return expected_paths[resolved]

    loaded = {}

    def fake_loader(model_dir, vlm_dir, *, device, n_action_steps):
        loaded.update(
            model_dir=Path(model_dir),
            vlm_dir=Path(vlm_dir),
            device=device,
            n_action_steps=n_action_steps,
        )
        return "policy", "preprocessor", "postprocessor"

    monkeypatch.setattr(live, "_sha256_file", fake_hash)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(smolvla, "load_published_smolvla", fake_loader)

    result = live.load_prepared_policy("smolvla_published", tmp_path)

    assert result[:3] == ("policy", "preprocessor", "postprocessor")
    assert loaded == {
        "model_dir": (tmp_path / reference["local_dir"]).resolve(),
        "vlm_dir": (tmp_path / learner_start["local_dir"]).resolve(),
        "device": "cpu",
        "n_action_steps": reference["selected_n_action_steps"],
    }
    assert set(seen_hash_paths) == set(expected_paths)
    assert result[3]["model_revision"] == reference["revision"]
    assert result[3]["training_data"] == reference["training_data"]
