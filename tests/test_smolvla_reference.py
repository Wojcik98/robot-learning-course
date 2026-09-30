"""Check the published SmolVLA loading recipe."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_lab.policy import smolvla


def test_published_loader_uses_saved_weights_and_processors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    vlm_dir = tmp_path / "vlm"
    model_dir.mkdir()
    vlm_dir.mkdir()
    (model_dir / "model.safetensors").touch()
    (model_dir / "policy_preprocessor.json").touch()
    (model_dir / "policy_postprocessor.json").touch()
    (vlm_dir / "config.json").touch()
    config = SimpleNamespace(chunk_size=50, n_action_steps=1)
    calls = {}

    def load_config(path):
        calls["config_path"] = path
        return config

    monkeypatch.setattr(smolvla.SmolVLAConfig, "from_pretrained", load_config)

    def load_policy(path, **kwargs):
        calls["policy"] = (path, kwargs)
        return SimpleNamespace(eval=lambda: SimpleNamespace(to=lambda device: "policy"))

    def load_processors(path, **kwargs):
        calls["processors"] = (path, kwargs)
        return "pre", "post"

    monkeypatch.setattr(smolvla.SmolVLAPolicy, "from_pretrained", load_policy)
    monkeypatch.setattr(smolvla, "load_pretrained_policy_processors", load_processors)

    policy, pre, post = smolvla.load_published_smolvla(
        model_dir, vlm_dir, device="cpu", n_action_steps=10
    )

    assert (policy, pre, post) == ("policy", "pre", "post")
    assert calls["policy"][1]["strict"] is True
    assert calls["policy"][1]["local_files_only"] is True
    assert config.vlm_model_name == str(vlm_dir.resolve())
    assert config.load_vlm_weights is False
    assert config.n_action_steps == 10
    assert calls["processors"][1]["preprocessor_overrides"]["tokenizer_processor"][
        "tokenizer_name"
    ] == str(vlm_dir.resolve())


def test_published_loader_rejects_action_count_above_saved_chunk(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_action_steps"):
        smolvla.load_published_smolvla(tmp_path, tmp_path, n_action_steps=51)
