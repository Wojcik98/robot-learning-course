"""Keep the learner stages and development states fixed."""

from pathlib import Path

import pytest
import yaml

from vla_lab.experiments.smolvla import load_smolvla_train_config

ROOT = Path(__file__).resolve().parents[1]


def test_learner_config_has_early_and_continued_stages() -> None:
    config = load_smolvla_train_config(
        ROOT / "configs" / "train" / "smolvla_bowl_plate.yaml", ROOT / "configs"
    )
    assert config.c1_steps == 100
    assert config.c2_steps == 2000
    assert config.state_ids == tuple(range(20))
    assert config.batch_size == 1


def test_learner_config_rejects_final_state_for_selection(tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / "configs" / "train" / "smolvla_bowl_plate.yaml").read_text())
    payload["development_state_ids"] = [0, 20]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="development"):
        load_smolvla_train_config(path, ROOT / "configs")


def test_learner_config_rejects_reversed_stages(tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / "configs" / "train" / "smolvla_bowl_plate.yaml").read_text())
    payload["c2_steps"] = payload["c1_steps"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="c2_steps"):
        load_smolvla_train_config(path, ROOT / "configs")
