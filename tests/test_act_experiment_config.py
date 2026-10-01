"""Keep ACT experiment variants on fixed development states."""

from pathlib import Path

import pytest
import yaml

from vla_lab.experiments.act import load_act_suite_config

ROOT = Path(__file__).resolve().parents[1]


def test_course_act_suite_uses_nested_demos_and_development_states() -> None:
    recipe = load_act_suite_config(ROOT / "configs" / "train" / "act_task.yaml", ROOT / "configs")
    assert [variant.episode_count for variant in recipe.variants] == [10, 39, 39]
    assert [variant.chunk_size for variant in recipe.variants] == [20, 20, 10]
    assert recipe.state_ids == tuple(range(20))


def test_act_suite_rejects_final_state_during_tuning(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs" / "train" / "act_task.yaml").read_text())
    config["development_state_ids"] = [0, 20]
    path = tmp_path / "act.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="development"):
        load_act_suite_config(path, ROOT / "configs")


def test_act_suite_rejects_duplicate_variant_name(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs" / "train" / "act_task.yaml").read_text())
    config["variants"][1]["name"] = config["variants"][0]["name"]
    path = tmp_path / "act.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="variant"):
        load_act_suite_config(path, ROOT / "configs")
