"""Check sweep config, deduplication, resume identity, and summaries on CPU."""

import json
from pathlib import Path

import pytest
import yaml

from vla_lab.experiments.runtime_modes import load_latency_config, load_reusable, save_atomic
from vla_lab.experiments.runtime_sweep import (
    condition_identity,
    expand_conditions,
    load_sweep_config,
    result_path,
    run_sweep,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "latency_sweep.yaml"


def write_config(tmp_path: Path, **changes) -> Path:
    payload = yaml.safe_load(CONFIG.read_text())
    for key, value in changes.items():
        section, _, name = key.partition("__")
        if name:
            payload[section][name] = value
        else:
            payload[section] = value
    path = tmp_path / "latency_sweep.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def load(path: Path = CONFIG) -> dict:
    return load_sweep_config(path, ROOT / "configs")


def fake_result(identity: dict, *, success: bool = False) -> dict:
    latency = {
        "samples": [30.0, 10.0, 12.0],
        "first_request_ms": 30.0,
        "warm_p50_ms": 11.0,
        "warm_p95_ms": 11.9,
        "warm_p99_ms": 11.98,
    }
    metrics = {
        "success": success,
        "steps": 300,
        "underflow": 2,
        "discontinuity_l2_mean": 0.5,
        "discontinuity_l2_max": 0.9,
        "requests": 3,
        "request_latency_ms": latency,
        "cuda_peak_allocated_bytes": None,
        "episode_wall_s": 4.0,
    }
    return {"identity": identity, "metrics": metrics}


def test_config_has_six_deduplicated_conditions_and_shared_pins() -> None:
    config = load()
    assert [item["name"] for item in config["conditions"]] == [
        "synchronous-s01-d0",
        "synchronous-s05-d0",
        "synchronous-s10-d0",
        "naive_async-s10-d1",
        "naive_async-s10-d2",
        "naive_async-s10-d4",
    ]
    latency = load_latency_config(ROOT / "configs/experiments/latency.yaml", ROOT / "configs")
    assert config["learner"] == latency["learner"]
    assert config["task_prompt"] == latency["task_prompt"]
    assert config["partition"] == "development"
    assert config["state_id"] == 0
    assert (config["reset_seed"], config["noise_seed"], config["max_steps"]) == (
        20260930,
        4100,
        300,
    )
    assert len(config["config_sha256"]) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"partition": "final"},
        {"state_id": 1},
        {"state_id": 20},
        {"reset_seed": 1},
        {"noise_seed": 1},
        {"max_steps": 100},
        {"learner__model_sha256": "0" * 64},
        {"learner__vlm_sha256": "0" * 64},
        {"learner__checkpoint": "other"},
        {"runtime__horizon": 40},
        {"runtime__num_steps": 10},
        {"task_prompt": "other"},
        {"grids": [{"mode": "synchronous", "strides": [5], "delays": [1]}]},
        {"grids": [{"mode": "rtc_hard", "strides": [5], "delays": [0]}]},
        {"grids": [{"mode": "naive_async", "strides": [51], "delays": [0]}]},
        {"grids": [{"mode": "naive_async", "strides": [5], "delays": [50]}]},
        {"grids": [{"mode": "naive_async", "strides": [5, 5], "delays": [1]}]},
        {"grids": []},
    ],
)
def test_config_rejects_bad_settings(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        load(write_config(tmp_path, **changes))


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing or unknown"):
        load(write_config(tmp_path, extra=1))


def test_first_listed_condition_wins_overlap() -> None:
    grids = [
        {"mode": "naive_async", "strides": [10], "delays": [0, 1]},
        {"mode": "synchronous", "strides": [10], "delays": [0]},
    ]
    names = [item["name"] for item in expand_conditions(grids, 50)]
    assert names == ["naive_async-s10-d0", "naive_async-s10-d1"]


def test_resume_identity_uses_all_settings(tmp_path: Path) -> None:
    config = load()
    name = "naive_async-s10-d2"
    identity = condition_identity(config, name)
    assert identity["stride"] == 10 and identity["delay"] == 2
    assert identity["reset_seed"] == 20260930 and identity["noise_seed"] == 4100
    path = result_path(tmp_path, name)
    assert load_reusable(path, identity) is None
    save_atomic(path, fake_result(identity))
    assert not path.with_name(path.name + ".tmp").exists()
    assert load_reusable(path, identity) is not None
    for key, value in {"delay": 4, "stride": 5, "num_steps": 10, "config_sha256": "x"}.items():
        assert load_reusable(path, {**identity, key: value}) is None
    other = load(write_config(tmp_path, runtime__ensemble_decay=0.5))
    assert condition_identity(other, name) != identity
    with pytest.raises(ValueError, match="outside the sweep"):
        condition_identity(config, "naive_async-s10-d0")


def test_summary_follows_config_order_and_names_missing(tmp_path: Path) -> None:
    config = load()
    done = ["naive_async-s10-d1", "synchronous-s01-d0"]
    results = [
        fake_result(condition_identity(config, name), success=name.startswith("sync"))
        for name in done
    ]
    summary = summarize(config, results)
    assert [row["condition"] for row in summary["conditions"]] == [
        "synchronous-s01-d0",
        "naive_async-s10-d1",
    ]
    assert summary["episodes"] == 2 and summary["successes"] == 1
    assert len(summary["missing"]) == 4
    row = summary["conditions"][1]
    assert row["underflow"] == 2 and row["discontinuity_l2_mean"] == 0.5
    assert row["warm_p50_ms"] == 11.0 and row["episode_wall_s"] == 4.0


def test_run_sweep_reuses_saved_results_without_gpu(tmp_path: Path) -> None:
    config = load()
    for item in config["conditions"]:
        identity = condition_identity(config, item["name"])
        save_atomic(result_path(tmp_path, item["name"]), fake_result(identity))
    summary = run_sweep(
        config_path=CONFIG,
        configs_root=ROOT / "configs",
        data_root=tmp_path / "unused",
        output_dir=tmp_path,
    )
    assert summary["episodes"] == 6 and summary["missing"] == []
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with pytest.raises(ValueError, match="outside the sweep"):
        run_sweep(
            config_path=CONFIG,
            configs_root=ROOT / "configs",
            data_root=tmp_path,
            output_dir=tmp_path,
            conditions=["bogus"],
        )
