"""Check latency config, result identity, resume, and summaries on CPU."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vla_lab.experiments.runtime_modes import (
    episode_identity,
    episode_metrics,
    load_latency_config,
    load_reusable,
    result_path,
    run_latency_experiment,
    save_atomic,
    summarize,
)
from vla_lab.runtime.experiment import MODES, BoundaryRecord, EpisodeResult, TickLog

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "latency.yaml"


def write_config(tmp_path: Path, **changes) -> Path:
    payload = yaml.safe_load(CONFIG.read_text())
    for key, value in changes.items():
        section, _, name = key.partition("__")
        if name:
            payload[section][name] = value
        else:
            payload[section] = value
    path = tmp_path / "latency.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def load(path: Path = CONFIG) -> dict:
    return load_latency_config(path, ROOT / "configs")


def test_config_pins_development_recipe() -> None:
    config = load()
    assert config["state_ids"] == [0, 1, 2, 3, 4]
    assert config["modes"] == list(MODES)
    assert config["runtime"]["horizon"] == 50
    assert len(config["config_sha256"]) == 64


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"state_ids": [0, 20]}, "partition"),
        ({"state_ids": [0, 0]}, "unique"),
        ({"partition": "final"}, "development"),
        ({"modes": ["rtc_soft", "synchronous"]}, "modes"),
        ({"learner__model_sha256": "0" * 64}, "pinned"),
        ({"learner__vlm_dir": ".local-data/models/other"}, "pinned"),
        ({"reset_seed_base": 1}, "seeds"),
        ({"noise_seed_base": 1}, "seeds"),
        ({"runtime__num_steps": 10}, "runtime"),
        ({"runtime__horizon": 40}, "runtime"),
        ({"task_prompt": "other task"}, "task prompt"),
        ({"train_split_sha256": "0" * 64}, "data split"),
        ({"unexpected": 1}, "keys"),
    ],
)
def test_config_rejects_changed_pins(tmp_path: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load(write_config(tmp_path, **changes))


def test_identity_derives_seeds_from_state() -> None:
    config = load()
    identity = episode_identity(config, "rtc_soft", 3)
    assert identity["reset_seed"] == 20260933
    assert identity["noise_seed"] == 7100
    assert identity["mode"] == "rtc_soft"
    assert identity["config_sha256"] == config["config_sha256"]
    with pytest.raises(ValueError, match="outside"):
        episode_identity(config, "rtc_soft", 20)


def make_result(*, success: bool = True, underflows: int = 1) -> EpisodeResult:
    ticks = tuple(
        TickLog(
            tick=index,
            action=np.full(7, 0.1 * index, dtype=np.float32),
            underflow=index < underflows,
            observation_age_ticks=None,
            queue_depth=0,
            accepted=(),
            rejected=(),
            step_success=False,
            discontinuity=None if index == 0 else 0.1 * np.sqrt(7),
        )
        for index in range(4)
    )
    return EpisodeResult("e", 0, success, ticks, (BoundaryRecord("r", 0, ()),))


def test_metrics_report_underflow_discontinuity_and_latency() -> None:
    metrics = episode_metrics(
        make_result(underflows=2),
        [10.0, 20.0, 30.0],
        peak_allocated_bytes=5,
        peak_reserved_bytes=9,
    )
    assert metrics["success"] is True
    assert metrics["steps"] == 4
    assert metrics["underflow"] == 2
    assert metrics["discontinuity_l2_mean"] == pytest.approx(0.1 * np.sqrt(7))
    assert metrics["request_latency_ms"]["p50"] == 20.0
    assert metrics["request_latency_ms"]["p95"] == pytest.approx(29.0)
    assert metrics["request_latency_ms"]["first_request_ms"] == 10.0
    assert metrics["request_latency_ms"]["warm_p50_ms"] == 25.0
    assert metrics["cuda_peak_reserved_bytes"] == 9


def test_metrics_reject_unbounded_action() -> None:
    result = make_result()
    object.__setattr__(result.ticks[1], "action", np.full(7, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="within"):
        episode_metrics(result, [1.0], peak_allocated_bytes=None, peak_reserved_bytes=None)


def saved(config: dict, mode: str, state_id: int, *, success: bool, latency: float) -> dict:
    metrics = episode_metrics(
        make_result(success=success),
        [latency, latency * 2],
        peak_allocated_bytes=100 + state_id,
        peak_reserved_bytes=200,
    )
    return {"identity": episode_identity(config, mode, state_id), "metrics": metrics}


def test_reuse_requires_exact_identity(tmp_path: Path) -> None:
    config = load()
    identity = episode_identity(config, "naive_async", 1)
    path = result_path(tmp_path, "naive_async", 1)
    assert load_reusable(path, identity) is None
    save_atomic(path, saved(config, "naive_async", 1, success=True, latency=5.0))
    assert not list(tmp_path.rglob("*.tmp"))
    assert load_reusable(path, identity) is not None
    stale = saved(config, "naive_async", 1, success=True, latency=5.0)
    stale["metrics"]["request_latency_ms"].pop("warm_p50_ms")
    save_atomic(path, stale)
    assert load_reusable(path, identity) is None
    save_atomic(path, saved(config, "naive_async", 1, success=True, latency=5.0))
    for key, value in (
        ("config_sha256", "0" * 64),
        ("model_sha256", "0" * 64),
        ("reset_seed", 1),
        ("noise_seed", 1),
        ("mode", "rtc_hard"),
        ("state_id", 2),
        ("num_steps", 4),
    ):
        assert load_reusable(path, {**identity, key: value}) is None
    path.write_text("{truncated")
    assert load_reusable(path, identity) is None


def test_summary_pools_latency_by_mode() -> None:
    config = load()
    rows = [
        saved(config, "synchronous", 0, success=True, latency=10.0),
        saved(config, "synchronous", 1, success=False, latency=30.0),
        saved(config, "rtc_soft", 0, success=True, latency=40.0),
    ]
    summary = summarize(rows)
    assert list(summary) == ["synchronous", "rtc_soft"]
    sync = summary["synchronous"]
    assert sync["episodes"] == 2
    assert sync["successes"] == 1
    assert sync["success_rate"] == 0.5
    assert sync["underflow_total"] == 2
    assert sync["request_latency_ms"]["p50"] == pytest.approx(25.0)
    assert sync["request_latency_ms"]["warm_p50_ms"] == pytest.approx(40.0)
    assert sync["cuda_peak_allocated_bytes"] == 101


def test_complete_results_resume_without_model_or_environment(tmp_path: Path) -> None:
    config = load()
    for state_id in config["state_ids"]:
        for mode in config["modes"]:
            save_atomic(
                result_path(tmp_path, mode, state_id),
                saved(config, mode, state_id, success=state_id % 2 == 0, latency=7.0),
            )
    summary = run_latency_experiment(
        config_path=CONFIG,
        configs_root=ROOT / "configs",
        data_root=tmp_path / "unused",
        output_dir=tmp_path,
    )
    assert summary["modes"]["rtc_hard"]["episodes"] == 5
    assert summary["modes"]["rtc_hard"]["successes"] == 3
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_run_rejects_final_state_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="partition"):
        run_latency_experiment(
            config_path=CONFIG,
            configs_root=ROOT / "configs",
            data_root=tmp_path,
            output_dir=tmp_path,
            state_ids=[20],
        )
