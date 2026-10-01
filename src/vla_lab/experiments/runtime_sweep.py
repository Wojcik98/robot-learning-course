"""Sweep stride and delay on development state 0 with one SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from vla_lab.evaluation.states import selected_state_ids
from vla_lab.experiments.runtime_modes import (
    PINNED_CHECKPOINT,
    PINNED_MODEL_SHA256,
    PINNED_NOISE_SEED_BASE,
    PINNED_RESET_SEED_BASE,
    PINNED_VLM_DIR,
    load_reusable,
    run_mode_episode,
    save_atomic,
    verify_model_source,
)

SCHEMA_VERSION = 1
PINNED_VLM_SHA256 = "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
SWEEP_MODES = ("synchronous", "naive_async")
TOP_KEYS = {
    "schema_version",
    "experiment",
    "task_alias",
    "partition",
    "state_id",
    "reset_seed",
    "noise_seed",
    "max_steps",
    "dataset_revision",
    "train_split_sha256",
    "task_prompt",
    "learner",
    "runtime",
    "grids",
}
LEARNER_KEYS = {"checkpoint", "vlm_dir", "model_sha256", "vlm_sha256"}
RUNTIME_KEYS = {"horizon", "num_steps", "ensemble_decay"}
GRID_KEYS = {"mode", "strides", "delays"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def condition_name(mode: str, stride: int, delay: int) -> str:
    return f"{mode}-s{stride:02d}-d{delay}"


def expand_conditions(grids: Sequence[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    """Check each grid, expand it, and drop a condition that repeats an earlier one."""
    if not isinstance(grids, list) or not grids:
        raise ValueError("Sweep needs a nonempty list of grids")
    conditions: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for grid in grids:
        if not isinstance(grid, dict) or set(grid) != GRID_KEYS:
            raise ValueError("Sweep grid has missing or unknown keys")
        mode = grid["mode"]
        if mode not in SWEEP_MODES:
            raise ValueError(f"Sweep mode must be one of {list(SWEEP_MODES)}")
        for key in ("strides", "delays"):
            values = grid[key]
            if (
                not isinstance(values, list)
                or not values
                or not all(_is_int(value) for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"Sweep {key} must be a nonempty list of unique integers")
        for stride in grid["strides"]:
            if not 1 <= stride <= horizon:
                raise ValueError("Sweep stride must be between 1 and the horizon")
            for delay in grid["delays"]:
                if not 0 <= delay < horizon:
                    raise ValueError("Sweep delay must be at least 0 and below the horizon")
                if mode == "synchronous" and delay != 0:
                    raise ValueError("Synchronous mode must use delay 0")
                # Synchronous mode ignores delay, so (stride, delay 0) is the same run in both modes.
                if (stride, delay) in seen:
                    continue
                seen.add((stride, delay))
                conditions.append(
                    {
                        "name": condition_name(mode, stride, delay),
                        "mode": mode,
                        "stride": stride,
                        "delay": delay,
                    }
                )
    return conditions


def load_sweep_config(path: str | Path, configs_root: str | Path) -> dict[str, Any]:
    """Read the sweep config, check its pins, and add the condition list and file hash."""
    content = Path(path).read_bytes()
    config = yaml.safe_load(content)
    if not isinstance(config, dict):
        raise TypeError("Sweep config must be a mapping")
    if set(config) != TOP_KEYS:
        raise ValueError("Sweep config has missing or unknown keys")
    if config["schema_version"] != SCHEMA_VERSION or config["experiment"] != "latency_sweep":
        raise ValueError("Sweep config needs schema version 1 and experiment latency_sweep")
    if config["partition"] != "development":
        raise ValueError("Sweep uses development states only")
    learner, runtime = config["learner"], config["runtime"]
    if (
        not isinstance(learner, dict)
        or not isinstance(runtime, dict)
        or set(learner) != LEARNER_KEYS
        or set(runtime) != RUNTIME_KEYS
    ):
        raise ValueError("Sweep learner or runtime section has missing or unknown keys")
    if (
        learner["checkpoint"] != PINNED_CHECKPOINT
        or learner["vlm_dir"] != PINNED_VLM_DIR
        or learner["model_sha256"] != PINNED_MODEL_SHA256
        or learner["vlm_sha256"] != PINNED_VLM_SHA256
    ):
        raise ValueError("Sweep learner source differs from the pinned SmolVLA checkpoint")
    if (
        config["reset_seed"] != PINNED_RESET_SEED_BASE
        or config["noise_seed"] != PINNED_NOISE_SEED_BASE
    ):
        raise ValueError("Sweep reset or noise seed differs from the pinned recipe")
    if runtime["horizon"] != 50 or runtime["num_steps"] != 5:
        raise ValueError("Sweep runtime needs horizon 50 and 5 steps")
    if not 0 < float(runtime["ensemble_decay"]) <= 1:
        raise ValueError("Sweep ensemble decay must be in (0, 1]")
    if not _is_int(config["max_steps"]) or config["max_steps"] != 300:
        raise ValueError("Sweep max_steps must be 300")
    if not _is_int(config["state_id"]) or config["state_id"] != 0:
        raise ValueError("Sweep runs development state 0 only")
    config["state_id"] = selected_state_ids(
        configs_root, str(config["task_alias"]), "development", [config["state_id"]]
    )[0]
    episodes = yaml.safe_load((Path(configs_root) / "data" / "episodes.yaml").read_text())
    task = episodes["tasks"][config["task_alias"]]
    if (
        config["dataset_revision"] != episodes["dataset_revision"]
        or config["train_split_sha256"] != task["split_sha256"]
        or config["task_prompt"] != task["task_name"]
    ):
        raise ValueError("Sweep data split or task prompt differs from the frozen data")
    config["conditions"] = expand_conditions(config["grids"], runtime["horizon"])
    config["config_sha256"] = hashlib.sha256(content).hexdigest()
    return config


def condition_identity(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Everything that must match before a saved condition result is reused."""
    matches = [item for item in config["conditions"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Condition is outside the sweep config: {name}")
    condition, runtime = matches[0], config["runtime"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": config["experiment"],
        "config_sha256": config["config_sha256"],
        "model_sha256": config["learner"]["model_sha256"],
        "vlm_sha256": config["learner"]["vlm_sha256"],
        "task_alias": config["task_alias"],
        "task_prompt": config["task_prompt"],
        "partition": config["partition"],
        "condition": name,
        "mode": condition["mode"],
        "state_id": config["state_id"],
        "reset_seed": config["reset_seed"],
        "noise_seed": config["noise_seed"],
        "horizon": runtime["horizon"],
        "stride": condition["stride"],
        "delay": condition["delay"],
        "num_steps": runtime["num_steps"],
        "ensemble_decay": float(runtime["ensemble_decay"]),
        "max_steps": config["max_steps"],
    }


def result_path(output: str | Path, name: str) -> Path:
    return Path(output) / f"{name}.json"


def summarize(config: dict[str, Any], results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """List saved results in config order. Missing conditions are named, not guessed."""
    by_name = {saved["identity"]["condition"]: saved for saved in results}
    rows = []
    for condition in config["conditions"]:
        saved = by_name.get(condition["name"])
        if saved is None:
            continue
        metrics = saved["metrics"]
        latency = metrics["request_latency_ms"]
        rows.append(
            {
                "condition": condition["name"],
                "mode": condition["mode"],
                "stride": condition["stride"],
                "delay": condition["delay"],
                "success": metrics["success"],
                "steps": metrics["steps"],
                "underflow": metrics["underflow"],
                "discontinuity_l2_mean": metrics["discontinuity_l2_mean"],
                "discontinuity_l2_max": metrics.get("discontinuity_l2_max"),
                "requests": metrics.get("requests"),
                "first_request_ms": latency["first_request_ms"],
                "warm_p50_ms": latency["warm_p50_ms"],
                "warm_p95_ms": latency["warm_p95_ms"],
                "episode_wall_s": metrics.get("episode_wall_s"),
            }
        )
    done = {row["condition"] for row in rows}
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": config["experiment"],
        "config_sha256": config["config_sha256"],
        "model_sha256": config["learner"]["model_sha256"],
        "vlm_sha256": config["learner"]["vlm_sha256"],
        "partition": config["partition"],
        "state_id": config["state_id"],
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "missing": [item["name"] for item in config["conditions"] if item["name"] not in done],
        "conditions": rows,
    }


def run_sweep(
    *,
    config_path: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run every missing condition, saving each one as soon as it finishes."""
    config = load_sweep_config(config_path, configs_root)
    names = [item["name"] for item in config["conditions"]]
    chosen = list(conditions) if conditions is not None else names
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(names):
        raise ValueError("Requested conditions are empty, repeated, or outside the sweep config")
    pending: list[dict[str, Any]] = []
    saved_results: list[dict[str, Any]] = []
    for name in chosen:
        identity = condition_identity(config, name)
        saved = load_reusable(result_path(output_dir, name), identity)
        if saved is None:
            pending.append(identity)
        else:
            saved_results.append(saved)
    if pending:
        from vla_lab.policy.smolvla_learner import load_learner_bundle
        from vla_lab.sim.episode import open_libero_task

        verify_model_source(config)
        learner = config["learner"]
        policy, preprocessor, postprocessor, _ = load_learner_bundle(
            learner["checkpoint"], learner["vlm_dir"]
        )
        policy.eval()
        policy.config.num_steps = config["runtime"]["num_steps"]
        for identity in pending:
            environment = open_libero_task(
                config["task_alias"], identity["state_id"], configs_root, data_root
            )
            # run_mode_episode reads stride and delay from the runtime section.
            episode_config = {
                **config,
                "runtime": {
                    **config["runtime"],
                    "stride": identity["stride"],
                    "delay": identity["delay"],
                },
            }
            try:
                start = time.perf_counter()
                metrics = run_mode_episode(
                    config=episode_config,
                    identity=identity,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    environment=environment,
                )
                metrics["episode_wall_s"] = time.perf_counter() - start
            finally:
                active_error = sys.exc_info()[0]
                try:
                    environment.close()
                except Exception:
                    if active_error is None:
                        raise
            saved = {"identity": identity, "metrics": metrics}
            save_atomic(result_path(output_dir, identity["condition"]), saved)
            saved_results.append(saved)
    summary = summarize(config, saved_results)
    save_atomic(Path(output_dir) / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the development stride and delay sweep; finished conditions are reused."""
    parser = argparse.ArgumentParser(description="Sweep P5 stride and delay on state 0")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/latency_sweep.yaml")
    )
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args(argv)
    summary = run_sweep(
        config_path=args.config,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
        conditions=args.conditions,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
