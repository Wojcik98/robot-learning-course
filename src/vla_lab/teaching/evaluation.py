"""Transparent case selection and paired-result aggregation for the course."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def development_cases(stage: str) -> list[int]:
    """Return the declared development states for exploration, method, or data checks."""
    groups = {
        "exploration": range(10),
        "method": range(10, 15),
        "intervention": range(15, 20),
    }
    try:
        return list(groups[stage])
    except KeyError as error:
        raise ValueError("stage must be exploration, method, or intervention") from error


def final_cases(*, data_unit_complete: bool, recipes_frozen: bool) -> list[int]:
    """Open final states only after the data unit and recipe freeze."""
    if not data_unit_complete or not recipes_frozen:
        raise ValueError("final states require completed data work and frozen recipes")
    return list(range(20, 50))


def validate_case_results(
    results: Sequence[Mapping[str, Any]], *, expected_case_ids: Sequence[int]
) -> list[Mapping[str, Any]]:
    """Check exact case coverage and retain every failed attempt in order."""
    expected = list(expected_case_ids)
    actual = [row["case_id"] for row in results]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected case IDs must be nonempty and unique")
    if len(actual) != len(set(actual)):
        raise ValueError("result rows contain duplicate cases")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"case coverage mismatch: missing={missing}, extra={extra}")
    by_id = {row["case_id"]: row for row in results}
    return [by_id[case_id] for case_id in expected]


def paired_summary(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[int],
) -> dict[str, Any]:
    """Summarize paired successes, failures, and disagreements without omissions."""
    a = validate_case_results(left, expected_case_ids=expected_case_ids)
    b = validate_case_results(right, expected_case_ids=expected_case_ids)
    by_a = {row["case_id"]: row for row in a}
    by_b = {row["case_id"]: row for row in b}
    ids = list(expected_case_ids)
    identity_fields = (
        "state_id",
        "reset_seed",
        "task_id",
        "partition",
        "prompt",
    )
    required_fields = (*identity_fields, "action_steps", "policy_revision", "success")
    for label, rows in (("left", a), ("right", b)):
        for row in rows:
            missing = [key for key in required_fields if key not in row]
            if missing:
                raise ValueError(f"{label} result row misses paired identity fields: {missing}")
        revisions = {row["policy_revision"] for row in rows}
        if len(revisions) != 1:
            raise ValueError(f"{label} results mix policy revisions")
        action_steps = {row["action_steps"] for row in rows}
        if len(action_steps) != 1:
            raise ValueError(f"{label} results mix action execution lengths")
    for case_id in ids:
        if any(by_a[case_id][key] != by_b[case_id][key] for key in identity_fields):
            raise ValueError(
                "paired rows differ in task, state, seed, prompt, partition, or action length"
            )
    a_only = b_only = 0
    for case_id in ids:
        a_ok = by_a[case_id]["success"] is True
        b_ok = by_b[case_id]["success"] is True
        a_only += int(a_ok and not b_ok)
        b_only += int(b_ok and not a_ok)
    a_successes = sum(by_a[case_id]["success"] is True for case_id in ids)
    b_successes = sum(by_b[case_id]["success"] is True for case_id in ids)
    return {
        "cases": len(ids),
        "case_ids": ids,
        "left_successes": a_successes,
        "right_successes": b_successes,
        "left_policy_revision": a[0]["policy_revision"],
        "right_policy_revision": b[0]["policy_revision"],
        "left_success_rate": a_successes / len(ids),
        "right_success_rate": b_successes / len(ids),
        "left_only": a_only,
        "right_only": b_only,
        "paired_difference": (a_successes - b_successes) / len(ids),
        "failed_attempts_included": True,
    }


def evaluate_frozen_method_timed(
    *,
    method: str,
    config_path: str,
    configs_root: str,
    data_root: str,
    output_dir: str,
) -> dict[str, Any]:
    """Run one frozen method and record synchronized policy-call latency.

    The checkpoint is loaded through the comparison module's hash and lineage
    checks. Per-call time wraps ``policy.select_action`` only. Episode wall time
    remains in the evaluator output and is not treated as inference time.
    """
    import torch
    import yaml

    from vla_lab.evaluation import comparison
    from vla_lab.sim.episode import open_libero_task, run_policy_episode

    config = comparison.load_comparison_config(config_path, configs_root)
    if method not in comparison.METHODS:
        raise ValueError(f"Unknown comparison method: {method}")
    recipe = config["policies"][method]
    episodes = yaml.safe_load((Path(configs_root) / "data" / "episodes.yaml").read_text())
    task = episodes["tasks"][config["task_alias"]]
    policy, preprocessor, postprocessor = comparison._load_method(
        method,
        recipe,
        config["task_alias"],
        task["split_sha256"],
        episodes["dataset_revision"],
    )
    parameter = next(policy.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    api_times: dict[int, list[float]] = {}
    inference_times: dict[int, list[float]] = {}
    select_action = policy.select_action
    inference_hook_name = (
        "_get_action_chunk" if method == "smolvla_learner" else "predict_action_chunk"
    )
    inference_hook = getattr(policy, inference_hook_name, None)
    if inference_hook is None:
        raise ValueError(f"{method} has no known action-generation hook for timing")

    def synchronized_call(callback: Any, samples: list[float], *args: Any, **kwargs: Any) -> Any:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = callback(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
        return result

    def timed_generation(*args: Any, **kwargs: Any) -> Any:
        return synchronized_call(inference_hook, inference_times[current_state], *args, **kwargs)

    setattr(policy, inference_hook_name, timed_generation)

    def timed_select_action(*args: Any, **kwargs: Any) -> Any:
        return synchronized_call(select_action, api_times[current_state], *args, **kwargs)

    policy.select_action = timed_select_action
    current_state = -1

    def run_one(state_id: int) -> Any:
        nonlocal current_state
        current_state = state_id
        api_times[state_id] = []
        inference_times[state_id] = []
        environment = open_libero_task(config["task_alias"], state_id, configs_root, data_root)
        try:
            return run_policy_episode(
                environment,
                policy,
                preprocessor,
                postprocessor,
                config["task_alias"],
                state_id,
                configs_root,
                partition="final",
                max_steps=config["max_steps"],
                task_prompt=config["task_prompt"] if method == "smolvla_learner" else None,
                reset_seed=int(config["base_seed"]) + state_id,
            )
        finally:
            environment.close()

    try:
        summary = comparison.evaluate_frozen_method(
            method=method,
            config_path=config_path,
            configs_root=configs_root,
            data_root=data_root,
            output_dir=output_dir,
            run_one=run_one,
        )
    finally:
        policy.select_action = select_action
        setattr(policy, inference_hook_name, inference_hook)
    timing_rows = []
    for state_id in config["state_ids"]:
        api_samples = api_times.get(int(state_id), [])
        inference_samples = inference_times.get(int(state_id), [])
        timing_rows.append(
            {
                "state_id": int(state_id),
                "reset_seed": int(config["base_seed"]) + int(state_id),
                "action_selection_api": {
                    **summarize_latency_samples(api_samples),
                    "samples_s": api_samples,
                },
                "action_generation": {
                    **summarize_latency_samples(inference_samples),
                    "samples_s": inference_samples,
                },
                "timing_available": bool(api_samples or inference_samples),
            }
        )
    timing = {
        "scope": {
            "action_selection_api": "policy.select_action wall time; includes queued action retrieval; CUDA synchronized when applicable",
            "action_generation": f"policy.{inference_hook_name} wall time; model action generation only; CUDA synchronized when applicable",
        },
        "attempts": timing_rows,
        "action_selection_api": summarize_latency_samples(
            [value for row in timing_rows for value in row["action_selection_api"]["samples_s"]]
        ),
        "action_generation": summarize_latency_samples(
            [value for row in timing_rows for value in row["action_generation"]["samples_s"]]
        ),
    }
    (Path(output_dir) / "inference-timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n"
    )
    return {"summary": summary, "timing": timing}


def summarize_latency_samples(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Report count, mean, p95, and maximum; empty samples remain unavailable."""
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latency samples must be a finite nonnegative vector")
    if not len(values):
        return {"count": 0, "mean_s": None, "p95_s": None, "max_s": None}
    return {
        "count": len(values),
        "mean_s": float(values.mean()),
        "p95_s": float(np.quantile(values, 0.95)),
        "max_s": float(values.max()),
    }
