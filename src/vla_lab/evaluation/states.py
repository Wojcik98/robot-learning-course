"""Evaluate exact simulator state IDs and keep every attempt in the result."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from vla_lab.contracts import EpisodeOutcome
from vla_lab.evaluation.errors import EvaluationRunError
from vla_lab.sim.episode import PolicyActionError


def selected_state_ids(
    configs_root: str | Path,
    task_id: str,
    partition: str,
    state_ids: Sequence[int],
    *,
    allow_final: bool = False,
) -> list[int]:
    """Check explicit state IDs without wrapping or repeating them."""
    if partition not in ("development", "final"):
        raise ValueError("Partition must be development or final")
    if partition == "final" and not allow_final:
        raise ValueError("Final states need an explicit final-evaluation flag")
    manifest = yaml.safe_load((Path(configs_root) / "eval" / "splits.yaml").read_text())
    try:
        allowed = set(manifest["tasks"][task_id][f"{partition}_state_ids"])
    except KeyError as error:
        raise ValueError(f"Unknown simulator task: {task_id}") from error
    selected = list(state_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("State IDs must be nonempty and unique")
    if any(state_id not in allowed for state_id in selected):
        raise ValueError(f"State ID is outside the {partition} partition")
    return selected


def evaluate_states(
    *,
    configs_root: str | Path,
    task_id: str,
    partition: str,
    state_ids: Sequence[int],
    policy_revision: str,
    output_dir: str | Path,
    run_one: Callable[[int], EpisodeOutcome],
    allow_final: bool = False,
    error_metrics: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run each selected state once and save success and failure rows."""
    selected = selected_state_ids(
        configs_root, task_id, partition, state_ids, allow_final=allow_final
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for state_id in selected:
        started = time.monotonic()
        error_text = None
        try:
            result = run_one(state_id)
            if (
                result.task_id != task_id
                or result.initial_state_id != state_id
                or result.partition != partition
                or result.policy_revision != policy_revision
            ):
                raise ValueError("Episode result identity differs from the selected state")
        except (PolicyActionError, EvaluationRunError) as error:
            policy_fault = isinstance(error, PolicyActionError) or error.policy_fault
            result = EpisodeOutcome(
                task_id,
                state_id,
                partition,
                policy_revision,
                "libero-goal-v1",
                False,
                "policy_error" if policy_fault else "infrastructure_error",
                error_metrics(state_id) if error_metrics is not None else {},
                error.artifact_paths if isinstance(error, EvaluationRunError) else {},
            )
            error_text = f"{type(error).__name__}: {error}"
        except Exception as error:  # noqa: BLE001
            result = EpisodeOutcome(
                task_id,
                state_id,
                partition,
                policy_revision,
                "libero-goal-v1",
                False,
                "infrastructure_error",
                error_metrics(state_id) if error_metrics is not None else {},
                {},
            )
            error_text = f"{type(error).__name__}: {error}"
        record = {
            "schema_version": result.schema_version,
            "task_id": result.task_id,
            "initial_state_id": result.initial_state_id,
            "partition": result.partition,
            "policy_revision": result.policy_revision,
            "success_predicate_version": result.success_predicate_version,
            "success": result.success,
            "termination_reason": result.termination_reason,
            "metrics": {**result.metrics, "elapsed_s": round(time.monotonic() - started, 3)},
            "artifact_paths": result.artifact_paths,
            "error": error_text,
        }
        records.append(record)
        with (output / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "task_id": task_id,
        "partition": partition,
        "policy_revision": policy_revision,
        "state_ids": selected,
        "attempts": len(selected),
        "successes": sum(row["success"] for row in records),
        "policy_errors": sum(row["termination_reason"] == "policy_error" for row in records),
        "infrastructure_errors": sum(
            row["termination_reason"] == "infrastructure_error" for row in records
        ),
    }
    summary["success_rate"] = summary["successes"] / summary["attempts"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def evaluate_act_states(
    *,
    checkpoint: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    state_ids: Sequence[int],
) -> dict[str, Any]:
    """Run ACT on explicit development states and save each live video."""
    from vla_lab.evaluation.demo import run_act_demo

    source = Path(checkpoint)
    bundle = json.loads((source / "manifest.json").read_text())
    task_id = bundle["task_alias"]
    model_digest = hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    output = Path(output_dir)

    def run_one(state_id: int) -> EpisodeOutcome:
        record = run_act_demo(
            task_alias=task_id,
            checkpoint=source,
            state_id=state_id,
            configs_root=configs_root,
            data_root=data_root,
            output_dir=output / f"state-{state_id:03d}",
        )
        return EpisodeOutcome(
            task_id=task_id,
            initial_state_id=state_id,
            partition="development",
            policy_revision=model_digest,
            success_predicate_version="libero-goal-v1",
            success=bool(record["success"]),
            termination_reason=str(record["termination_reason"]),
            metrics={
                "steps": int(record["steps"]),
                "clipped_values": int(record["clipped_values"]),
                "reset_seed": int(record["reset_seed"]),
            },
            artifact_paths={
                "video": str(record["video"]),
                "run": str(output / f"state-{state_id:03d}" / "run.json"),
            },
        )

    return evaluate_states(
        configs_root=configs_root,
        task_id=task_id,
        partition="development",
        state_ids=state_ids,
        policy_revision=model_digest,
        output_dir=output,
        run_one=run_one,
    )
