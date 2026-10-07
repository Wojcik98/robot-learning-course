"""Run records, resource meters, bounded adaptation runs, and recipe checks for the final lessons."""

import gc
import hashlib
import json
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict
from uuid import uuid4

import numpy as np
import torch
import yaml
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline, load_pretrained_policy_processors
from numpy.typing import ArrayLike, NDArray
from safetensors import safe_open
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_collate

from vla_lab.evaluation.diffusion import evaluate_diffusion_reference
from vla_lab.evaluation.smolvla_learner import evaluate_learner_reference
from vla_lab.evaluation.states import evaluate_act_states
from vla_lab.policy.bundle import load_act_bundle, save_act_bundle
from vla_lab.policy.smolvla_learner import load_learner_bundle, save_learner_bundle
from vla_lab.teaching.act_model import open_libero_dataset
from vla_lab.teaching.records import (
    AlignedTransition,
    CaseResult,
    ComparisonConfig,
    InterventionCase,
)
from vla_lab.timing import LIBERO_CONTROL_HZ, frame_offset_seconds
from vla_lab.training.diffusion import (
    FPS as DIFFUSION_STORAGE_FPS,
)
from vla_lab.training.diffusion import (
    diffusion_delta_timestamps,
    eligible_frame_indices,
    load_diffusion_bundle,
    save_diffusion_bundle,
    validate_window_batch,
)
from vla_lab.types import JSONValue

type AnyPolicy = ACTPolicy | DiffusionPolicy | SmolVLAPolicy

MAX_TRAINING_SECONDS = 7200.0
BASE_SEED = 20260930
ACT_CHUNK_SIZE = 20
DEVELOPMENT_METHOD_STATES = (10, 11, 12, 13, 14)


class TrainingRate(TypedDict):
    steps: int
    warmup_steps: int
    steady_steps_per_s: float
    total_seconds: float
    last_loss: float


class ActionQuality(TypedDict):
    steps: int
    duration_s: float
    idle_fraction: float
    saturated_fraction: float
    gripper_switches: int
    max_step_change: float


class RealtimeReport(TypedDict):
    period_s: float
    mean_call_s: float
    amortized_s_per_tick: float
    late_tick_fraction: float
    async_fit_fraction: float


class TrainingExposure(TypedDict):
    samples_seen: int
    samples_per_second: float
    projected_hours: float


class Interval(TypedDict):
    low: float
    high: float


class RunIdentity(TypedDict):
    run_id: str
    created_unix_s: float
    fields: dict[str, JSONValue]


class ComparisonRow(TypedDict):
    method: str
    successes: int
    attempts: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    policy_errors: int
    infrastructure_errors: int
    mean_simulated_episode_s: float | None
    mean_wall_episode_s: float | None
    generation_calls: int
    generation_mean_s: float | None
    generation_p95_s: float | None
    generation_max_s: float | None
    peak_allocated_mib: float
    peak_reserved_mib: float
    robot_training_steps: int
    adaptation_updates: int
    adaptation_wall_s: float
    adaptation_peak_allocated_mib: float
    n_action_steps: int
    pretraining: str
    parameters: NotRequired[int]


class EpisodeTimes(TypedDict):
    mean_simulated_s: float | None
    mean_wall_s: float | None
    mean_inference_wall_s: float | None
    wall_over_simulated: float | None


class AdaptationRecord(TypedDict):
    method: str
    source_checkpoint: str
    source_model_sha256: str
    source_robot_steps: int
    checkpoint: str
    model_sha256: str
    robot_training_steps: int
    adaptation_updates: int
    samples_seen: int
    batch_size: int
    seed: int
    learning_rate: float
    max_seconds: float
    stopped_by: str
    train_seconds: float
    last_loss: float
    heldout_loss_before: float
    heldout_loss_after: float
    heldout_samples: int
    peak_allocated_mib: float
    peak_reserved_mib: float
    new_sample_fraction: float
    metrics_path: str
    run_path: str


class DataUnitCompletion(TypedDict):
    run_dir: str
    task_alias: str
    dataset_revision: str
    train_split_sha256: str
    state_ids: list[int]
    created_unix_s: float


class UpdateLog(TypedDict):
    updates: int
    stopped_by: str
    seconds: float
    last_loss: float


class ActionBatch(TypedDict):
    action: NDArray[np.float32]
    action_is_pad: NDArray[np.bool_]
    episode_ids: list[str]
    rows: list[int]


@dataclass
class ResourceUse:
    """Wall time and PyTorch CUDA allocator peaks for one measured block."""

    seconds: float = 0.0
    peak_allocated_mib: float = 0.0
    peak_reserved_mib: float = 0.0


def sha256_file(path: str | Path) -> str:
    """Hash one file in one-megabyte blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, JSONValue]]:
    """Read one JSON object per nonempty line."""
    records: list[dict[str, JSONValue]] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: each line must be a JSON object")
            records.append(record)
    return records


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(key): item for key, item in value.items()}


def _object_field(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Lesson 21 artifact field {name!r} must be an object")
    return {str(key): item for key, item in value.items()}


def _numeric_field(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Lesson 21 artifact field {name!r} must be numeric")
    return float(value)


def verify_completed_data_unit_run(
    run_dir: str | Path,
    *,
    task_alias: str,
    dataset_revision: str,
    train_split_sha256: str,
) -> DataUnitCompletion:
    """Validate the saved Lesson 21 training and evaluation artifacts for this task split."""
    root = Path(run_dir).expanduser().resolve()
    required = ("run-identity.json", "training.json", "reload.json", "paired-summary.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Lesson 21 run is incomplete at {root}; missing {', '.join(missing)}"
        )

    identity = _read_json_object(root / "run-identity.json")
    fields = _object_field(identity.get("fields"), name="run-identity.fields")
    expected = {
        "task_alias": task_alias,
        "dataset_revision": dataset_revision,
        "train_split_sha256": train_split_sha256,
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(
                f"Lesson 21 {key} {fields.get(key)!r} does not match comparison {value!r}"
            )
    created_unix_s = _numeric_field(
        identity.get("created_unix_s"), name="run-identity.created_unix_s"
    )

    training = _read_json_object(root / "training.json")
    arms = _object_field(training.get("arms"), name="training.arms")
    if set(arms) != {"baseline", "intervention"}:
        raise ValueError("Lesson 21 must finish both baseline and intervention training arms")
    arm_rows: dict[str, dict[str, object]] = {}
    for label in ("baseline", "intervention"):
        row = _object_field(arms[label], name=f"training.arms.{label}")
        updates = _numeric_field(row.get("adaptation_updates"), name=f"{label}.updates")
        if updates < 1 or row.get("stopped_by") not in {"updates", "deadline"}:
            raise ValueError(f"Lesson 21 {label} arm has no completed updates")
        checkpoint = Path(str(row.get("checkpoint", "")))
        if not checkpoint.is_absolute():
            raise ValueError(f"Lesson 21 {label} checkpoint path is not absolute")
        model_path = checkpoint / "model.safetensors"
        if not model_path.is_file() or sha256_file(model_path) != row.get("model_sha256"):
            raise ValueError(f"Lesson 21 {label} checkpoint is missing or has a different hash")
        arm_rows[label] = row
    baseline = arm_rows["baseline"]
    intervention = arm_rows["intervention"]
    if baseline.get("source_model_sha256") != intervention.get("source_model_sha256"):
        raise ValueError("Lesson 21 arms do not start from the same model")
    if (
        _numeric_field(baseline.get("new_sample_fraction"), name="baseline.new_sample_fraction")
        != 0
    ):
        raise ValueError("Lesson 21 baseline arm must contain no recovery samples")
    if (
        _numeric_field(
            intervention.get("new_sample_fraction"), name="intervention.new_sample_fraction"
        )
        <= 0
    ):
        raise ValueError("Lesson 21 intervention arm must use recovery samples")

    reloads = _read_json_object(root / "reload.json")
    if set(reloads) != {"baseline", "intervention"}:
        raise ValueError("Lesson 21 must reload both trained checkpoints")
    for label, raw in reloads.items():
        row = _object_field(raw, name=f"reload.{label}")
        difference = _numeric_field(row.get("difference"), name=f"reload.{label}.difference")
        if difference > 1e-4:
            raise ValueError(f"Lesson 21 {label} checkpoint did not pass the reload loss check")

    report = _read_json_object(root / "paired-summary.json")
    summaries = _object_field(report.get("summaries"), name="paired-summary.summaries")
    labels = ("start", "baseline", "intervention")
    if set(summaries) != set(labels):
        raise ValueError("Lesson 21 must evaluate start, baseline, and intervention policies")
    expected_states = list(range(15, 20))
    for label in labels:
        summary = _object_field(summaries[label], name=f"paired-summary.summaries.{label}")
        if (
            summary.get("task_id") != task_alias
            or summary.get("partition") != "development"
            or summary.get("state_ids") != expected_states
            or summary.get("attempts") != len(expected_states)
        ):
            raise ValueError(f"Lesson 21 {label} evaluation has incomplete or mismatched cases")
        episode_path = root / f"eval-{label}" / "episodes.jsonl"
        rows = read_jsonl(episode_path)
        if len(rows) != len(expected_states):
            raise ValueError(f"Lesson 21 {label} episode records do not cover states 15..19")
        if [row.get("initial_state_id") for row in rows] != expected_states or any(
            row.get("task_id") != task_alias for row in rows
        ):
            raise ValueError(f"Lesson 21 {label} episode records have the wrong task or states")
    effect = _object_field(report.get("effect"), name="paired-summary.effect")
    if (
        effect.get("cases") != len(expected_states)
        or effect.get("failed_attempts_included") is not True
    ):
        raise ValueError("Lesson 21 paired result is incomplete")
    return {
        "run_dir": str(root),
        "task_alias": task_alias,
        "dataset_revision": dataset_revision,
        "train_split_sha256": train_split_sha256,
        "state_ids": expected_states,
        "created_unix_s": created_unix_s,
    }


def find_completed_data_unit_run(
    results_root: str | Path,
    *,
    task_alias: str,
    dataset_revision: str,
    train_split_sha256: str,
    override_path: str | Path | None = None,
) -> DataUnitCompletion:
    """Return the latest complete matching Lesson 21 run, or validate an explicit path."""
    if override_path is not None:
        return verify_completed_data_unit_run(
            override_path,
            task_alias=task_alias,
            dataset_revision=dataset_revision,
            train_split_sha256=train_split_sha256,
        )
    candidates: list[DataUnitCompletion] = []
    for candidate in Path(results_root).expanduser().resolve().glob("data-quality-*"):
        if not candidate.is_dir():
            continue
        try:
            candidates.append(
                verify_completed_data_unit_run(
                    candidate,
                    task_alias=task_alias,
                    dataset_revision=dataset_revision,
                    train_split_sha256=train_split_sha256,
                )
            )
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
            continue
    if not candidates:
        raise FileNotFoundError(
            "No complete Lesson 21 run matches this task and training split. "
            "Finish Lesson 21 or set VLA_LESSON21_RUN to a completed run directory."
        )
    return max(candidates, key=lambda row: row["created_unix_s"])


def case_id(state_id: int, reset_seed: int) -> str:
    """Name one paired case by its simulator state and reset seed."""
    return f"state-{state_id}-seed-{reset_seed}"


def new_run_dir(root: str | Path, prefix: str) -> Path:
    """Create a run directory whose name has a time stamp and a random suffix."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(root) / f"{prefix}-{stamp}-{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def write_run_identity(output_dir: str | Path, **fields: JSONValue) -> Path:
    """Save the identity fields of a run next to its results."""
    target = Path(output_dir)
    identity: RunIdentity = {
        "run_id": target.name,
        "created_unix_s": time.time(),
        "fields": dict(fields),
    }
    path = target / "run-identity.json"
    if path.exists():
        raise FileExistsError(f"Run identity already exists: {path}")
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return path


def tick_vectors(
    observation_ticks: Sequence[int] | NDArray[np.int64],
    action_ticks: Sequence[int] | NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Check tick shapes and return ``(observation_ticks, action_ticks)`` as vectors.

    A recording has N action ticks and N+1 observation ticks. Observation ticks
    start at zero and increase by one.
    """
    obs = np.asarray(observation_ticks, dtype=np.int64)
    act = np.asarray(action_ticks, dtype=np.int64)
    if obs.ndim != 1 or act.ndim != 1 or len(act) == 0 or len(obs) != len(act) + 1:
        raise ValueError("need N action ticks and N+1 observation ticks")
    if obs[0] != 0 or np.any(np.diff(obs) != 1):
        raise ValueError("observation ticks must start at zero and increase by one")
    return obs, act


def _number(value: JSONValue) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _result_seed(record: Mapping[str, JSONValue], state_id: int, base_seed: int) -> int:
    metrics = record.get("metrics")
    saved = None
    if isinstance(metrics, dict):
        saved = metrics.get("seed", metrics.get("reset_seed"))
    expected = base_seed + state_id
    if saved is not None and saved != expected:
        raise ValueError(
            f"Saved reset seed {saved} differs from the frozen rule for state {state_id}"
        )
    return expected


def intervention_case_rows(
    records: Sequence[Mapping[str, JSONValue]], *, base_seed: int, prompt: str | None = None
) -> list[InterventionCase]:
    """Turn saved evaluation records into paired intervention rows.

    Error rows can have no metrics. They stay in the result with success False.
    """
    rows: list[InterventionCase] = []
    for record in records:
        state_id = int(_number(record["initial_state_id"]))
        seed = _result_seed(record, state_id, base_seed)
        row: InterventionCase = {
            "case_id": case_id(state_id, seed),
            "state_id": state_id,
            "reset_seed": seed,
            "task_id": str(record["task_id"]),
            "success": record["success"] is True,
        }
        if prompt is not None:
            row["prompt"] = prompt
        rows.append(row)
    return rows


def episode_case_rows(
    records: Sequence[Mapping[str, JSONValue]],
    *,
    base_seed: int,
    prompt: str,
    action_steps: int,
) -> list[CaseResult]:
    """Turn saved evaluation records into rows for ``paired_summary``."""
    rows: list[CaseResult] = []
    for record in records:
        state_id = int(_number(record["initial_state_id"]))
        seed = _result_seed(record, state_id, base_seed)
        rows.append(
            {
                "case_id": case_id(state_id, seed),
                "state_id": state_id,
                "reset_seed": seed,
                "task_id": str(record["task_id"]),
                "partition": str(record["partition"]),
                "prompt": prompt,
                "action_steps": action_steps,
                "policy_revision": str(record["policy_revision"]),
                "success": record["success"] is True,
            }
        )
    return rows


def max_leaf_difference(
    saved: Mapping[str, object], replayed: Mapping[str, object], prefix: str = ""
) -> dict[str, float]:
    """Return the largest absolute difference for each numeric leaf of two observations."""
    if saved.keys() != replayed.keys():
        raise ValueError("observations have different keys")
    result: dict[str, float] = {}
    for key, value in saved.items():
        other = replayed[key]
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and isinstance(other, Mapping):
            result.update(max_leaf_difference(value, other, prefix=f"{name}."))
        else:
            left = np.asarray(value, dtype=np.float64)
            right = np.asarray(other, dtype=np.float64)
            if left.shape != right.shape:
                raise ValueError(f"{name}: shapes differ")
            result[name] = float(np.max(np.abs(left - right))) if left.size else 0.0
    return result


def training_rate(metrics_path: str | Path, *, warmup_steps: int) -> TrainingRate:
    """Read steady-state step time from a trainer's ``metrics.jsonl`` file.

    The first ``warmup_steps`` steps are excluded because they include one-time
    CUDA and data-loader start-up work.
    """
    rows = read_jsonl(metrics_path)
    if warmup_steps < 0 or len(rows) <= warmup_steps + 1:
        raise ValueError("need more logged steps than the warm-up count plus one")
    elapsed = [_number(row["elapsed_s"]) for row in rows]
    steady_steps = len(rows) - 1 - warmup_steps
    steady_seconds = elapsed[-1] - elapsed[warmup_steps]
    if steady_seconds <= 0:
        raise ValueError("elapsed time must increase after the warm-up steps")
    return {
        "steps": len(rows),
        "warmup_steps": warmup_steps,
        "steady_steps_per_s": steady_steps / steady_seconds,
        "total_seconds": elapsed[-1],
        "last_loss": _number(rows[-1]["loss"]),
    }


@contextmanager
def measure_cuda_use() -> Iterator[ResourceUse]:
    """Measure wall time and PyTorch CUDA peaks in this process.

    The allocator counters exclude other processes and non-PyTorch memory.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This measurement needs a CUDA GPU")
    use = ResourceUse()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        yield use
    finally:
        torch.cuda.synchronize()
        use.seconds = time.perf_counter() - started
        use.peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
        use.peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2


def count_safetensors_parameters(path: str | Path) -> int:
    """Count tensor elements in a safetensors file without loading the weights."""
    total = 0
    with safe_open(str(path), framework="pt") as stream:
        for name in stream.keys():  # noqa: SIM118 - safe_open is not a mapping
            total += int(np.prod(stream.get_slice(name).get_shape(), dtype=np.int64))
    return total


def freeze_comparison_recipe(
    source: str | Path, *, workspace: str | Path, output_dir: str | Path
) -> Path:
    """Copy a comparison recipe with absolute checkpoint paths into a new run directory."""
    root = Path(workspace).resolve()
    recipe = yaml.safe_load(Path(source).read_text())
    for item in recipe["policies"].values():
        for key in ("checkpoint", "vlm_dir"):
            if key in item:
                item[key] = str((root / item[key]).resolve())
    target = Path(output_dir) / "comparison.yaml"
    if target.exists():
        raise FileExistsError(f"Frozen recipe already exists: {target}")
    target.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return target


def verify_recipe_files(config: ComparisonConfig) -> dict[str, str]:
    """Check every model hash in a frozen recipe and return the hashes."""
    hashes: dict[str, str] = {}
    for method, recipe in config["policies"].items():
        checkpoint = Path(recipe["checkpoint"])
        weights = checkpoint / "model.safetensors"
        if not checkpoint.is_absolute() or not weights.is_file():
            raise FileNotFoundError(f"{method} checkpoint is missing or not absolute: {checkpoint}")
        digest = sha256_file(weights)
        if digest != recipe["model_sha256"]:
            raise ValueError(f"{method} model hash differs from the frozen recipe")
        if "vlm_dir" in recipe:
            vlm_weights = Path(recipe["vlm_dir"]) / "model.safetensors"
            if not vlm_weights.is_file() or sha256_file(vlm_weights) != recipe["vlm_sha256"]:
                raise ValueError(f"{method} VLM weights differ from the frozen recipe")
        hashes[method] = digest
    return hashes


# --- Supplied pieces for Lesson 21 ---------------------------------------------------------------

ACT_SAMPLE_KEYS = (
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "action",
    "action_is_pad",
)


def aligned_transition_rows[ObservationT](
    observations: Sequence[ObservationT],
    actions: ArrayLike,
    observation_ticks: Sequence[int] | NDArray[np.int64],
    action_ticks: Sequence[int] | NDArray[np.int64],
) -> list[AlignedTransition[ObservationT]]:
    """Pair observation ``i`` with action ``i`` and refuse a recording with a tick offset.

    The final observation has no action and is not a row. Each action is copied.
    """
    observed, acted = tick_vectors(observation_ticks, action_ticks)
    offsets = sorted({int(value) for value in acted - observed[:-1]})
    if offsets != [0]:
        raise ValueError(f"actions are offset by {offsets} control ticks")
    action_array = np.asarray(actions, dtype=np.float32)
    count = len(acted)
    if action_array.ndim != 2 or len(action_array) != count:
        raise ValueError("actions must have shape (N, action_dim)")
    if len(observations) != count + 1:
        raise ValueError("observations must have N + 1 items")
    return [
        {
            "tick": int(acted[index]),
            "observation": observations[index],
            "action": np.array(action_array[index], dtype=np.float32, copy=True),
        }
        for index in range(count)
    ]


# --- Bounded adaptation runs --------------------------------------------------------------------


class SampleSource(Protocol):
    """Anything that holds indexed training samples."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, object]: ...


class KeyProjection(Dataset[dict[str, object]]):
    """Keep only the named sample keys, so two data sources can share one batch."""

    def __init__(self, source: SampleSource, keys: Sequence[str]) -> None:
        self.source = source
        self.keys = tuple(keys)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.source[index]
        missing = [key for key in self.keys if key not in sample]
        if missing:
            raise KeyError(f"sample {index} has no keys {missing}")
        return {key: sample[key] for key in self.keys}


@dataclass(frozen=True)
class AdaptationSettings:
    """The one protocol that every adaptation run of a comparison shares."""

    seed: int = 20261006
    batch_size: int = 8
    max_updates: int = 1000
    min_updates: int = 500
    max_seconds: float = 1800.0
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    heldout_samples: int = 32
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.heldout_samples < 1:
            raise ValueError("batch_size and heldout_samples must be positive")
        if self.heldout_samples % self.batch_size:
            raise ValueError("heldout_samples must be a multiple of batch_size")
        if not 1 <= self.min_updates <= self.max_updates:
            raise ValueError("need 1 <= min_updates <= max_updates")
        if not 0 < self.max_seconds <= MAX_TRAINING_SECONDS:
            raise ValueError(f"max_seconds must be in (0, {MAX_TRAINING_SECONDS:g}]")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay not negative")


def epoch_batch_indices(length: int, batch_size: int, seed: int, step: int) -> list[int]:
    """Choose training step ``step`` (counted from 1) from one fixed permutation per epoch."""
    per_epoch = length // batch_size
    if per_epoch < 1 or step < 1:
        raise ValueError("data must hold one batch and step must be positive")
    epoch, batch = divmod(step - 1, per_epoch)
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch))
    return order[batch * batch_size : (batch + 1) * batch_size].tolist()


def run_bounded_updates(
    update: Callable[[int], float],
    *,
    max_updates: int,
    max_seconds: float,
    min_updates: int,
    metrics_path: str | Path,
) -> UpdateLog:
    """Run ``update(step)`` for step 1, 2, ... until the update budget or the deadline.

    The deadline is tested before every update, so a run overshoots it by at most one
    update. A run that ends before ``min_updates`` is an error, not a result.
    """
    if not 0 < max_seconds <= MAX_TRAINING_SECONDS:
        raise ValueError(f"max_seconds must be in (0, {MAX_TRAINING_SECONDS:g}]")
    if not 1 <= min_updates <= max_updates:
        raise ValueError("need 1 <= min_updates <= max_updates")
    started = time.monotonic()
    done = 0
    last_loss = float("nan")
    stopped_by = "updates"
    with Path(metrics_path).open("w") as stream:
        for step in range(1, max_updates + 1):
            if time.monotonic() - started >= max_seconds:
                stopped_by = "deadline"
                break
            last_loss = update(step)
            done = step
            row = {
                "step": step,
                "loss": last_loss,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    seconds = time.monotonic() - started
    if done < min_updates:
        raise RuntimeError(
            f"stopped by {stopped_by} after {done} updates; at least {min_updates} are needed "
            "for an adaptation result. Keep this run as incomplete."
        )
    return {"updates": done, "stopped_by": stopped_by, "seconds": seconds, "last_loss": last_loss}


@contextmanager
def measure_use(device: str) -> Iterator[ResourceUse]:
    """Measure wall time, and PyTorch CUDA peaks when ``device`` is a CUDA device."""
    if device.startswith("cuda"):
        with measure_cuda_use() as use:
            yield use
        return
    use = ResourceUse()
    started = time.perf_counter()
    try:
        yield use
    finally:
        use.seconds = time.perf_counter() - started


def limit_gpu_memory(limit_gb: float = 48.0) -> float:
    """Cap this process at ``limit_gb`` of GPU memory and return the cap in GiB."""
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment needs a CUDA GPU")
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(limit_gb * 1024**3)
    if total > cap:
        torch.cuda.set_per_process_memory_fraction(cap / total)
    return min(total, cap) / 1024**3


def release_gpu_memory() -> None:
    """Collect garbage and return cached GPU blocks, so the next model starts clean."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_assets(workspace: str | Path, needs: Mapping[str, str]) -> None:
    """Fail once, listing every missing path with the command that creates it."""
    root = Path(workspace)
    missing = [f"  - {path}: {fix}" for path, fix in needs.items() if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(
            "This experiment needs local files that are missing. Nothing downloads here.\n"
            + "\n".join(missing)
            + "\nRun `just doctor` from the workspace to see the full local-file report."
        )


@dataclass
class _Loaded:
    policy: AnyPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    manifest: Mapping[str, object]
    deltas: dict[str, list[float]]
    drop_last: int


def _load(method: str, source: Path, vlm_dir: Path | None, device: str) -> _Loaded:
    """Load one saved bundle and its processors on ``device``, from local files only."""
    overrides: dict[str, dict[str, str]] = {"device_processor": {"device": device}}
    policy: AnyPolicy
    manifest: Mapping[str, object]
    if method == "act":
        policy, _, _, manifest = load_act_bundle(source)
        if not isinstance(policy.config, ACTConfig):
            raise TypeError("ACT bundle has no ACT config")
        deltas = {"action": frame_offset_seconds(list(range(policy.config.chunk_size)))}
        drop_last = 0
    elif method == "diffusion":
        policy, _, _, manifest = load_diffusion_bundle(source)
        if not isinstance(policy.config, DiffusionConfig):
            raise TypeError("Diffusion bundle has no Diffusion config")
        deltas = diffusion_delta_timestamps(policy.config, DIFFUSION_STORAGE_FPS)
        drop_last = policy.config.drop_n_last_frames
    elif method == "smolvla_learner":
        if vlm_dir is None:
            raise ValueError("smolvla_learner needs vlm_dir")
        policy, _, _, manifest = load_learner_bundle(source, vlm_dir, device=device)
        overrides["tokenizer_processor"] = {"tokenizer_name": str(vlm_dir.resolve())}
        if not isinstance(policy.config, SmolVLAConfig):
            raise TypeError("Learner bundle has no SmolVLA config")
        deltas = {"action": frame_offset_seconds(list(range(policy.config.chunk_size)))}
        drop_last = 0
    else:
        raise ValueError(f"Unknown method: {method}")
    policy.config.device = device
    policy = policy.to(device)
    preprocessor, postprocessor = load_pretrained_policy_processors(
        str(source), preprocessor_overrides=overrides
    )
    return _Loaded(policy, preprocessor, postprocessor, manifest, deltas, drop_last)


def _sample_positions(dataset: SampleSource, drop_last: int) -> list[int]:
    """Dataset rows that may be a current frame. Diffusion drops each episode's last rows."""
    if drop_last == 0:
        return list(range(len(dataset)))
    table = getattr(dataset, "hf_dataset")  # noqa: B009 - LeRobotDataset exposes this at run time
    return eligible_frame_indices(
        [int(value) for value in table["episode_index"]],
        [int(value) for value in table["frame_index"]],
        drop_last,
    )


def _heldout_loss(loaded: _Loaded, validation: SampleSource, settings: AdaptationSettings) -> float:
    """Mean loss on the same validation rows, noise, and time samples every time."""
    eligible = _sample_positions(validation, loaded.drop_last)
    if len(eligible) < settings.heldout_samples:
        raise ValueError("the validation split is smaller than heldout_samples")
    spread = np.linspace(0, len(eligible) - 1, settings.heldout_samples).round().astype(int)
    rows = [eligible[int(index)] for index in spread]
    policy = loaded.policy
    was_training = policy.training
    policy.eval()
    torch.manual_seed(settings.seed + 1)
    losses: list[float] = []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), settings.batch_size):
                raw = default_collate(
                    [validation[row] for row in rows[start : start + settings.batch_size]]
                )
                loss, _ = policy(loaded.preprocessor(raw))
                losses.append(float(loss.detach().cpu()))
    finally:
        policy.train(was_training)
    return float(np.mean(losses))


def _optimization_step(
    policy: AnyPolicy,
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    batch: Mapping[str, object],
    step: int,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss, _ = policy(dict(batch))
    if not torch.isfinite(loss):
        raise FloatingPointError(f"loss is not finite at step {step}")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"gradient is not finite at step {step}")
    optimizer.step()
    return float(loss.detach().cpu())


def _save_adapted(
    method: str,
    loaded: _Loaded,
    destination: Path,
    *,
    task_alias: str,
    configs_root: Path,
    vlm_dir: Path | None,
    robot_steps: int,
    parent_sha256: str,
) -> None:
    policy = loaded.policy
    if method == "act" and isinstance(policy, ACTPolicy):
        save_act_bundle(
            policy, loaded.preprocessor, loaded.postprocessor, destination, task_alias, configs_root
        )
    elif method == "diffusion" and isinstance(policy, DiffusionPolicy):
        save_diffusion_bundle(
            policy, loaded.preprocessor, loaded.postprocessor, destination, task_alias, configs_root
        )
    elif method == "smolvla_learner" and isinstance(policy, SmolVLAPolicy) and vlm_dir is not None:
        save_learner_bundle(
            policy=policy,
            preprocessor=loaded.preprocessor,
            postprocessor=loaded.postprocessor,
            destination=destination,
            task_alias=task_alias,
            configs_root=configs_root,
            vlm_dir=vlm_dir,
            vlm_revision=str(loaded.manifest["vlm_revision"]),
            vlm_sha256=str(loaded.manifest["vlm_sha256"]),
            stage="c2",
            step=robot_steps,
            parent_model_sha256=parent_sha256,
        )
    else:
        raise ValueError(f"Cannot save a {method} bundle from a {type(policy).__name__}")


def _adapt_policy(
    method: str,
    *,
    source: str | Path,
    source_robot_steps: int,
    output_dir: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    settings: AdaptationSettings,
    vlm_dir: str | Path | None = None,
    task_alias: str = "bowl_plate",
    train_data: SampleSource | None = None,
    new_data_start: int | None = None,
) -> AdaptationRecord:
    """Continue one saved policy on the frozen training split under a deadline.

    The windows come from frame offsets of the dataset rows. The optimizer is a new
    AdamW over every trainable parameter, with the same settings for every method.
    The saved checkpoint keeps the loader format of its method, so the comparison
    evaluator can load it. ``train_data`` replaces the frozen training split (the
    Lesson 21 intervention). Indices ``>= new_data_start`` then count as new data.
    """
    source_path = Path(source).expanduser().resolve()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Adaptation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    configs = Path(configs_root)
    data = Path(data_root)
    vlm = None if vlm_dir is None else Path(vlm_dir).expanduser().resolve()
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if settings.device.startswith("cuda"):
        torch.cuda.manual_seed_all(settings.seed)

    split = yaml.safe_load((configs / "data" / "episodes.yaml").read_text())
    task = split["tasks"][task_alias]
    loaded = _load(method, source_path, vlm, settings.device)
    policy = loaded.policy
    source_sha256 = sha256_file(source_path / "model.safetensors")
    train: SampleSource = (
        open_libero_dataset(configs, data, task_alias, "train", loaded.deltas)
        if train_data is None
        else train_data
    )
    validation = open_libero_dataset(configs, data, task_alias, "validation", loaded.deltas)
    positions = _sample_positions(train, loaded.drop_last if train_data is None else 0)
    if len(positions) < settings.batch_size:
        raise ValueError("training data has fewer samples than one batch")
    heldout_before = _heldout_loss(loaded, validation, settings)

    policy.train()
    if isinstance(policy, SmolVLAPolicy):
        policy.model.vlm_with_expert.vlm.eval()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    drawn = 0
    drawn_new = 0

    def update(step: int) -> float:
        nonlocal drawn, drawn_new
        chosen = [
            positions[i]
            for i in epoch_batch_indices(len(positions), settings.batch_size, settings.seed, step)
        ]
        drawn += len(chosen)
        if new_data_start is not None:
            drawn_new += sum(index >= new_data_start for index in chosen)
        raw = default_collate([train[index] for index in chosen])
        if isinstance(policy, DiffusionPolicy) and isinstance(policy.config, DiffusionConfig):
            validate_window_batch(raw, policy.config)
        return _optimization_step(policy, optimizer, parameters, loaded.preprocessor(raw), step)

    metrics_path = output / "metrics.jsonl"
    with measure_use(settings.device) as use:
        log = run_bounded_updates(
            update,
            max_updates=settings.max_updates,
            max_seconds=settings.max_seconds,
            min_updates=settings.min_updates,
            metrics_path=metrics_path,
        )
    heldout_after = _heldout_loss(loaded, validation, settings)
    robot_steps = source_robot_steps + log["updates"]
    checkpoint = output / f"checkpoint-{robot_steps:06d}"
    _save_adapted(
        method,
        loaded,
        checkpoint,
        task_alias=task_alias,
        configs_root=configs,
        vlm_dir=vlm,
        robot_steps=robot_steps,
        parent_sha256=source_sha256,
    )
    model_sha256 = sha256_file(checkpoint / "model.safetensors")
    run_path = output / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": method,
                "task_alias": task_alias,
                "dataset_revision": split["dataset_revision"],
                "train_episode_ids": task["train_episode_ids"],
                "validation_episode_ids": task["validation_episode_ids"],
                "split_sha256": task["split_sha256"],
                "steps": robot_steps,
                "source_robot_steps": source_robot_steps,
                "adaptation_updates": log["updates"],
                "stopped_by": log["stopped_by"],
                "seed": settings.seed,
                "batch_size": settings.batch_size,
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "max_updates": settings.max_updates,
                "max_seconds": settings.max_seconds,
                "optimizer": "AdamW, new state, every trainable parameter",
                "sampler": "epoch_randperm_v1",
                "window_offsets": {
                    key: [round(value * DIFFUSION_STORAGE_FPS) for value in values]
                    for key, values in loaded.deltas.items()
                },
                "storage_label_fps": DIFFUSION_STORAGE_FPS,
                "source_checkpoint": str(source_path),
                "source_model_sha256": source_sha256,
                "device": settings.device,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    record: AdaptationRecord = {
        "method": method,
        "source_checkpoint": str(source_path),
        "source_model_sha256": source_sha256,
        "source_robot_steps": source_robot_steps,
        "checkpoint": str(checkpoint),
        "model_sha256": model_sha256,
        "robot_training_steps": robot_steps,
        "adaptation_updates": log["updates"],
        "samples_seen": log["updates"] * settings.batch_size,
        "batch_size": settings.batch_size,
        "seed": settings.seed,
        "learning_rate": settings.learning_rate,
        "max_seconds": settings.max_seconds,
        "stopped_by": log["stopped_by"],
        "train_seconds": log["seconds"],
        "last_loss": log["last_loss"],
        "heldout_loss_before": heldout_before,
        "heldout_loss_after": heldout_after,
        "heldout_samples": settings.heldout_samples,
        "peak_allocated_mib": use.peak_allocated_mib,
        "peak_reserved_mib": use.peak_reserved_mib,
        "new_sample_fraction": drawn_new / drawn if drawn else 0.0,
        "metrics_path": str(metrics_path),
        "run_path": str(run_path),
    }
    return record


def adapt_policy(
    method: str,
    *,
    source: str | Path,
    source_robot_steps: int,
    output_dir: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    settings: AdaptationSettings,
    vlm_dir: str | Path | None = None,
    task_alias: str = "bowl_plate",
    train_data: SampleSource | None = None,
    new_data_start: int | None = None,
) -> AdaptationRecord:
    """Continue one saved policy on the frozen training split under a deadline.

    The windows come from frame offsets of the dataset rows. The optimizer is a new
    AdamW over every trainable parameter, with the same settings for every method.
    The saved checkpoint keeps the loader format of its method, so the comparison
    evaluator can load it. ``train_data`` replaces the frozen training split (the
    Lesson 21 intervention). Indices ``>= new_data_start`` then count as new data.
    The model is released from GPU memory before this function returns.
    """
    try:
        return _adapt_policy(
            method,
            source=source,
            source_robot_steps=source_robot_steps,
            output_dir=output_dir,
            configs_root=configs_root,
            data_root=data_root,
            settings=settings,
            vlm_dir=vlm_dir,
            task_alias=task_alias,
            train_data=train_data,
            new_data_start=new_data_start,
        )
    finally:
        release_gpu_memory()


def checkpoint_heldout_loss(
    method: str,
    checkpoint: str | Path,
    *,
    configs_root: str | Path,
    data_root: str | Path,
    settings: AdaptationSettings,
    vlm_dir: str | Path | None = None,
    task_alias: str = "bowl_plate",
) -> float:
    """Reload a saved bundle from disk and score it on the held-out validation rows."""
    loaded = _load(
        method,
        Path(checkpoint).expanduser().resolve(),
        None if vlm_dir is None else Path(vlm_dir).expanduser().resolve(),
        settings.device,
    )
    validation = open_libero_dataset(
        configs_root, data_root, task_alias, "validation", loaded.deltas
    )
    value = _heldout_loss(loaded, validation, settings)
    del loaded
    release_gpu_memory()
    return value


def continue_act_matched(
    *,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    updates: int,
    seed: int,
    base_robot_steps: int,
    extra_data: SampleSource | None = None,
    device: str = "cuda",
    max_seconds: float = 1800.0,
    batch_size: int = 8,
) -> AdaptationRecord:
    """Continue ACT for exactly ``updates`` updates from the same weights, optimizer, and seed.

    Without ``extra_data`` the run continues on the original training split. With it, the
    run continues on the original split followed by the extra rows. Nothing else differs.
    A deadline that cuts the run short is an error, because a short arm breaks the match.
    """
    deltas = {"action": frame_offset_seconds(list(range(ACT_CHUNK_SIZE)))}
    original = open_libero_dataset(configs_root, data_root, "bowl_plate", "train", deltas)
    keep = KeyProjection(original, ACT_SAMPLE_KEYS)
    data: SampleSource = keep
    start = None
    if extra_data is not None:
        data = ConcatDataset([keep, KeyProjection(extra_data, ACT_SAMPLE_KEYS)])
        start = len(keep)
    settings = AdaptationSettings(
        seed=seed,
        batch_size=batch_size,
        max_updates=updates,
        min_updates=updates,
        max_seconds=max_seconds,
        device=device,
    )
    return adapt_policy(
        "act",
        source=base_checkpoint,
        source_robot_steps=base_robot_steps,
        output_dir=output_dir,
        configs_root=configs_root,
        data_root=data_root,
        settings=settings,
        train_data=data,
        new_data_start=start,
    )


def evaluate_development(
    method: str,
    checkpoint: str | Path,
    *,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    state_ids: Sequence[int] = DEVELOPMENT_METHOD_STATES,
    vlm_dir: str | Path | None = None,
    inference_steps: int = 20,
) -> dict[str, JSONValue]:
    """Run one saved checkpoint on development states 10-14 with the frozen seed rule.

    Final states are refused here. They belong to the one evaluation after the freeze.
    """
    if not state_ids or not set(state_ids) <= set(DEVELOPMENT_METHOD_STATES):
        raise ValueError(f"before/after checks use states {list(DEVELOPMENT_METHOD_STATES)} only")
    if method == "act":
        return evaluate_act_states(
            checkpoint=checkpoint,
            configs_root=configs_root,
            data_root=data_root,
            output_dir=output_dir,
            state_ids=state_ids,
        )
    if method == "diffusion":
        return evaluate_diffusion_reference(
            checkpoint=checkpoint,
            configs_root=configs_root,
            data_root=data_root,
            output_dir=output_dir,
            state_ids=state_ids,
            seed=BASE_SEED,
            inference_steps=inference_steps,
        )
    if method == "smolvla_learner" and vlm_dir is not None:
        return evaluate_learner_reference(
            checkpoint=checkpoint,
            vlm_dir=vlm_dir,
            configs_root=configs_root,
            data_root=data_root,
            output_dir=output_dir,
            state_ids=state_ids,
            seed=BASE_SEED,
        )
    raise ValueError(f"Cannot evaluate {method} (smolvla_learner needs vlm_dir)")


def adapted_recipe(
    start: ComparisonConfig, records: Mapping[str, AdaptationRecord]
) -> dict[str, JSONValue]:
    """Write the new comparison recipe: the start recipe with the adapted checkpoints."""
    recipe: dict[str, JSONValue] = json.loads(json.dumps(start))
    policies = recipe["policies"]
    if not isinstance(policies, dict) or set(records) != set(policies):
        raise ValueError("need one adaptation record for every method in the recipe")
    for method, record in records.items():
        entry = policies[method]
        if not isinstance(entry, dict):
            raise ValueError(f"{method} recipe is not a mapping")
        entry.update(
            {
                "checkpoint": record["checkpoint"],
                "model_sha256": record["model_sha256"],
                "robot_training_steps": record["robot_training_steps"],
                "start_checkpoint": record["source_checkpoint"],
                "start_model_sha256": record["source_model_sha256"],
                "adaptation_updates": record["adaptation_updates"],
                "pretraining": (
                    f"{entry['pretraining']}; then {record['adaptation_updates']} adaptation "
                    f"updates on the bowl_plate train split (batch {record['batch_size']})"
                ),
            }
        )
    return recipe


def episode_time_summary(records: Sequence[Mapping[str, JSONValue]]) -> EpisodeTimes:
    """Separate simulated time (ticks / 20 Hz) from wall time and model-call wall time."""
    simulated: list[float] = []
    wall: list[float] = []
    inference: list[float] = []
    both: list[tuple[float, float]] = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        ticks = metrics.get("steps")
        elapsed = metrics.get("elapsed_s")
        model_wall = metrics.get("inference_wall_s")
        if ticks is not None:
            simulated.append(_number(ticks) / LIBERO_CONTROL_HZ)
        if elapsed is not None:
            wall.append(_number(elapsed))
        if model_wall is not None:
            inference.append(_number(model_wall))
        if ticks is not None and elapsed is not None:
            both.append((_number(ticks) / LIBERO_CONTROL_HZ, _number(elapsed)))

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    simulated_total = sum(item[0] for item in both)
    return {
        "mean_simulated_s": mean(simulated),
        "mean_wall_s": mean(wall),
        "mean_inference_wall_s": mean(inference),
        "wall_over_simulated": (
            sum(item[1] for item in both) / simulated_total if simulated_total > 0 else None
        ),
    }


def describe_adaptation(record: AdaptationRecord) -> str:
    """One printable line with what an adaptation run really did."""
    return (
        f"{record['method']} | start steps: {record['source_robot_steps']} "
        f"| updates: {record['adaptation_updates']} (stopped by {record['stopped_by']}) "
        f"| samples seen: {record['samples_seen']} | seconds: {record['train_seconds']:.1f} "
        f"| peak allocated MiB: {record['peak_allocated_mib']:.0f} "
        f"| held-out loss {record['heldout_loss_before']:.4f} -> {record['heldout_loss_after']:.4f}"
    )
