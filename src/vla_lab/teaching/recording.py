"""Small transition checks for recorded robot episodes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from vla_lab.teaching.records import (
    AlignedTransition,
    InterventionCase,
    InterventionSummary,
    TransitionAlignment,
)


def transition_alignment(
    observation_ticks: Sequence[int] | np.ndarray, action_ticks: Sequence[int] | np.ndarray
) -> TransitionAlignment:
    """Check that action i was applied after observation i on the same tick."""
    obs = np.asarray(observation_ticks, dtype=np.int64)
    act = np.asarray(action_ticks, dtype=np.int64)
    if obs.ndim != 1 or act.ndim != 1 or len(obs) != len(act) + 1 or len(act) == 0:
        raise ValueError("need N action ticks and N+1 observation ticks")
    if obs[0] != 0 or np.any(np.diff(obs) != 1):
        raise ValueError("observation ticks must be contiguous and start at zero")
    expected = obs[:-1]
    offsets = act - expected
    return {
        "aligned": bool(np.array_equal(act, expected)),
        "offset_ticks": sorted(set(map(int, offsets))),
        "action_hz": 20,
        "recorded_dataset_hz": 10,
        "dataset_stride_ticks": 2,
    }


def dataset_tick_indices(control_ticks: Sequence[int]) -> np.ndarray:
    """Map tick timestamps to 10 Hz row numbers; this does not change execution."""
    ticks = np.asarray(control_ticks, dtype=np.int64)
    if ticks.ndim != 1 or np.any(ticks < 0):
        raise ValueError("control_ticks must be a one-dimensional nonnegative sequence")
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("control_ticks must increase")
    if np.any(ticks % 2):
        raise ValueError("10 Hz rows align to every second 20 Hz control tick")
    return ticks // 2


def aligned_transition_indices(
    observation_ticks: Sequence[int] | np.ndarray, action_ticks: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Return action row indexes paired with their pre-action observations.

    This reference path requires exact same-tick alignment. It does not shift
    rows to hide a timing defect.
    """
    report = transition_alignment(observation_ticks, action_ticks)
    if not report["aligned"]:
        raise ValueError(f"recording actions are misaligned by {report['offset_ticks']} ticks")
    return np.arange(len(action_ticks), dtype=np.int64)


def aligned_transition_rows[ObservationT](
    observations: Sequence[ObservationT],
    actions: ArrayLike,
    observation_ticks: Sequence[int] | np.ndarray,
    action_ticks: Sequence[int] | np.ndarray,
) -> list[AlignedTransition[ObservationT]]:
    """Build state/action rows only after exact temporal alignment passes."""
    action_array = np.asarray(actions, dtype=np.float32)
    indices = aligned_transition_indices(observation_ticks, action_ticks)
    if action_array.ndim != 2 or len(action_array) != len(indices):
        raise ValueError("actions must have shape (transitions, action_dim)")
    if len(observations) != len(indices) + 1:
        raise ValueError("observations must contain one final state after the last action")
    rows: list[AlignedTransition[ObservationT]] = []
    for index in indices:
        rows.append(
            {
                "tick": int(action_ticks[int(index)]),
                "observation": observations[int(index)],
                "action": action_array[int(index)].copy(),
            }
        )
    return rows


def resample_aligned_transitions_10hz[ObservationT](
    rows: Sequence[AlignedTransition[ObservationT]], *, control_hz: int = 20, dataset_hz: int = 10
) -> list[AlignedTransition[ObservationT]]:
    """Select aligned rows at the saved 10 Hz cadence from 20 Hz control rows.

    This conversion is for newly recorded recovery data. It does not change
    the historical runner, which applies one recorded action per controller
    tick. Rows must contain contiguous control ticks beginning at zero.
    """
    if control_hz < 1 or dataset_hz < 1 or control_hz % dataset_hz:
        raise ValueError("dataset_hz must be a positive divisor of control_hz")
    if not rows:
        raise ValueError("at least one aligned transition is required")
    stride = control_hz // dataset_hz
    ticks = [int(row["tick"]) for row in rows]
    if ticks != list(range(len(rows))):
        raise ValueError("transition rows must have contiguous aligned control ticks from zero")
    return [rows[index].copy() for index in range(0, len(rows), stride)]


def intervention_effect(
    baseline: Sequence[InterventionCase],
    changed_data: Sequence[InterventionCase],
    *,
    case_ids: Sequence[str | int],
) -> InterventionSummary:
    """Compare a data intervention on the same development cases."""
    wanted = list(case_ids)
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("case_ids must be nonempty and unique")
    left_ids = [row["case_id"] for row in baseline]
    right_ids = [row["case_id"] for row in changed_data]
    if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        raise ValueError("each run must include every case exactly once")
    if any("success" not in row for row in (*baseline, *changed_data)):
        raise ValueError("each intervention result needs a success value")
    left_rows = {row["case_id"]: row for row in baseline}
    right_rows = {row["case_id"]: row for row in changed_data}
    identity_fields = ("state_id", "reset_seed", "task_id", "prompt")

    def identity(row: InterventionCase) -> tuple[int | None, int | None, str | None, str | None]:
        return row.get("state_id"), row.get("reset_seed"), row.get("task_id"), row.get("prompt")

    for case_id in wanted:
        a, b = left_rows[case_id], right_rows[case_id]
        present = [key for key in identity_fields if key in a or key in b]
        if any(key not in a or key not in b for key in present):
            raise ValueError(
                "paired intervention rows need matching state, seed, task, and prompt identity"
            )
        if identity(a) != identity(b):
            raise ValueError(
                "paired intervention rows differ in state, seed, task, or prompt identity"
            )
    left = {case_id: row["success"] is True for case_id, row in left_rows.items()}
    right = {case_id: row["success"] is True for case_id, row in right_rows.items()}
    if set(left) != set(wanted) or set(right) != set(wanted):
        raise ValueError("both runs must include each selected case exactly once")
    gains = [case for case in wanted if right[case] and not left[case]]
    losses = [case for case in wanted if left[case] and not right[case]]
    return {
        "cases": len(wanted),
        "baseline_successes": sum(left.values()),
        "changed_data_successes": sum(right.values()),
        "gains": gains,
        "losses": losses,
        "failed_attempts_included": True,
    }
