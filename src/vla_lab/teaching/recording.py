"""Small transition checks for recorded robot episodes."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from vla_lab.teaching.records import (
    AlignedTransition,
    InterventionCase,
    InterventionSummary,
    TransitionAlignment,
)
from vla_lab.timing import LIBERO_CONTROL_HZ, SOURCE_STORAGE_FPS, require_libero_rate


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
    return TransitionAlignment(
        aligned=bool(np.array_equal(act, expected)),
        offset_ticks=sorted(set(map(int, offsets))),
        control_hz=LIBERO_CONTROL_HZ,
        source_storage_fps=SOURCE_STORAGE_FPS,
        simulated_duration_s=len(act) / LIBERO_CONTROL_HZ,
    )


def dataset_tick_indices(control_ticks: Sequence[int]) -> np.ndarray:
    """Map saved control ticks to row offsets without dropping transitions."""
    ticks = np.asarray(control_ticks, dtype=np.int64)
    if ticks.ndim != 1 or np.any(ticks < 0):
        raise ValueError("control_ticks must be a one-dimensional nonnegative sequence")
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("control_ticks must increase")
    return ticks.copy()


def aligned_transition_indices(
    observation_ticks: Sequence[int] | np.ndarray, action_ticks: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Return action row indexes paired with their pre-action observations.

    This reference path requires exact same-tick alignment. It does not shift
    rows to hide a timing defect.
    """
    report = transition_alignment(observation_ticks, action_ticks)
    if not report.aligned:
        raise ValueError(f"recording actions are misaligned by {report.offset_ticks} ticks")
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
            AlignedTransition(
                tick=int(action_ticks[int(index)]),
                observation=observations[int(index)],
                action=action_array[int(index)].copy(),
            )
        )
    return rows


def preserve_aligned_transitions[ObservationT](
    rows: Sequence[AlignedTransition[ObservationT]], *, control_hz: int = LIBERO_CONTROL_HZ
) -> list[AlignedTransition[ObservationT]]:
    """Save every aligned control transition in a new recording.

    Source dataset storage labels do not set a row stride. One saved action
    corresponds to one simulator step at the declared control rate.
    """
    require_libero_rate(control_hz)
    if not rows:
        raise ValueError("at least one aligned transition is required")
    ticks = [int(row.tick) for row in rows]
    if ticks != list(range(len(rows))):
        raise ValueError("transition rows must have contiguous aligned control ticks from zero")
    return [replace(row) for row in rows]


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
    left_ids = [row.case_id for row in baseline]
    right_ids = [row.case_id for row in changed_data]
    if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        raise ValueError("each run must include every case exactly once")
    left_rows = {row.case_id: row for row in baseline}
    right_rows = {row.case_id: row for row in changed_data}

    def identity(row: InterventionCase) -> tuple[int | None, int | None, str | None, str | None]:
        return row.state_id, row.reset_seed, row.task_id, row.prompt

    for case_id in wanted:
        a, b = identity(left_rows[case_id]), identity(right_rows[case_id])
        if any((x is None) != (y is None) for x, y in zip(a, b, strict=True)):
            raise ValueError(
                "paired intervention rows need matching state, seed, task, and prompt identity"
            )
        if a != b:
            raise ValueError(
                "paired intervention rows differ in state, seed, task, or prompt identity"
            )
    left = {case_id: row.success is True for case_id, row in left_rows.items()}
    right = {case_id: row.success is True for case_id, row in right_rows.items()}
    if set(left) != set(wanted) or set(right) != set(wanted):
        raise ValueError("both runs must include each selected case exactly once")
    gains = [case for case in wanted if right[case] and not left[case]]
    losses = [case for case in wanted if left[case] and not right[case]]
    return InterventionSummary(
        cases=len(wanted),
        baseline_successes=sum(left.values()),
        changed_data_successes=sum(right.values()),
        gains=gains,
        losses=losses,
        failed_attempts_included=True,
    )
