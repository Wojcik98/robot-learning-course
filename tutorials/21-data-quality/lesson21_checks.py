"""Behaviour checks for the Lesson 21 exercises.

These add cases the notebook checks do not reach: early (negative) tick labels, a defect
in the middle of an episode, windows that end before the episode does, per-column action
bounds, several gripper switches, case order and partially filled case identities.
Read a failing assertion message first: it states the expected behaviour.
"""

import math
from types import SimpleNamespace

import numpy as np

from vla_lab.data.recording import RecordedEpisode
from vla_lab.teaching.records import InterventionCase


def _done(message: str) -> None:
    print(f"✓ {message}")


def _expect_value_error(call, message: str, must_mention: str | None = None) -> None:
    try:
        call()
    except ValueError as error:
        if must_mention is not None:
            assert must_mention in str(error), (
                f"{message}; the error should name {must_mention!r}, got: {error}"
            )
        return
    raise AssertionError(message)


def check_transition_alignment(transition_alignment) -> None:
    """Exercise 21.1: offsets, their types and order, and the duration."""
    early = transition_alignment(np.arange(5), np.array([0, 0, 1, 3]))
    assert early.aligned is False, "any non-zero offset means not aligned"
    assert early.offset_ticks == [-1, 0], (
        f"offsets are action_ticks - observation_ticks[:N], sorted and distinct: expected [-1, 0], "
        f"got {early.offset_ticks}"
    )
    assert all(type(v) is int for v in early.offset_ticks), (
        "offsets must be Python ints (NumPy input too)"
    )
    assert type(early.aligned) is bool, "aligned must be a Python bool"
    _done("negative offsets (early labels) are found; offsets are sorted Python ints")

    long = transition_alignment(np.arange(41), np.arange(40))
    assert long.aligned is True and long.offset_ticks == [0]
    assert math.isclose(long.simulated_duration_s, 2.0), "40 actions at 20 Hz last 2.0 s"
    assert long.control_hz == 20 and long.source_storage_fps == 10, (
        "control_hz is LIBERO_CONTROL_HZ (20); source_storage_fps is SOURCE_STORAGE_FPS (10)"
    )
    one = transition_alignment([0, 1], [0])
    assert math.isclose(one.simulated_duration_s, 0.05), "one action lasts one 50 ms tick"
    _done("duration is N / 20 s; the two rates are reported separately")


def _episode(steps: int, episode_id: str, seed: int) -> RecordedEpisode:
    generator = np.random.default_rng(seed)
    return RecordedEpisode(
        episode_id=episode_id,
        task_id="bowl_plate",
        actions=generator.uniform(-0.5, 0.5, size=(steps, 7)).astype(np.float32),
        action_ticks=np.arange(steps),
        observation_ticks=np.arange(steps + 1),
        observations=tuple(
            {"state": np.full(8, tick, dtype=np.float32)} for tick in range(steps + 1)
        ),
        reset_seed=seed,
    )


def check_prepare_action_batch(prepare_action_batch) -> None:
    """Exercise 21.2: full windows, the default horizon, and a defect in the middle."""
    long_episode, short_episode = _episode(10, "long", 1), _episode(3, "short", 2)
    originals = (long_episode.actions.copy(), short_episode.actions.copy())
    batch = prepare_action_batch([long_episode, short_episode], [(0, 2), (0, 2), (1, 0)], horizon=4)
    np.testing.assert_array_equal(
        batch["action"][0],
        long_episode.actions[2:6],
        err_msg="a window that ends inside the episode holds exactly `horizon` real rows",
    )
    assert batch["action_is_pad"][0].tolist() == [False] * 4, "a full window has no padding"
    np.testing.assert_array_equal(
        batch["action"][1], batch["action"][0], err_msg="the same pick gives the same window"
    )
    assert batch["action_is_pad"][2].tolist() == [False, False, False, True]
    assert batch["episode_ids"] == ["long", "long", "short"] and batch["rows"] == [2, 2, 0]
    assert all(type(r) is int for r in batch["rows"]), "rows must be Python ints"
    assert np.array_equal(long_episode.actions, originals[0]) and np.array_equal(
        short_episode.actions, originals[1]
    ), "do not modify the recordings"
    _done("windows inside an episode are unpadded; repeated picks and pick order are kept")

    default = prepare_action_batch([long_episode], [(0, 0)])
    assert default["action"].shape == (1, 20, 7), "the default horizon is ACT_CHUNK_SIZE (20)"
    assert default["action_is_pad"][0].tolist() == [False] * 10 + [True] * 10
    _done("the default horizon is ACT_CHUNK_SIZE")

    ticks = np.arange(10)
    ticks[6] = 7  # one label in the middle is one tick late
    # RecordedEpisode refuses such a recording, so a stand-in with the same fields carries it.
    middle_defect = SimpleNamespace(
        episode_id="middle-defect",
        actions=long_episode.actions,
        observation_ticks=long_episode.observation_ticks,
        action_ticks=ticks,
    )
    _expect_value_error(
        lambda: prepare_action_batch([short_episode, middle_defect], [(0, 0)], horizon=4),
        "a defect in the middle of any episode (even an unused one) must raise ValueError",
        must_mention="middle-defect",
    )
    _done("a single late label in the middle of an unused episode is rejected, naming the episode")


def check_action_quality_report(action_quality_report) -> None:
    """Exercise 21.3: per-column bounds, the gripper column, and Python number types."""
    lower = [-1.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    upper = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    actions = np.zeros((5, 7))
    actions[:, 1] = [1.9, 0.0, 0.0, -1.97, 0.0]  # column 1 has range 4
    actions[:, 0] = [0.0, 0.3, 0.0, 0.0, 0.0]
    actions[:, 6] = [-1.0, 1.0, -1.0, 0.0, 1.0]  # > 0: F T F F T -> 3 switches
    report = action_quality_report(actions, lower=lower, upper=upper, edge_margin=0.01)
    # Column 1 edges: <= -2 + 0.04 or >= 2 - 0.04. 1.9 is not saturated; -1.97 is.
    assert math.isclose(report["saturated_fraction"], 1 / 30), (
        f"saturation uses each column's own bounds and range (edge_margin * (u - l)); "
        f"expected 1/30, got {report['saturated_fraction']}"
    )
    wide = action_quality_report(actions, lower=lower, upper=upper, edge_margin=0.03)
    assert math.isclose(wide["saturated_fraction"], 2 / 30), (
        "a larger edge_margin marks more values"
    )
    assert report["gripper_switches"] == 3, (
        f"count changes of (gripper > 0) between ticks; expected 3, got {report['gripper_switches']}"
    )
    assert math.isclose(report["max_step_change"], 1.97), (
        f"max_step_change uses the six arm columns only (gripper jumps of 2 are ignored); "
        f"got {report['max_step_change']}"
    )
    # Idle: max |arm| < 0.05 -> ticks 2 and 4 (tick 0 has 1.9, tick 1 has 0.3, tick 3 has -1.97).
    assert math.isclose(report["idle_fraction"], 2 / 5), (
        f"idle ignores the gripper column; expected 2/5, got {report['idle_fraction']}"
    )
    assert type(report["steps"]) is int and type(report["gripper_switches"]) is int
    assert all(
        type(report[key]) is float
        for key in ("duration_s", "idle_fraction", "saturated_fraction", "max_step_change")
    ), "return Python floats"
    assert math.isclose(report["duration_s"], 0.25)
    _done("per-column bounds, gripper excluded from idle/saturation/jumps, 3 gripper switches")

    boundary = np.zeros((2, 7))
    boundary[0, 0] = 0.05  # not idle: the rule is strictly below the threshold
    edge = action_quality_report(boundary, lower=lower, upper=upper)
    assert math.isclose(edge["idle_fraction"], 0.5), (
        "a tick is idle only when max |arm| < idle_threshold"
    )
    _done("the idle rule is strict (< idle_threshold)")


def check_intervention_effect(intervention_effect) -> None:
    """Exercise 21.4: case order, extra cases, partial identities, and strict success."""

    def row(case, success, **fields):
        return InterventionCase(case_id=case, success=success, **fields)

    ids = [12, 3, 7]
    baseline = [row(3, False), row(7, True), row(12, False)]
    changed = [row(3, True), row(7, False), row(12, True)]
    summary = intervention_effect(baseline, changed, case_ids=ids)
    assert summary.gains == [12, 3], f"gains follow case_ids order ([12, 3]), got {summary.gains}"
    assert summary.losses == [7] and summary.cases == 3
    assert summary.baseline_successes == 1 and summary.changed_data_successes == 2
    assert summary.failed_attempts_included is True
    _done("gains and losses are listed in case_ids order; integer case IDs work")

    not_true = [row(3, "yes"), row(7, True), row(12, False)]  # only True itself is a success
    summary = intervention_effect(not_true, changed, case_ids=ids)
    assert summary.baseline_successes == 1 and 3 in summary.gains, (
        "only success is True counts as a success"
    )
    _done("only success is True counts as a success")

    _expect_value_error(
        lambda: intervention_effect(baseline, changed, case_ids=[]),
        "empty case_ids must raise ValueError",
    )
    _expect_value_error(
        lambda: intervention_effect(baseline, changed, case_ids=[3, 3, 7, 12]),
        "repeated case_ids must raise ValueError",
    )
    _expect_value_error(
        lambda: intervention_effect(baseline + [row(99, True)], changed, case_ids=ids),
        "a run with a case that was not selected must raise ValueError",
    )
    _expect_value_error(
        lambda: intervention_effect(baseline, changed, case_ids=[3, 7]),
        "a run with more cases than selected must raise ValueError",
    )
    with_prompt = [row(3, False, prompt="pick"), row(7, True), row(12, False)]
    _expect_value_error(
        lambda: intervention_effect(with_prompt, changed, case_ids=ids),
        "a prompt set in one run and None in the other must raise ValueError",
    )
    same_prompt = [row(3, True, prompt="pick"), row(7, False), row(12, True)]
    assert intervention_effect(with_prompt, same_prompt, case_ids=ids).gains == [12, 3], (
        "equal identity fields in both runs are fine"
    )
    _done("bad case lists, extra cases and a one-sided identity field raise ValueError")
