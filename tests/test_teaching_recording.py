import numpy as np
import pytest

from vla_lab.teaching.recording import (
    aligned_transition_rows,
    dataset_tick_indices,
    intervention_effect,
    preserve_aligned_transitions,
    transition_alignment,
)


def test_transition_alignment_detects_one_tick_defect_and_preserves_frame_offsets():
    report = transition_alignment([0, 1, 2, 3], [1, 2, 3])
    assert report["aligned"] is False
    assert report["offset_ticks"] == [1]
    np.testing.assert_array_equal(dataset_tick_indices([0, 1, 2, 3]), [0, 1, 2, 3])


def test_aligned_transition_rows_pair_pre_action_observation_and_action():
    rows = aligned_transition_rows(
        [{"state": 0}, {"state": 1}, {"state": 2}],
        [[1], [2]],
        [0, 1, 2],
        [0, 1],
    )
    assert rows[1]["observation"] == {"state": 1}
    np.testing.assert_array_equal(rows[1]["action"], [2])


def test_recorded_rows_preserve_every_control_transition():
    rows = [{"tick": tick, "observation": tick, "action": [tick]} for tick in range(5)]
    saved = preserve_aligned_transitions(rows, control_hz=20)
    assert [row["tick"] for row in saved] == [0, 1, 2, 3, 4]
    assert [row["action"] for row in saved] == [[0], [1], [2], [3], [4]]
    report = transition_alignment(range(6), range(5))
    assert report["control_hz"] == 20
    assert report["source_storage_fps"] == 10
    assert report["simulated_duration_s"] == 0.25


def test_recorded_rows_reject_missing_control_transition():
    with pytest.raises(ValueError, match="contiguous"):
        preserve_aligned_transitions([{"tick": 0}, {"tick": 2}])
    with pytest.raises(ValueError, match="20 Hz"):
        preserve_aligned_transitions([], control_hz=10)


def test_intervention_effect_rejects_duplicate_cases():
    with pytest.raises(ValueError, match="exactly once"):
        intervention_effect(
            [{"case_id": 15, "success": True}, {"case_id": 15, "success": False}],
            [{"case_id": 15, "success": True}],
            case_ids=[15],
        )


def test_intervention_effect_checks_state_and_seed_identity():
    base = {
        "case_id": "state-15-seed-10",
        "state_id": 15,
        "reset_seed": 10,
        "task_id": "bowl_plate",
        "success": False,
    }
    changed = {**base, "reset_seed": 11, "success": True}
    with pytest.raises(ValueError, match="identity"):
        intervention_effect([base], [changed], case_ids=[base["case_id"]])
