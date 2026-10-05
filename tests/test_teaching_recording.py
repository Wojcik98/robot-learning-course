import numpy as np
import pytest

from vla_lab.teaching.recording import (
    aligned_transition_rows,
    dataset_tick_indices,
    intervention_effect,
    resample_aligned_transitions_10hz,
    transition_alignment,
)


def test_transition_alignment_detects_one_tick_defect_and_preserves_rate_mapping():
    report = transition_alignment([0, 1, 2, 3], [1, 2, 3])
    assert report["aligned"] is False
    assert report["offset_ticks"] == [1]
    np.testing.assert_array_equal(dataset_tick_indices([0, 2, 4]), [0, 1, 2])


def test_aligned_transition_rows_pair_pre_action_observation_and_action():
    rows = aligned_transition_rows(
        [{"state": 0}, {"state": 1}, {"state": 2}],
        [[1], [2]],
        [0, 1, 2],
        [0, 1],
    )
    assert rows[1]["observation"] == {"state": 1}
    np.testing.assert_array_equal(rows[1]["action"], [2])


def test_resample_aligned_control_rows_to_dataset_rate():
    rows = [{"tick": tick, "observation": tick, "action": [tick]} for tick in range(5)]
    sampled = resample_aligned_transitions_10hz(rows)
    assert [row["tick"] for row in sampled] == [0, 2, 4]
    assert [row["action"] for row in sampled] == [[0], [2], [4]]


def test_resample_rejects_noncontiguous_or_wrong_rate_rows():
    with pytest.raises(ValueError, match="contiguous"):
        resample_aligned_transitions_10hz(
            [{"tick": 0}, {"tick": 2}],
        )
    with pytest.raises(ValueError, match="divisor"):
        resample_aligned_transitions_10hz([], control_hz=20, dataset_hz=7)


def test_dataset_tick_mapping_rejects_odd_controller_tick():
    with pytest.raises(ValueError, match="every second"):
        dataset_tick_indices([0, 1, 2])


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
