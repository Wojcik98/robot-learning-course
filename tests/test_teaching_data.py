import numpy as np
import pytest

from vla_lab.teaching.data import (
    action_window,
    build_act_batches,
    extract_episode_arrays,
    fit_training_statistics,
    order_episode_rows,
    split_episode_ids,
)


def test_order_rows_and_reject_duplicate_frame():
    rows = [{"episode_index": 2, "frame_index": 0}, {"episode_index": 1, "frame_index": 1}]
    assert [row["episode_index"] for row in order_episode_rows(rows)] == [1, 2]
    with pytest.raises(ValueError, match="unique"):
        order_episode_rows([rows[0], rows[0]])


def test_extract_episode_arrays_reads_ordered_raw_action_and_state_columns():
    rows = [
        {
            "episode_index": 3,
            "frame_index": 1,
            "action": np.ones(7),
            "observation.state": np.ones(8),
        },
        {
            "episode_index": 3,
            "frame_index": 0,
            "action": np.zeros(7),
            "observation.state": np.zeros(8),
        },
    ]
    extracted = extract_episode_arrays(rows)
    np.testing.assert_array_equal(extracted["frame_indices"], [0, 1])
    np.testing.assert_array_equal(extracted["actions"][0], np.zeros(7))
    np.testing.assert_array_equal(extracted["states"][1], np.ones(8))


def test_action_window_never_crosses_episode_end_and_marks_padding():
    window, mask = action_window([[1, 2], [3, 4]], start=1, horizon=4)
    np.testing.assert_array_equal(window, [[3, 4], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(mask, [False, True, True, True])


def test_split_keeps_complete_ids_and_statistics_ignore_validation():
    train_ids, validation_ids = split_episode_ids([0, 1, 2, 3, 4], validation_fraction=0.2)
    assert train_ids == [0, 1, 2, 3] and validation_ids == [4]
    stats = fit_training_statistics({0: [[0.0], [2.0]]}, validation_episodes={9: [[1000.0]]})
    np.testing.assert_allclose(stats.mean, [1.0])
    np.testing.assert_allclose(stats.std, [1.0])


def test_student_window_and_stats_build_the_batches_used_by_training():
    image = np.zeros((3, 2, 2), dtype=np.float32)
    training = {
        0: {
            "actions": np.arange(21 * 7).reshape(21, 7),
            "states": np.arange(21 * 8).reshape(21, 8),
            "samples": [
                {"start": 20, "observation.images.image": image, "observation.images.image2": image}
            ],
        }
    }
    validation = {
        1: {
            "actions": np.ones((2, 7)),
            "states": np.ones((2, 8)),
            "samples": [
                {"start": 1, "observation.images.image": image, "observation.images.image2": image}
            ],
        }
    }
    built = build_act_batches(
        training,
        validation,
        horizon=4,
        batch_size=1,
        window_fn=action_window,
        stats_fn=fit_training_statistics,
    )
    batch = built["train"][0]
    assert tuple(batch["action"].shape) == (1, 4, 7)
    assert batch["action_is_pad"].tolist() == [[False, True, True, True]]
    assert batch["observation.state"].shape == (1, 8)
    assert batch["observation.images.image"].shape == (1, 3, 2, 2)
