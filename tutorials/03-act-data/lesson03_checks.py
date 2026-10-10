"""Behaviour checks for the Lesson 3 exercises.

Each ``check_*`` function receives your implementation and compares its output with
values computed independently here, on small episodes whose numbers are easy to follow.
Read a failing assertion message first: it states the expected behaviour.
"""

from types import SimpleNamespace

import numpy as np
import torch

from vla_lab.teaching.records import CameraFrame, FeatureKey, TrainingEpisode

STATE_DIM, ACTION_DIM = 8, 7


def _done(message: str) -> None:
    print(f"✓ {message}")


def random_episode(
    length: int, seed: int, *, image_frames: tuple[int, ...] = (), offset: float = 0.0
) -> TrainingEpisode:
    """An episode with distinct values in every feature column and frame-specific images.

    Image values encode the frame: ``image`` is filled with ``frame / 100`` and ``image2``
    with ``0.5 + frame / 100``, so a check can tell which frame and which camera was read.
    """
    rng = np.random.default_rng(seed)
    states = (rng.normal(size=(length, STATE_DIM)) * np.arange(1, STATE_DIM + 1) + offset).astype(
        np.float32
    )
    actions = (rng.normal(size=(length, ACTION_DIM)) * 0.3 - np.arange(ACTION_DIM) + offset).astype(
        np.float32
    )
    images = {
        frame: CameraFrame(
            image=np.full((3, 4, 5), frame / 100, dtype=np.float32),
            image2=np.full((3, 4, 5), 0.5 + frame / 100, dtype=np.float32),
        )
        for frame in image_frames
    }
    return TrainingEpisode(actions=actions, states=states, images=images)


def _expect_error(call, errors: tuple[type[Exception], ...], message: str) -> None:
    try:
        call()
    except errors:
        return
    except Exception as error:  # noqa: BLE001 - report the unexpected type
        raise AssertionError(f"{message}; got {type(error).__name__}: {error}") from error
    raise AssertionError(message)


def check_fit_feature_stats(fit_feature_stats) -> None:
    """Exercise 3.1: per-feature population statistics over training frames only."""
    episodes = {
        3: random_episode(5, seed=0),
        7: random_episode(2, seed=1, offset=4.0),
        9: random_episode(4, seed=2, offset=-50.0),  # validation: must be ignored
    }
    stats = fit_feature_stats(episodes, [3, 7])
    for name, dim in (("states", STATE_DIM), ("actions", ACTION_DIM)):
        rows = np.concatenate([getattr(episodes[i], name) for i in (3, 7)]).astype(np.float64)
        fitted = getattr(stats, name)
        assert np.shape(fitted.mean) == (dim,) and np.shape(fitted.std) == (dim,), (
            f"{name}: mean and std must have shape ({dim},)"
        )
        np.testing.assert_allclose(
            fitted.mean,
            rows.mean(axis=0),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{name}: mean must be computed per feature over all training frames",
        )
        np.testing.assert_allclose(
            fitted.std,
            rows.std(axis=0, ddof=0),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{name}: std must be the population (ddof=0) std per feature",
        )
    _done("per-feature population mean and std, states and actions fitted separately")

    # Frames are pooled: a long episode weighs more than a short one.
    pooled = {
        0: TrainingEpisode(
            actions=np.zeros((4, ACTION_DIM), np.float32),
            states=np.zeros((4, STATE_DIM), np.float32),
            images={},
        ),
        1: TrainingEpisode(
            actions=np.full((1, ACTION_DIM), 10, np.float32),
            states=np.full((1, STATE_DIM), 10, np.float32),
            images={},
        ),
    }
    stats = fit_feature_stats(pooled, [0, 1])
    np.testing.assert_allclose(
        stats.actions.mean,
        2.0,
        atol=1e-6,
        err_msg="pool all frames: mean of [0, 0, 0, 0, 10] is 2, not the mean of episode means (5)",
    )
    np.testing.assert_allclose(stats.actions.std, 4.0, atol=1e-5)
    _done("frames from all training episodes are pooled before reducing")

    # Order of training IDs and other episodes in the mapping do not matter.
    baseline = fit_feature_stats(episodes, [3, 7])
    reordered = fit_feature_stats(episodes, [7, 3])
    changed = fit_feature_stats({**episodes, 9: random_episode(4, seed=99, offset=1e6)}, [3, 7])
    for other in (reordered, changed):
        for name in ("states", "actions"):
            np.testing.assert_allclose(
                getattr(other, name).mean,
                getattr(baseline, name).mean,
                rtol=1e-5,
                err_msg="statistics must depend only on the listed training episodes",
            )
    _done("validation episodes and train_ids order do not change the statistics")

    # Only constant features get scale 1; other features keep their real std.
    states = np.tile(np.arange(STATE_DIM, dtype=np.float32), (3, 1))
    states[:, 2] = [1.0, 2.0, 3.0]
    actions = np.zeros((3, ACTION_DIM), np.float32)
    actions[:, 0] = [0.0, 0.0, 6.0]
    stats = fit_feature_stats({0: TrainingEpisode(actions=actions, states=states, images={})}, [0])
    expected_state_std = np.ones(STATE_DIM)
    expected_state_std[2] = np.std([1.0, 2.0, 3.0])
    expected_action_std = np.ones(ACTION_DIM)
    expected_action_std[0] = np.std([0.0, 0.0, 6.0])
    np.testing.assert_allclose(
        stats.states.std,
        expected_state_std,
        rtol=1e-5,
        err_msg="replace the std only of constant features with 1",
    )
    np.testing.assert_allclose(stats.actions.std, expected_action_std, rtol=1e-5)
    np.testing.assert_allclose(stats.states.mean, states.mean(axis=0), rtol=1e-6)
    _done("constant features get scale 1; the others keep their std")

    # Invalid input is rejected.
    nan_episode = random_episode(3, seed=4)
    nan_episode.actions[1, 2] = np.nan
    _expect_error(
        lambda: fit_feature_stats(episodes, []),
        (ValueError,),
        "empty train_ids must raise ValueError",
    )
    _expect_error(
        lambda: fit_feature_stats(episodes, [3, 3]),
        (ValueError,),
        "duplicate train_ids must raise ValueError",
    )
    _expect_error(
        lambda: fit_feature_stats({0: nan_episode}, [0]),
        (ValueError,),
        "a NaN in a training episode must raise ValueError",
    )
    _done("empty, duplicate, or nonfinite input raises ValueError")


def _stats(episodes: dict[int, TrainingEpisode], ids: list[int]) -> SimpleNamespace:
    """Independent statistics, with deliberately different values for every feature."""
    states = np.concatenate([episodes[i].states for i in ids]).astype(np.float64)
    actions = np.concatenate([episodes[i].actions for i in ids]).astype(np.float64)
    return SimpleNamespace(
        states=SimpleNamespace(mean=states.mean(0), std=states.std(0)),
        actions=SimpleNamespace(mean=actions.mean(0), std=actions.std(0)),
    )


def check_build_training_batch(build_training_batch) -> None:
    """Exercise 3.2: cameras at the start, normalized state, episode-bounded padded targets."""
    episodes = {
        4: random_episode(10, seed=10, image_frames=(0, 3, 9)),
        8: random_episode(6, seed=11, image_frames=(0, 2, 5), offset=3.0),
    }
    stats = _stats(episodes, [4, 8])
    starts = [(4, 3), (8, 5), (4, 0), (8, 2)]
    horizon = 5
    try:
        batch = build_training_batch(episodes, starts, stats, horizon=horizon)
    except Exception as error:
        raise AssertionError(
            f"build_training_batch failed with {type(error).__name__}: {error}. Starts {starts} "
            f"with horizon={horizon} include windows that end before the episode does, so only "
            "the first `horizon` action rows may be used, and windows that run past the end."
        ) from error

    expected_keys = {
        FeatureKey.IMAGE,
        FeatureKey.IMAGE2,
        FeatureKey.STATE,
        FeatureKey.ACTION,
        FeatureKey.ACTION_IS_PAD,
    }
    assert set(batch) == expected_keys, (
        f"batch keys must be exactly {sorted(map(str, expected_keys))}"
    )
    for key in (FeatureKey.IMAGE, FeatureKey.IMAGE2, FeatureKey.STATE, FeatureKey.ACTION):
        assert batch[key].dtype == torch.float32, f"{key} must be float32, got {batch[key].dtype}"
    assert batch[FeatureKey.ACTION_IS_PAD].dtype == torch.bool, "action_is_pad must be bool"

    for row, (episode_id, start) in enumerate(starts):
        episode = episodes[episode_id]
        label = f"example {row} (episode {episode_id}, start {start})"
        torch.testing.assert_close(
            batch[FeatureKey.IMAGE][row],
            torch.full((3, 4, 5), start / 100),
            msg=f"{label}: IMAGE must be CameraFrame.image at the start frame",
        )
        torch.testing.assert_close(
            batch[FeatureKey.IMAGE2][row],
            torch.full((3, 4, 5), 0.5 + start / 100),
            msg=f"{label}: IMAGE2 must be CameraFrame.image2 at the start frame",
        )
        expected_state = (episode.states[start] - stats.states.mean) / stats.states.std
        np.testing.assert_allclose(
            batch[FeatureKey.STATE][row].numpy(),
            expected_state,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"{label}: state must be (x - state mean) / state std, per feature",
        )
        count = min(horizon, episode.frame_count - start)
        expected_target = np.zeros((horizon, ACTION_DIM))
        expected_target[:count] = (
            episode.actions[start : start + count] - stats.actions.mean
        ) / stats.actions.std
        np.testing.assert_allclose(
            batch[FeatureKey.ACTION][row].numpy(),
            expected_target,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"{label}: target slot j must hold the normalized action at frame start + j, "
            "and zero after the episode end",
        )
        expected_mask = [j >= count for j in range(horizon)]
        assert batch[FeatureKey.ACTION_IS_PAD][row].tolist() == expected_mask, (
            f"{label}: action_is_pad must be {expected_mask}"
        )
    _done("each example reads its own start frame, cameras, and normalized state")
    _done("targets take exactly `horizon` rows when available and pad only past the episode end")

    # A window never reads from another episode, even an adjacent one in the mapping.
    adjacent = {
        8: random_episode(6, seed=11, image_frames=(4,), offset=3.0),
        9: random_episode(6, seed=12, image_frames=(0,)),
    }
    tail = build_training_batch(adjacent, [(8, 4)], stats, horizon=4)
    np.testing.assert_array_equal(
        tail[FeatureKey.ACTION][0, 2:].numpy(),
        0.0,
        err_msg="slots after the episode end must be zero, never actions from another episode",
    )
    assert tail[FeatureKey.ACTION_IS_PAD][0].tolist() == [False, False, True, True]
    _done("windows stay inside their own episode")

    # Horizon 1 keeps just the current action.
    one = build_training_batch(episodes, [(4, 9)], stats, horizon=1)
    assert tuple(one[FeatureKey.ACTION].shape) == (1, 1, ACTION_DIM)
    assert one[FeatureKey.ACTION_IS_PAD].tolist() == [[False]]
    _done("horizon=1 gives one unpadded target slot")

    # Invalid starts must fail clearly rather than read the wrong frame.
    _expect_error(
        lambda: build_training_batch(episodes, [(8, 6)], stats, horizon=3),
        (ValueError,),
        "a start equal to the episode length must raise ValueError",
    )
    _expect_error(
        lambda: build_training_batch(episodes, [(8, -1)], stats, horizon=3),
        (ValueError,),
        "a negative start must raise ValueError (a negative index would silently read the last frame)",
    )
    _expect_error(
        lambda: build_training_batch(episodes, [(8, 1)], stats, horizon=3),
        (ValueError, KeyError),
        "a start without decoded images must raise an error",
    )
    _done("starts outside the episode or without decoded images are rejected")

    # Inputs are not modified.
    before = episodes[4].actions.copy()
    build_training_batch(episodes, [(4, 0)], stats, horizon=3)
    np.testing.assert_array_equal(
        episodes[4].actions, before, err_msg="do not modify the episode arrays"
    )
    _done("episode arrays are left unchanged")
