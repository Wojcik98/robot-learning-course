"""Check keyboard recording, saved replay data, and timing alignment."""

import numpy as np
import pytest

from vla_lab.data.recording import (
    CONTROL_PERIOD_MS,
    KeyboardActionMapper,
    apply_synthetic_timing_defect,
    record_action_episode,
    record_episode,
    repair_synthetic_timing_defect,
    replay_episode,
)


class FakeEnvironment:
    """Small deterministic environment for recording tests."""

    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []

    def reset(self, seed=None):
        self.seed = seed
        return {"state": np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        self.applied.append(np.array(action, copy=True))
        value = float(len(self.applied))
        return {"state": np.array([value], dtype=np.float32)}, 0.0, False, False, {}


def test_keyboard_actions_are_bounded_and_combine() -> None:
    mapper = KeyboardActionMapper()

    action = mapper.action_for({"w", "d", "open"})

    assert action.shape == (7,)
    assert action.tolist() == pytest.approx([0.25, 0.25, 0.0, 0.0, 0.0, 0.0, -1.0])
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert np.all(mapper.action_for({"w", "s", "close", "open"}) <= 1.0)


def test_recording_pairs_applied_action_with_observation_tick() -> None:
    env = FakeEnvironment()
    key_events = iter(({"w"}, {"d"}))

    episode = record_episode(
        env,
        lambda _tick, _observation: next(key_events),
        task_id="bowl_plate",
        max_steps=2,
        episode_id="test-episode",
    )

    assert CONTROL_PERIOD_MS == 50
    assert episode.observation_ticks.tolist() == [0, 1, 2]
    assert episode.action_ticks.tolist() == [0, 1]
    assert np.allclose(episode.actions[:, :2], [[0.25, 0.0], [0.0, 0.25]])
    assert episode.observations[0]["state"].tolist() == [0.0]
    assert episode.observations[1]["state"].tolist() == [1.0]
    assert np.array_equal(env.applied[0], episode.actions[0])


def test_action_recorder_saves_the_clipped_action_that_reached_the_controller() -> None:
    env = FakeEnvironment()
    episode = record_action_episode(
        env,
        lambda tick, _observation: (
            np.array([1.5, -0.5, 0, 0, 0, 0, -2]) if tick == 0 else np.zeros(7)
        ),
        task_id="bowl_plate",
        max_steps=2,
        episode_id="teacher-episode",
        reset_seed=17,
    )

    assert episode.action_ticks.tolist() == [0, 1]
    assert episode.observation_ticks.tolist() == [0, 1, 2]
    assert episode.actions[0].tolist() == pytest.approx([1, -0.5, 0, 0, 0, 0, -1])
    assert np.array_equal(env.applied[0], episode.actions[0])


def test_saved_episode_can_be_loaded_and_replayed(tmp_path) -> None:
    env = FakeEnvironment()
    episode = record_episode(
        env,
        lambda tick, _observation: {"w"} if tick == 0 else set(),
        task_id="bowl_plate",
        max_steps=2,
        episode_id="saved-episode",
        reset_seed=17,
    )
    path = tmp_path / "episode.npz"
    episode.save(path)
    loaded = type(episode).load(path)

    replay_env = FakeEnvironment()
    replayed = replay_episode(replay_env, loaded)

    assert len(replayed) == 3
    assert replay_env.seed == 17
    assert np.array_equal(replay_env.applied[0], loaded.actions[0])
    assert replayed[-1]["state"].tolist() == [2.0]


def test_synthetic_tick_offset_has_label_and_repairs_alignment() -> None:
    env = FakeEnvironment()
    episode = record_episode(
        env,
        lambda _tick, _observation: {"w"},
        task_id="bowl_plate",
        max_steps=2,
        episode_id="timing-episode",
    )

    defective = apply_synthetic_timing_defect(episode, offset_ticks=1)
    repaired = repair_synthetic_timing_defect(defective)

    assert defective.defect["label"] == "synthetic timing defect"
    assert defective.action_ticks.tolist() == [1, 2]
    assert repaired.defect is None
    assert repaired.action_ticks.tolist() == repaired.observation_ticks[:-1].tolist()
    assert np.array_equal(repaired.actions, episode.actions)
