"""Check the keyboard interface without a simulator or display."""

from pathlib import Path

import numpy as np
import pytest

from vla_lab.data.keyboard import (
    KeyboardEvent,
    run_keyboard_recording,
    validate_development_state,
)
from vla_lab.data.recording import KeyboardActionMapper, RecordedEpisode


class FakeEnvironment:
    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []
        self.closed = False

    def reset(self, seed=None):
        self.seed = seed
        return {
            "state": np.array([0], dtype=np.float32),
            "pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)},
        }, {}

    def step(self, action):
        self.applied.append(np.array(action, copy=True))
        return (
            {
                "state": np.array([len(self.applied)], dtype=np.float32),
                "pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)},
            },
            0.0,
            False,
            False,
            {},
        )

    def close(self):
        self.closed = True


class FakeInterface:
    def __init__(self, events: list[KeyboardEvent]) -> None:
        self.events = iter(events)
        self.closed = False

    def tick(self, _frame: np.ndarray, _delay_ms: int) -> KeyboardEvent:
        return next(self.events)

    def close(self) -> None:
        self.closed = True


def test_mapper_combines_and_bounds_keys() -> None:
    action = KeyboardActionMapper().action_for({"w", "s", "d", "close"})

    assert action.shape == (7,)
    assert action.tolist() == pytest.approx([0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert KeyboardActionMapper().action_for({"open"})[6] == -1.0
    assert np.all((-1.0 <= action) & (action <= 1.0))


def test_save_event_stops_and_writes_recorded_data(tmp_path: Path) -> None:
    env = FakeEnvironment()
    interface = FakeInterface([KeyboardEvent(keys=frozenset({"w"})), KeyboardEvent(save=True)])

    result = run_keyboard_recording(
        env,
        interface,
        configs_root=Path(__file__).resolve().parents[1] / "configs",
        task_alias="bowl_plate",
        state_id=0,
        output_path=tmp_path / "episode.npz",
        episode_id="keyboard-test",
    )

    assert result.path == tmp_path / "episode.npz"
    assert env.closed and interface.closed
    assert len(env.applied) == 2
    assert env.applied[0].tolist() == pytest.approx([0.25, 0, 0, 0, 0, 0, 0])
    assert env.applied[1].tolist() == pytest.approx([0, 0, 0, 0, 0, 0, 0])
    loaded = RecordedEpisode.load(result.path)
    assert loaded.actions.shape == (2, 7)
    assert loaded.observation_ticks.tolist() == [0, 1, 2]
    assert loaded.task_id == "bowl_plate"
    assert loaded.reset_seed == 20260930
    assert env.seed == 20260930


def test_quit_event_stops_without_saving(tmp_path: Path) -> None:
    env = FakeEnvironment()
    interface = FakeInterface([KeyboardEvent(quit=True)])

    result = run_keyboard_recording(
        env,
        interface,
        configs_root=Path(__file__).resolve().parents[1] / "configs",
        task_alias="bowl_plate",
        state_id=0,
        output_path=tmp_path / "discarded.npz",
        episode_id="quit-test",
    )

    assert result.path is None
    assert env.closed and interface.closed
    assert len(env.applied) == 1
    assert not (tmp_path / "discarded.npz").exists()


def test_state_validation_only_accepts_development_ids() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"

    assert validate_development_state("bowl_plate", 0, configs) == 8
    with pytest.raises(ValueError, match="development"):
        validate_development_state("bowl_plate", 20, configs)
