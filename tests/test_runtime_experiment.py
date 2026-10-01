"""Check the virtual-tick episode runner with a CPU fake environment."""

import numpy as np
import pytest

from vla_lab.runtime.experiment import EpisodeConfig, EpisodeRunner, run_episode

HORIZON = 8


class FakeEnv:
    """Count steps, record every action, and report success at a chosen step."""

    def __init__(self, success_at: int | None = None) -> None:
        self.success_at = success_at
        self.seeds: list[int] = []
        self.actions: list[np.ndarray] = []

    def reset(self, seed: int) -> tuple[int, dict]:
        self.seeds.append(seed)
        self.actions = []
        return 0, {}

    def step(self, action: np.ndarray) -> tuple[int, float, bool, bool, dict]:
        self.actions.append(action.copy())
        count = len(self.actions)
        success = self.success_at is not None and count >= self.success_at
        return count, float(success), success, False, {"is_success": success}


class Predictor:
    """Encode request tick and chunk index in each action value."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, np.ndarray | None, int | None]] = []

    def __call__(self, observation, observation_tick, previous, origin):
        self.calls.append((observation, observation_tick, previous, origin))
        index = np.arange(HORIZON, dtype=np.float32)
        normalized = np.zeros((HORIZON, 7), dtype=np.float32)
        normalized[:, 0] = (observation_tick * 10 + index) / 1000
        return normalized, normalized * 0.5


def config(mode: str = "naive_async", **kwargs) -> EpisodeConfig:
    values = {
        "mode": mode,
        "horizon": HORIZON,
        "stride": 4,
        "delay": 2,
        "max_steps": 12,
        "seed": 7,
    }
    values.update(kwargs)
    return EpisodeConfig(**values)


def test_delay_uses_chunk_index_tick_minus_request() -> None:
    env = FakeEnv()
    result = run_episode(env, Predictor(), config())

    xs = [row.action[0] for row in result.ticks]
    # Ticks 0 and 1 wait for the first reply; tick 2 uses index 2 of request 0.
    assert xs[:2] == [0, 0]
    assert [row.underflow for row in result.ticks[:3]] == [True, True, False]
    assert xs[2] == pytest.approx(0.5 * 0.002)
    assert xs[5] == pytest.approx(0.5 * 0.005)
    assert result.ticks[2].observation_age_ticks == 2


def test_new_chunk_replaces_old_one_at_its_arrival_tick() -> None:
    result = run_episode(FakeEnv(), Predictor(), config())

    xs = [row.action[0] for row in result.ticks]
    assert xs[6] == pytest.approx(0.5 * (4 * 10 + 2) / 1000)
    assert result.ticks[6].accepted == ("request-4",)


def test_modes_choose_delay_ensemble_and_previous_chunk() -> None:
    sync = run_episode(FakeEnv(), Predictor(), config("synchronous"))
    assert sync.ticks[0].underflow is False
    assert sync.ticks[0].action[0] == pytest.approx(0.0)

    naive = run_episode(FakeEnv(), Predictor(), config("naive_async"))
    ensemble = run_episode(FakeEnv(), Predictor(), config("temporal_ensemble"))
    assert naive.ticks[6].action[0] != pytest.approx(ensemble.ticks[6].action[0])

    for mode, expects_previous in [
        ("synchronous", False),
        ("naive_async", False),
        ("temporal_ensemble", False),
        ("rtc_hard", True),
        ("rtc_soft", True),
    ]:
        predictor = Predictor()
        run_episode(FakeEnv(), predictor, config(mode))
        assert predictor.calls[0][2:] == (None, None)
        assert (predictor.calls[1][2] is not None) is expects_previous


def test_rtc_previous_chunk_starts_at_current_request_tick() -> None:
    predictor = Predictor()
    run_episode(FakeEnv(), predictor, config("rtc_hard"))

    _, tick, previous, origin = predictor.calls[1]
    assert (tick, origin) == (4, 0)
    assert previous.shape == (HORIZON - 4, 7)
    # Request 0 index 4 is the first action aligned to tick 4.
    assert previous[:4, 0] == pytest.approx([(4 + i) / 1000 for i in range(4)])


def test_rtc_rejects_bad_delay_and_stride() -> None:
    with pytest.raises(ValueError):
        config("rtc_hard", delay=5, stride=4)
    with pytest.raises(ValueError):
        config("rtc_soft", delay=2, stride=7)
    config("rtc_soft", delay=2, stride=6)
    with pytest.raises(ValueError):
        config(max_steps=0)


def test_predictor_shape_is_checked() -> None:
    def bad(observation, observation_tick, previous, origin):
        return np.zeros((3, 7), np.float32), np.zeros((3, 7), np.float32)

    with pytest.raises(ValueError):
        run_episode(FakeEnv(), bad, config())


def test_success_stops_episode_and_is_recorded() -> None:
    result = run_episode(FakeEnv(success_at=5), Predictor(), config())

    assert result.success is True
    assert len(result.ticks) == 5
    assert result.ticks[-1].step_success is True
    assert all(not row.step_success for row in result.ticks[:-1])


def test_no_success_runs_to_max_steps() -> None:
    result = run_episode(FakeEnv(), Predictor(), config())

    assert result.success is False
    assert len(result.ticks) == 12


def test_records_queue_depth_and_boundary_discontinuity() -> None:
    result = run_episode(FakeEnv(), Predictor(), config())

    assert result.ticks[0].queue_depth == 0
    assert result.ticks[2].queue_depth == 1
    assert [boundary.request_id for boundary in result.boundaries] == [
        "request-0",
        "request-4",
        "request-8",
    ]
    first = result.boundaries[1]
    assert first.start_tick == 6
    assert len(first.values) == 6
    # Tick 5 uses request 0 index 5; tick 6 uses request 4 index 2.
    assert first.values[0] == pytest.approx(0.5 * abs((42 - 5) / 1000))


def test_each_run_resets_env_and_drops_old_chunks() -> None:
    env = FakeEnv()
    runner = EpisodeRunner(config(max_steps=3))
    first = runner.run(env, Predictor())
    second = runner.run(env, Predictor())

    assert env.seeds == [7, 7]
    assert second.reset_counter == first.reset_counter + 1
    assert [row.underflow for row in second.ticks] == [True, True, False]
    assert second.ticks[0].action[0] == 0


def test_termination_without_success_is_not_counted_as_success() -> None:
    class TerminatedEnv(FakeEnv):
        def step(self, action):
            observation, reward, _, _, _ = super().step(action)
            return observation, reward, len(self.actions) == 3, False, {"is_success": False}

    result = run_episode(TerminatedEnv(), Predictor(), config())

    assert result.success is False
    assert len(result.ticks) == 3
