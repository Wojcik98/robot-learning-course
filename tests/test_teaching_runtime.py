from __future__ import annotations

import numpy as np
import pytest
import torch

from vla_lab.teaching.runtime import (
    reset_environment,
    run_chunked_episode,
    select_chunk_action,
)


class FakeEnvironment:
    def __init__(self, *, stop_at: int = 4, stop: str = "timeout"):
        self.stop_at = stop_at
        self.stop = stop
        self.steps = 0
        self.reset_seed = None
        self.actions = []

    def reset(self, seed=None):
        self.reset_seed = seed
        self.steps = 0
        return {"step": 0}, {"seed": seed}

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        terminated = self.stop == "terminated" and self.steps == self.stop_at
        truncated = self.stop == "truncated" and self.steps == self.stop_at
        success = self.stop == "success" and self.steps == self.stop_at
        return {"step": self.steps}, 0.0, terminated, truncated, {"is_success": success}


class FakePolicy:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def test_reset_environment_resets_both_environment_and_policy():
    environment = FakeEnvironment()
    policy = FakePolicy()
    observation = reset_environment(environment, policy, seed=9)
    assert observation == {"step": 0}
    assert environment.reset_seed == 9
    assert policy.reset_count == 1


def test_select_chunk_action_checks_batch_and_offset():
    chunk = torch.arange(21).reshape(1, 3, 7)
    assert torch.equal(select_chunk_action(chunk, 1), chunk[0, 1])
    with pytest.raises(IndexError, match="outside"):
        select_chunk_action(chunk, 3)
    with pytest.raises(ValueError, match="shape"):
        select_chunk_action(torch.zeros(2, 3, 7), 0)


@pytest.mark.parametrize(
    ("stop", "reason"),
    [("success", "success"), ("terminated", "terminated"), ("truncated", "truncated")],
)
def test_rollout_stops_on_each_environment_end_condition(stop, reason):
    environment = FakeEnvironment(stop_at=2, stop=stop)
    policy = FakePolicy()
    calls = []

    def predict(_policy, batch):
        calls.append(batch["step"])
        return torch.ones((1, 4, 7)) * 0.25

    result = run_chunked_episode(
        environment,
        policy,
        lambda observation: observation,
        predict,
        lambda action: action,
        max_steps=8,
        execute_steps=3,
        seed=22,
    )
    assert result.termination_reason == reason
    assert result.steps == 2
    assert result.policy_calls == 1
    assert result.success is (stop == "success")
    assert policy.reset_count == 1
    assert environment.reset_seed == 22
    assert len(environment.actions) == 2
    assert calls == [0]


def test_rollout_replans_after_execute_count_and_clips_actions():
    environment = FakeEnvironment(stop_at=4)
    policy = FakePolicy()
    prediction_inputs = []

    def predict(_policy, observation):
        prediction_inputs.append(observation["step"])
        return np.full((1, 3, 7), 1.5, dtype=np.float32)

    result = run_chunked_episode(
        environment,
        policy,
        lambda observation: observation,
        predict,
        lambda action: action,
        max_steps=4,
        execute_steps=2,
    )
    assert prediction_inputs == [0, 2]
    assert result.policy_calls == 2
    assert result.steps == 4
    assert result.clipped_values == 28
    assert all(np.all(action == 1.0) for action in environment.actions)


def test_rollout_rejects_bad_predictions_and_non_finite_actions():
    environment = FakeEnvironment()
    with pytest.raises(ValueError, match=r"\[1, time, 7\]"):
        run_chunked_episode(
            environment,
            FakePolicy(),
            lambda observation: observation,
            lambda _policy, _batch: torch.zeros((1, 3, 6)),
            lambda action: action,
            max_steps=2,
            execute_steps=1,
        )

    with pytest.raises(ValueError, match="seven finite"):
        run_chunked_episode(
            FakeEnvironment(),
            FakePolicy(),
            lambda observation: observation,
            lambda _policy, _batch: torch.zeros((1, 3, 7)),
            lambda action: torch.full_like(action, float("nan")),
            max_steps=2,
            execute_steps=1,
        )
