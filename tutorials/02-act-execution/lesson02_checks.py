"""Behaviour checks for the Lesson 2 exercises.

Each ``check_*`` function receives your implementation and runs it against small,
deterministic fakes. The fakes record what your function sent to the environment and
the policy, so these checks test the control-loop logic, not only result shapes.
Read a failing assertion message first: it states the expected behaviour.
"""

from typing import Any

import numpy as np
import torch

ACTION_DIM = 7


class RecordingEnvironment:
    """Fake environment that records every call and ends on a chosen step.

    ``end_after`` is the 1-based step on which the episode ends; ``end_kind`` is one of
    ``"success"``, ``"terminated"``, ``"truncated"``, ``"success+terminated"`` or
    ``None`` (never ends). Observations are ``{"step": n}`` dictionaries, so a check
    can tell which observation a function used.
    """

    def __init__(self, end_after: int | None = None, end_kind: str | None = None) -> None:
        self.end_after, self.end_kind = end_after, end_kind
        self.steps = 0
        self.actions: list[np.ndarray] = []
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        self.reset_seeds.append(seed)
        self.steps, self.actions = 0, []
        return {"step": 0}, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        if self.end_after is not None and self.steps >= self.end_after:
            raise AssertionError(
                f"environment.step was called after the episode ended at step {self.end_after}"
            )
        self.actions.append(np.array(action, copy=True))
        self.steps += 1
        ended = self.end_after is not None and self.steps >= self.end_after
        kind = self.end_kind or ""
        info: dict[str, Any] = {"is_success": ended and "success" in kind}
        return (
            {"step": self.steps},
            0.5 * self.steps,
            ended and ("terminated" in kind),
            ended and kind == "truncated",
            info,
        )


class RecordingPolicy:
    """Fake policy that returns a fixed chunk and records every batch it receives."""

    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk
        self.batches: list[dict[str, Any]] = []
        self.grad_enabled: list[bool] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        self.batches.append(batch)
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.chunk.clone()


def numbered_chunk(length: int, scale: float = 0.01) -> torch.Tensor:
    """A (1, T, 7) chunk whose row t is filled with ``t * scale`` (row 0 is all zeros).

    Distinct rows let a check see which chunk rows were sent, and in which order.
    """
    rows = torch.arange(length, dtype=torch.float32)[:, None].repeat(1, ACTION_DIM) * scale
    rows += torch.arange(ACTION_DIM, dtype=torch.float32) * scale / 10
    return rows[None]


def call_predict(policy: RecordingPolicy, batch: dict[str, Any]) -> torch.Tensor:
    return policy.predict_action_chunk(batch)


def identity(action: torch.Tensor) -> torch.Tensor:
    return action


def _as_numpy(action: Any) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _done(message: str) -> None:
    print(f"✓ {message}")


def check_predict_and_step(predict_and_step) -> None:
    """Exercise 1: one observation → one prediction → first action → one environment step."""
    # The policy must see exactly the batch built from the given observation.
    chunk = torch.tensor([[[0.2, -0.4, 0.6, 0.0, 0.1, -0.1, 0.3], [9, 9, 9, 9, 9, 9, 9.0]]])
    policy = RecordingPolicy(chunk)
    environment = RecordingEnvironment(end_after=1, end_kind="terminated")
    prepared: list[Any] = []

    def prepare(observation: Any) -> dict[str, Any]:
        prepared.append(observation)
        return {"from": observation["name"]}

    converted: list[torch.Tensor] = []

    def to_controller(action: torch.Tensor) -> torch.Tensor:
        converted.append(action.detach().clone())
        return action * 2.0  # a stand-in postprocessor: controller = 2 * normalized

    result = predict_and_step(
        environment, policy, {"name": "start"}, prepare, call_predict, to_controller
    )
    assert prepared == [{"name": "start"}], (
        "prepare_observation must be called once, on the given observation"
    )
    assert policy.batches == [{"from": "start"}], "the policy must receive the prepared batch"
    assert policy.grad_enabled == [False], (
        "the policy call must run without building a gradient graph"
    )
    assert len(converted) == 1, "convert only the first action of the chunk"
    torch.testing.assert_close(
        converted[0], chunk[0, 0], msg="the converted action must be chunk[0, 0]"
    )
    assert environment.steps == 1, "environment.step must be called exactly once"
    expected = np.clip(chunk[0, 0].numpy() * 2.0, -1, 1)
    np.testing.assert_allclose(
        environment.actions[0].reshape(-1),
        expected,
        rtol=0,
        atol=1e-6,
        err_msg="send the converted, then clipped, first action (not the raw normalized one)",
    )
    np.testing.assert_allclose(
        _as_numpy(result.action),
        expected,
        atol=1e-6,
        err_msg="result.action must be the sent action",
    )
    assert result.observation == {"step": 1}, (
        "result.observation must be the observation returned by environment.step"
    )
    assert result.reward == 0.5, "keep the environment reward"
    assert bool(result.terminated) is True and bool(result.truncated) is False, (
        "keep the terminated/truncated flags"
    )
    assert bool(result.success) is False, (
        "success must come from info['is_success'], not from terminated"
    )
    torch.testing.assert_close(result.chunk, chunk, msg="result.chunk must be the predicted chunk")
    _done("uses the prepared batch, no gradients, only chunk[0, 0], converted then clipped")

    # Success comes from info; a missing key means no success.
    success_env = RecordingEnvironment(end_after=1, end_kind="success")
    result = predict_and_step(
        success_env, RecordingPolicy(chunk), {"name": "a"}, lambda o: {}, call_predict, identity
    )
    assert bool(result.success) is True and bool(result.terminated) is False
    no_info_env = RecordingEnvironment()
    result = predict_and_step(
        no_info_env, RecordingPolicy(chunk), {"name": "a"}, lambda o: {}, call_predict, identity
    )
    assert (
        bool(result.success) is False
        and bool(result.terminated) is False
        and bool(result.truncated) is False
    )
    _done("success, termination and truncation are reported separately")

    # Values outside [-1, 1] are clipped, values inside are unchanged; dtype is float32.
    wide = torch.tensor([[[1.5, -3.0, 1.0, -1.0, 0.999, -0.25, 7.0]]])
    env = RecordingEnvironment()
    result = predict_and_step(
        env, RecordingPolicy(wide), {"name": "a"}, lambda o: {}, call_predict, identity
    )
    np.testing.assert_array_equal(
        env.actions[0].reshape(-1), np.array([1, -1, 1, -1, 0.999, -0.25, 1], dtype=np.float32)
    )
    assert _as_numpy(result.action).shape == (ACTION_DIM,), "the sent action has shape (7,)"
    assert np.asarray(env.actions[0]).dtype == np.float32, "send a float32 action"
    _done("clipping changes only values outside [-1, 1]")


def check_execute_chunk_prefix(execute_chunk_prefix) -> None:
    """Exercise 2: execute a bounded, ordered prefix and stop for the right reason."""

    def run(
        env: RecordingEnvironment,
        chunk: torch.Tensor,
        *,
        execute_steps: int,
        max_steps: int,
        convert=identity,
    ):
        return execute_chunk_prefix(
            env, {"step": 0}, chunk, convert, execute_steps=execute_steps, max_steps=max_steps
        )

    # The prefix length is the smallest of T, execute_steps and max_steps.
    cases = [
        # (T, execute_steps, max_steps, expected steps, expected stop reason)
        (20, 6, 4, 4, "step_limit"),
        (20, 4, 6, 4, "chunk_end"),
        (3, 10, 10, 3, "chunk_end"),
        (5, 5, 5, 5, "step_limit"),
        (8, 1, 50, 1, "chunk_end"),
    ]
    for length, execute_steps, max_steps, expected_steps, expected_reason in cases:
        env = RecordingEnvironment()
        chunk = numbered_chunk(length)
        result = run(env, chunk, execute_steps=execute_steps, max_steps=max_steps)
        label = f"T={length}, execute_steps={execute_steps}, max_steps={max_steps}"
        assert env.steps == expected_steps, (
            f"{label}: expected {expected_steps} environment steps, got {env.steps}"
        )
        assert result.steps == expected_steps, f"{label}: result.steps must count environment steps"
        assert result.stop_reason == expected_reason, (
            f"{label}: expected stop reason {expected_reason!r}, got {result.stop_reason!r}"
        )
        assert not result.done and not result.success, f"{label}: the episode has not ended"
        sent = np.stack([a.reshape(-1) for a in env.actions])
        np.testing.assert_allclose(
            sent,
            chunk[0, :expected_steps].numpy(),
            atol=1e-6,
            err_msg=f"{label}: send chunk rows 0, 1, 2, ... in order",
        )
        np.testing.assert_allclose(
            np.stack([_as_numpy(a) for a in result.actions]),
            sent,
            atol=1e-6,
            err_msg=f"{label}: result.actions must be the sent actions, in order",
        )
        assert result.observation == {"step": expected_steps}, (
            f"{label}: return the latest observation"
        )
    _done("prefix length = min(T, execute_steps, max_steps), rows sent in order")

    # The environment can end the prefix early; no further step may follow.
    for kind, reason in (
        ("success", "success"),
        ("terminated", "terminated"),
        ("truncated", "truncated"),
        ("success+terminated", "success"),
    ):
        env = RecordingEnvironment(end_after=2, end_kind=kind)
        result = run(env, numbered_chunk(10), execute_steps=6, max_steps=6)
        assert env.steps == 2, f"{kind}: stop right after the step that ended the episode"
        assert result.steps == 2 and bool(result.done) is True, (
            f"{kind}: report 2 steps and done=True"
        )
        assert result.stop_reason == reason, (
            f"{kind}: expected stop reason {reason!r}, got {result.stop_reason!r}"
        )
        assert bool(result.success) is ("success" in kind), f"{kind}: success flag"
        assert bool(result.terminated) is ("terminated" in kind), f"{kind}: terminated flag"
        assert bool(result.truncated) is (kind == "truncated"), f"{kind}: truncated flag"
        assert result.observation == {"step": 2}
    _done("stops at success, termination or truncation; success wins over termination")

    # An episode ending on the last allowed step reports the environment reason.
    env = RecordingEnvironment(end_after=3, end_kind="success")
    result = run(env, numbered_chunk(10), execute_steps=3, max_steps=3)
    assert result.stop_reason == "success" and bool(result.done) is True
    _done("an environment end on the last budgeted step is not a step-limit stop")

    # Conversion happens per row; clipping is counted per value.
    def to_controller(action: torch.Tensor) -> torch.Tensor:
        return action * 10.0 - 0.5

    env = RecordingEnvironment()
    chunk = numbered_chunk(4, scale=0.1)
    result = run(env, chunk, execute_steps=4, max_steps=10, convert=to_controller)
    proposed = chunk[0].numpy() * 10.0 - 0.5
    expected = np.clip(proposed, -1, 1)
    sent = np.stack([a.reshape(-1) for a in env.actions])
    np.testing.assert_allclose(sent, expected, atol=1e-5, err_msg="convert each row, then clip it")
    expected_clipped = int(np.count_nonzero(np.abs(proposed - expected) > 0))
    assert result.clipped_values == expected_clipped, (
        f"clipped_values counts changed values: expected {expected_clipped}, got {result.clipped_values}"
    )
    no_clip = run(RecordingEnvironment(), numbered_chunk(3), execute_steps=3, max_steps=3)
    assert no_clip.clipped_values == 0, "values inside [-1, 1] are not clipped"
    _done("each row is converted, clipped, and its clipped values counted")

    # Invalid limits are rejected before any step.
    for kwargs in ({"execute_steps": 0, "max_steps": 5}, {"execute_steps": -1, "max_steps": 5}):
        env = RecordingEnvironment()
        try:
            run(env, numbered_chunk(3), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs}: a non-positive execute_steps must raise ValueError")
        assert env.steps == 0, "reject invalid limits before stepping the environment"
    _done("non-positive execute_steps raises ValueError")


def check_run_act_episode(run_act_episode) -> None:
    """Exercise 3: predict, execute a prefix, re-predict from the latest observation."""

    def run(env, policy, *, max_steps: int, execute_steps: int, seed=None, convert=identity):
        prepared: list[Any] = []

        def prepare(observation: Any) -> dict[str, Any]:
            prepared.append(observation)
            return {"observation": observation}

        result = run_act_episode(
            env,
            policy,
            prepare,
            call_predict,
            convert,
            max_steps=max_steps,
            execute_steps=execute_steps,
            seed=seed,
        )
        return result, prepared

    # Budget-limited episode: 3 predictions of at most 2 actions, the last one cut to 1.
    env, policy = RecordingEnvironment(), RecordingPolicy(numbered_chunk(6))
    result, prepared = run(env, policy, max_steps=5, execute_steps=2, seed=123)
    assert env.reset_seeds == [123], "reset the environment once, with the given seed"
    assert policy.resets == 1, "reset the policy once at the start of the episode"
    assert env.steps == 5 and result.steps == 5, (
        "the episode must use exactly max_steps environment steps"
    )
    assert result.policy_calls == 3 == len(policy.batches), (
        "5 steps in prefixes of 2 need 3 predictions"
    )
    assert prepared == [{"step": 0}, {"step": 2}, {"step": 4}], (
        f"each prediction must use the latest observation; prepared {prepared}"
    )
    assert all(not enabled for enabled in policy.grad_enabled), "predict without gradients"
    assert result.stop_reason == "step_limit" and bool(result.success) is False
    sent = np.stack([a.reshape(-1) for a in env.actions])
    rows = numbered_chunk(6)[0].numpy()
    np.testing.assert_allclose(
        sent,
        rows[[0, 1, 0, 1, 0]],
        atol=1e-6,
        err_msg="each new chunk restarts at its own row 0",
    )
    np.testing.assert_allclose(np.stack([_as_numpy(a) for a in result.actions]), sent, atol=1e-6)
    _done("re-predicts from the latest observation and never exceeds max_steps")

    # A short chunk limits each prefix even when execute_steps is larger.
    env, policy = RecordingEnvironment(), RecordingPolicy(numbered_chunk(3))
    result, prepared = run(env, policy, max_steps=7, execute_steps=10)
    assert result.steps == 7 and result.policy_calls == 3, "T=3 chunks: 3 + 3 + 1 steps"
    assert prepared == [{"step": 0}, {"step": 3}, {"step": 6}]
    _done("a chunk shorter than execute_steps is fully used, then re-predicted")

    # The environment can end the episode in the middle of a prefix.
    for kind, reason in (
        ("success", "success"),
        ("terminated", "terminated"),
        ("truncated", "truncated"),
    ):
        env, policy = (
            RecordingEnvironment(end_after=5, end_kind=kind),
            RecordingPolicy(numbered_chunk(4)),
        )
        result, _ = run(env, policy, max_steps=100, execute_steps=4)
        assert env.steps == 5 and result.steps == 5, f"{kind}: stop after the ending step"
        assert result.policy_calls == 2, f"{kind}: no prediction after the episode ended"
        assert result.stop_reason == reason, (
            f"{kind}: expected {reason!r}, got {result.stop_reason!r}"
        )
        assert bool(result.success) is (kind == "success"), f"{kind}: success flag"
    _done("stops when the environment ends the episode, with its stop reason")

    # Clipping counts add up over the episode.
    env, policy = RecordingEnvironment(), RecordingPolicy(torch.full((1, 4, ACTION_DIM), 1.5))
    result, _ = run(env, policy, max_steps=6, execute_steps=4)
    assert result.clipped_values == 6 * ACTION_DIM, (
        f"6 steps x 7 clipped values = 42, got {result.clipped_values}"
    )
    assert all(np.all(a == 1.0) for a in env.actions), "send clipped actions"
    _done("clipped values are summed over all prefixes")

    # The sum of steps matches the actions; reset happens before the first prediction.
    order: list[str] = []

    class OrderEnv(RecordingEnvironment):
        def reset(self, seed=None):
            order.append("env.reset")
            return super().reset(seed)

    class OrderPolicy(RecordingPolicy):
        def reset(self) -> None:
            order.append("policy.reset")
            super().reset()

        def predict_action_chunk(self, batch):
            order.append("predict")
            return super().predict_action_chunk(batch)

    result, _ = run(OrderEnv(), OrderPolicy(numbered_chunk(2)), max_steps=2, execute_steps=2)
    assert order.index("predict") > max(order.index("env.reset"), order.index("policy.reset")), (
        "reset the environment and the policy before the first prediction"
    )
    assert len(result.actions) == result.steps
    _done("resets happen before the first prediction")
