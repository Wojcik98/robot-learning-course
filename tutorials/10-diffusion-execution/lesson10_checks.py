"""Behaviour checks for the Lesson 10 exercise.

The check feeds your ``next_queued_action`` a predictor that returns a different, numbered
chunk on every call, so it can see which slots were queued, in which order, and when a
new prediction was requested. Read a failing assertion message first.
"""

from collections import deque

import torch


def _done(message: str) -> None:
    print(f"✓ {message}")


class _NumberedPredictor:
    """Call k returns a (B, H, A) chunk whose slot j holds 100 * k + j in every action value."""

    def __init__(self, batch: int, horizon: int, action_dim: int) -> None:
        self.batch, self.horizon, self.action_dim = batch, horizon, action_dim
        self.calls = 0

    def __call__(self) -> torch.Tensor:
        self.calls += 1
        slots = torch.arange(self.horizon, dtype=torch.float32) + 100 * self.calls
        return (
            slots.view(1, self.horizon, 1).expand(self.batch, self.horizon, self.action_dim).clone()
        )


def _slot(action: torch.Tensor) -> float:
    return float(action.flatten()[0])


def check_next_queued_action(next_queued_action) -> None:
    """Exercise 10.1: predict only when empty; queue slots n_o-1 .. n_o+n_a-2; pop in order."""
    predictor = _NumberedPredictor(batch=2, horizon=8, action_dim=3)
    queue: deque = deque()
    popped = [next_queued_action(queue, predictor, 2, 3) for _ in range(7)]
    assert all(tuple(a.shape) == (2, 3) for a in popped), "each action has shape (B, A)"
    got = [_slot(a) for a in popped]
    assert got == [101, 102, 103, 201, 202, 203, 301], (
        f"n_obs_steps=2, n_action_steps=3: expected slots 1, 2, 3 of each new chunk "
        f"(values 101, 102, 103, then 201, ...); got {got}"
    )
    assert predictor.calls == 3, (
        f"predict only when the queue is empty: 7 pops need 3 predictions, saw {predictor.calls}"
    )
    assert len(queue) == 2, "the remaining two actions of the third chunk stay queued"
    assert torch.equal(popped[0][1], torch.full((3,), 101.0)), "keep every batch row of the action"
    _done("queues slots n_o-1 .. n_o+n_a-2, pops them first-in-first-out, predicts only when empty")

    predictor = _NumberedPredictor(batch=1, horizon=4, action_dim=2)
    queue = deque()
    got = [_slot(next_queued_action(queue, predictor, 1, 4)) for _ in range(5)]
    assert got == [100, 101, 102, 103, 200], (
        f"n_obs_steps=1 starts at slot 0 and may use the whole horizon; got {got}"
    )
    _done("n_obs_steps = 1 starts at slot 0; a prefix that exactly fills the horizon is allowed")

    queue = deque([torch.full((1, 2), -5.0), torch.full((1, 2), -6.0)])
    predictor = _NumberedPredictor(batch=1, horizon=4, action_dim=2)
    first = next_queued_action(queue, predictor, 2, 2)
    assert predictor.calls == 0 and _slot(first) == -5.0, (
        "a non-empty queue is used before any new prediction"
    )
    _done("an existing queue is consumed from the front without predicting")

    predictor = _NumberedPredictor(batch=1, horizon=4, action_dim=2)
    queue = deque()
    try:
        next_queued_action(queue, predictor, 3, 3)  # needs slots 2..4, but the horizon has 0..3
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a prefix that does not fit the horizon (slots 2..4 of 0..3) must raise ValueError"
        )
    assert len(queue) == 0, "do not queue a partial prefix"
    _done("a prefix that does not fit the predicted horizon raises ValueError")
