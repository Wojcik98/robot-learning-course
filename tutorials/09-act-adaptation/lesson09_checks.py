"""Behaviour checks for the Lesson 9 exercise.

The check drives your ``fine_tune_updates`` with a tiny ACT-shaped model whose loss and
gradients are known exactly, and with plain SGD, so it can tell whether the KL weight,
the gradient clipping, the batch order and the time budget were applied.
Read a failing assertion message first: it states the expected behaviour.
"""

import math
import time as wall_time

import torch
from torch import nn

CPU = torch.device("cpu")


def _done(message: str) -> None:
    print(f"✓ {message}")


class _TinyACT(nn.Module):
    """Returns (prediction, (mu, logvar)) like ACT. ``bias`` predicts actions; ``mu`` is the latent mean."""

    def __init__(self, mu: float = 0.0, delay: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))
        self.mu = nn.Parameter(torch.full((3,), mu))
        self.delay = delay
        self.seen: list[int] = []

    def forward(self, batch):
        if self.delay:
            wall_time.sleep(self.delay)
        self.seen.append(int(batch["id"]))
        target = batch["action"]
        prediction = self.bias.view(1, 1, 2).expand_as(target)
        mu = self.mu.expand(target.shape[0], 3)
        return prediction, (mu, torch.zeros_like(mu))


def _batch(index: int, value: float) -> dict[str, torch.Tensor]:
    return {
        "id": torch.tensor(index),
        "action": torch.full((2, 3, 2), value),
        "action_is_pad": torch.tensor([[False, False, False], [False, True, True]]),
    }


def check_fine_tune_updates(fine_tune_updates) -> None:
    """Exercise: masked L1 + kl_weight * KL, clipped, one row per update, in batch order."""
    batches = [_batch(0, 1.0), _batch(1, -1.0), _batch(2, 0.5)]
    originals = [{k: v.clone() for k, v in b.items()} for b in batches]
    model = _TinyACT(mu=2.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = fine_tune_updates(
        model, batches, optimizer, steps=5, device=CPU, kl_weight=0.5, max_grad_norm=1e6
    )
    assert model.seen == [0, 1, 2, 0, 1], f"use the batches in order, cycling; saw ids {model.seen}"
    assert len(history) == 5, "return one metric dictionary per completed update"
    # First update: prediction 0, target 1 → L1 = 1; KL per example = 0.5 * 3 * mu^2 = 6.
    first = history[0]
    assert math.isclose(first["loss"], 1.0 + 0.5 * 6.0, abs_tol=1e-5), (
        f"the first loss must be masked L1 (1.0) + kl_weight (0.5) * KL (6.0) = 4.0, got {first['loss']}"
    )
    for key in ("action_l1", "kl"):
        assert key in first, f"keep the ACT loss metrics; missing {key!r}"
    for original, batch in zip(originals, batches, strict=True):
        assert all(torch.equal(original[k], batch[k]) for k in original), (
            "do not modify the batches"
        )
    _done("loss = masked L1 + kl_weight * KL; batches cycle in order; inputs unchanged")

    # The KL gradient on mu is kl_weight * mu, so with kl_weight = 0, mu must not move.
    model = _TinyACT(mu=2.0)
    fine_tune_updates(
        model,
        [batches[0]],
        torch.optim.SGD(model.parameters(), lr=0.1),
        steps=1,
        device=CPU,
        kl_weight=0.0,
        max_grad_norm=1e6,
    )
    assert torch.allclose(model.mu.detach(), torch.full((3,), 2.0)), (
        "kl_weight = 0 must remove the KL gradient"
    )
    model = _TinyACT(mu=2.0)
    fine_tune_updates(
        model,
        [batches[0]],
        torch.optim.SGD(model.parameters(), lr=0.1),
        steps=1,
        device=CPU,
        kl_weight=0.5,
        max_grad_norm=1e6,
    )
    torch.testing.assert_close(
        model.mu.detach(),
        torch.full((3,), 2.0 - 0.1 * 0.5 * 2.0),
        msg="one SGD step on mu must be lr * kl_weight * mu (pass kl_weight to the ACT loss)",
    )
    _done("kl_weight scales the KL gradient")

    # Clipping: with mu = 100 the gradient norm is about 87; max_grad_norm = 1 caps the step.
    model = _TinyACT(mu=100.0)
    before = torch.cat([p.detach().clone().flatten() for p in model.parameters()])
    fine_tune_updates(
        model,
        [batches[0]],
        torch.optim.SGD(model.parameters(), lr=0.1),
        steps=1,
        device=CPU,
        kl_weight=1.0,
        max_grad_norm=1.0,
    )
    after = torch.cat([p.detach().flatten() for p in model.parameters()])
    step = (after - before).norm().item()
    assert math.isclose(step, 0.1, rel_tol=1e-3), (
        f"with max_grad_norm = 1 and lr = 0.1, one SGD step must have norm 0.1; got {step:.4f}"
    )
    _done("max_grad_norm clips the total gradient norm")

    model = _TinyACT()
    assert (
        fine_tune_updates(
            model,
            batches,
            torch.optim.SGD(model.parameters(), lr=0.1),
            steps=0,
            device=CPU,
            kl_weight=0.1,
            max_grad_norm=1.0,
        )
        == []
    )
    assert not model.seen and torch.all(model.bias == 0), (
        "steps = 0 must not run or change the model"
    )
    _done("steps = 0 runs no update")

    model = _TinyACT(delay=0.05)
    started = wall_time.monotonic()
    history = fine_tune_updates(
        model,
        batches,
        torch.optim.SGD(model.parameters(), lr=0.1),
        steps=100,
        device=CPU,
        kl_weight=0.1,
        max_grad_norm=1.0,
        max_seconds=0.2,
    )
    elapsed = wall_time.monotonic() - started
    assert 1 <= len(history) < 100, f"stop when max_seconds is used up; ran {len(history)} updates"
    assert elapsed < 0.2 + 0.05 * 3, (
        f"check the time budget before each update; took {elapsed:.2f} s"
    )
    assert len(model.seen) == len(history), "return a row for every completed update"
    _done("max_seconds stops the loop early and keeps the completed rows")
