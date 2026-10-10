"""Behaviour checks for the Lesson 4 exercises.

Each ``check_*`` function receives your implementation. Losses are compared with values
computed independently here; the update checks use a tiny linear model whose gradient
is known exactly, so they can tell whether gradients were cleared, clipped, and applied.
Read a failing assertion message first: it states the expected behaviour.
"""

import math
import time as wall_time
from typing import Any

import torch
from torch import nn


def _same(actual: object, expected: object) -> bool:
    """True when ``actual`` is ``expected`` or holds equal values (tensors, lists, dicts)."""
    if actual is expected:
        return True
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and actual.shape == expected.shape
            and torch.equal(actual.cpu(), expected.cpu())
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_same(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same(actual[k], expected[k]) for k in expected)
        )
    return actual == expected


def _done(message: str) -> None:
    print(f"✓ {message}")


def _expect_error(call, errors: tuple[type[Exception], ...], message: str) -> None:
    try:
        call()
    except errors:
        return
    except Exception as error:  # noqa: BLE001 - report the unexpected type
        raise AssertionError(f"{message}; got {type(error).__name__}: {error}") from error
    raise AssertionError(message)


def check_masked_action_l1(masked_action_l1) -> None:
    """Exercise 1: mean absolute error over valid (b, t, d) elements only."""
    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(3, 5, 4, generator=generator)
    target = torch.randn(3, 5, 4, generator=generator)
    is_pad = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, True, True, True],
            [False, True, True, True, True],
        ]
    )
    valid = ~is_pad
    expected = (prediction - target).abs()[valid].mean()
    loss = masked_action_l1(prediction, target, is_pad)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "return a 0-dimensional tensor"
    torch.testing.assert_close(
        loss, expected, msg=f"expected the mean over the 9 valid steps x 4 values = {expected:.6f}"
    )
    per_example_mean = torch.stack(
        [(prediction[b] - target[b]).abs()[valid[b]].mean() for b in range(3)]
    ).mean()
    assert not torch.isclose(loss, per_example_mean), (
        "average over all valid elements together, not the mean of per-example means"
    )
    _done("averages |prediction - target| over all valid elements together")

    no_pad = torch.zeros(3, 5, dtype=torch.bool)
    torch.testing.assert_close(
        masked_action_l1(prediction, target, no_pad),
        (prediction - target).abs().mean(),
        msg="with no padding, the loss is the ordinary mean absolute error",
    )
    _done("no padding gives the plain mean absolute error")

    # Padded predictions get no gradient; valid ones get sign(diff) / count.
    leaf = prediction.clone().requires_grad_(True)
    masked_action_l1(leaf, target, is_pad).backward()
    assert leaf.grad is not None, "the loss must keep the graph to the prediction"
    count = int(valid.sum()) * prediction.shape[-1]
    expected_grad = torch.sign(prediction - target) / count * valid.unsqueeze(-1)
    torch.testing.assert_close(
        leaf.grad,
        expected_grad,
        msg="gradient must be sign(prediction - target) / (valid elements) on valid steps, 0 on padding",
    )
    _done("padded positions get zero gradient; valid ones get sign / count")

    changed = prediction.clone()
    changed[is_pad] = 1e4
    torch.testing.assert_close(masked_action_l1(changed, target, is_pad), expected)
    _done("padded values do not change the loss")

    all_pad = torch.ones(3, 5, dtype=torch.bool)
    leaf = prediction.clone().requires_grad_(True)
    zero = masked_action_l1(leaf, target, all_pad)
    assert torch.isfinite(zero) and zero.item() == 0.0, "an all-padding batch must give exactly 0"
    zero.backward()
    assert leaf.grad is not None and torch.all(leaf.grad == 0), "all-padding: gradient must be zero"
    _done("an all-padding batch returns 0 without dividing by zero")


def check_diagonal_gaussian_kl(diagonal_gaussian_kl) -> None:
    """Exercise 2: KL(N(mu, exp(logvar)) || N(0, I)), summed over latents, mean over batch."""
    generator = torch.Generator().manual_seed(1)
    mu = torch.randn(4, 6, generator=generator)
    logvar = torch.randn(4, 6, generator=generator)
    variance = logvar.exp()
    # Closed form for each latent: 0.5 * (variance + mu^2 - 1 - log variance).
    per_latent = 0.5 * (variance + mu**2 - 1 - logvar)
    expected = per_latent.sum(dim=1).mean()
    value = diagonal_gaussian_kl(mu, logvar)
    assert isinstance(value, torch.Tensor) and value.ndim == 0, "return a 0-dimensional tensor"
    torch.testing.assert_close(value, expected, msg=f"expected {expected:.6f} nats")
    assert not torch.isclose(value, per_latent.mean()), (
        "sum over latent dimensions first, then average over the batch (not a mean over both)"
    )
    _done("matches the closed form: sum over latents, mean over batch")

    # logvar is the log of the variance, not the standard deviation.
    one = diagonal_gaussian_kl(torch.zeros(1, 1), torch.tensor([[math.log(4.0)]]))
    torch.testing.assert_close(
        one,
        torch.tensor(0.5 * (4.0 - 1.0 - math.log(4.0))),
        msg="variance 4 (logvar = log 4) must give 0.5 * (4 - 1 - log 4)",
    )
    shifted = diagonal_gaussian_kl(torch.tensor([[3.0]]), torch.zeros(1, 1))
    torch.testing.assert_close(shifted, torch.tensor(4.5), msg="mu = 3, variance 1 gives 4.5")
    _done("uses exp(logvar) as the variance and mu^2 for the mean shift")

    assert torch.all(
        torch.stack([diagonal_gaussian_kl(mu * s, logvar * s) for s in (0.1, 1.0, 3.0)]) >= 0
    ), "KL divergence is never negative"
    torch.testing.assert_close(
        diagonal_gaussian_kl(torch.zeros(5, 3), torch.zeros(5, 3)), torch.tensor(0.0)
    )
    _done("non-negative, and zero for the standard normal")

    leaf_mu = mu.clone().requires_grad_(True)
    leaf_logvar = logvar.clone().requires_grad_(True)
    diagonal_gaussian_kl(leaf_mu, leaf_logvar).backward()
    assert leaf_mu.grad is not None and leaf_logvar.grad is not None, "keep the autograd graph"
    torch.testing.assert_close(leaf_mu.grad, mu / 4, msg="d KL / d mu must be mu / B")
    torch.testing.assert_close(
        leaf_logvar.grad, 0.5 * (variance - 1) / 4, msg="d KL / d logvar must be (var - 1) / (2B)"
    )
    _done("gradients match the analytic derivatives")


class LinearProbe(nn.Module):
    """A model whose loss gradient is exactly ``coefficients``: loss = weight · coefficients."""

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))


def linear_loss(coefficients: torch.Tensor, record: list[dict[str, Any]] | None = None):
    """A loss callback following the Lesson 4 contract that records how it was called."""

    def loss_fn(model: LinearProbe, batch, *, noise=None, time=None):
        if record is not None:
            record.append(
                {"training": model.training, "batch": batch, "noise": noise, "time": time}
            )
        loss = (model.weight * coefficients).sum() + batch.get("offset", torch.tensor(0.0))
        return loss, {"tensor_metric": loss.detach() * 2, "float_metric": 1.5}

    return loss_fn


def check_optimizer_update(optimizer_update) -> None:
    """Exercise 3: train mode → zero grads → loss → backward → optional clip → step."""
    coefficients = torch.tensor([3.0, 4.0])  # gradient norm 5
    record: list[dict[str, Any]] = []
    model = LinearProbe()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([1.0, -1.0]))
    model.eval()
    model.weight.grad = torch.full((2,), 100.0)  # a stale gradient from an earlier step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    noise, time = torch.tensor([7.0]), torch.tensor([0.25])
    batch = {"offset": torch.tensor(2.0)}
    result = optimizer_update(
        model, batch, optimizer, linear_loss(coefficients, record), noise=noise, time=time
    )

    assert len(record) == 1, "call loss_fn exactly once"
    assert record[0]["training"] is True, "put the model in training mode before calling loss_fn"
    assert _same(record[0]["batch"], batch), "pass the batch to loss_fn"
    assert _same(record[0]["noise"], noise) and _same(record[0]["time"], time), (
        "pass noise and time through"
    )
    torch.testing.assert_close(
        model.weight.detach(),
        torch.tensor([1.0, -1.0]) - 0.1 * coefficients,
        msg="one SGD step must use only this loss's gradient: clear old gradients before backward()",
    )
    _done("trains the model, clears stale gradients, and applies exactly one step")

    # The returned loss is the value before the update, as plain Python floats.
    assert set(result) == {"loss", "tensor_metric", "float_metric"}, f"keys: {sorted(result)}"
    assert all(type(value) is float for value in result.values()), "return Python floats"
    expected_loss = 3.0 * 1.0 + 4.0 * -1.0 + 2.0
    assert math.isclose(result["loss"], expected_loss, abs_tol=1e-6), (
        f"loss must be the callback's loss ({expected_loss}), got {result['loss']}"
    )
    assert math.isclose(result["tensor_metric"], 2 * expected_loss, abs_tol=1e-6)
    assert math.isclose(result["float_metric"], 1.5)
    _done("returns the loss and every metric as Python floats")

    # Keyword noise/time default to None and must still be passed.
    record.clear()
    optimizer_update(
        LinearProbe(),
        {},
        torch.optim.SGD(LinearProbe().parameters(), lr=0.1),
        linear_loss(coefficients, record),
    )
    assert record[0]["noise"] is None and record[0]["time"] is None
    _done("passes noise=None and time=None when they are not given")

    # Clipping rescales the whole gradient to max_grad_norm.
    model = LinearProbe()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_update(model, {}, optimizer, linear_loss(coefficients), max_grad_norm=1.0)
    torch.testing.assert_close(
        model.weight.detach(),
        -0.1 * coefficients / 5.0,
        msg="with max_grad_norm=1 the gradient [3, 4] (norm 5) must be clipped to [0.6, 0.8]",
    )
    model = LinearProbe()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_update(model, {}, optimizer, linear_loss(coefficients), max_grad_norm=10.0)
    torch.testing.assert_close(
        model.weight.detach(),
        -0.1 * coefficients,
        msg="a gradient below max_grad_norm must not change",
    )
    _done("max_grad_norm clips the total gradient norm, after backward()")

    # Invalid losses are rejected before the weights change.
    for bad_loss, description in (
        (torch.tensor(float("nan")), "NaN"),
        (torch.tensor(float("inf")), "inf"),
    ):
        model = LinearProbe()

        def broken(model, batch, *, noise=None, time=None, bad=bad_loss):
            return (model.weight.sum() * 0 + bad), {}

        _expect_error(
            lambda model=model, broken=broken: optimizer_update(
                model, {}, torch.optim.SGD(model.parameters(), lr=0.1), broken
            ),
            (ValueError,),
            f"a {description} loss must raise ValueError",
        )
        assert torch.all(model.weight == 0), f"a {description} loss must not update the weights"

    def vector_loss(model, batch, *, noise=None, time=None):
        return model.weight * 2, {}

    _expect_error(
        lambda: optimizer_update(
            LinearProbe(), {}, torch.optim.SGD(LinearProbe().parameters(), lr=0.1), vector_loss
        ),
        (ValueError, RuntimeError),
        "a non-scalar loss must be rejected",
    )
    _expect_error(
        lambda: optimizer_update(
            LinearProbe(),
            {},
            torch.optim.SGD(LinearProbe().parameters(), lr=0.1),
            linear_loss(coefficients),
            max_grad_norm=0.0,
        ),
        (ValueError,),
        "max_grad_norm <= 0 must raise ValueError",
    )
    _done("non-finite or non-scalar losses and a non-positive max_grad_norm are rejected")


def check_train_loop(train_loop) -> None:
    """Exercise 4: cycle batches in order, one metric row per update, honour the time budget."""
    coefficients = torch.tensor([1.0, 0.0])
    record: list[dict[str, Any]] = []
    batches = [{"offset": torch.tensor(float(i)), "id": torch.tensor(i)} for i in range(3)]
    model = LinearProbe()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    history = train_loop(
        model, iter(batches), optimizer, linear_loss(coefficients, record), steps=7
    )
    ids = [int(call["batch"]["id"]) for call in record]
    assert ids == [0, 1, 2, 0, 1, 2, 0], f"cycle the batches in order; used batch ids {ids}"
    assert len(history) == 7, "return one metric dictionary per update"
    # weight[0] starts at 0 and drops by 0.5 per update, so loss_i = -0.5 * i + offset_i.
    expected = [-0.5 * i + (i % 3) for i in range(7)]
    got = [row["loss"] for row in history]
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got, expected, strict=True)), (
        f"history must hold each update's metrics in order: expected losses {expected}, got {got}"
    )
    assert math.isclose(model.weight.detach()[0].item(), -3.5), (
        "7 updates must each change the weights"
    )
    _done("cycles batches in order (works with a one-shot iterator), one row per update")

    model = LinearProbe()
    assert (
        train_loop(
            model,
            batches,
            torch.optim.SGD(model.parameters(), lr=0.5),
            linear_loss(coefficients),
            steps=0,
        )
        == []
    )
    assert torch.all(model.weight == 0), "steps=0 must not update the model"
    _done("steps=0 returns an empty history")

    model = LinearProbe()
    train_loop(
        model,
        batches,
        torch.optim.SGD(model.parameters(), lr=1.0),
        linear_loss(torch.tensor([30.0, 40.0])),
        steps=3,
        max_grad_norm=1.0,
    )
    torch.testing.assert_close(
        model.weight.detach(),
        torch.tensor([-1.8, -2.4]),
        msg="pass max_grad_norm to every update (3 clipped steps of [0.6, 0.8])",
    )
    _done("max_grad_norm reaches every update")

    def slow_loss(model, batch, *, noise=None, time=None):
        wall_time.sleep(0.05)
        return (model.weight * coefficients).sum(), {}

    model = LinearProbe()
    started = wall_time.monotonic()
    history = train_loop(
        model,
        batches,
        torch.optim.SGD(model.parameters(), lr=0.1),
        slow_loss,
        steps=100,
        max_seconds=0.2,
    )
    elapsed = wall_time.monotonic() - started
    assert 1 <= len(history) < 100, f"stop when the time budget is used; ran {len(history)} updates"
    assert elapsed < 0.2 + 0.05 * 3, f"check the budget before each update; took {elapsed:.2f} s"
    assert math.isclose(model.weight.detach()[0].item(), -0.1 * len(history), abs_tol=1e-6), (
        "return every completed update"
    )
    _done("max_seconds stops the loop early and returns the completed updates")

    model = LinearProbe()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = linear_loss(coefficients)
    _expect_error(
        lambda: train_loop(model, [], optimizer, loss_fn, steps=1),
        (ValueError,),
        "an empty batch list must raise ValueError",
    )
    _expect_error(
        lambda: train_loop(model, batches, optimizer, loss_fn, steps=-1),
        (ValueError,),
        "negative steps must raise ValueError",
    )
    for budget in (0.0, -1.0):
        _expect_error(
            lambda budget=budget: train_loop(
                model, batches, optimizer, loss_fn, steps=1, max_seconds=budget
            ),
            (ValueError,),
            f"max_seconds={budget} must raise ValueError",
        )
    assert torch.all(model.weight == 0), "reject invalid arguments before any update"
    _done("empty batches, negative steps, and a non-positive time budget are rejected")
