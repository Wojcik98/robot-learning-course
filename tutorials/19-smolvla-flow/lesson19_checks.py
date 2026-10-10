"""Behaviour checks for the Lesson 19 exercise.

The notebook checks one scalar example. These checks use several examples, several real
and padded action columns, and a recording velocity function that shows which times and
which states the Euler loop visited. Read a failing assertion message first.
"""

import torch


def _done(message: str) -> None:
    print(f"✓ {message}")


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    actual = torch.as_tensor(actual)
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach().double() - expected.detach().double()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def check_masked_flow_mse(masked_flow_mse) -> None:
    """Exercise 19.1: squared error over real steps and the first action_dim columns only."""
    generator = torch.Generator().manual_seed(120)
    prediction = torch.randn(2, 4, 5, generator=generator, requires_grad=True)
    target = torch.randn(2, 4, 5, generator=generator)
    is_pad = torch.tensor([[False, False, False, True], [False, True, True, True]])
    loss = masked_flow_mse(prediction, target, is_pad, 3)
    valid = ~is_pad
    expected = (prediction - target)[..., :3].square()[valid].sum() / (int(valid.sum()) * 3)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "return a scalar tensor"
    _close(
        loss,
        expected,
        "loss = sum over real steps and the first action_dim columns / (N_valid * action_dim)",
    )
    loss.backward()
    grad = prediction.grad
    assert grad is not None, "keep the gradient path to the prediction"
    assert torch.all(grad[..., 3:] == 0), (
        "padded columns (beyond action_dim) must get zero gradient"
    )
    assert torch.all(grad[is_pad] == 0), "padded steps must get zero gradient"
    _done("matches the formula with mixed padding; padded steps and columns get no gradient")

    changed = target.clone()
    changed[..., 3:] = 1e3
    changed[is_pad] = -1e3
    _close(
        masked_flow_mse(prediction.detach(), changed, is_pad, 3),
        expected,
        "padded values must not change the loss",
    )
    zero = masked_flow_mse(prediction.detach(), target, torch.ones(2, 4, dtype=torch.bool), 3)
    assert torch.isfinite(zero) and zero.item() == 0.0, "all padding: return 0"
    full = masked_flow_mse(prediction.detach(), target, torch.zeros(2, 4, dtype=torch.bool), 5)
    _close(
        full,
        (prediction.detach() - target).square().mean(),
        "no padding and all columns: plain MSE",
    )
    _done("padded values are ignored; all padding gives 0; no padding gives plain MSE")


def check_euler_to_action(euler_to_action) -> None:
    """Exercise 19.1: n steps of x += dt * v(x, t) with dt = -1/n, t = 1, 1 - 1/n, ..., 1/n."""
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def linear_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        seen.append((x.clone(), t.clone()))
        return x * 2.0 + t.view(-1, 1, 1)

    noise = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(121))
    original = noise.clone()
    out = euler_to_action(linear_velocity, noise, 4)
    times = [t for _, t in seen]
    assert len(seen) == 4, f"call velocity_fn once per step (4), saw {len(seen)}"
    assert all(t.shape == (2,) and t.dtype == torch.float32 for t in times), (
        "t is a float32 (B,) tensor"
    )
    got = [float(t[0]) for t in times]
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, [1.0, 0.75, 0.5, 0.25], strict=True)), (
        f"with 4 steps the times are 1, 0.75, 0.5, 0.25; got {got}"
    )
    expected = noise.clone()
    for step in range(4):
        _close(
            seen[step][0],
            expected,
            f"step {step} must receive the state produced by the previous step",
        )
        expected = expected - 0.25 * (expected * 2.0 + (1.0 - 0.25 * step))
    _close(out, expected, "return the state after the last step")
    assert torch.equal(noise, original), "do not modify the noise in place"
    _done("visits t = 1, 1 - 1/n, ..., 1/n and carries the state from step to step")

    seen.clear()
    one = euler_to_action(linear_velocity, noise, 1)
    _close(one, noise - (noise * 2.0 + 1.0), "one step: x = noise - v(noise, 1)")
    for bad in (0, -1):
        try:
            euler_to_action(linear_velocity, noise, bad)
        except ValueError:
            continue
        raise AssertionError(f"num_steps={bad} must raise ValueError")
    _done("one step works; num_steps < 1 raises ValueError")
