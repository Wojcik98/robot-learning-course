"""Behaviour checks for the Lesson 11 exercises.

These add to the notebook's oracle checks. Values are compared with formulas written out
here, on random inputs with several timesteps and padding patterns per batch, and the loss
callback is driven by recording models that reveal what it computed.
Read a failing assertion message first: it states the expected behaviour.
"""

import torch
from torch import nn

from vla_lab.teaching.diffusion import DDPMSchedule


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


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-6) -> None:
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach().double() - expected.detach().double()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def check_cumulative_alphas(cumulative_alphas) -> None:
    """Exercise 11.1: an inclusive running product of (1 - beta)."""
    betas = torch.rand(30, generator=torch.Generator().manual_seed(40)) * 0.3 + 0.01
    original = betas.clone()
    alpha_bar = cumulative_alphas(betas)
    expected, running = [], 1.0
    for beta in betas.tolist():
        running *= 1.0 - beta
        expected.append(running)
    _close(
        alpha_bar,
        torch.tensor(expected),
        "alpha_bar[t] must be prod_{s<=t} (1 - betas[s])",
        atol=1e-6,
    )
    assert torch.equal(betas, original), "do not modify betas in place"
    _close(
        cumulative_alphas(torch.tensor([0.2])), torch.tensor([0.8]), "a single step gives 1 - beta"
    )
    _done("matches the running product for random betas, inclusive of step t")


def check_forward_noise(forward_noise) -> None:
    """Exercise 11.2: per-row timesteps, the closed form, gradients, inputs unchanged."""
    generator = torch.Generator().manual_seed(41)
    alpha_bar = torch.linspace(0.99, 0.01, 10)
    x0 = torch.randn(5, 3, 2, generator=generator, requires_grad=True)
    noise = torch.randn(5, 3, 2, generator=generator, requires_grad=True)
    timesteps = torch.tensor([9, 0, 4, 4, 7])
    originals = (x0.detach().clone(), noise.detach().clone())
    out = forward_noise(x0, noise, timesteps, alpha_bar)
    abar = alpha_bar[timesteps].view(5, 1, 1)
    _close(
        out,
        abar.sqrt() * x0 + (1 - abar).sqrt() * noise,
        "x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) noise, per row",
        atol=1e-6,
    )
    out.sum().backward()
    assert x0.grad is not None and noise.grad is not None, "gradients must flow to x0 and noise"
    _close(x0.grad, abar.sqrt().expand(5, 3, 2), "d x_t / d x0 = sqrt(abar_t) for each row's own t")
    _close(noise.grad, (1 - abar).sqrt().expand(5, 3, 2), "d x_t / d noise = sqrt(1 - abar_t)")
    assert torch.equal(x0.detach(), originals[0]) and torch.equal(noise.detach(), originals[1]), (
        "do not change x0 or noise in place"
    )
    _done("each row uses its own timestep; gradients reach x0 and noise")

    one_step = forward_noise(
        torch.ones(1, 1, 1), torch.zeros(1, 1, 1), torch.tensor([3]), alpha_bar
    )
    _close(one_step, alpha_bar[3].sqrt().view(1, 1, 1), "with zero noise, x_t = sqrt(abar_t) x0")
    _done("zero noise scales x0 by sqrt(abar_t)")


def check_sample_noise_and_timesteps(sample_noise_and_timesteps) -> None:
    """Exercise 11.3: uniform timesteps over 0..T-1, dtype follows actions."""
    actions = torch.zeros(20000, 2, 1)
    _, timesteps = sample_noise_and_timesteps(actions, 5, torch.Generator().manual_seed(42))
    counts = torch.bincount(timesteps, minlength=5)
    assert counts.shape == (5,) and counts.min() > 3600 and counts.max() < 4400, (
        f"timesteps must be uniform over 0..4: counts {counts.tolist()} (expected about 4000 each)"
    )
    _, single = sample_noise_and_timesteps(
        torch.zeros(50, 2, 1), 1, torch.Generator().manual_seed(1)
    )
    assert torch.equal(single, torch.zeros(50, dtype=torch.long)), "with T = 1 every timestep is 0"
    _done("timesteps are uniform over 0..T-1")

    noise64, t64 = sample_noise_and_timesteps(
        torch.zeros(4, 3, 2, dtype=torch.float64), 10, torch.Generator().manual_seed(2)
    )
    assert noise64.dtype == torch.float64 and noise64.shape == (4, 3, 2), (
        "noise takes the dtype and shape of actions"
    )
    assert t64.dtype == torch.long and t64.shape == (4,), "timesteps are long, one per row"
    noise, timesteps = sample_noise_and_timesteps(
        torch.zeros(8, 3, 2), 10, torch.Generator().manual_seed(5)
    )
    expected_generator = torch.Generator().manual_seed(5)
    expected_noise = torch.randn((8, 3, 2), generator=expected_generator)
    expected_t = torch.randint(0, 10, (8,), generator=expected_generator)
    assert torch.equal(noise, expected_noise) and torch.equal(timesteps, expected_t), (
        "draw both the noise and then the timesteps from the supplied generator"
    )
    _done("noise follows the actions' dtype; the supplied generator is used")


def check_masked_epsilon_loss(masked_epsilon_loss) -> None:
    """Exercise 11.4: squared error averaged over valid scalar values only."""
    generator = torch.Generator().manual_seed(43)
    prediction = torch.randn(3, 5, 4, generator=generator, requires_grad=True)
    target = torch.randn(3, 5, 4, generator=generator)
    is_pad = torch.tensor([[False] * 5, [False, False, True, True, True], [False] + [True] * 4])
    valid = ~is_pad
    count = int(valid.sum()) * 4
    loss = masked_epsilon_loss(prediction, target, is_pad)
    expected = (prediction - target).square()[valid].sum() / count
    _close(loss, expected, "loss = sum of valid squared errors / (valid steps * A)", atol=1e-6)
    loss.backward()
    expected_grad = 2 * (prediction - target) / count * valid.unsqueeze(-1)
    _close(
        prediction.grad,
        expected_grad,
        "gradient = 2 (prediction - target) / count on valid steps, 0 on padding",
    )
    _done("matches the formula with mixed padding; gradient is exact and zero on padding")

    shaken_target = target.clone()
    shaken_target[is_pad] = 1e3
    leaf = prediction.detach().clone().requires_grad_(True)
    masked_epsilon_loss(leaf, shaken_target, is_pad).backward()
    _close(leaf.grad, expected_grad, "changing padded targets must not change the gradient")
    no_pad = torch.zeros(3, 5, dtype=torch.bool)
    _close(
        masked_epsilon_loss(prediction.detach(), target, no_pad),
        (prediction.detach() - target).square().mean(),
        "with no padding, the loss is the ordinary mean squared error",
    )
    leaf = prediction.detach().clone().requires_grad_(True)
    zero = masked_epsilon_loss(leaf, target, torch.ones(3, 5, dtype=torch.bool))
    zero.backward()
    assert zero.item() == 0.0 and torch.isfinite(leaf.grad).all(), (
        "all padding: loss 0, finite gradient"
    )
    _done("padded targets do not matter; no padding is plain MSE; all padding gives 0")


class _RecordingDenoiser(nn.Module):
    """Returns ``predict(noisy, timesteps)`` and records its inputs; has a real schedule."""

    def __init__(self, predict) -> None:
        super().__init__()
        self.schedule = DDPMSchedule.cosine(20)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.predict = predict
        self.calls: list[tuple] = []

    def forward(self, noisy, timesteps, batch):
        self.calls.append((noisy.detach().clone(), timesteps, batch))
        return self.predict(noisy, timesteps) * self.scale


def check_epsilon_loss_fn(epsilon_loss_fn) -> None:
    """Exercise 11.5: noise the clean actions, predict noise, masked MSE against the noise."""
    generator = torch.Generator().manual_seed(44)
    actions = torch.randn(3, 4, 2, generator=generator)
    is_pad = torch.tensor([[False] * 4, [False, False, True, True], [False, True, True, True]])
    batch = {"action": actions, "action_is_pad": is_pad, "observation.state": torch.randn(3, 5)}
    noise = torch.randn(3, 4, 2, generator=generator)
    timesteps = torch.tensor([0, 9, 19])
    model = _RecordingDenoiser(lambda noisy, t: torch.zeros_like(noisy))
    loss, metrics = epsilon_loss_fn(model, batch, noise=noise, time=timesteps)

    assert len(model.calls) == 1, "call the model exactly once"
    noisy, got_t, got_batch = model.calls[0]
    abar = model.schedule.alpha_bar[timesteps].view(3, 1, 1)
    _close(
        noisy,
        abar.sqrt() * actions + (1 - abar).sqrt() * noise,
        "the model input must be the actions noised at the given timesteps",
        atol=1e-6,
    )
    assert torch.equal(got_t, timesteps), "pass the timesteps to the model"
    assert _same(got_batch, batch), "pass the batch to the model (it holds the observations)"
    valid = ~is_pad
    expected = noise.square()[valid].sum() / (int(valid.sum()) * 2)  # prediction is 0
    _close(
        loss,
        expected,
        "loss = masked MSE between the predicted noise and the noise (not the actions)",
    )
    assert set(metrics) == {"epsilon_mse"}, (
        f"metrics must be {{'epsilon_mse': loss}}, got keys {sorted(metrics)}"
    )
    _close(torch.as_tensor(metrics["epsilon_mse"]), loss, "epsilon_mse must equal the loss")
    loss.backward()
    assert model.scale.grad is not None, "the loss must keep the gradient path to the model"
    _done("noises the actions with the given noise and timesteps; masked MSE against the noise")

    # Without noise/time: draw them. A model that recovers the drawn noise exactly has zero loss.
    schedule = DDPMSchedule.cosine(20)

    def recover_noise(noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        abar = schedule.alpha_bar[t].view(-1, 1, 1)
        return (noisy - abar.sqrt() * actions) / (1 - abar).sqrt()

    model = _RecordingDenoiser(recover_noise)
    loss, _ = epsilon_loss_fn(model, batch)
    _, drawn_t, _ = model.calls[0]
    assert drawn_t.dtype == torch.long and drawn_t.shape == (3,), "draw one long timestep per row"
    assert int(drawn_t.min()) >= 0 and int(drawn_t.max()) < 20, (
        "draw timesteps in 0..T-1 from model.schedule"
    )
    assert loss.item() < 1e-8, "without arguments, draw noise and use that same noise as the target"
    _done("draws its own noise and timesteps when none are given, and trains against that noise")
