"""Behaviour checks for the Lesson 14 exercises.

These add to the notebook's oracle checks: the worked example, other schedule lengths,
dtype rules, a custom clip range, and the one-step sampler.
Read a failing assertion message first: it states the expected behaviour.
"""

import torch

from vla_lab.teaching.diffusion import DDPMSchedule


def _done(message: str) -> None:
    print(f"✓ {message}")


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach().double() - expected.detach().double()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def check_posterior_coefficients(posterior_coefficients) -> None:
    """Exercise 14.1: the worked example, float32 output, and the final step."""
    coef_x0, coef_xt, variance = posterior_coefficients(0.5, 0.8)
    # alpha = 0.625, beta = 0.375: c0 = sqrt(0.8) * 0.375 / 0.5, ct = sqrt(0.625) * 0.2 / 0.5.
    _close(coef_x0, torch.tensor(0.8**0.5 * 0.375 / 0.5), "worked example: coef_x0 ≈ 0.671")
    _close(coef_xt, torch.tensor(0.625**0.5 * 0.2 / 0.5), "worked example: coef_xt ≈ 0.316")
    _close(variance, torch.tensor(0.15), "worked example: variance = 0.4 * 0.375 = 0.15")
    _done("reproduces the worked example (0.671, 0.316, 0.150)")

    outputs = posterior_coefficients(
        torch.tensor(0.3, dtype=torch.float64), torch.tensor(0.6, dtype=torch.float64)
    )
    assert all(o.dtype == torch.float32 and o.shape == () for o in outputs), (
        "return three 0-d float32 tensors, also for float64 inputs"
    )
    _done("returns 0-d float32 tensors for float64 inputs")

    coef_x0, coef_xt, variance = posterior_coefficients(0.37, 1.0)
    _close(coef_x0, torch.tensor(1.0), "abar_prev = 1: coef_x0 = 1", atol=1e-6)
    _close(coef_xt, torch.tensor(0.0), "abar_prev = 1: coef_xt = 0", atol=1e-6)
    assert 0 < float(variance) <= 1e-19, "the zero final-step variance is clamped to 1e-20"
    _done("the final step gives coef_x0 = 1, coef_xt = 0 and a clamped variance")


def check_ddpm_reverse_step(ddpm_reverse_step) -> None:
    """Exercise 14.2: custom clip range, per-element formula, dtype, inputs unchanged."""
    alpha_bar = DDPMSchedule.cosine(100).alpha_bar
    generator = torch.Generator().manual_seed(60)
    x = torch.randn(2, 3, 4, generator=generator) * 2
    eps = torch.randn(2, 3, 4, generator=generator)
    noise = torch.randn(2, 3, 4, generator=generator)
    originals = (x.clone(), eps.clone(), noise.clone())
    out = ddpm_reverse_step(x, eps, 40, 30, alpha_bar, noise, 0.5)
    abar_t, abar_prev = alpha_bar[40], alpha_bar[30]
    x0_hat = ((x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()).clamp(-0.5, 0.5)
    alpha = abar_t / abar_prev
    beta = 1 - alpha
    mean = (
        abar_prev.sqrt() * beta / (1 - abar_t) * x0_hat
        + alpha.sqrt() * (1 - abar_prev) / (1 - abar_t) * x
    )
    variance = (1 - abar_prev) / (1 - abar_t) * beta
    _close(
        out,
        mean + variance.sqrt() * noise,
        "a skipped step 40 -> 30 with clip_range = 0.5",
        atol=1e-5,
    )
    for before, after, name in zip(
        originals, (x, eps, noise), ("x_t", "epsilon", "noise"), strict=True
    ):
        assert torch.equal(before, after), f"do not modify {name} in place"
    _done("clamps the clean estimate to the given clip_range, then mean + sqrt(variance) * noise")

    final = ddpm_reverse_step(x, eps, 3, -1, alpha_bar, None, 0.5)
    x0_final = ((x - (1 - alpha_bar[3]).sqrt() * eps) / alpha_bar[3].sqrt()).clamp(-0.5, 0.5)
    _close(final, x0_final, "the final step returns the clean estimate clipped to clip_range")
    _done("the final step returns the clipped clean estimate")

    out64 = ddpm_reverse_step(x.double(), eps.double(), 40, 30, alpha_bar, noise.double(), 1.0)
    assert out64.dtype == torch.float64, "keep the dtype of x_t"
    _done("keeps float64 inputs in float64")


def check_inference_timesteps(inference_timesteps) -> None:
    """Exercise 14.3: stride T // n for other T, including uneven division."""
    cases = {
        (10, 3): [6, 3, 0],
        (10, 10): [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        (7, 2): [3, 0],
        (1, 1): [0],
        (1000, 4): [750, 500, 250, 0],
    }
    for (total, steps), expected in cases.items():
        got = inference_timesteps(total, steps)
        assert got == expected and all(type(v) is int for v in got), (
            f"T={total}, n={steps}: expected {expected} (Python ints), got {got}"
        )
    for total, steps in ((10, 0), (10, 11), (10, -2)):
        try:
            inference_timesteps(total, steps)
        except ValueError:
            continue
        raise AssertionError(f"T={total}, n={steps} must raise ValueError")
    _done("uses stride T // n, ends at 0, and rejects n < 1 or n > T")


def check_sample_actions(sample_actions) -> None:
    """Exercise 14.4: clip_range is passed on; a one-step schedule needs no step noise."""
    schedule = DDPMSchedule.cosine(100)
    start = torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(61)) * 3
    original = start.clone()

    def big_epsilon(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x  # pushes the clean estimate far outside [-1, 1]

    one_step = sample_actions(big_epsilon, start, schedule, 1, [], 1.0)
    abar = schedule.alpha_bar[0]
    expected = ((start + (1 - abar).sqrt() * start) / abar.sqrt()).clamp(-1, 1)
    _close(one_step, expected, "one inference step: t = 0 is the final step and needs no noise")
    assert torch.equal(start, original), "do not modify initial_noise in place"

    clipped = sample_actions(big_epsilon, start, schedule, 1, [], 0.25)
    unclipped = sample_actions(big_epsilon, start, schedule, 1, [], None)
    assert float(clipped.abs().max()) <= 0.25 + 1e-6, "pass clip_range to every reverse step"
    assert float(unclipped.abs().max()) > 1.0, "clip_range=None must not clip"
    _done("a one-step schedule uses no step noise; clip_range reaches the reverse step")

    try:
        sample_actions(big_epsilon, start, schedule, 1, [torch.zeros_like(start)], 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "one inference step takes zero step noises; one extra must raise ValueError"
        )
    _done("an extra step noise is rejected")
