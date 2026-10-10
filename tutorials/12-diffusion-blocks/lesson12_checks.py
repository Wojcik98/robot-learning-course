"""Behaviour checks for the Lesson 12 exercises.

These add to the notebook's oracle checks. The embedding is compared with the formula
for sizes and timestep values the oracle check does not use; the residual block is run
on stand-in modules with simple known effects, so a failure says which step is wrong.
Read a failing assertion message first: it states the expected behaviour.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn


def _done(message: str) -> None:
    print(f"✓ {message}")


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach().double() - expected.detach().double()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def check_sinusoidal_timestep_embedding(sinusoidal_timestep_embedding) -> None:
    """Exercise 12.1: sines first then cosines, f_k = exp(-k ln(10000) / (n - 1))."""

    def formula(t: torch.Tensor, dim: int) -> torch.Tensor:
        n = dim // 2
        k = torch.arange(n, dtype=torch.float64)
        f = torch.exp(-k * math.log(10000) / (n - 1))
        angle = t.double()[:, None] * f[None]
        return torch.cat((angle.sin(), angle.cos()), dim=-1)

    for dim in (4, 6, 10):
        t = torch.tensor([0, 3, 42, 999])
        table = sinusoidal_timestep_embedding(t, dim)
        assert table.shape == (4, dim) and table.dtype == torch.float32, (
            f"dim {dim}: float32 (B, dim)"
        )
        _close(
            table,
            formula(t, dim),
            f"dim {dim}: sines of all n frequencies, then cosines",
            atol=1e-4,
        )
    _done("matches the formula for dim 4, 6 and 10, including t = 999")

    halves = torch.tensor([0.5, 2.5])
    _close(
        sinusoidal_timestep_embedding(halves, 8),
        formula(halves, 8),
        "float timesteps keep their fraction",
    )
    float64 = sinusoidal_timestep_embedding(torch.tensor([7.0], dtype=torch.float64), 8)
    assert float64.dtype == torch.float32, "return float32 even for float64 timesteps"
    _done("float timesteps work and the result is float32")

    rows = sinusoidal_timestep_embedding(torch.arange(5), 12)
    _close(
        rows[:, :6].square() + rows[:, 6:].square(),
        torch.ones(5, 6),
        "sin^2 + cos^2 = 1 for each frequency k",
    )
    _done("column k and column n + k share one frequency")


class _Times(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _Plus(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.offset


def check_film_residual_forward(film_residual_forward) -> None:
    """Exercise 12.2: conv1 → FiLM (scale * out + bias, or out + embed) → conv2 → + residual(x)."""
    torch.manual_seed(50)
    batch, channels, length, cond_dim = 2, 3, 4, 5
    x, cond = torch.randn(batch, channels, length), torch.randn(batch, cond_dim)
    for use_scale in (True, False):
        encoder = nn.Linear(cond_dim, 2 * channels if use_scale else channels)
        block = SimpleNamespace(
            conv1=_Times(2.0),  # stands in for Conv1d → GroupNorm → Mish
            conv2=_Plus(3.0),
            cond_encoder=encoder,
            residual_conv=_Times(-1.0),
            out_channels=channels,
            use_film_scale_modulation=use_scale,
        )
        out = film_residual_forward(block, x, cond)
        embed = encoder(cond)[:, :, None]
        if use_scale:
            modulated = embed[:, :channels] * (2.0 * x) + embed[:, channels:]
        else:
            modulated = 2.0 * x + embed
        expected = (modulated + 3.0) + (-1.0 * x)
        label = "scale modulation" if use_scale else "bias only"
        _close(
            out,
            expected,
            f"{label}: expected conv2(FiLM(conv1(x))) + residual_conv(x), with scale = embed[:, :C], "
            "bias = embed[:, C:] broadcast over time (scale * out + bias, not (1 + scale) * out)",
        )
    _done("FiLM sits between conv1 and conv2; scale is the first C values, bias the last C")
    _done("the shortcut adds residual_conv(x) of the block input")

    block = SimpleNamespace(
        conv1=_Times(1.0),
        conv2=_Times(1.0),
        cond_encoder=nn.Linear(cond_dim, 2 * channels),
        residual_conv=nn.Identity(),
        out_channels=channels,
        use_film_scale_modulation=True,
    )
    out_a = film_residual_forward(block, x, cond)
    out_b = film_residual_forward(block, x[:1], cond[:1])
    _close(out_b, out_a[:1], "each example uses its own condition (no mixing across the batch)")
    _done("conditions are applied per example")
