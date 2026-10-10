"""Behaviour checks for the Lesson 18 exercise.

A recording adapter shows what each ``velocity`` call received, so the check can test that
only the state changed, that the caller's inputs were left alone, and that the result is
the mean absolute difference. Read a failing assertion message first.
"""

import torch


def _done(message: str) -> None:
    print(f"✓ {message}")


class _RecordingAdapter:
    """velocity = x_t * (1 + t) - state @ w (broadcast over steps); records every call."""

    def __init__(self, state_dim: int) -> None:
        self.weights = torch.linspace(-1.0, 2.0, state_dim, dtype=torch.float64)
        self.calls: list[dict] = []

    def velocity(self, inputs, x_t, time):
        self.calls.append(
            {"inputs": inputs, "state": inputs["state"].clone(), "x_t": x_t, "time": time}
        )
        shift = (inputs["state"].double() @ self.weights).to(x_t.dtype).view(-1, 1, 1)
        return x_t * (1 + time.view(-1, 1, 1)) - shift


def check_state_condition_delta(state_condition_delta) -> None:
    """Exercise 18.1: |v(s) - v(0)| averaged over every value; only the state changes."""
    generator = torch.Generator().manual_seed(110)
    images = [torch.rand(2, 3, 4, 4, generator=generator)]
    state = torch.tensor([[1.0, -2.0, 0.5], [-3.0, 0.0, 1.0]], dtype=torch.float64)
    inputs = {
        "images": images,
        "img_masks": [torch.ones(2, dtype=torch.bool)],
        "lang_tokens": torch.tensor([[5, 6, 0], [7, 0, 0]]),
        "lang_masks": torch.tensor([[True, True, False], [True, False, False]]),
        "state": state,
    }
    original_state = state.clone()
    x_t = torch.randn(2, 4, 3, generator=generator)
    time = torch.tensor([0.3, 0.9])
    adapter = _RecordingAdapter(3)
    delta = state_condition_delta(adapter, inputs, x_t, time)

    assert isinstance(delta, torch.Tensor) and delta.ndim == 0, "return a 0-dimensional tensor"
    assert len(adapter.calls) == 2, (
        f"call adapter.velocity twice (with and without state), saw {len(adapter.calls)}"
    )
    states = [call["state"] for call in adapter.calls]
    with_state = [s for s in states if torch.equal(s, original_state)]
    zero_state = [s for s in states if not s.any()]
    assert with_state and zero_state, (
        "one call must use the supplied state and one an all-zero state"
    )
    assert zero_state[0].dtype == torch.float64 and zero_state[0].shape == (2, 3), (
        "the zero state must have the shape, dtype and device of inputs['state']"
    )
    for call in adapter.calls:
        assert torch.equal(call["x_t"], x_t) and torch.equal(call["time"], time), (
            "use the same x_t and time for both calls"
        )
        for key in ("images", "img_masks", "lang_tokens", "lang_masks"):
            assert call["inputs"][key] is inputs[key], (
                f"keep inputs[{key!r}] unchanged in both calls"
            )
    assert torch.equal(inputs["state"], original_state) and inputs["state"] is state, (
        "do not modify the caller's inputs dict or its state tensor (copy the dict)"
    )
    _done("two velocity calls that differ only in the state; the caller's inputs are untouched")

    # For this adapter v(s) - v(0) = -(state @ w), the same for every value of one example.
    per_example = (state @ adapter.weights).abs()  # (B,)
    expected = per_example.mean().to(delta.dtype)
    assert torch.isclose(delta, expected, atol=1e-6), (
        f"delta must be the mean of |v(s) - v(0)| over all values; expected {float(expected):.6f}, "
        f"got {float(delta):.6f}"
    )
    _done("returns the mean absolute difference (negative differences count positively)")
