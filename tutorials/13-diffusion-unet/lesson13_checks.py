"""Behaviour checks for the Lesson 13 exercises.

These add to the notebook's oracle checks. The plan is compared with values written out
by hand for edge cases (one stage, another action width). The forward pass is run on a
stand-in U-Net whose blocks record what they received, so a failure says which block got
the wrong condition or the wrong skip feature.
Read a failing assertion message first: it states the expected behaviour.
"""

from types import SimpleNamespace

import torch
from torch import nn


def _done(message: str) -> None:
    print(f"✓ {message}")


def check_unet_channel_plan(unet_channel_plan) -> None:
    """Exercise 13.1: hand-written plans for one, two and four stages."""
    cases = {
        (7, (16,)): {"down": [(7, 16, False)], "mid": [(16, 16), (16, 16)], "up": []},
        (2, (4, 8)): {
            "down": [(2, 4, True), (4, 8, False)],
            "mid": [(8, 8), (8, 8)],
            "up": [(16, 4, True)],
        },
        (7, (16, 32, 64, 128)): {
            "down": [(7, 16, True), (16, 32, True), (32, 64, True), (64, 128, False)],
            "mid": [(128, 128), (128, 128)],
            "up": [(256, 64, True), (128, 32, True), (64, 16, True)],
        },
    }
    for (action_dim, down_dims), expected in cases.items():
        plan = unet_channel_plan(action_dim, down_dims)
        for key in ("down", "mid", "up"):
            got = [tuple(entry) for entry in plan[key]]
            want = expected[key]
            assert got == want, (
                f"action_dim={action_dim}, down_dims={down_dims}: {key!r} must be {want}, got {got}"
            )
    _done("one stage has no decoder stage; the last encoder stage never downsamples")
    _done("decoder stages read pairs in reverse, without the first, with 2x input channels")


class _Block(nn.Module):
    """Stand-in residual block: records (name, input channels, cond) and maps to ``out`` channels."""

    def __init__(self, name: str, out_channels: int, log: list) -> None:
        super().__init__()
        self.name, self.out_channels, self.log = name, out_channels, log

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        self.log.append((self.name, x.shape[1], x.shape[2], cond.detach().clone()))
        return x.mean(1, keepdim=True).expand(-1, self.out_channels, -1) + 1


class _Down(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., ::2]


class _Up(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(2, dim=-1)


class _Head(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1].expand(-1, self.action_dim, -1) * 1.0


def check_unet_forward(unet_forward) -> None:
    """Exercise 13.2: the condition reaches every block; skips are saved and popped in order."""
    log: list = []
    time_calls: list = []

    def time_mlp(timestep: torch.Tensor) -> torch.Tensor:
        time_calls.append(timestep)
        return torch.stack([timestep.float(), -timestep.float()], dim=-1)  # (B, 2)

    unet = SimpleNamespace(
        time_mlp=time_mlp,
        # Three encoder stages (channels 4, 6, 8) on a length-8 window: skips have
        # (channels, length) = (4, 8), (6, 4), (8, 2); the decoder uses (8, 2) then (6, 4).
        down_blocks=[
            (_Block("down0.res1", 4, log), _Block("down0.res2", 4, log), _Down()),
            (_Block("down1.res1", 6, log), _Block("down1.res2", 6, log), _Down()),
            (_Block("down2.res1", 8, log), _Block("down2.res2", 8, log), nn.Identity()),
        ],
        mid_blocks=[_Block("mid0", 8, log), _Block("mid1", 8, log)],
        up_blocks=[
            (_Block("up0.res1", 6, log), _Block("up0.res2", 6, log), _Up()),
            (_Block("up1.res1", 4, log), _Block("up1.res2", 4, log), _Up()),
        ],
        head=_Head(3),
    )
    x = torch.randn(2, 8, 3)  # (B, H, A)
    timestep, global_cond = torch.tensor([5, 9]), torch.randn(2, 4)
    original = x.clone()
    try:
        out = unet_forward(unet, x, timestep, global_cond)
    except RuntimeError as error:
        raise AssertionError(
            f"the stand-in U-Net failed ({error}). Save each skip after res2 and before resampling, "
            "and join the most recent skip (skips.pop()) before each decoder stage."
        ) from error
    assert out.shape == (2, 8, 3), f"return (B, H, A) = (2, 8, 3), got {tuple(out.shape)}"
    assert torch.equal(x, original), "do not modify x in place"
    assert len(time_calls) == 1 and torch.equal(time_calls[0], timestep), (
        "call time_mlp once with the timesteps"
    )

    names = [entry[0] for entry in log]
    expected_order = [
        "down0.res1",
        "down0.res2",
        "down1.res1",
        "down1.res2",
        "down2.res1",
        "down2.res2",
        "mid0",
        "mid1",
        "up0.res1",
        "up0.res2",
        "up1.res1",
        "up1.res2",
    ]
    assert names == expected_order, f"blocks must run in order {expected_order}; got {names}"
    expected_cond = torch.cat([time_mlp(timestep), global_cond], dim=-1)
    for name, _, _, cond in log:
        assert torch.allclose(cond, expected_cond), (
            f"{name}: cond must be [time_mlp(timestep), global_cond] concatenated on the last axis"
        )
    first = log[0]
    assert first[1] == 3 and first[2] == 8, (
        "the first block must see channels-first input (B, A, H) = (2, 3, 8)"
    )
    for name, channels, length in (("up0.res1", 8 + 8, 2), ("up1.res1", 6 + 6, 4)):
        entry = log[names.index(name)]
        assert (entry[1], entry[2]) == (channels, length), (
            f"{name} must see [x, newest skip] with {channels} channels at length {length}; "
            f"got {entry[1]} channels at length {entry[2]}"
        )
    _done("channels-first input; one cond, built once, reaches every block")
    _done(
        "the newest skip (saved after res2, before resampling) is joined before the decoder blocks"
    )

    log.clear()
    time_calls.clear()
    unet_forward(unet, x, timestep, None)
    assert all(torch.allclose(entry[3], time_mlp(timestep)) for entry in log), (
        "with global_cond=None, cond is the time vector alone"
    )
    _done("global_cond=None uses the time vector alone")
