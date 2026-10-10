"""Behaviour checks for the Lesson 8 exercises.

These add to the oracle checks in the notebook. Each check builds a small stand-in model
from plain modules and recording callables, so it can see exactly which tokens, masks
and positions your function built and passed on, and can test cases the notebook's real
model does not reach (other sizes, other dtypes, cameras of different sizes).
Read a failing assertion message first: it states the expected behaviour.
"""

import math
from types import SimpleNamespace

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


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach().double() - expected.detach().double()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def check_camera_position_embedding_2d(camera_position_embedding_2d) -> None:
    """Exercise 8.1: the formula for non-square grids, row half first, column half second."""

    def axis_code(count: int, per_axis: int) -> torch.Tensor:
        index = torch.arange(1, count + 1, dtype=torch.float64)
        scaled = index / (count + 1e-6) * 2 * math.pi
        j = torch.arange(per_axis, dtype=torch.float64)
        angle = scaled[:, None] / 10000 ** (2 * (j // 2) / per_axis)
        return torch.where(j % 2 == 0, angle.sin(), angle.cos())  # (count, per_axis)

    for height, width, dim in ((1, 1, 4), (2, 7, 8), (6, 3, 12), (4, 4, 4)):
        table = camera_position_embedding_2d(height, width, dim)
        assert table.shape == (1, dim, height, width), f"({height}, {width}, {dim}): shape"
        assert table.dtype == torch.float32, "return float32"
        d = dim // 2
        rows, columns = axis_code(height, d), axis_code(width, d)
        expected = torch.cat(
            (
                rows[:, None, :].expand(height, width, d),
                columns[None, :, :].expand(height, width, d),
            ),
            dim=-1,
        ).permute(2, 0, 1)[None]
        _close(
            table,
            expected,
            f"{height}x{width}, dim {dim}: channels 0..d-1 encode the row (1..height), "
            "d..dim-1 the column (1..width), sin on even and cos on odd channels",
            atol=1e-5,
        )
    _done("matches the formula for 1x1, wide, tall and square grids")

    tall = camera_position_embedding_2d(5, 2, 8)
    wide = camera_position_embedding_2d(2, 5, 8)
    _close(
        tall[0, :4, :, 0],
        wide[0, 4:, 0, :],
        "the row code of a 5x2 grid equals the column code of a 2x5 grid",
    )
    _done("row and column halves are not swapped")


def _posterior_model(state_dim: int, action_dim: int, d_model: int, latent: int, chunk: int):
    """Plain modules with the attribute names of the real model, and a recording encoder."""
    torch.manual_seed(30)
    calls: list[dict] = []

    def vae_encoder(tokens, pos_embed, key_padding_mask):
        calls.append(
            {"tokens": tokens.detach().clone(), "pos": pos_embed, "mask": key_padding_mask}
        )
        # Output row i = token i * (i + 1): reading the wrong position gives a different value.
        scale = torch.arange(1, tokens.shape[1] + 1, dtype=tokens.dtype)[None, :, None]
        return tokens * scale

    model = SimpleNamespace(
        vae_cls_embed=nn.Embedding(1, d_model),
        vae_state_proj=nn.Linear(state_dim, d_model),
        vae_action_proj=nn.Linear(action_dim, d_model),
        vae_encoder=vae_encoder,
        vae_pos_embed=torch.randn(1, chunk + 2, d_model),
        vae_latent_proj=nn.Linear(d_model, 2 * latent),
        training=True,
    )
    return model, calls


def check_encode_posterior(encode_posterior) -> None:
    """Exercise 8.2: [cls, state, actions] tokens, the mask, position 0, then mu | logvar."""
    model, calls = _posterior_model(state_dim=5, action_dim=3, d_model=8, latent=2, chunk=4)
    state, actions = torch.randn(2, 5), torch.randn(2, 4, 3)
    pad = torch.tensor([[False, False, False, False], [False, True, True, True]])
    mu, logvar = encode_posterior(model, state, actions, pad)
    assert len(calls) == 1, "call vae_encoder exactly once"
    tokens = calls[0]["tokens"]
    assert tokens.shape == (2, 6, 8), (
        f"tokens must be (B, T + 2, D) = (2, 6, 8), got {tuple(tokens.shape)}"
    )
    _close(
        tokens[:, 0],
        model.vae_cls_embed.weight.expand(2, 8),
        "token 0 must be the CLS embedding, for every example",
    )
    _close(tokens[:, 1], model.vae_state_proj(state), "token 1 must be vae_state_proj(state)")
    _close(
        tokens[:, 2:],
        model.vae_action_proj(actions),
        "tokens 2.. must be vae_action_proj(actions), in order",
    )
    assert _same(calls[0]["pos"], model.vae_pos_embed), "pass model.vae_pos_embed as the position"
    mask = calls[0]["mask"]
    assert mask.dtype == torch.bool, "the key padding mask must be bool"
    assert mask.tolist() == [[False] * 6, [False, False, False, True, True, True]], (
        f"the mask must be [False, False, *action_is_pad]; got {mask.tolist()}"
    )
    _done(
        "tokens are [cls, state, action_1..T]; mask is [False, False, *action_is_pad]; pos is vae_pos_embed"
    )

    projected = model.vae_latent_proj(tokens[:, 0] * 1)  # encoder output row 0 = token 0 * 1
    _close(
        mu, projected[:, :2], "mu must be the first L outputs of vae_latent_proj(encoder_out[:, 0])"
    )
    _close(logvar, projected[:, 2:], "logvar must be the last L outputs")
    _done("reads the CLS output (position 0) and splits mu first, logvar second")


def check_select_latent(select_latent) -> None:
    """Exercise 8.3: dtype and device of the zero latent, and no posterior work at inference."""
    mu = torch.tensor([[0.5, -1.0, 2.0]])
    logvar = torch.tensor([[0.0, math.log(4.0), -2.0]])
    calls: list[tuple] = []

    def posterior(model, state, actions, action_is_pad):
        calls.append((state, actions, action_is_pad))
        return mu, logvar

    model = SimpleNamespace(
        training=True,
        settings=SimpleNamespace(latent_dim=3),
        hooks=SimpleNamespace(posterior=posterior),
    )
    batch = {
        "observation.state": torch.randn(1, 4),
        "action": torch.randn(1, 5, 2),
        "action_is_pad": torch.zeros(1, 5, dtype=torch.bool),
    }
    noise = torch.tensor([[1.0, 0.5, -1.0]])
    z, got_mu, got_logvar = select_latent(model, batch, noise)
    assert len(calls) == 1, "call the posterior hook once in training"
    state, actions, is_pad = calls[0]
    assert (
        _same(state, batch["observation.state"])
        and _same(actions, batch["action"])
        and _same(is_pad, batch["action_is_pad"])
    ), "pass the batch's state, action and action_is_pad to the posterior hook"
    _close(z, torch.tensor([[1.5, 0.0, 2.0 - math.exp(-1.0)]]), "z = mu + exp(logvar / 2) * noise")
    assert _same(got_mu, mu) and _same(got_logvar, logvar), "return the hook's mu and logvar"
    _done("training: z = mu + exp(logvar / 2) * noise from the hook's outputs")

    calls.clear()
    model.training = False
    state64 = {
        "observation.state": torch.randn(4, 4, dtype=torch.float64),
        **{k: v for k, v in batch.items() if k != "observation.state"},
    }
    rng_before = torch.get_rng_state()
    z, got_mu, got_logvar = select_latent(model, state64)
    assert not calls, "evaluation mode must not run the posterior"
    assert torch.equal(torch.get_rng_state(), rng_before), (
        "evaluation mode must not draw random numbers"
    )
    assert z.shape == (4, 3) and z.dtype == torch.float64, "zeros (B, L) with the state's dtype"
    assert not z.any() and got_mu is None and got_logvar is None
    _done("evaluation: zeros (B, L) in the state's dtype, no posterior call, no random numbers")


def check_build_encoder_tokens(build_encoder_tokens) -> None:
    """Exercise 8.4: cameras of different sizes, one hook call each, positions in feature dtype."""
    torch.manual_seed(31)
    d_model = 6
    seen: list[tuple[int, int, int]] = []

    def camera_positions(height: int, width: int, dim: int) -> torch.Tensor:
        seen.append((height, width, dim))
        # float64 on purpose: your function must convert it to the feature dtype.
        values = torch.arange(dim * height * width, dtype=torch.float64).reshape(
            1, dim, height, width
        )
        return values / 100 + 1000 * len(seen)

    model = SimpleNamespace(
        latent_proj=nn.Linear(3, d_model),
        state_proj=nn.Linear(4, d_model),
        camera_proj=nn.Conv2d(5, d_model, 1),
        pos_1d=nn.Embedding(2, d_model),
        settings=SimpleNamespace(dim_model=d_model),
        hooks=SimpleNamespace(camera_positions=camera_positions),
    )
    latent, state = torch.randn(2, 3), torch.randn(2, 4)
    maps = [torch.randn(2, 5, 2, 3), torch.randn(2, 5, 3, 1)]
    tokens, pos = build_encoder_tokens(model, latent, state, maps)
    assert tokens.shape == (2, 2 + 6 + 3, d_model) and pos.shape == (1, 11, d_model), (
        f"expected tokens (2, 11, {d_model}) and pos (1, 11, {d_model}) for 2x3 and 3x1 cameras"
    )
    assert seen == [(2, 3, d_model), (3, 1, d_model)], (
        f"call camera_positions once per camera with that camera's (h, w, D); saw {seen}"
    )
    assert pos.dtype == torch.float32, "convert camera positions to the features' dtype"
    _close(tokens[:, 0], model.latent_proj(latent), "token 0 is latent_proj(latent)")
    _close(tokens[:, 1], model.state_proj(state), "token 1 is state_proj(state)")
    _close(pos[0, :2], model.pos_1d.weight, "positions 0 and 1 are the rows of pos_1d")
    offset = 2
    for index, features in enumerate(maps):
        projected = model.camera_proj(features)  # (B, D, h, w)
        cam_pos = torch.arange(
            d_model * features.shape[2] * features.shape[3], dtype=torch.float64
        ).reshape(1, d_model, features.shape[2], features.shape[3]) / 100 + 1000 * (index + 1)
        cells = features.shape[2] * features.shape[3]
        _close(
            tokens[:, offset : offset + cells],
            projected.flatten(2).transpose(1, 2),
            f"camera {index + 1}: tokens must be its h*w cells, row by row, after the earlier tokens",
        )
        _close(
            pos[:, offset : offset + cells],
            cam_pos.flatten(2).transpose(1, 2),
            f"camera {index + 1}: positions must be its own hook output, flattened row by row",
            atol=1e-3,
        )
        offset += cells
    _done("each camera uses its own size and positions, in order, row by row")


class _RecordingDecoder(nn.Module):
    def forward(self, x, memory, query_pos=None, memory_pos=None, memory_key_padding_mask=None):
        self.args = (x, memory, query_pos, memory_pos)
        return x + memory.mean(1, keepdim=True) + query_pos


def check_decode_actions(decode_actions) -> None:
    """Exercise 8.5: zero queries in the memory's dtype, positions passed, head applied."""
    torch.manual_seed(32)
    model = SimpleNamespace(
        settings=SimpleNamespace(chunk_size=4, dim_model=6),
        decoder=_RecordingDecoder(),
        query_embed=nn.Embedding(4, 6).double(),
        action_head=nn.Linear(6, 3).double(),
    )
    memory, pos = (
        torch.randn(2, 5, 6, dtype=torch.float64),
        torch.randn(1, 5, 6, dtype=torch.float64),
    )
    actions = decode_actions(model, memory, pos)
    x, got_memory, query_pos, memory_pos = model.decoder.args
    assert x.shape == (2, 4, 6) and x.dtype == torch.float64 and not x.any(), (
        "the decoder input must be zeros (B, T, D) with the encoder output's dtype"
    )
    assert _same(got_memory, memory) and _same(memory_pos, pos), (
        "pass encoder_out as memory and pos as memory_pos"
    )
    _close(query_pos, model.query_embed.weight[None], "query_pos must be query_embed.weight[None]")
    expected = model.action_head(memory.mean(1, keepdim=True) + model.query_embed.weight[None])
    _close(actions, expected, "return action_head(decoder output)")
    _done("zero queries (B, T, D) in the memory dtype; learned query positions; head on the output")
