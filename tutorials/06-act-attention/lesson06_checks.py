"""Behaviour checks for the Lesson 6 exercises.

These add to the oracle checks in the notebook. They test the parts an oracle comparison
can miss: the returned weights themselves, numerical stability, which input feeds the
key and which the value, how dropout and masks are passed on, and exactly where each
norm, residual and dropout sits in the encoder layer.
Read a failing assertion message first: it states the expected behaviour.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

import vla_lab.teaching.attention as ref


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


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach() - expected.detach()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def check_scaled_dot_product_attention(scaled_dot_product_attention) -> None:
    """Exercise 6.1: the returned weights, stability, and untouched inputs."""
    torch.manual_seed(10)
    q, k, v = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 6, 5), torch.randn(2, 3, 6, 5)
    pad = torch.tensor([[False] * 6, [False, False, False, True, True, True]])
    originals = [t.clone() for t in (q, k, v, pad)]
    out, weights = scaled_dot_product_attention(q, k, v, pad)
    scores = q @ k.transpose(-2, -1) / math.sqrt(5)
    expected_weights = scores.masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
    _close(
        weights,
        expected_weights,
        "weights must be softmax(q k^T / sqrt(Dh)) over the real keys, 0 on padded keys",
    )
    _close(out, expected_weights @ v, "output must be weights @ value")
    for before, after, name in zip(
        originals, (q, k, v, pad), ("query", "key", "value", "mask"), strict=True
    ):
        assert torch.equal(before, after), f"do not modify {name} in place"
    _done("returns the masked softmax weights themselves, and leaves inputs unchanged")

    # Scale by the feature width per head (Dh), not by the number of keys or heads.
    q1, k1 = torch.randn(1, 1, 3, 16), torch.randn(1, 1, 7, 16)
    _, w1 = scaled_dot_product_attention(q1, k1, torch.randn(1, 1, 7, 16))
    _close(w1, (q1 @ k1.transpose(-2, -1) / 4.0).softmax(-1), "with Dh = 16 the scale is 1 / 4")
    _done("scores are divided by sqrt(Dh), the last axis of query")

    # Large scores must not overflow: a naive exp() without the softmax max-shift gives nan.
    big_q, big_k = q * 40.0, k * 40.0
    big_out, big_w = scaled_dot_product_attention(big_q, big_k, v, pad)
    assert torch.isfinite(big_out).all() and torch.isfinite(big_w).all(), (
        "large scores must not give inf or nan (use softmax, not a hand-written exp)"
    )
    allowed = (~pad)[:, None, None, :].expand(2, 3, 4, 6)
    _close(
        big_out,
        F.scaled_dot_product_attention(big_q, big_k, v, attn_mask=allowed),
        "large scores",
        atol=1e-4,
    )
    _done("stays finite and exact for large scores")

    # Works in float64 too (the blocked score must fit the input dtype).
    out64, w64 = scaled_dot_product_attention(q.double(), k.double(), v.double(), pad)
    assert out64.dtype == torch.float64 and w64.dtype == torch.float64, "keep the input dtype"
    _close(out64.float(), out, "float64 result", atol=1e-5)
    _done("keeps float64 inputs in float64")

    # One-key case: every query must read that key's value exactly.
    single_out, single_w = scaled_dot_product_attention(q, k[:, :, :1], v[:, :, :1])
    _close(single_w, torch.ones(2, 3, 4, 1), "one key gets weight 1")
    _close(single_out, v[:, :, :1].expand(2, 3, 4, 5), "one key: output equals its value")
    _done("a single key gets weight 1")


def check_split_merge_heads(split_heads, merge_heads) -> None:
    """Exercise 6.2: exact index mapping both ways, for any input layout."""
    torch.manual_seed(11)
    x = torch.randn(2, 3, 12)
    heads = split_heads(x, 4)
    assert heads.shape == (2, 4, 3, 3)
    for b in range(2):
        for h in range(4):
            for t in range(3):
                assert torch.equal(heads[b, h, t], x[b, t, 3 * h : 3 * h + 3]), (
                    f"heads[{b}, {h}, {t}] must be x[{b}, {t}, {3 * h}:{3 * h + 3}]"
                )
    _done("split_heads: heads[b, h, t] = x[b, t, h*Dh:(h+1)*Dh]")

    arbitrary = torch.randn(2, 4, 3, 5)  # not produced by split_heads
    merged = merge_heads(arbitrary)
    assert merged.shape == (2, 3, 20)
    for h in range(4):
        assert torch.equal(merged[:, :, 5 * h : 5 * h + 5], arbitrary[:, h]), (
            f"merge_heads must place head {h} at features {5 * h}:{5 * h + 5}"
        )
    assert torch.equal(split_heads(merged, 4), arbitrary), "split_heads must undo merge_heads"
    _done("merge_heads places head h at features h*Dh:(h+1)*Dh, and split undoes it")

    non_contiguous = torch.randn(12, 3, 2).permute(2, 1, 0)  # (2, 3, 12), not contiguous
    assert torch.equal(merge_heads(split_heads(non_contiguous, 3)), non_contiguous), (
        "work on non-contiguous tensors (use reshape, not view)"
    )
    leaf = torch.randn(1, 2, 6, requires_grad=True)
    merge_heads(split_heads(leaf, 2)).sum().backward()
    assert leaf.grad is not None and torch.equal(leaf.grad, torch.ones_like(leaf)), "keep gradients"
    _done("handles non-contiguous inputs and keeps gradients")

    for width, n_heads in ((10, 4), (7, 2)):
        _expect_error(
            lambda width=width, n_heads=n_heads: split_heads(torch.zeros(1, 2, width), n_heads),
            (ValueError,),
            f"split_heads must raise ValueError for D={width}, H={n_heads}",
        )
    _done("raises ValueError when D is not divisible by H")


class _RecordingAttend:
    """An attend function with the Exercise 6.1 contract that records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, query, key, value, key_padding_mask=None, dropout_p=0.0):
        self.calls.append(
            {
                "shapes": (tuple(query.shape), tuple(key.shape), tuple(value.shape)),
                "mask": key_padding_mask,
                "dropout_p": dropout_p,
            }
        )
        return ref.scaled_dot_product_attention(query, key, value, key_padding_mask)


def check_multi_head_forward(multi_head_forward) -> None:
    """Exercise 6.3: separate key and value inputs, mask and dropout passed to attend."""
    torch.manual_seed(12)
    d_model, n_heads = 12, 3
    oracle = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
    attend = _RecordingAttend()
    mine = ref.MultiHeadAttention(d_model, n_heads, forward_fn=multi_head_forward, attend_fn=attend)
    ref.load_lerobot_attention(mine, oracle)
    query, key, value = (
        torch.randn(2, 4, d_model),
        torch.randn(2, 6, d_model),
        torch.randn(2, 6, d_model),
    )
    pad = torch.tensor([[False] * 6, [False] * 4 + [True] * 2])
    out = mine.eval()(query, key, value, pad)
    expected, _ = oracle(query, key, value, key_padding_mask=pad)
    _close(
        out,
        expected,
        "different key and value inputs: output differs from torch (is value projected from `value`?)",
    )
    assert len(attend.calls) == 1, (
        f"call attend exactly once per forward, saw {len(attend.calls)} calls"
    )
    call = attend.calls[0]
    assert call["shapes"] == ((2, 3, 4, 4), (2, 3, 6, 4), (2, 3, 6, 4)), (
        f"attend must receive head-split (B, H, T, Dh) tensors, got {call['shapes']}"
    )
    assert call["mask"] is pad, "pass key_padding_mask to attend unchanged"
    _done("key and value are projected from their own inputs; attend gets split heads and the mask")

    attend.calls.clear()
    mine.dropout = 0.25
    mine.train()(query, key, value, pad)
    mine.eval()(query, key, value, pad)
    used = [c["dropout_p"] for c in attend.calls]
    assert used == [0.25, 0.0], (
        f"pass dropout_p = attention.dropout in training and 0.0 in eval; got {used}"
    )
    _done("dropout_p is attention.dropout in training and 0 in eval")

    attend.calls.clear()
    _, weights = multi_head_forward(mine.eval(), query, key, value, None, attend)
    assert attend.calls[0]["mask"] is None, "with no mask, pass None"
    assert weights.shape == (2, 3, 4, 6), "return attend's weights (B, H, Tq, Tk)"
    _done("returns the head weights from attend")


def check_sinusoidal_positions(sinusoidal_positions) -> None:
    """Exercise 6.4: the formula at small sizes and large positions."""

    def formula(n: int, d: int) -> torch.Tensor:
        p = torch.arange(n, dtype=torch.float64)[:, None]
        j = torch.arange(d, dtype=torch.float64)[None, :]
        angle = p / 10000 ** (2 * (j // 2) / d)
        return torch.where(j % 2 == 0, angle.sin(), angle.cos())

    for n, d in ((1, 2), (5, 2), (3, 6), (50, 10)):
        table = sinusoidal_positions(n, d)
        assert table.shape == (n, d) and table.dtype == torch.float32, f"({n}, {d}) float32 table"
        _close(table.double(), formula(n, d), f"table ({n}, {d})", atol=1e-6)
    _done("small tables match the formula, including dimension 2 and one position")

    table = sinusoidal_positions(2000, 16)
    _close(table.double(), formula(2000, 16), "positions up to 1999", atol=1e-3)
    _done("large positions stay accurate")

    table = sinusoidal_positions(30, 8)
    _close(
        table.square()[:, 0::2] + table.square()[:, 1::2],
        torch.ones(30, 4),
        "sin^2 + cos^2 = 1 per pair",
    )
    _done("each sin/cos column pair shares one frequency")


class _Scale(nn.Module):
    """Stands in for a dropout module: multiplies by a known factor so its position shows."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _RecordingSelfAttention:
    """A self-attention stand-in: returns 0.3 q + 0.2 k + 0.5 v and records its inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, query, key, value, key_padding_mask=None):
        self.calls.append(
            (query.detach().clone(), key.detach().clone(), value.detach().clone(), key_padding_mask)
        )
        return 0.3 * query + 0.2 * key + 0.5 * value


class _Layer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn = _RecordingSelfAttention()
        self.linear1, self.linear2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        for norm in (self.norm1, self.norm2):  # non-trivial affine parameters
            nn.init.normal_(norm.weight, 1.0, 0.3)
            nn.init.normal_(norm.bias, 0.0, 0.3)
        self.dropout, self.dropout1, self.dropout2 = _Scale(0.9), _Scale(1.1), _Scale(1.3)
        self.activation = F.relu
        self.pre_norm = pre_norm


def check_encoder_layer_forward(encoder_layer_forward) -> None:
    """Exercise 6.5: positions on q/k only; norm, residual and dropout in the right places."""
    torch.manual_seed(13)
    x = torch.randn(2, 5, 8)
    pos = torch.randn(1, 5, 8)
    pad = torch.tensor([[False] * 5, [False, False, False, True, True]])
    for pre_norm in (False, True):
        layer = _Layer(8, 16, pre_norm)
        out = encoder_layer_forward(layer, x, pos, pad)

        def ff(h: torch.Tensor, layer: _Layer = layer) -> torch.Tensor:
            return layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))

        if pre_norm:
            normed = layer.norm1(x)
            attn_in = (normed + pos, normed + pos, normed)
            h = x + layer.dropout1(0.3 * (normed + pos) + 0.2 * (normed + pos) + 0.5 * normed)
            expected = h + layer.dropout2(ff(layer.norm2(h)))
        else:
            attn_in = (x + pos, x + pos, x)
            h = layer.norm1(x + layer.dropout1(0.3 * (x + pos) + 0.2 * (x + pos) + 0.5 * x))
            expected = layer.norm2(h + layer.dropout2(ff(h)))
        name = "pre-norm" if pre_norm else "post-norm"
        assert len(layer.self_attn.calls) == 1, f"{name}: call self_attn exactly once"
        query, key, value, mask = layer.self_attn.calls[0]
        _close(
            query,
            attn_in[0],
            f"{name}: the query must be {'norm1(x)' if pre_norm else 'x'} + pos_embed",
        )
        _close(
            key,
            attn_in[1],
            f"{name}: the key must be {'norm1(x)' if pre_norm else 'x'} + pos_embed",
        )
        _close(
            value,
            attn_in[2],
            f"{name}: the value must be {'norm1(x)' if pre_norm else 'x'}, without pos_embed",
        )
        assert mask is pad, f"{name}: pass key_padding_mask to self_attn"
        _close(
            out,
            expected,
            f"{name}: output differs; check dropout1 on attention, dropout inside ff, dropout2 on ff, "
            "and where norm1/norm2 sit relative to each residual add",
        )
    _done("post-norm and pre-norm: exact placement of norms, residuals and the three dropouts")
    _done("pos_embed is added to the query and key only; the mask reaches self_attn")

    layer = _Layer(8, 16, False)
    out = encoder_layer_forward(layer, x, None, None)
    query, key, value, mask = layer.self_attn.calls[0]
    _close(query, x, "without pos_embed the query is x")
    _close(value, x, "without pos_embed the value is x")
    assert mask is None and out.shape == x.shape
    _done("pos_embed=None and key_padding_mask=None are handled")
