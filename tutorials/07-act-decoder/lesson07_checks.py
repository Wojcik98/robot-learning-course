"""Behaviour checks for the Lesson 7 exercises.

These add to the oracle checks in the notebook. They use stand-in attention modules that
record their inputs, and stand-in dropout modules that multiply by known factors, so a
check can see exactly which tensor reached which block and where each norm and dropout
was applied. Read a failing assertion message first: it states the expected behaviour.
"""

import torch
import torch.nn.functional as F
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
    worst = (actual.detach() - expected.detach()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


class _RecordingAttention:
    """Attention stand-in: returns 0.3 q + 0.2 mean_k(k) + 0.5 mean_k(v) and records its inputs.

    Averaging keys and values over their positions keeps the output shaped like the query,
    so it works for cross-attention with a different memory length.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, query, key, value, key_padding_mask=None):
        self.calls.append(
            (query.detach().clone(), key.detach().clone(), value.detach().clone(), key_padding_mask)
        )
        return 0.3 * query + 0.2 * key.mean(1, keepdim=True) + 0.5 * value.mean(1, keepdim=True)


def _stand_in(query, key, value):
    return 0.3 * query + 0.2 * key.mean(1, keepdim=True) + 0.5 * value.mean(1, keepdim=True)


def check_attend_with_positions(attend_with_positions) -> None:
    """Exercise 7.1: every combination of positions, one call, inputs left unchanged."""
    torch.manual_seed(20)
    x, memory = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
    query_pos, key_pos = torch.randn(1, 3, 4), torch.randn(1, 5, 4)
    mask = torch.tensor([[False] * 5, [False, False, True, True, True]])
    originals = (x.clone(), memory.clone())
    for use_q, use_k in ((True, False), (False, True), (True, True)):
        attention = _RecordingAttention()
        out = attend_with_positions(
            attention, x, memory, query_pos if use_q else None, key_pos if use_k else None, mask
        )
        label = f"query_pos={'set' if use_q else 'None'}, key_pos={'set' if use_k else 'None'}"
        assert len(attention.calls) == 1, f"{label}: call attention exactly once"
        q, k, v, passed = attention.calls[0]
        _close(q, x + query_pos if use_q else x, f"{label}: query")
        _close(k, memory + key_pos if use_k else memory, f"{label}: key")
        _close(v, memory, f"{label}: the value must be key_value_input, never with a position")
        assert _same(passed, mask), f"{label}: pass key_padding_mask through unchanged"
        _close(out, _stand_in(q, k, v), f"{label}: return the attention output")
    assert torch.equal(x, originals[0]) and torch.equal(memory, originals[1]), (
        "do not modify the inputs in place (use x + pos, not x += pos)"
    )
    _done("query_pos goes to the query, key_pos to the key, neither to the value")
    _done("one attention call; mask passed through; inputs unchanged")


class _Scale(nn.Module):
    """Stands in for a dropout module: multiplies by a known factor so its position shows."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn, self.cross_attn = _RecordingAttention(), _RecordingAttention()
        self.linear1, self.linear2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        for norm in (self.norm1, self.norm2, self.norm3):  # non-trivial affine parameters
            nn.init.normal_(norm.weight, 1.0, 0.3)
            nn.init.normal_(norm.bias, 0.0, 0.3)
        self.dropout = _Scale(0.9)
        self.dropout1, self.dropout2, self.dropout3 = _Scale(1.1), _Scale(1.2), _Scale(1.3)
        self.activation = F.relu
        self.pre_norm = pre_norm


def check_decoder_layer_forward(decoder_layer_forward) -> None:
    """Exercise 7.2: which tensors reach each block, and where norms and dropouts sit."""
    torch.manual_seed(21)
    x, memory = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    query_pos, memory_pos = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    mask = torch.tensor([[False] * 5, [False, False, False, True, True]])
    for pre_norm in (False, True):
        layer = _DecoderLayer(8, 16, pre_norm)
        out = decoder_layer_forward(layer, x, memory, query_pos, memory_pos, mask)
        name = "pre-norm" if pre_norm else "post-norm"

        def ff(h: torch.Tensor, layer: _DecoderLayer = layer) -> torch.Tensor:
            return layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))

        if pre_norm:
            s_in = layer.norm1(x)
            h1 = x + layer.dropout1(_stand_in(s_in + query_pos, s_in + query_pos, s_in))
            c_in = layer.norm2(h1)
            h2 = h1 + layer.dropout2(_stand_in(c_in + query_pos, memory + memory_pos, memory))
            expected = h2 + layer.dropout3(ff(layer.norm3(h2)))
        else:
            s_in = x
            h1 = layer.norm1(x + layer.dropout1(_stand_in(x + query_pos, x + query_pos, x)))
            c_in = h1
            h2 = layer.norm2(
                h1 + layer.dropout2(_stand_in(h1 + query_pos, memory + memory_pos, memory))
            )
            expected = layer.norm3(h2 + layer.dropout3(ff(h2)))

        assert len(layer.self_attn.calls) == 1 and len(layer.cross_attn.calls) == 1, (
            f"{name}: call self_attn once and cross_attn once"
        )
        q, k, v, passed = layer.self_attn.calls[0]
        _close(
            q,
            s_in + query_pos,
            f"{name}: self-attention query must be {'norm1(x)' if pre_norm else 'x'} + query_pos",
        )
        _close(k, s_in + query_pos, f"{name}: self-attention key must also use query_pos")
        _close(v, s_in, f"{name}: self-attention value has no position")
        assert passed is None, f"{name}: self-attention over the queries takes no padding mask"
        q, k, v, passed = layer.cross_attn.calls[0]
        _close(
            q,
            c_in + query_pos,
            f"{name}: cross-attention query must be the self-attention block's output + query_pos",
        )
        _close(k, memory + memory_pos, f"{name}: cross-attention key must be memory + memory_pos")
        _close(v, memory, f"{name}: cross-attention value must be memory, without a position")
        assert _same(passed, mask), f"{name}: pass memory_key_padding_mask to cross-attention"
        _close(
            out,
            expected,
            f"{name}: output differs; check dropout1/2/3 after each block, dropout inside ff, and "
            "where norm1/2/3 sit relative to each residual add",
        )
    _done(
        "self-attention: query_pos on q and k, no mask; cross-attention: memory_pos on the key, mask passed"
    )
    _done("post-norm and pre-norm: exact placement of the three norms, residuals and dropouts")

    layer = _DecoderLayer(8, 16, False)
    out = decoder_layer_forward(layer, x, memory)
    q, k, v, passed = layer.cross_attn.calls[0]
    _close(k, memory, "without memory_pos the cross-attention key is memory")
    assert passed is None and out.shape == x.shape
    _done("positions and mask may all be None")
