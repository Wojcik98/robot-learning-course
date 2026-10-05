"""Transformer blocks written with plain tensor operations.

Layout: every sequence tensor is batch first, ``(B, T, D)``. Padding masks are
``(B, T)`` booleans where ``True`` means "this position is padding".

The functions named ``*_forward`` and ``scaled_dot_product_attention`` are the
reference answers for course units 7 and 8A. Each module stores a hook, so a
learner function can replace its reference function without editing the package.
The installed LeRobot ACT blocks use ``nn.MultiheadAttention`` and a
sequence-first layout. ``load_lerobot_*`` copies their weights into these modules.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Unpack, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_lab.teaching.blocks import (
    AttentionOperation,
    DecoderForward,
    DecoderHooks,
    DecoderParts,
    EncoderForward,
    EncoderHooks,
    EncoderParts,
    MultiHeadForward,
    ProjectedAttention,
    ScaledAttention,
)

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import (
        ACTDecoder,
        ACTDecoderLayer,
        ACTEncoder,
        ACTEncoderLayer,
    )


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Attention from Vaswani et al. 2017, section 3.2.1.

    Shapes: ``query (B, H, Tq, Dh)``, ``key`` and ``value (B, H, Tk, Dh)``,
    ``key_padding_mask (B, Tk)`` with ``True`` for padding keys. Returns
    ``(output (B, H, Tq, Dh), weights (B, H, Tq, Tk))``. Padding keys get weight 0.
    A query row whose keys are all padding gets all-zero weights and a zero output.
    ``dropout_p`` drops weights during training; the returned weights have no dropout.
    """
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = query @ key.transpose(-2, -1) * scale
    if key_padding_mask is not None:
        blocked = key_padding_mask[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1)
    if key_padding_mask is not None:
        weights = weights.masked_fill(blocked, 0.0)
    used = F.dropout(weights, p=dropout_p, training=True) if dropout_p > 0.0 else weights
    return used @ value, weights


def split_heads(x: Tensor, n_heads: int) -> Tensor:
    """``(B, T, D)`` to ``(B, H, T, D // H)``."""
    batch, length, width = x.shape
    if width % n_heads != 0:
        raise ValueError("model width must be divisible by the number of heads")
    return x.reshape(batch, length, n_heads, width // n_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """``(B, H, T, Dh)`` to ``(B, T, H * Dh)``."""
    batch, heads, length, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_width)


def multi_head_forward(
    attention: ProjectedAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Tensor | None = None,
    attend: ScaledAttention = scaled_dot_product_attention,
) -> tuple[Tensor, Tensor]:
    """Multi-head attention (section 3.2.2): project, split heads, attend, merge, project.

    ``attention`` has ``q_proj``, ``k_proj``, ``v_proj``, ``out_proj`` (``nn.Linear``),
    ``n_heads`` and ``dropout``. Returns ``(output (B, Tq, D), weights (B, H, Tq, Tk))``.
    """
    heads = attention.n_heads
    q = split_heads(attention.q_proj(query), heads)
    k = split_heads(attention.k_proj(key), heads)
    v = split_heads(attention.v_proj(value), heads)
    dropout_p = attention.dropout if attention.training else 0.0
    out, weights = attend(q, k, v, key_padding_mask=key_padding_mask, dropout_p=dropout_p)
    return attention.out_proj(merge_heads(out)), weights


class MultiHeadAttention(nn.Module):
    """Four linear maps and a hook that decides how they are combined."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        forward_fn: MultiHeadForward = multi_head_forward,
        attend_fn: ScaledAttention = scaled_dot_product_attention,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.dropout = d_model, n_heads, dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.forward_fn = forward_fn
        self.attend_fn = attend_fn

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        output, weights = self.forward_fn(self, query, key, value, key_padding_mask, self.attend_fn)
        self.last_weights = weights.detach()
        return output


def sinusoidal_positions(num_positions: int, dimension: int) -> Tensor:
    """Fixed positional encoding (section 3.5), shape ``(num_positions, dimension)``.

    ``angle = pos / 10000 ** (2 * (j // 2) / dimension)``. Even columns use sin and odd
    columns use cos. This is the table the pinned ACT posterior encoder adds to its tokens.
    """
    positions = torch.arange(num_positions, dtype=torch.float64)[:, None]
    columns = torch.arange(dimension, dtype=torch.float64)[None, :]
    angles = positions / 10000 ** (2 * (columns // 2) / dimension)
    table = torch.zeros_like(angles)
    table[:, 0::2] = angles[:, 0::2].sin()
    table[:, 1::2] = angles[:, 1::2].cos()
    return table.float()


def feedforward_activation(name: str) -> Callable[[Tensor], Tensor]:
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"activation must be relu or gelu, not {name}")


def encoder_layer_forward(
    layer: EncoderParts,
    x: Tensor,
    pos_embed: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
) -> Tensor:
    """One encoder layer (section 3.1): attention and feed-forward, each with residual + norm.

    ``x (B, T, D)``; ``pos_embed`` broadcasts to ``(B, T, D)`` and is added to the query and
    key only, never to the value. ``layer`` has ``self_attn``, ``linear1``, ``linear2``,
    ``norm1``, ``norm2``, ``dropout``, ``dropout1``, ``dropout2``, ``activation``, ``pre_norm``.
    Post-norm: ``x = norm1(x + attn(x))``, then ``x = norm2(x + ff(x))``. Pre-norm applies each
    norm before its sub-layer and leaves the residual stream unnormalized.
    """
    skip = x
    if layer.pre_norm:
        x = layer.norm1(x)
    qk = x if pos_embed is None else x + pos_embed
    x = layer.self_attn(qk, qk, x, key_padding_mask)
    x = skip + layer.dropout1(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm2(x)
    else:
        x = layer.norm1(x)
        skip = x
    x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = skip + layer.dropout2(x)
    if not layer.pre_norm:
        x = layer.norm2(x)
    return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        *,
        activation: str = "relu",
        pre_norm: bool = False,
        forward_fn: EncoderForward = encoder_layer_forward,
        attend_fn: ScaledAttention = scaled_dot_product_attention,
        attention_forward_fn: MultiHeadForward = multi_head_forward,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, forward_fn=attention_forward_fn, attend_fn=attend_fn
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = feedforward_activation(activation)
        self.pre_norm = pre_norm
        self.forward_fn = forward_fn

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        return self.forward_fn(self, x, pos_embed, key_padding_mask)


class TransformerEncoder(nn.Module):
    """A stack of encoder layers, then a final LayerNorm when ``pre_norm`` is set."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        *,
        activation: str = "relu",
        pre_norm: bool = False,
        **hooks: Unpack[EncoderHooks],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                    **hooks,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed, key_padding_mask)
        return self.norm(x)


def attend_with_positions(
    attention: AttentionOperation,
    query_input: Tensor,
    key_value_input: Tensor,
    query_pos: Tensor | None = None,
    key_pos: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
) -> Tensor:
    """Attention where positions help the match but are not copied into the result.

    ``query_input (B, Tq, D)`` and ``key_value_input (B, Tk, D)``. ``query_pos`` broadcasts to
    ``(B, Tq, D)`` and ``key_pos`` to ``(B, Tk, D)``; either may be ``None``. The query is
    ``query_input + query_pos``, the key is ``key_value_input + key_pos`` and the value is
    ``key_value_input`` with **no** position. Return ``attention(query, key, value, key_padding_mask)``.
    Self-attention passes the same tensor twice; cross-attention passes encoder memory as the second.
    """
    query = query_input if query_pos is None else query_input + query_pos
    key = key_value_input if key_pos is None else key_value_input + key_pos
    return attention(query, key, key_value_input, key_padding_mask)


def decoder_layer_forward(
    layer: DecoderParts,
    x: Tensor,
    memory: Tensor,
    query_pos: Tensor | None = None,
    memory_pos: Tensor | None = None,
    memory_key_padding_mask: Tensor | None = None,
) -> Tensor:
    """One decoder layer (section 3.1): self-attention, cross-attention, feed-forward.

    ``x (B, Tq, D)`` are the queries, ``memory (B, Tm, D)`` is the encoder output.
    Self-attention uses ``query_pos`` on both sides; cross-attention uses ``query_pos`` on
    the query and ``memory_pos`` on the memory key (``attend_with_positions``). Values never
    receive a position. Each of the three blocks has a residual connection and a norm, placed as
    in ``encoder_layer_forward`` (post-norm by default, ``layer.pre_norm`` for pre-norm), with
    ``norm1``/``dropout1`` after self-attention, ``norm2``/``dropout2`` after cross-attention and
    ``norm3``/``dropout3`` after the feed-forward block.
    Layer attributes: ``self_attn``, ``cross_attn``, ``linear1``, ``linear2``, ``norm1..3``,
    ``dropout``, ``dropout1..3``, ``activation``, ``pre_norm``.
    """
    skip = x
    if layer.pre_norm:
        x = layer.norm1(x)
    x = attend_with_positions(layer.self_attn, x, x, query_pos, query_pos)
    x = skip + layer.dropout1(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm2(x)
    else:
        x = layer.norm1(x)
        skip = x
    x = attend_with_positions(
        layer.cross_attn, x, memory, query_pos, memory_pos, memory_key_padding_mask
    )
    x = skip + layer.dropout2(x)
    if layer.pre_norm:
        skip = x
        x = layer.norm3(x)
    else:
        x = layer.norm2(x)
        skip = x
    x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = skip + layer.dropout3(x)
    if not layer.pre_norm:
        x = layer.norm3(x)
    return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        *,
        activation: str = "relu",
        pre_norm: bool = False,
        forward_fn: DecoderForward = decoder_layer_forward,
        attend_fn: ScaledAttention = scaled_dot_product_attention,
        attention_forward_fn: MultiHeadForward = multi_head_forward,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, forward_fn=attention_forward_fn, attend_fn=attend_fn
        )
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, dropout, forward_fn=attention_forward_fn, attend_fn=attend_fn
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = feedforward_activation(activation)
        self.pre_norm = pre_norm
        self.forward_fn = forward_fn

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        query_pos: Tensor | None = None,
        memory_pos: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_fn(self, x, memory, query_pos, memory_pos, memory_key_padding_mask)


class TransformerDecoder(nn.Module):
    """A stack of decoder layers followed by a LayerNorm (the pinned ACT decoder always norms)."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        *,
        activation: str = "relu",
        pre_norm: bool = False,
        **hooks: Unpack[DecoderHooks],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                    **hooks,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        query_pos: Tensor | None = None,
        memory_pos: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, query_pos, memory_pos, memory_key_padding_mask)
        return self.norm(x)


# --- Weight converters from the installed LeRobot ACT blocks ------------------------------------


def _copy(target: Tensor, source: Tensor, name: str) -> None:
    if target.shape != source.shape:
        raise ValueError(f"{name}: shape {tuple(source.shape)} does not fit {tuple(target.shape)}")
    with torch.no_grad():
        target.copy_(source)


def load_lerobot_attention(target: MultiHeadAttention, source: nn.MultiheadAttention) -> None:
    """Split ``in_proj_weight`` (rows: query, key, value) into three linear layers."""
    if target.d_model != source.embed_dim or target.n_heads != source.num_heads:
        raise ValueError("attention width or head count differs")
    width = target.d_model
    weights = source.in_proj_weight.detach().split(width, dim=0)
    biases = source.in_proj_bias.detach().split(width, dim=0)
    for name, weight, bias in zip(("q_proj", "k_proj", "v_proj"), weights, biases, strict=True):
        _copy(getattr(target, name).weight, weight, f"{name}.weight")
        _copy(getattr(target, name).bias, bias, f"{name}.bias")
    _copy(target.out_proj.weight, source.out_proj.weight.detach(), "out_proj.weight")
    _copy(target.out_proj.bias, source.out_proj.bias.detach(), "out_proj.bias")


def _load_layer_common(target: nn.Module, source: nn.Module, norms: tuple[str, ...]) -> None:
    for name in ("linear1", "linear2", *norms):
        _copy(getattr(target, name).weight, getattr(source, name).weight.detach(), f"{name}.weight")
        _copy(getattr(target, name).bias, getattr(source, name).bias.detach(), f"{name}.bias")


def load_lerobot_encoder_layer(target: EncoderLayer, source: ACTEncoderLayer) -> None:
    load_lerobot_attention(target.self_attn, source.self_attn)
    _load_layer_common(target, source, ("norm1", "norm2"))


def load_lerobot_decoder_layer(target: DecoderLayer, source: ACTDecoderLayer) -> None:
    load_lerobot_attention(target.self_attn, source.self_attn)
    load_lerobot_attention(target.cross_attn, source.multihead_attn)
    _load_layer_common(target, source, ("norm1", "norm2", "norm3"))


def load_lerobot_encoder(target: TransformerEncoder, source: ACTEncoder) -> None:
    """Copy a LeRobot ``ACTEncoder`` into a ``TransformerEncoder`` of the same shape."""
    if len(target.layers) != len(source.layers):
        raise ValueError("encoder layer count differs")
    if type(target.norm) is not type(source.norm):
        raise ValueError("pre_norm setting differs: final norm type differs")
    for learner_layer, reference_layer in zip(target.layers, source.layers, strict=True):
        learner_layer = cast(EncoderLayer, learner_layer)
        reference_layer = cast("ACTEncoderLayer", reference_layer)
        if learner_layer.pre_norm != reference_layer.pre_norm:
            raise ValueError("pre_norm setting differs")
        load_lerobot_encoder_layer(learner_layer, reference_layer)
    if isinstance(target.norm, nn.LayerNorm) and isinstance(source.norm, nn.LayerNorm):
        _copy(target.norm.weight, source.norm.weight.detach(), "norm.weight")
        _copy(target.norm.bias, source.norm.bias.detach(), "norm.bias")


def load_lerobot_decoder(target: TransformerDecoder, source: ACTDecoder) -> None:
    """Copy a LeRobot ``ACTDecoder`` into a ``TransformerDecoder`` of the same shape."""
    if len(target.layers) != len(source.layers):
        raise ValueError("decoder layer count differs")
    for learner_layer, reference_layer in zip(target.layers, source.layers, strict=True):
        learner_layer = cast(DecoderLayer, learner_layer)
        reference_layer = cast("ACTDecoderLayer", reference_layer)
        if learner_layer.pre_norm != reference_layer.pre_norm:
            raise ValueError("pre_norm setting differs")
        load_lerobot_decoder_layer(learner_layer, reference_layer)
    _copy(target.norm.weight, source.norm.weight.detach(), "norm.weight")
    _copy(target.norm.bias, source.norm.bias.detach(), "norm.bias")


def to_sequence_first(x: Tensor) -> Tensor:
    """``(B, T, D)`` to the ``(T, B, D)`` layout used by LeRobot's ``nn.MultiheadAttention``."""
    return x.transpose(0, 1)
