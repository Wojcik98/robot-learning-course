"""Check the hand-written attention blocks against torch and the pinned LeRobot ACT blocks."""

import pytest
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    create_sinusoidal_pos_embedding,
)
from torch import nn

from vla_lab.teaching.attention import (
    DecoderLayer,
    EncoderLayer,
    MultiHeadAttention,
    TransformerDecoder,
    TransformerEncoder,
    attend_with_positions,
    load_lerobot_attention,
    load_lerobot_decoder,
    load_lerobot_encoder,
    merge_heads,
    scaled_dot_product_attention,
    sinusoidal_positions,
    split_heads,
    to_sequence_first,
)

D, H, FF = 16, 4, 32


def _padding(batch=3, length=6):
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[1, 4:] = True
    mask[2, 2:] = True
    return mask


def test_sdpa_matches_torch_with_padding_mask():
    torch.manual_seed(0)
    q, k, v = (torch.randn(3, H, n, D // H) for n in (5, 6, 6))
    mask = _padding()
    out, weights = scaled_dot_product_attention(q, k, v, mask)
    allowed = (~mask)[:, None, None, :].expand(3, H, 5, 6)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(weights.sum(-1), torch.ones(3, H, 5), atol=1e-6)
    assert (
        torch.count_nonzero(weights.masked_select(mask[:, None, None, :].expand_as(weights))) == 0
    )


def test_sdpa_gradient_matches_torch_and_padded_keys_get_no_gradient():
    torch.manual_seed(1)
    q, k, v = (torch.randn(3, H, n, D // H, requires_grad=True) for n in (5, 6, 6))
    mask = _padding()
    out, _ = scaled_dot_product_attention(q, k, v, mask)
    out.square().sum().backward()
    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    allowed = (~mask)[:, None, None, :].expand(3, H, 5, 6)
    expected = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, attn_mask=allowed)
    expected.square().sum().backward()
    for mine, theirs in ((q, q2), (k, k2), (v, v2)):
        assert torch.allclose(mine.grad, theirs.grad, atol=1e-5)
    padded = mask[:, None, :, None]
    assert torch.count_nonzero(k.grad.masked_select(padded.expand_as(k.grad))) == 0
    assert torch.count_nonzero(v.grad.masked_select(padded.expand_as(v.grad))) == 0


def test_sdpa_all_padding_row_is_finite_and_zero():
    q, k, v = (torch.randn(2, H, 3, D // H, requires_grad=True) for _ in range(3))
    mask = torch.tensor([[True, True, True], [False, False, True]])
    out, _weights = scaled_dot_product_attention(q, k, v, mask)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.count_nonzero(out[0]) == 0
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


def test_split_and_merge_heads_round_trip():
    x = torch.randn(2, 5, D)
    heads = split_heads(x, H)
    assert heads.shape == (2, H, 5, D // H)
    assert torch.equal(merge_heads(heads), x)
    assert torch.equal(heads[1, 2, 3], x[1, 3, 2 * (D // H) : 3 * (D // H)])


def test_multihead_matches_torch_output_and_gradient():
    torch.manual_seed(2)
    reference = nn.MultiheadAttention(D, H, dropout=0.0, batch_first=True)
    mine = MultiHeadAttention(D, H)
    load_lerobot_attention(mine, reference)
    x = torch.randn(3, 6, D, requires_grad=True)
    memory = torch.randn(3, 6, D)
    mask = _padding()
    out = mine(x, memory, memory, mask)
    expected, _ = reference(x, memory, memory, key_padding_mask=mask)
    assert torch.allclose(out, expected, atol=1e-5)
    out.square().sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    reference(x2, memory, memory, key_padding_mask=mask)[0].square().sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-4)
    packed = torch.cat([mine.q_proj.weight.grad, mine.k_proj.weight.grad, mine.v_proj.weight.grad])
    assert torch.allclose(packed, reference.in_proj_weight.grad, atol=1e-4)


def test_sinusoidal_positions_match_lerobot_table():
    for length, width in ((7, 16), (21, 32)):
        assert torch.allclose(
            sinusoidal_positions(length, width), create_sinusoidal_pos_embedding(length, width)
        )


def _config(pre_norm):
    return ACTConfig(
        dim_model=D,
        n_heads=H,
        dim_feedforward=FF,
        n_encoder_layers=2,
        n_decoder_layers=2,
        pre_norm=pre_norm,
        dropout=0.0,
        chunk_size=4,
        n_action_steps=4,
        input_features={},
        output_features={},
    )


@pytest.mark.parametrize("pre_norm", [False, True])
def test_encoder_matches_lerobot_with_pos_embed_mask_and_gradient(pre_norm):
    torch.manual_seed(3)
    reference = ACTEncoder(_config(pre_norm)).eval()
    mine = TransformerEncoder(2, D, H, FF, pre_norm=pre_norm).eval()
    load_lerobot_encoder(mine, reference)
    x = torch.randn(3, 6, D, requires_grad=True)
    pos = torch.randn(1, 6, D)
    mask = _padding()
    out = mine(x, pos, mask)
    expected = reference(
        to_sequence_first(x),
        pos_embed=to_sequence_first(pos.expand(3, -1, -1)),
        key_padding_mask=mask,
    ).transpose(0, 1)
    valid = ~mask[..., None]
    assert torch.allclose(out * valid, expected * valid, atol=1e-5)
    (out * valid).square().sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    (
        reference(
            to_sequence_first(x2),
            pos_embed=to_sequence_first(pos.expand(3, -1, -1)),
            key_padding_mask=mask,
        ).transpose(0, 1)
        * valid
    ).square().sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-4)


def test_encoder_output_at_valid_positions_ignores_padded_values():
    torch.manual_seed(4)
    encoder = TransformerEncoder(2, D, H, FF).eval()
    x = torch.randn(3, 6, D)
    mask = _padding()
    changed = x.clone()
    changed[mask] = 100.0
    valid = ~mask[..., None]
    assert torch.allclose(
        encoder(x, None, mask) * valid, encoder(changed, None, mask) * valid, atol=1e-5
    )
    assert not torch.allclose(encoder(x) * valid, encoder(changed) * valid, atol=1e-2)


@pytest.mark.parametrize("pre_norm", [False, True])
def test_decoder_matches_lerobot_cross_attention(pre_norm):
    torch.manual_seed(5)
    reference = ACTDecoder(_config(pre_norm)).eval()
    mine = TransformerDecoder(2, D, H, FF, pre_norm=pre_norm).eval()
    load_lerobot_decoder(mine, reference)
    queries = torch.zeros(3, 4, D)
    memory = torch.randn(3, 7, D, requires_grad=True)
    query_pos, memory_pos = torch.randn(1, 4, D), torch.randn(1, 7, D)
    out = mine(queries, memory, query_pos, memory_pos)
    expected = reference(
        to_sequence_first(queries),
        to_sequence_first(memory),
        decoder_pos_embed=to_sequence_first(query_pos),
        encoder_pos_embed=to_sequence_first(memory_pos),
    ).transpose(0, 1)
    assert torch.allclose(out, expected, atol=1e-5)
    out.square().sum().backward()
    memory2 = memory.detach().clone().requires_grad_(True)
    reference(
        to_sequence_first(queries),
        to_sequence_first(memory2),
        decoder_pos_embed=to_sequence_first(query_pos),
        encoder_pos_embed=to_sequence_first(memory_pos),
    ).transpose(0, 1).square().sum().backward()
    assert torch.allclose(memory.grad, memory2.grad, atol=1e-4)


def test_hooks_replace_the_reference_function_in_the_real_forward_path():
    calls = []

    def counting_attend(*args, **kwargs):
        calls.append("attend")
        return scaled_dot_product_attention(*args, **kwargs)

    layer = EncoderLayer(D, H, FF, attend_fn=counting_attend)
    layer(torch.randn(2, 3, D))
    assert calls == ["attend"]
    decoder = DecoderLayer(D, H, FF, attend_fn=counting_attend)
    decoder(torch.randn(2, 3, D), torch.randn(2, 5, D))
    assert calls == ["attend"] * 3


def test_load_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        load_lerobot_attention(MultiHeadAttention(D, H), nn.MultiheadAttention(D * 2, H))


def test_attend_with_positions_adds_positions_to_query_and_key_only():
    torch.manual_seed(8)
    attention = MultiHeadAttention(D, H)
    x, memory = torch.randn(2, 4, D), torch.randn(2, 6, D)
    query_pos, key_pos = torch.randn(1, 4, D), torch.randn(1, 6, D)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 4:] = True
    out = attend_with_positions(attention, x, memory, query_pos, key_pos, mask)
    assert torch.allclose(out, attention(x + query_pos, memory + key_pos, memory, mask))
    assert torch.allclose(attend_with_positions(attention, x, x), attention(x, x, x))
    shifted = attend_with_positions(attention, x, memory, None, key_pos)
    assert not torch.allclose(shifted, attend_with_positions(attention, x, memory))
