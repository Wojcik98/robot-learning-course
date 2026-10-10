"""Behaviour checks for the Lesson 16 exercise.

The notebook checks constant white images. These checks use images with varying content
and odd sizes, so they test the resize itself, where the padding goes, and the [-1, 1]
mapping. ``encode_instruction`` is driven by a recording tokenizer that shows exactly
which texts and options it received. Read a failing assertion message first.
"""

import torch
import torch.nn.functional as F


def _done(message: str) -> None:
    print(f"✓ {message}")


def _close(actual: torch.Tensor, expected: torch.Tensor, message: str, atol: float = 1e-5) -> None:
    assert actual.shape == expected.shape, (
        f"{message}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    )
    worst = (actual.detach() - expected.detach()).abs().max().item()
    assert worst <= atol, f"{message}: largest difference {worst:.2e} is above {atol:.0e}"


def _expected_fit(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    h, w = image.shape[-2:]
    ratio = max(w / width, h / height)
    new_h, new_w = int(h / ratio), int(w / ratio)
    resized = image
    if (new_h, new_w) != (h, w):
        resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = torch.zeros(image.shape[0], image.shape[1], height, width)
    out[:, :, height - new_h :, width - new_w :] = resized
    return out * 2 - 1


def check_fit_image(fit_image) -> None:
    """Exercise 16.1: aspect-preserving bilinear resize, top/left padding, then 2p - 1."""
    generator = torch.Generator().manual_seed(90)
    for shape, out_size, note in (
        ((2, 3, 30, 40), (24, 24), "a wide image is shrunk to 18x24 with 6 padding rows on top"),
        (
            (1, 3, 50, 20),
            (40, 40),
            "a tall image is shrunk to 40x16 with 24 padding columns on the left",
        ),
        ((1, 3, 8, 16), (32, 32), "a small image is enlarged to 16x32 with 16 padding rows on top"),
        ((1, 3, 12, 12), (12, 12), "an image of the output size is only normalized"),
    ):
        image = torch.rand(shape, generator=generator)
        original = image.clone()
        out = fit_image(image, *out_size)
        _close(out, _expected_fit(image, *out_size), f"{tuple(shape)} -> {out_size}: {note}")
        assert torch.equal(image, original), "do not modify the input image in place"
        assert float(out.min()) >= -1 - 1e-6 and float(out.max()) <= 1 + 1e-6, (
            "values must stay in [-1, 1]"
        )
    _done("bilinear resize that keeps the aspect ratio; padding on top or left becomes -1")

    stripes = torch.zeros(1, 3, 16, 32)
    stripes[:, 0] = 1.0  # red channel only
    out = fit_image(stripes, 32, 32)
    assert out[:, 0, 16:].eq(1).all() and out[:, 1:, 16:].eq(-1).all(), (
        "each channel is resized on its own (red stays 1, green and blue stay -1)"
    )
    _done("channels are not mixed")


class _RecordingTokenizer:
    """Tokenizer stand-in: one ID per word (its length), int32 IDs and an int64 mask."""

    def __init__(self) -> None:
        self.padding_side = "left"  # a real tokenizer may default to left padding
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs):
        side = kwargs.get("padding_side", self.padding_side)
        self.calls.append({"texts": texts, "side": side, **kwargs})
        limit = kwargs.get("max_length") if kwargs.get("truncation") else None
        rows = [[len(word) for word in text.split()][:limit] for text in texts]
        width = max(len(row) for row in rows)
        ids, masks = [], []
        for row in rows:
            pad = width - len(row)
            if side == "right":
                ids.append(row + [0] * pad)
                masks.append([1] * len(row) + [0] * pad)
            else:
                ids.append([0] * pad + row)
                masks.append([0] * pad + [1] * len(row))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.int32),
            "attention_mask": torch.tensor(masks, dtype=torch.int64),
        }


def check_encode_instruction(encode_instruction) -> None:
    """Exercise 16.1: newline once, right padding, truncation at max_length, long/bool outputs."""
    tokenizer = _RecordingTokenizer()
    ids, mask = encode_instruction(tokenizer, ["pick up the bowl", "stop\n"], 3)
    call = tokenizer.calls[-1]
    assert list(call["texts"]) == ["pick up the bowl\n", "stop\n"], (
        f"add a final newline only when it is missing; the tokenizer received {list(call['texts'])}"
    )
    assert call["side"] == "right", "pad on the right"
    assert call.get("truncation") is True and call.get("max_length") == 3, "truncate at max_length"
    assert ids.dtype == torch.long and mask.dtype == torch.bool, "return long IDs and a bool mask"
    assert ids.tolist() == [[4, 2, 3], [4, 0, 0]], (
        f"IDs must be truncated and right-padded; got {ids.tolist()}"
    )
    assert mask.tolist() == [[True, True, True], [True, False, False]], "True marks real tokens"
    _done("adds one final newline, pads on the right, truncates at max_length")

    tokenizer = _RecordingTokenizer()
    ids, mask = encode_instruction(tokenizer, "open the drawer", 10)
    assert list(tokenizer.calls[-1]["texts"]) == ["open the drawer\n"], (
        "a single string prompt becomes a list of one string"
    )
    assert ids.shape == (1, 3) and mask.shape == (1, 3), "a single prompt gives a batch of one"
    _done("a single string prompt is treated as a batch of one")
