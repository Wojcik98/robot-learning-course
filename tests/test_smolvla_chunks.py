"""Check the SmolVLA chunk adapter without loading model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_lab.runtime.smolvla_chunks import SmolVLAChunkPredictor


class FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(chunk_size=6, max_action_dim=8, device="cpu", num_steps=5)
        self.requests: list[dict[str, object]] = []

    def predict_action_chunk(self, _batch, *, noise, **kwargs):
        self.requests.append({"noise": noise.clone(), **kwargs})
        return torch.ones((1, 6, 7), dtype=torch.float32)


def test_chunk_predictor_uses_same_noise_for_same_tick_and_physical_processor(monkeypatch) -> None:
    monkeypatch.setattr(
        "vla_lab.runtime.smolvla_chunks.prepare_policy_batch",
        lambda _observation, _prompt, _processor: {"observation.state": torch.zeros(1, 8)},
    )
    policy = FakePolicy()
    predictor = SmolVLAChunkPredictor(
        policy,
        preprocessor=object(),
        postprocessor=lambda action: action * 0.25,
        task_prompt="put the bowl on the plate",
        mode="naive_async",
        stride=2,
        delay=1,
        noise_seed=51,
    )

    normalized, physical = predictor({}, 4, None, None)
    predictor({}, 4, None, None)

    assert normalized.shape == physical.shape == (6, 7)
    assert np.allclose(normalized, 1)
    assert np.allclose(physical, 0.25)
    assert torch.equal(policy.requests[0]["noise"], policy.requests[1]["noise"])
    assert policy.requests[0]["noise"].shape == (1, 6, 8)
    assert "prev_chunk_left_over" not in policy.requests[0]


@pytest.mark.parametrize("mode", ["rtc_hard", "rtc_soft"])
def test_rtc_predictor_passes_aligned_previous_normalized_actions(monkeypatch, mode) -> None:
    monkeypatch.setattr(
        "vla_lab.runtime.smolvla_chunks.prepare_policy_batch",
        lambda _observation, _prompt, _processor: {"observation.state": torch.zeros(1, 8)},
    )
    policy = FakePolicy()
    predictor = SmolVLAChunkPredictor(
        policy,
        preprocessor=object(),
        postprocessor=lambda action: action,
        task_prompt="put the bowl on the plate",
        mode=mode,
        stride=2,
        delay=1,
        noise_seed=51,
    )
    previous = np.full((4, 7), 0.5, dtype=np.float32)

    predictor({}, 2, previous, 0)

    request = policy.requests[0]
    assert request["inference_delay"] == 1
    assert request["execution_horizon"] == 2
    assert torch.allclose(request["prev_chunk_left_over"], torch.from_numpy(previous)[None])
