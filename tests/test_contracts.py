"""Reject malformed packets before a policy can control the simulator."""

import numpy as np
import pytest

from vla_lab.contracts import (
    ActionChunk,
    ActionSchema,
    EpisodeOutcome,
    ObservationPacket,
    PolicyCapabilities,
    PolicyRequest,
)


def _observation() -> ObservationPacket:
    return ObservationPacket(
        episode_id="episode-1",
        reset_counter=0,
        sequence=0,
        observation_tick=0,
        capture_time_s=1.0,
        cameras={
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        state=np.zeros(8, dtype=np.float32),
        state_schema=tuple(f"state_{index}" for index in range(8)),
        task_id="bowl_plate",
    )


def test_observation_rejects_swapped_or_missing_camera_order() -> None:
    valid = _observation()
    with pytest.raises(ValueError, match="camera order"):
        ObservationPacket(**{**vars(valid), "cameras": dict(reversed(list(valid.cameras.items())))})
    with pytest.raises(ValueError, match="camera order"):
        ObservationPacket(
            **{**vars(valid), "cameras": {"agentview_image": valid.cameras["agentview_image"]}}
        )


def test_observation_rejects_bad_state_and_image() -> None:
    valid = _observation()
    state = valid.state.copy()
    state[0] = np.nan
    with pytest.raises(ValueError, match="state"):
        ObservationPacket(**{**vars(valid), "state": state})
    with pytest.raises(ValueError, match="camera"):
        ObservationPacket(
            **{
                **vars(valid),
                "cameras": {**valid.cameras, "robot0_eye_in_hand_image": np.zeros((2, 2, 1))},
            }
        )


def test_policy_request_rejects_unsupported_language_and_horizon() -> None:
    observation = _observation()
    capabilities = PolicyCapabilities(False, False, True, False, 20)
    request = PolicyRequest(observation, "act-reference", "request-1", 20, 10)
    capabilities.validate(request)
    with pytest.raises(ValueError, match="trained horizon"):
        capabilities.validate(PolicyRequest(observation, "act-reference", "request-1", 16, 10))
    with pytest.raises(ValueError, match="language"):
        capabilities.validate(
            PolicyRequest(
                ObservationPacket(**{**vars(observation), "instruction": "put bowl"}),
                "act-reference",
                "request-1",
                20,
                10,
            )
        )


def test_action_chunk_rejects_nonfinite_and_out_of_range_actions() -> None:
    schema = ActionSchema.libero_relative()
    base = {
        "episode_id": "episode-1",
        "reset_counter": 0,
        "request_id": "request-1",
        "observation_tick": 0,
        "availability_tick": 0,
        "controller_period_ms": 50,
        "schema": schema,
    }
    with pytest.raises(ValueError, match="finite"):
        ActionChunk(**base, actions=np.full((2, 7), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="bounds"):
        ActionChunk(**base, actions=np.full((2, 7), 2.0, dtype=np.float32))


def test_episode_result_keeps_failure_category() -> None:
    failed = EpisodeOutcome(
        "bowl_plate",
        3,
        "development",
        "act-v1",
        "libero-v1",
        False,
        "policy_error",
        {"steps": 7},
        {"video": "run/rollout.mp4"},
    )
    assert failed.termination_reason == "policy_error"
    with pytest.raises(ValueError, match="differ"):
        EpisodeOutcome(
            "bowl_plate",
            3,
            "development",
            "act-v1",
            "libero-v1",
            True,
            "policy_error",
            {"steps": 7},
            {},
        )
