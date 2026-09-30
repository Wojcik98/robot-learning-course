"""Check the ACT action path before using the live simulator."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vla_lab.policy.act import make_act_config, make_act_processors
from vla_lab.sim.episode import run_act_episode, run_policy_episode

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class FixedPolicy:
    def __init__(self) -> None:
        self.observation = None
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, observation):
        self.observation = observation
        return torch.zeros((1, 7))


class OneStepEnv:
    def __init__(self) -> None:
        self.action = None
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]
        self.observation = {
            "pixels": {"image": image, "image2": image.copy()},
            "robot_state": {
                "eef": {
                    "pos": np.array([-0.05, 0.03, 1.0]),
                    "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                    "mat": np.eye(3),
                },
                "gripper": {"qpos": np.array([0.02, -0.02]), "qvel": np.zeros(2)},
                "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
            },
        }

    def reset(self):
        return self.observation, {"is_success": False}

    def step(self, action):
        self.action = action
        return self.observation, 1.0, True, False, {"is_success": True}


def test_act_episode_uses_le_robot_transforms_and_physical_action() -> None:
    policy = FixedPolicy()
    environment = OneStepEnv()
    preprocessor, postprocessor = make_act_processors(
        make_act_config(device="cpu"), "bowl_plate", CONFIGS
    )

    result = run_act_episode(
        environment, policy, preprocessor, postprocessor, "bowl_plate", 0, CONFIGS, max_steps=2
    )

    assert result.success is True
    assert result.steps == 1
    assert policy.reset_count == 1
    assert policy.observation["observation.state"].shape == (1, 8)
    assert policy.observation["observation.images.image"][0, 0, -1, -1] == 1.0
    assert environment.action.shape == (7,)
    assert environment.action[0] == pytest.approx(0.1972228995947525, abs=1e-6)


def test_act_episode_refuses_final_state_during_development() -> None:
    preprocessor, postprocessor = make_act_processors(
        make_act_config(device="cpu"), "bowl_plate", CONFIGS
    )
    with pytest.raises(ValueError, match="development"):
        run_act_episode(
            OneStepEnv(),
            FixedPolicy(),
            preprocessor,
            postprocessor,
            "bowl_plate",
            20,
            CONFIGS,
            max_steps=1,
        )


def test_act_episode_emits_visual_frames_at_reset_and_after_action() -> None:
    preprocessor, postprocessor = make_act_processors(
        make_act_config(device="cpu"), "bowl_plate", CONFIGS
    )
    frames = []
    run_act_episode(
        OneStepEnv(),
        FixedPolicy(),
        preprocessor,
        postprocessor,
        "bowl_plate",
        0,
        CONFIGS,
        max_steps=1,
        on_frame=frames.append,
    )
    assert len(frames) == 2
    assert frames[0].shape == (256, 256, 3)
    assert frames[0][-1, -1, 0] == 255


def test_language_policy_receives_prompt_for_each_action() -> None:
    policy = FixedPolicy()
    environment = OneStepEnv()
    preprocessor, postprocessor = make_act_processors(
        make_act_config(device="cpu"), "bowl_plate", CONFIGS
    )

    result = run_policy_episode(
        environment,
        policy,
        preprocessor,
        postprocessor,
        "bowl_plate",
        0,
        CONFIGS,
        task_prompt="put the bowl on the plate",
        max_steps=1,
    )

    assert result.success is True
    assert policy.observation["task"] == ["put the bowl on the plate"]
