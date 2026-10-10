"""Behaviour checks for the Lesson 20 exercise.

``course_input_fn`` is run with a stand-in adapter and, for the length of the check only,
recording stand-ins for the notebook's ``prepare_vla_inputs`` and ``TOKENIZER`` (the real
tokenizer loads in an experiment cell). The loss and sampler reuse the Lesson 17 and
Lesson 19 checks. Read a failing assertion message first.
"""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent


def _done(message: str) -> None:
    print(f"✓ {message}")


def _load(folder: str, module: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module, HERE.parent / folder / f"{module}.py")
    assert spec is not None and spec.loader is not None, f"cannot find {folder}/{module}.py"
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class _Policy:
    """Stand-in for the policy's processors: state * 10 and action + 100, both recorded."""

    def __init__(self) -> None:
        self.state_batches: list = []
        self.action_batches: list = []

    def prepare_state(self, batch):
        self.state_batches.append(batch)
        return batch["observation.state"] * 10

    def prepare_action(self, batch):
        self.action_batches.append(batch)
        return batch["action"] + 100


def check_course_input_fn(course_input_fn) -> None:
    """Exercise 20.1: raw cameras, processed state and prompt go to prepare_vla_inputs."""
    calls: list[dict] = []
    tokenizer = object()

    def prepare_vla_inputs(observation, prompts, given_tokenizer, config):
        calls.append(
            {
                "observation": dict(observation),
                "prompts": prompts,
                "tokenizer": given_tokenizer,
                "config": config,
            }
        )
        return {
            "images": [observation[k] * 2 - 1 for k in config.image_features],
            "state": observation["observation.state"],
        }

    config = SimpleNamespace(
        image_features={"observation.images.image": None, "observation.images.image2": None}
    )
    generator = torch.Generator().manual_seed(130)
    batch = {
        "observation.images.image": torch.rand(2, 3, 8, 8, generator=generator),
        "observation.images.image2": torch.rand(2, 3, 8, 8, generator=generator),
        "observation.state": torch.randn(2, 8, generator=generator),
        "action": torch.randn(2, 5, 7, generator=generator),
        "task": ["put the bowl on the plate", "put the bowl on the plate"],
    }
    namespace = course_input_fn.__globals__
    saved = {name: namespace.get(name) for name in ("prepare_vla_inputs", "TOKENIZER")}
    try:
        namespace["prepare_vla_inputs"] = prepare_vla_inputs
        namespace["TOKENIZER"] = tokenizer
        adapter = SimpleNamespace(config=config, policy=_Policy())
        inputs = course_input_fn(adapter, batch)
        assert len(calls) == 1, "call prepare_vla_inputs exactly once"
        call = calls[0]
        observation = call["observation"]
        for key in config.image_features:
            assert key in observation and torch.equal(observation[key], batch[key]), (
                f"pass the raw [0, 1] camera tensor {key!r} (prepare_vla_inputs resizes and normalizes it)"
            )
        assert torch.equal(observation["observation.state"], batch["observation.state"] * 10), (
            "observation['observation.state'] must come from adapter.policy.prepare_state(batch)"
        )
        assert call["prompts"] == batch["task"], "pass the batch's task prompt(s)"
        assert call["tokenizer"] is tokenizer and call["config"] is config, (
            "use the local TOKENIZER and adapter.config"
        )
        assert "actions" in inputs and torch.equal(inputs["actions"], batch["action"] + 100), (
            "with an action in the batch, add inputs['actions'] = adapter.policy.prepare_action(batch)"
        )
        assert torch.equal(inputs["images"][0], batch["observation.images.image"] * 2 - 1), (
            "return the dict built by prepare_vla_inputs"
        )
        _done(
            "raw cameras, policy-processed state, and the prompt reach prepare_vla_inputs; actions are added"
        )

        calls.clear()
        no_action = {key: value for key, value in batch.items() if key != "action"}
        no_action["task"] = "put the bowl on the plate"
        adapter = SimpleNamespace(config=config, policy=_Policy())
        inputs = course_input_fn(adapter, no_action)
        assert "actions" not in inputs and not adapter.policy.action_batches, (
            "without an action in the batch, do not add actions"
        )
        assert calls[0]["prompts"] in (
            "put the bowl on the plate",
            ["put the bowl on the plate"],
        ), "pass a single prompt string on (as the string or a one-item list)"
        _done("without actions the inputs have no 'actions' entry; a single prompt string works")

        missing = {key: value for key, value in batch.items() if key != "observation.images.image2"}
        try:
            course_input_fn(SimpleNamespace(config=config, policy=_Policy()), missing)
        except (KeyError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("a missing camera tensor must raise an error")
        _done("a missing camera raises an error")
    finally:
        for name, value in saved.items():
            if value is None:
                namespace.pop(name, None)
            else:
                namespace[name] = value


def check_learner_flow_loss(learner_flow_loss) -> None:
    """The Lesson 17 checks, run on this notebook's learner_flow_loss."""
    _load("17-smolvla-training", "lesson17_checks").check_learner_flow_loss(learner_flow_loss)


def check_euler_to_action(euler_to_action) -> None:
    """The Lesson 19 checks, run on this notebook's euler_to_action."""
    _load("19-smolvla-flow", "lesson19_checks").check_euler_to_action(euler_to_action)
