"""Exercise timed frozen evaluation with a small real Torch policy."""

import json

import torch
import yaml

from vla_lab.evaluation import comparison
from vla_lab.sim.episode import EpisodeResult
from vla_lab.teaching import evaluation


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def predict_action_chunk(self, value):
        return value * self.weight

    def select_action(self, value):
        return self.predict_action_chunk(value)


class Environment:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_timed_frozen_evaluation_keeps_failures_and_restores_hooks(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "data").mkdir(parents=True)
    (configs / "eval").mkdir()
    (configs / "data" / "episodes.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset_revision": "dataset-r1",
                "tasks": {
                    "bowl_plate": {
                        "split_sha256": "split-hash",
                        "task_name": "put the bowl on the plate",
                    }
                },
            }
        )
    )
    (configs / "eval" / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "bowl_plate": {
                        "development_state_ids": list(range(20)),
                        "final_state_ids": [20, 21],
                    }
                }
            }
        )
    )
    recipe = {
        "model_sha256": "a" * 64,
        "checkpoint": str(tmp_path / "frozen-checkpoint"),
        "robot_training_steps": 7,
        "n_action_steps": 1,
    }
    config = {
        "policies": {"act": recipe},
        "task_alias": "bowl_plate",
        "state_ids": [20, 21],
        "base_seed": 20260930,
        "max_steps": 2,
        "task_prompt": "put the bowl on the plate",
        "comparison_type": "practical_recipe",
        "train_split_sha256": "split-hash",
        "dataset_revision": "dataset-r1",
    }
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(comparison, "load_comparison_config", lambda *_args: config)
    policy = SmallPolicy()
    original_select = policy.select_action
    original_generation = policy.predict_action_chunk
    loader_calls = []

    def load_method(method, selected_recipe, task_alias, split_hash, revision):
        loader_calls.append((method, selected_recipe, task_alias, split_hash, revision))
        return policy, object(), object()

    monkeypatch.setattr(comparison, "_load_method", load_method)
    environments = []

    def open_task(*_args):
        env = Environment()
        environments.append(env)
        return env

    def run_episode(_environment, active_policy, _pre, _post, _task, state_id, *_args, **_kwargs):
        result = active_policy.select_action(torch.tensor(3.0))
        assert result.item() == 6.0
        if state_id == 21:
            raise RuntimeError("synthetic infrastructure fault")
        return EpisodeResult("bowl_plate", state_id, True, 1, 0, "success")

    monkeypatch.setattr("vla_lab.sim.episode.open_libero_task", open_task)
    monkeypatch.setattr("vla_lab.sim.episode.run_policy_episode", run_episode)
    output = tmp_path / "results"

    result = evaluation.evaluate_frozen_method_timed(
        method="act",
        config_path=str(config_path),
        configs_root=str(configs),
        data_root=str(tmp_path / "data"),
        output_dir=str(output),
    )

    assert loader_calls == [("act", recipe, "bowl_plate", "split-hash", "dataset-r1")]
    assert result["summary"]["attempts"] == 2
    assert result["summary"]["successes"] == 1
    assert result["summary"]["infrastructure_errors"] == 1
    assert result["timing"]["action_selection_api"]["count"] == 2
    assert result["timing"]["action_generation"]["count"] == 2
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert [row["termination_reason"] for row in rows] == ["success", "infrastructure_error"]
    assert rows[1]["metrics"]["seed"] == 20260951
    assert all(env.closed for env in environments)
    assert policy.select_action.__func__ is original_select.__func__
    assert policy.predict_action_chunk.__func__ is original_generation.__func__
    timing_path = output / "inference-timing.json"
    assert timing_path.is_file()
    assert json.loads(timing_path.read_text())["action_generation"]["count"] == 2
