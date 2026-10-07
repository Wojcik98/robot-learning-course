import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from vla_lab.teaching.course_comparison import (
    AdaptationSettings,
    KeyProjection,
    adapt_policy,
    adapted_recipe,
    aligned_transition_rows,
    case_id,
    checkpoint_heldout_loss,
    count_safetensors_parameters,
    describe_adaptation,
    episode_case_rows,
    episode_time_summary,
    epoch_batch_indices,
    evaluate_development,
    find_completed_data_unit_run,
    freeze_comparison_recipe,
    intervention_case_rows,
    max_leaf_difference,
    measure_cuda_use,
    measure_use,
    new_run_dir,
    read_jsonl,
    require_assets,
    run_bounded_updates,
    sha256_file,
    tick_vectors,
    training_rate,
    verify_completed_data_unit_run,
    verify_recipe_files,
    write_run_identity,
)


def record(state_id, success, seed=None):
    metrics = {} if seed is None else {"seed": seed}
    return {
        "task_id": "bowl_plate",
        "initial_state_id": state_id,
        "partition": "final",
        "policy_revision": "abc",
        "success": success,
        "metrics": metrics,
    }


def test_tick_vectors_checks_counts_and_start():
    obs, act = tick_vectors([0, 1, 2], [0, 1])
    assert obs.tolist() == [0, 1, 2] and act.tolist() == [0, 1]
    with pytest.raises(ValueError, match="N action ticks"):
        tick_vectors([0, 1], [0, 1])
    with pytest.raises(ValueError, match="start at zero"):
        tick_vectors([1, 2, 3], [1, 2])


def test_case_rows_keep_failures_and_check_the_seed_rule():
    rows = intervention_case_rows(
        [record(15, False), record(16, True, 20260946)], base_seed=20260930
    )
    assert [row["case_id"] for row in rows] == [case_id(15, 20260945), case_id(16, 20260946)]
    assert [row["success"] for row in rows] == [False, True]
    with pytest.raises(ValueError, match="frozen rule"):
        intervention_case_rows([record(15, True, 5)], base_seed=20260930)
    paired = episode_case_rows([record(20, True)], base_seed=20260930, prompt="p", action_steps=10)
    assert paired[0]["partition"] == "final" and paired[0]["action_steps"] == 10


def test_unique_run_directories_and_identity(tmp_path):
    first, second = new_run_dir(tmp_path, "unit"), new_run_dir(tmp_path, "unit")
    assert first != second and first.is_dir()
    path = write_run_identity(first, seed=3, states=[1, 2])
    assert json.loads(path.read_text())["fields"] == {"seed": 3, "states": [1, 2]}
    with pytest.raises(FileExistsError):
        write_run_identity(first, seed=3)


def _write_data_unit_run(
    root: Path,
    *,
    created_unix_s: float,
    task_alias: str = "bowl_plate",
    dataset_revision: str = "revision-a",
    train_split_sha256: str = "split-a",
) -> None:
    root.mkdir(parents=True)
    (root / "run-identity.json").write_text(
        json.dumps(
            {
                "created_unix_s": created_unix_s,
                "fields": {
                    "task_alias": task_alias,
                    "dataset_revision": dataset_revision,
                    "train_split_sha256": train_split_sha256,
                },
            }
        )
    )
    arms = {}
    for label, new_sample_fraction in (("baseline", 0.0), ("intervention", 0.25)):
        checkpoint = root / f"checkpoint-{label}"
        checkpoint.mkdir()
        weights = checkpoint / "model.safetensors"
        weights.write_bytes(label.encode())
        arms[label] = {
            "adaptation_updates": 300,
            "stopped_by": "updates",
            "checkpoint": str(checkpoint),
            "model_sha256": sha256_file(weights),
            "source_model_sha256": "shared-start-hash",
            "new_sample_fraction": new_sample_fraction,
        }
    (root / "training.json").write_text(json.dumps({"arms": arms}))
    (root / "reload.json").write_text(json.dumps({label: {"difference": 0.0} for label in arms}))
    summaries = {}
    for label in ("start", "baseline", "intervention"):
        summaries[label] = {
            "task_id": task_alias,
            "partition": "development",
            "state_ids": list(range(15, 20)),
            "attempts": 5,
        }
        evaluation = root / f"eval-{label}"
        evaluation.mkdir()
        rows = [{"task_id": task_alias, "initial_state_id": state_id} for state_id in range(15, 20)]
        (evaluation / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "paired-summary.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "effect": {"cases": 5, "failed_attempts_included": True},
            }
        )
    )


def test_completed_data_unit_matches_split_and_checkpoints(tmp_path):
    run = tmp_path / "data-quality-run"
    _write_data_unit_run(run, created_unix_s=1.0)
    completion = verify_completed_data_unit_run(
        run,
        task_alias="bowl_plate",
        dataset_revision="revision-a",
        train_split_sha256="split-a",
    )
    assert completion["state_ids"] == list(range(15, 20))
    assert completion["run_dir"] == str(run.resolve())
    with pytest.raises(ValueError, match="train_split_sha256"):
        verify_completed_data_unit_run(
            run,
            task_alias="bowl_plate",
            dataset_revision="revision-a",
            train_split_sha256="other-split",
        )


def test_latest_data_unit_skips_incomplete_and_mismatched_runs(tmp_path):
    results = tmp_path / "course-results"
    results.mkdir()
    _write_data_unit_run(results / "data-quality-old", created_unix_s=1.0)
    _write_data_unit_run(results / "data-quality-new", created_unix_s=2.0)
    mismatch = results / "data-quality-other-task"
    _write_data_unit_run(mismatch, created_unix_s=3.0, dataset_revision="other-revision")
    incomplete = results / "data-quality-incomplete"
    incomplete.mkdir()
    (incomplete / "run-identity.json").write_text("{}")

    latest = find_completed_data_unit_run(
        results,
        task_alias="bowl_plate",
        dataset_revision="revision-a",
        train_split_sha256="split-a",
    )
    assert latest["run_dir"] == str((results / "data-quality-new").resolve())
    with pytest.raises(ValueError, match="dataset_revision"):
        find_completed_data_unit_run(
            results,
            task_alias="bowl_plate",
            dataset_revision="revision-a",
            train_split_sha256="split-a",
            override_path=mismatch,
        )


def test_max_leaf_difference_and_training_rate(tmp_path):
    saved = {"state": np.array([1.0, 2.0]), "pixels": {"image": np.zeros((2, 2))}}
    replay = {"state": np.array([1.0, 2.5]), "pixels": {"image": np.zeros((2, 2))}}
    assert max_leaf_difference(saved, replay) == {"state": 0.5, "pixels.image": 0.0}
    log = tmp_path / "metrics.jsonl"
    log.write_text(
        "".join(
            json.dumps({"step": i, "loss": 1.0 / (i + 1), "elapsed_s": 1.0 + 0.5 * i}) + "\n"
            for i in range(11)
        )
    )
    rate = training_rate(log, warmup_steps=2)
    assert rate["steady_steps_per_s"] == pytest.approx(2.0) and rate["steps"] == 11
    with pytest.raises(ValueError):
        training_rate(log, warmup_steps=10)


def test_parameter_count_and_recipe_freeze(tmp_path):
    weights = tmp_path / "model.safetensors"
    save_file({"a": torch.zeros(2, 3), "b": torch.zeros(4)}, str(weights))
    assert count_safetensors_parameters(weights) == 10
    recipe = {
        "policies": {
            "act": {"checkpoint": "ckpt", "model_sha256": sha256_file(weights)},
        }
    }
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "model.safetensors").write_bytes(weights.read_bytes())
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(recipe))
    out = tmp_path / "out"
    out.mkdir()
    frozen = freeze_comparison_recipe(source, workspace=tmp_path, output_dir=out)
    config = yaml.safe_load(frozen.read_text())
    assert config["policies"]["act"]["checkpoint"] == str(tmp_path / "ckpt")
    assert verify_recipe_files(config) == {"act": sha256_file(weights)}
    config["policies"]["act"]["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        verify_recipe_files(config)
    with pytest.raises(FileExistsError):
        freeze_comparison_recipe(source, workspace=tmp_path, output_dir=out)
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n")
    assert read_jsonl(blank) == []


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU error")
def test_cuda_meter_fails_without_a_gpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        with measure_cuda_use():
            pass


def test_bounded_updates_test_the_deadline_inside_the_loop(tmp_path):
    calls = []

    def slow(step):
        calls.append(step)
        time.sleep(0.02)
        return 1.0 / step

    log = run_bounded_updates(
        slow, max_updates=1000, max_seconds=0.2, min_updates=3, metrics_path=tmp_path / "m.jsonl"
    )
    assert log["stopped_by"] == "deadline" and 3 <= log["updates"] < 1000
    assert log["updates"] == len(calls) and log["seconds"] < 1.0
    rows = read_jsonl(tmp_path / "m.jsonl")
    assert [row["step"] for row in rows] == list(range(1, log["updates"] + 1))
    done = run_bounded_updates(
        lambda step: 0.5,
        max_updates=4,
        max_seconds=10,
        min_updates=2,
        metrics_path=tmp_path / "n.jsonl",
    )
    assert done["stopped_by"] == "updates" and done["updates"] == 4


def test_bounded_updates_reject_short_runs_and_long_deadlines(tmp_path):
    with pytest.raises(RuntimeError, match="at least 2 are needed"):
        run_bounded_updates(
            lambda step: time.sleep(0.05) or 1.0,
            max_updates=100,
            max_seconds=0.01,
            min_updates=2,
            metrics_path=tmp_path / "m.jsonl",
        )
    with pytest.raises(ValueError, match="max_seconds"):
        run_bounded_updates(
            lambda step: 1.0,
            max_updates=2,
            max_seconds=7201,
            min_updates=1,
            metrics_path=tmp_path / "x",
        )
    with pytest.raises(ValueError, match="max_seconds"):
        AdaptationSettings(max_seconds=7201.0)
    with pytest.raises(ValueError, match="multiple"):
        AdaptationSettings(batch_size=8, heldout_samples=12)
    with pytest.raises(ValueError, match="min_updates"):
        AdaptationSettings(max_updates=10, min_updates=11)
    assert AdaptationSettings().max_seconds <= 7200 and AdaptationSettings().min_updates >= 500


def test_epoch_batches_cover_each_row_once_per_epoch_and_are_repeatable():
    first = [i for step in (1, 2, 3) for i in epoch_batch_indices(12, 4, seed=5, step=step)]
    assert sorted(first) == list(range(12))
    assert epoch_batch_indices(12, 4, seed=5, step=2) == epoch_batch_indices(12, 4, seed=5, step=2)
    assert epoch_batch_indices(12, 4, seed=5, step=4) != first[:4]
    with pytest.raises(ValueError):
        epoch_batch_indices(3, 4, seed=5, step=1)


def test_key_projection_keeps_shared_keys_and_names_missing_ones():
    rows = [{"a": 1, "b": 2, "extra": "x"}, {"a": 3, "b": 4, "extra": "y"}]
    view = KeyProjection(rows, ("a", "b"))
    assert len(view) == 2 and view[1] == {"a": 3, "b": 4}
    with pytest.raises(KeyError, match="missing"):
        KeyProjection(rows, ("a", "missing"))[0]


def test_aligned_rows_copy_actions_and_refuse_offsets():
    source = np.array([[4.0], [5.0]])
    rows = aligned_transition_rows(["s0", "s1", "s2"], source, [0, 1, 2], [0, 1])
    source[1, 0] = 99.0
    assert [row["tick"] for row in rows] == [0, 1] and rows[1]["observation"] == "s1"
    assert rows[1]["action"].dtype == np.float32 and rows[1]["action"][0] == 5.0
    with pytest.raises(ValueError, match=r"offset by \[1\]"):
        aligned_transition_rows(["s0", "s1", "s2"], [[4.0], [5.0]], [0, 1, 2], [1, 2])
    with pytest.raises(ValueError):
        aligned_transition_rows(["s0", "s1"], [[4.0], [5.0]], [0, 1, 2], [0, 1])


def test_adapted_recipe_replaces_checkpoints_and_keeps_the_rest():
    start = {
        "schema_version": 1,
        "state_ids": [20],
        "policies": {
            "act": {
                "checkpoint": "a",
                "model_sha256": "0",
                "robot_training_steps": 200,
                "n_action_steps": 10,
                "pretraining": "none",
            },
        },
    }
    record = {
        "checkpoint": "/new",
        "model_sha256": "1" * 64,
        "robot_training_steps": 1200,
        "source_checkpoint": "/old",
        "source_model_sha256": "0",
        "adaptation_updates": 1000,
        "batch_size": 8,
    }
    recipe = adapted_recipe(start, {"act": record})
    entry = recipe["policies"]["act"]
    assert entry["checkpoint"] == "/new" and entry["robot_training_steps"] == 1200
    assert entry["n_action_steps"] == 10 and entry["start_checkpoint"] == "/old"
    assert entry["pretraining"].startswith("none; then 1000 adaptation updates")
    assert start["policies"]["act"]["checkpoint"] == "a", "The start recipe is not changed"
    with pytest.raises(ValueError, match="every method"):
        adapted_recipe(start, {})


def test_episode_times_keep_simulated_and_wall_time_apart():
    records = [
        {"metrics": {"steps": 100, "elapsed_s": 10.0, "inference_wall_s": 2.0}},
        {"metrics": {"steps": 300, "elapsed_s": 30.0}},
        {"metrics": {}},
    ]
    times = episode_time_summary(records)
    assert times["mean_simulated_s"] == pytest.approx((5.0 + 15.0) / 2)
    assert times["mean_wall_s"] == pytest.approx(20.0)
    assert times["mean_inference_wall_s"] == pytest.approx(2.0)
    assert times["wall_over_simulated"] == pytest.approx(40.0 / 20.0)
    empty = episode_time_summary([{"metrics": {}}])
    assert empty["mean_simulated_s"] is None and empty["wall_over_simulated"] is None


def test_missing_assets_are_all_listed_with_their_fix(tmp_path):
    (tmp_path / "present").mkdir()
    with pytest.raises(FileNotFoundError) as error:
        require_assets(
            tmp_path, {"present": "n/a", "gone/a": "run `just one`", "gone/b": "run `just two`"}
        )
    text = str(error.value)
    assert "gone/a" in text and "just one" in text and "gone/b" in text and "just two" in text
    assert "present" not in text.replace("Nothing downloads", "")
    require_assets(tmp_path, {"present": "n/a"})


def test_development_checks_refuse_final_states(tmp_path):
    for states in ([20], [10, 25], [], [0]):
        with pytest.raises(ValueError, match="10, 11, 12, 13, 14"):
            evaluate_development(
                "act",
                tmp_path,
                configs_root=tmp_path,
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                state_ids=states,
            )
    with pytest.raises(ValueError, match="Cannot evaluate"):
        evaluate_development(
            "smolvla_learner",
            tmp_path,
            configs_root=tmp_path,
            data_root=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_cpu_meter_reports_time_and_describe_names_the_deadline():
    with measure_use("cpu") as use:
        time.sleep(0.01)
    assert use.seconds >= 0.01 and use.peak_allocated_mib == 0.0
    record = {
        "method": "act",
        "source_robot_steps": 200,
        "adaptation_updates": 700,
        "stopped_by": "deadline",
        "samples_seen": 5600,
        "train_seconds": 1500.2,
        "peak_allocated_mib": 4096.0,
        "heldout_loss_before": 0.5,
        "heldout_loss_after": 0.4,
    }
    text = describe_adaptation(record)
    assert "700" in text and "deadline" in text and "5600" in text and "0.5000 -> 0.4000" in text


ACT_START = Path(".local-data/runs/act-bowl-plate-probe-200/checkpoint-000200")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not ACT_START.is_dir(),
    reason="needs the local ACT checkpoint, the data, and CUDA to build the saved processors",
)
def test_tiny_real_act_adaptation_saves_a_checkpoint_that_reloads(tmp_path):
    settings = AdaptationSettings(
        batch_size=2, max_updates=3, min_updates=2, max_seconds=600, heldout_samples=2, device="cpu"
    )
    record = adapt_policy(
        "act",
        source=ACT_START,
        source_robot_steps=200,
        output_dir=tmp_path / "run",
        configs_root="configs",
        data_root=".local-data",
        settings=settings,
    )
    assert record["adaptation_updates"] == 3 and record["robot_training_steps"] == 203
    assert record["model_sha256"] != record["source_model_sha256"]
    run = json.loads(Path(record["run_path"]).read_text())
    assert run["steps"] == 203 and run["window_offsets"]["action"] == list(range(20))
    reloaded = checkpoint_heldout_loss(
        "act",
        record["checkpoint"],
        configs_root="configs",
        data_root=".local-data",
        settings=settings,
    )
    assert reloaded == pytest.approx(record["heldout_loss_after"], abs=1e-4)
