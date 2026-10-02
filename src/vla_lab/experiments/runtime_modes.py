"""Compare runtime modes on development states with one SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vla_lab.evaluation.states import selected_state_ids
from vla_lab.runtime.experiment import MODES, RTC_MODES, EpisodeConfig, EpisodeResult

SCHEMA_VERSION = 1
PINNED_CHECKPOINT = ".local-data/runs/smolvla-c2-5000-development/checkpoint-005000"
PINNED_VLM_DIR = ".local-data/models/smolvlm2-500m-video-instruct"
PINNED_MODEL_SHA256 = "5d71344fbc12bd55ef26fae2114da5ca1ffd1424c5ebabfd87298fd5acaf96b1"
PINNED_RESET_SEED_BASE = 20260930
PINNED_NOISE_SEED_BASE = 4100
PINNED_NOISE_SEED_STRIDE = 1000
TOP_KEYS = {
    "schema_version",
    "experiment",
    "task_alias",
    "partition",
    "state_ids",
    "reset_seed_base",
    "noise_seed_base",
    "noise_seed_state_stride",
    "max_steps",
    "dataset_revision",
    "train_split_sha256",
    "task_prompt",
    "learner",
    "runtime",
    "modes",
}
FINAL_EXTRA_KEYS = {"development_report_sha256"}
LEARNER_KEYS = {"checkpoint", "vlm_dir", "model_sha256", "vlm_sha256"}
RUNTIME_KEYS = {"horizon", "stride", "delay", "num_steps", "ensemble_decay"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_latency_config(path: str | Path, configs_root: str | Path) -> dict[str, Any]:
    """Read the development or frozen final config and check its source pins."""
    content = Path(path).read_bytes()
    config = yaml.safe_load(content)
    if not isinstance(config, dict):
        raise TypeError("Latency config must be a mapping")
    partition = config.get("partition")
    expected_keys = TOP_KEYS | FINAL_EXTRA_KEYS if partition == "final" else TOP_KEYS
    if set(config) != expected_keys:
        raise ValueError("Latency config has missing or unknown keys")
    expected_experiment = "latency_final" if partition == "final" else "latency"
    if config["schema_version"] != SCHEMA_VERSION or config["experiment"] != expected_experiment:
        raise ValueError("Latency config needs schema version 1 and the correct experiment")
    if partition not in ("development", "final"):
        raise ValueError("Latency partition must be development or final")
    if partition == "final":
        report_path = (
            Path(configs_root).resolve().parent
            / "artifacts/reference/runtime/development_modes.json"
        )
        if _sha256(report_path) != config["development_report_sha256"]:
            raise ValueError("Final latency development report hash differs from frozen evidence")
    learner, runtime = config["learner"], config["runtime"]
    if set(learner) != LEARNER_KEYS or set(runtime) != RUNTIME_KEYS:
        raise ValueError("Latency learner or runtime section has missing or unknown keys")
    if (
        learner["checkpoint"] != PINNED_CHECKPOINT
        or learner["vlm_dir"] != PINNED_VLM_DIR
        or learner["model_sha256"] != PINNED_MODEL_SHA256
        or not isinstance(learner["vlm_sha256"], str)
        or len(learner["vlm_sha256"]) != 64
    ):
        raise ValueError("Latency learner source differs from the pinned SmolVLA checkpoint")
    if (
        config["reset_seed_base"] != PINNED_RESET_SEED_BASE
        or config["noise_seed_base"] != PINNED_NOISE_SEED_BASE
        or config["noise_seed_state_stride"] != PINNED_NOISE_SEED_STRIDE
    ):
        raise ValueError("Latency reset or noise seeds differ from the pinned recipe")
    if (
        runtime["horizon"] != 50
        or runtime["stride"] != 10
        or runtime["delay"] != 2
        or runtime["num_steps"] != 5
        or not 0 < float(runtime["ensemble_decay"]) <= 1
    ):
        raise ValueError("Latency runtime needs horizon 50, stride 10, delay 2, and 5 steps")
    if config["modes"] != list(MODES):
        raise ValueError(f"Latency modes must be exactly {list(MODES)}")
    if not isinstance(config["max_steps"], int) or config["max_steps"] < 1:
        raise ValueError("Latency max_steps must be positive")
    task_alias = str(config["task_alias"])
    config["state_ids"] = selected_state_ids(
        configs_root, task_alias, partition, config["state_ids"], allow_final=partition == "final"
    )
    if partition == "final" and config["state_ids"] != list(range(20, 30)):
        raise ValueError("Final latency states differ from the frozen matrix")
    episodes = yaml.safe_load((Path(configs_root) / "data" / "episodes.yaml").read_text())
    task = episodes["tasks"][task_alias]
    if (
        config["dataset_revision"] != episodes["dataset_revision"]
        or config["train_split_sha256"] != task["split_sha256"]
        or config["task_prompt"] != task["task_name"]
    ):
        raise ValueError("Latency data split or task prompt differs from the frozen data")
    config["config_sha256"] = hashlib.sha256(content).hexdigest()
    return config


def verify_model_source(config: dict[str, Any]) -> dict[str, Any]:
    """Check the local checkpoint and VLM files against the pins; return the manifest."""
    learner = config["learner"]
    source = Path(learner["checkpoint"])
    if _sha256(source / "model.safetensors") != learner["model_sha256"]:
        raise ValueError("Latency model hash differs from the local checkpoint")
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest.get("policy") != "smolvla_learner"
        or manifest["model_sha256"] != learner["model_sha256"]
        or manifest["vlm_sha256"] != learner["vlm_sha256"]
        or manifest["task_alias"] != config["task_alias"]
        or manifest["split_sha256"] != config["train_split_sha256"]
        or manifest["chunk_size"] != config["runtime"]["horizon"]
    ):
        raise ValueError("Latency checkpoint manifest differs from the pinned model source")
    if _sha256(Path(learner["vlm_dir"]) / "model.safetensors") != learner["vlm_sha256"]:
        raise ValueError("Latency VLM weights differ from the pinned source")
    return manifest


def episode_identity(config: dict[str, Any], mode: str, state_id: int) -> dict[str, Any]:
    """Everything that must match before a saved episode result is reused."""
    if mode not in config["modes"] or state_id not in config["state_ids"]:
        raise ValueError("Mode or state is outside the latency config")
    runtime = config["runtime"]
    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "model_sha256": config["learner"]["model_sha256"],
        "task_alias": config["task_alias"],
        "partition": config["partition"],
        "mode": mode,
        "state_id": state_id,
        "reset_seed": config["reset_seed_base"] + state_id,
        "noise_seed": config["noise_seed_base"] + state_id * config["noise_seed_state_stride"],
        "horizon": runtime["horizon"],
        "stride": runtime["stride"],
        "delay": runtime["delay"],
        "num_steps": runtime["num_steps"],
        "max_steps": config["max_steps"],
    }


def result_path(output: str | Path, mode: str, state_id: int) -> Path:
    return Path(output) / mode / f"state-{state_id:03d}.json"


def load_reusable(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Return a saved result only if its identity is exactly the requested one."""
    try:
        saved = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict) or saved.get("identity") != identity:
        return None
    metrics = saved.get("metrics")
    if (
        not isinstance(metrics, dict)
        or not {
            "success",
            "steps",
            "underflow",
            "discontinuity_l2_mean",
            "request_latency_ms",
            "cuda_peak_allocated_bytes",
        }
        <= metrics.keys()
    ):
        return None
    latency = metrics["request_latency_ms"]
    if (
        not isinstance(latency, dict)
        or not {
            "samples",
            "first_request_ms",
            "warm_p50_ms",
            "warm_p95_ms",
            "warm_p99_ms",
        }
        <= latency.keys()
    ):
        return None
    return saved


def save_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write beside the target and rename, so a crash never leaves a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    p50, p95, p99 = np.percentile(np.asarray(values, dtype=np.float64), [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}


def episode_metrics(
    result: EpisodeResult,
    latencies_ms: Sequence[float],
    *,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
) -> dict[str, Any]:
    """Reduce one episode to runtime metrics."""
    if not result.ticks:
        raise ValueError("Episode has no controller ticks")
    for tick in result.ticks:
        if tick.action.shape != (7,) or not np.all(np.abs(tick.action) <= 1.0):
            raise ValueError("Controller action must be seven values within [-1, 1]")
    jumps = [tick.discontinuity for tick in result.ticks if tick.discontinuity is not None]
    return {
        "success": bool(result.success),
        "steps": len(result.ticks),
        "underflow": sum(tick.underflow for tick in result.ticks),
        "discontinuity_l2_mean": float(np.mean(jumps)) if jumps else None,
        "discontinuity_l2_max": float(np.max(jumps)) if jumps else None,
        "requests": len(latencies_ms),
        "request_latency_ms": {
            **_percentiles(latencies_ms),
            "first_request_ms": float(latencies_ms[0]) if latencies_ms else None,
            "warm_p50_ms": _percentiles(latencies_ms[1:])["p50"],
            "warm_p95_ms": _percentiles(latencies_ms[1:])["p95"],
            "warm_p99_ms": _percentiles(latencies_ms[1:])["p99"],
            "samples": list(latencies_ms),
        },
        "cuda_peak_allocated_bytes": peak_allocated_bytes,
        "cuda_peak_reserved_bytes": peak_reserved_bytes,
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Group saved per-state results by mode, pooling every request latency."""
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for saved in results:
        by_mode.setdefault(saved["identity"]["mode"], []).append(saved)
    summary: dict[str, Any] = {}
    for mode in MODES:
        rows = by_mode.get(mode)
        if not rows:
            continue
        metrics = [row["metrics"] for row in rows]
        pooled = [value for item in metrics for value in item["request_latency_ms"]["samples"]]
        warm = [value for item in metrics for value in item["request_latency_ms"]["samples"][1:]]
        jumps = [
            item["discontinuity_l2_mean"]
            for item in metrics
            if item["discontinuity_l2_mean"] is not None
        ]
        peaks = [
            item["cuda_peak_allocated_bytes"]
            for item in metrics
            if item["cuda_peak_allocated_bytes"] is not None
        ]
        reserved = [
            item["cuda_peak_reserved_bytes"]
            for item in metrics
            if item["cuda_peak_reserved_bytes"] is not None
        ]
        summary[mode] = {
            "state_ids": sorted(row["identity"]["state_id"] for row in rows),
            "episodes": len(rows),
            "successes": sum(item["success"] for item in metrics),
            "success_rate": sum(item["success"] for item in metrics) / len(rows),
            "mean_steps": float(np.mean([item["steps"] for item in metrics])),
            "underflow_total": sum(item["underflow"] for item in metrics),
            "discontinuity_l2_mean": float(np.mean(jumps)) if jumps else None,
            "request_latency_ms": {
                **_percentiles(pooled),
                "warm_p50_ms": _percentiles(warm)["p50"],
                "warm_p95_ms": _percentiles(warm)["p95"],
                "warm_p99_ms": _percentiles(warm)["p99"],
            },
            "cuda_peak_allocated_bytes": max(peaks) if peaks else None,
            "cuda_peak_reserved_bytes": max(reserved) if reserved else None,
        }
    return summary


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimedPredictor:
    """Time each chunk request after CUDA work has finished."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor
        self.latencies_ms: list[float] = []

    def __call__(self, *args: Any) -> tuple[np.ndarray, np.ndarray]:
        _synchronize()
        start = time.perf_counter()
        chunks = self.predictor(*args)
        _synchronize()
        self.latencies_ms.append((time.perf_counter() - start) * 1000)
        return chunks


def _configure_rtc(policy: Any, mode: str, stride: int) -> None:
    """Install checked guided RTC for RTC modes; leave it off for every other mode."""
    if mode in RTC_MODES:
        from vla_lab.rtc.guidance import install_guided_rtc

        install_guided_rtc(policy, execution_horizon=stride, mask_mode=mode.removeprefix("rtc_"))
    else:
        policy.config.rtc_config = None
        policy.rtc_processor = None
        policy.model.rtc_processor = None


def run_mode_episode(
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    environment: Any,
) -> dict[str, Any]:
    """Run one mode on one state with a fresh policy reset and timed requests."""
    from vla_lab.runtime.experiment import EpisodeRunner
    from vla_lab.runtime.smolvla_chunks import SmolVLAChunkPredictor

    runtime, mode = config["runtime"], identity["mode"]
    _configure_rtc(policy, mode, runtime["stride"])
    policy.reset()
    predictor = TimedPredictor(
        SmolVLAChunkPredictor(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_prompt=config["task_prompt"],
            mode=mode,
            stride=runtime["stride"],
            delay=runtime["delay"],
            noise_seed=identity["noise_seed"],
        )
    )
    episode = EpisodeConfig(
        mode=mode,
        horizon=runtime["horizon"],
        stride=runtime["stride"],
        delay=runtime["delay"],
        max_steps=config["max_steps"],
        seed=identity["reset_seed"],
        episode_id=f"{mode}-state-{identity['state_id']:03d}",
        task_id=config["task_alias"],
        ensemble_decay=float(runtime["ensemble_decay"]),
    )
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    result = EpisodeRunner(episode).run(environment, predictor)
    _synchronize()
    return episode_metrics(
        result,
        predictor.latencies_ms,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
    )


def run_latency_experiment(
    *,
    config_path: str | Path,
    configs_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    modes: Sequence[str] | None = None,
    state_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run every missing development episode, saving each one as soon as it finishes."""
    config = load_latency_config(config_path, configs_root)
    chosen_modes = list(modes) if modes is not None else list(config["modes"])
    chosen_states = (
        selected_state_ids(
            configs_root,
            config["task_alias"],
            config["partition"],
            state_ids,
            allow_final=config["partition"] == "final",
        )
        if state_ids is not None
        else list(config["state_ids"])
    )
    if not set(chosen_modes) <= set(config["modes"]) or not set(chosen_states) <= set(
        config["state_ids"]
    ):
        raise ValueError("Requested modes or states are outside the latency config")
    pending = []
    saved_results = []
    for state_id in chosen_states:
        for mode in chosen_modes:
            identity = episode_identity(config, mode, state_id)
            saved = load_reusable(result_path(output_dir, mode, state_id), identity)
            if saved is None:
                pending.append(identity)
            else:
                saved_results.append(saved)
    if pending:
        from vla_lab.policy.smolvla_learner import load_learner_bundle
        from vla_lab.sim.episode import open_libero_task

        verify_model_source(config)
        learner = config["learner"]
        policy, preprocessor, postprocessor, _ = load_learner_bundle(
            learner["checkpoint"], learner["vlm_dir"]
        )
        policy.eval()
        policy.config.num_steps = config["runtime"]["num_steps"]
        for identity in pending:
            environment = open_libero_task(
                config["task_alias"], identity["state_id"], configs_root, data_root
            )
            try:
                metrics = run_mode_episode(
                    config=config,
                    identity=identity,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    environment=environment,
                )
            finally:
                active_error = sys.exc_info()[0]
                try:
                    environment.close()
                except Exception:
                    if active_error is None:
                        raise
            saved = {"identity": identity, "metrics": metrics}
            save_atomic(result_path(output_dir, identity["mode"], identity["state_id"]), saved)
            saved_results.append(saved)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "model_sha256": config["learner"]["model_sha256"],
        "partition": config["partition"],
        "modes": summarize(saved_results),
    }
    save_atomic(Path(output_dir) / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Compare runtime modes on declared states; finished episodes are reused."""
    parser = argparse.ArgumentParser(description="Compare runtime modes on declared states")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/latency.yaml"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--data-root", type=Path, default=Path(".local-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--state-ids", nargs="+", type=int)
    args = parser.parse_args(argv)
    summary = run_latency_experiment(
        config_path=args.config,
        configs_root=args.configs_root,
        data_root=args.data_root,
        output_dir=args.output,
        modes=args.modes,
        state_ids=args.state_ids,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
