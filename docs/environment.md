# Project environment

The project uses Python 3.12 and one `uv` environment. `uv.lock` pins the Python packages. The LeRobot source is pinned to the Git revision in [the core manifest](../manifests/core.yaml). The local GPU path uses PyTorch 2.11.0 with CUDA 12.8. Video decoding uses the TorchCodec 0.11.0 CPU wheel. This choice keeps GPU policy work separate from video decode and avoids a missing CUDA NPP library in the TorchCodec GPU wheel on this host.

## Local setup

Use Ubuntu 24.04, `uv`, `just`, the NVIDIA driver, and the FFmpeg and EGL shared libraries listed in [the initial setup checklist](initial-setup.md). Keep the data root on a disk with at least 60 GB free before large downloads. The default root is `.local-data/` in this repository. It is ignored by Git.

```bash
UV_CACHE_DIR=.local-data/uv-cache uv sync --locked --extra dev
```

Network access is needed for the first install and for public Hub assets. Later `uv run --locked` commands use the lock. Keep `HF_HOME` at its default if you used `hf auth login` without an override. Set `HF_HUB_CACHE` and `HF_DATASETS_CACHE` under `.local-data/` to keep large files on the selected disk. The training recipe sets these cache paths. `prepare_local_libero()` writes the benchmark paths and selects EGL before LIBERO is imported. The installed `hf-libero 0.1.4` ignores the asset path in its config file, so `select_local_assets()` sets its asset cache to the pinned local download before the simulator opens.

Run `.venv/bin/robot-lab data fetch --scope demo` for the playground. It downloads the pinned simulator assets and one real AV1 sample used by `doctor`. Run `.venv/bin/robot-lab data fetch --scope training` before a training experiment. That scope adds the Parquet data, metadata, and the selected two-camera video pairs. Use `--offline` to check existing files without network access. The command uses the existing Hugging Face login store if one is present and keeps large files under `.local-data/`.

Before the VLA experiments in Lesson 16, run `.venv/bin/robot-lab data fetch-models`. It gets the exact published SmolVLA and SmolVLM2 Hub commits in [the VLA manifest](../manifests/smolvla_reference.yaml), then checks the model hashes and required processor files. Use `data fetch-models --offline` to verify local files without a download.

## ACT reference commands

The selected dataset shards and LIBERO assets must be in `.local-data/` before these commands run. The paths and source revisions are in [the core manifest](../manifests/core.yaml). Run GPU commands through a terminal that can see the RTX 3070.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m vla_lab.training.act --task bowl_plate --steps 200 \
  --batch-size 8 --checkpoint-every 200 \
  --output .local-data/runs/act-bowl-plate-probe-200

PYTHONPATH=src python -m vla_lab.evaluation.act \
  --checkpoint .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200 \
  --output .local-data/runs/act-bowl-plate-probe-200/eval-development
```

Use the activated project environment for `python`, or replace it with `.venv/bin/python`. Change `bowl_plate` to `bowl_stove` for the second task. Give each run a new output directory. The evaluator accepts development initial states only. Its JSONL file contains one result for each state. These commands use the demonstration training partition. Dataset timestamps record 10 Hz labels; the simulator executes actions at 20 Hz. The code does not resample actions.

## Disk and GPU use

The selected source download uses about 20 MB of action/state Parquet files, 371 MB of task videos, and 409 MB of simulator assets. The reference `uv` cache uses about 9.5 GB. A 200-step ACT training run uses about 179 MB for its model, optimizer, and records. These figures are selected-run measurements, not the full course size. Keep additional space for other models, videos, and experiments.

Use a terminal with GPU access for CUDA and EGL checks. A restricted command environment can hide the GPU. Keep storage timing, simulator execution time, and wallclock inference time distinct.

## Playground and checkpoint transfer

Use `.venv/bin/robot-lab doctor --live` to check the GPU, one real AV1 frame, both ACT bundles, and one EGL reset. Then start [Lesson 1](../tutorials/01-playground/01-playground.ipynb) from the repository root. Use the marked live cells to run a policy in the simulator.

The two complete ACT inference bundles contain 14 files. `robot-lab references pack` writes a 115,596,447-byte ZIP archive at `.local-data/bundles/act-references.zip`. `robot-lab references install --archive <path>` checks all 14 hashes against [the reference manifest](../manifests/act_references.yaml) before it writes checkpoint files. The [transport record](../artifacts/reference/act/transport.json) holds the archive hash. This archive is local only; a distribution source for new learner machines is still required. It contains no dataset or simulator assets.

## Bash completion

Load completion in the current Bash terminal without a shell startup change:

```bash
source <(_ROBOT_LAB_COMPLETE=source_bash .venv/bin/robot-lab)
source <(just --completions bash)
```

`robot-lab demo` completes task aliases, development state IDs, the local profile, options, and file paths. The installed `just` completion supplies recipe names. Use quotation marks around file paths with spaces when you enter a command.
