# Project environment

The project uses Python 3.12 and one `uv` environment. `uv.lock` pins the Python packages. The LeRobot source is pinned to the Git revision in [the core manifest](../manifests/core.yaml). The local GPU path uses PyTorch 2.11.0 with CUDA 12.8. Video decoding uses the TorchCodec 0.11.0 CPU wheel. This choice keeps GPU policy work separate from video decode and avoids a missing CUDA NPP library in the TorchCodec GPU wheel on this host.

## Local setup

Use Ubuntu 24.04, `uv`, `just`, the NVIDIA driver, and the FFmpeg and EGL shared libraries listed in [the initial setup checklist](initial-setup.md). Keep the data root on a disk with at least 60 GB free before large downloads. The default root is `.local-data/` in this repository. It is ignored by Git.

```bash
UV_CACHE_DIR=.local-data/uv-cache uv sync --locked --extra dev
```

Network access is needed for the first install and for public Hub assets. Later `uv run --locked` commands use the lock. Keep `HF_HOME` at its default if you used `hf auth login` without an override. Set `HF_HUB_CACHE` and `HF_DATASETS_CACHE` under `.local-data/` to keep large files on the selected disk. The training recipe sets these cache paths. `prepare_local_libero()` writes the benchmark paths and selects EGL before LIBERO is imported. The installed `hf-libero 0.1.4` ignores the asset path in its config file, so `select_local_assets()` sets its asset cache to the pinned local download before the simulator opens.

Run `.venv/bin/robot-lab data fetch --scope demo` for the first lab. It downloads the pinned simulator assets and one real AV1 sample used by `doctor`. Run `.venv/bin/robot-lab data fetch --scope training` before a training lab. That scope adds the Parquet data, metadata, and the selected two-camera video pairs. Use `--offline` to check existing files without network access. The command uses the existing Hugging Face login store if one is present and keeps large files under `.local-data/`.

Before Labs 4–5, run `.venv/bin/robot-lab data fetch-models`. It gets the exact published SmolVLA and SmolVLM2 Hub commits in [the VLA manifest](../manifests/smolvla_reference.yaml), then checks the model hashes and required processor files. Use `data fetch-models --offline` to verify local files without a download. The offline check passed for both pinned model directories on this host.

A new demo data root was downloaded and checked on 2026-10-01. The asset download took about 85 seconds. The offline hash check, CUDA, AV1 decode, and EGL reset passed. A live ACT rollout with that data root succeeded in 74 steps. See the [fresh data record](../artifacts/reference/data/fresh_demo.json). The training scope passed an online cached check; a full new-root training download has not run yet.

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

Use the activated project environment for `python`, or replace it with `.venv/bin/python`. Change `bowl_plate` to `bowl_stove` for the second task. Give each run a new output directory. The evaluator accepts development initial states only. Its JSONL file contains one result for each state. These commands use the demonstration training partition. The action timestamps use the dataset's 10 Hz labels; the simulator still executes one action at 20 Hz. The code does not resample actions.

## Measured host checks on 2026-09-30

| Check | Result |
|---|---|
| Free workspace space before downloads | 166 GB |
| Free workspace space after the first setup | 155 GB |
| GPU through approved host execution | RTX 3070, 8192 MiB total, 5277 MiB free at check time |
| CUDA tensor and matrix operation | Passed; finite output |
| FFmpeg libraries | FFmpeg 6 libraries and AV1 decoder present |
| TorchCodec AV1 decode | Passed on two real 256 × 256 camera videos |
| LIBERO-Goal reset and one action | Passed with EGL, 20 Hz relative control, two rendered cameras |
| ACT with simulator and LeRobot processors | One live inference step and two 200-step training runs passed |
| ACT development rollout | Bowl-on-plate 19/20 with explicit reset seeds; bowl-on-stove 20/20 from the older unseeded run |
| Published SmolVLA with simulator | Passed a live step; selected ten-action setting succeeded on 16/20 seeded development states |
| SmolVLA learner | C1 0/20 and selected C2 14/20 seeded development states; saved bundles reloaded |
| Diffusion Policy with simulator | Finite action and one live step; saved CPU probe reloaded |

The restricted shell cannot see the GPU. Use the approved host execution route for GPU and EGL checks. A CPU-only result does not pass the GPU gate. The simulator reports a 20 Hz controller. Dataset timestamps are 10 Hz labels; they do not set the simulator control period.

The initial selected download used 20 MB of action/state Parquet shards, 371 MB of selected task video shards, and 409 MB of pinned simulator assets. The `uv` cache used about 9.5 GB after installation. Each 200-step ACT run used about 179 MB, including a 62 MB model file and a 124 MB optimizer file. These are setup measurements, not a full course disk bound. The exact files and source revisions are in the core manifest.

## Lab 0 launch and checkpoint transfer

Use `.venv/bin/robot-lab doctor --live` to check the GPU, one real AV1 frame, both ACT bundles, and one EGL reset. Then start [Lab 0](../tutorials/00-first-run/00-first-run.ipynb) from the repository root. The default notebook runs development state 0 twice and shows each live video. On 2026-10-01, a fresh nbclient kernel ran all four code cells without an error. Both runs succeeded in 74 steps. Their recorded policy times were 4.347 s and 3.368 s; these times exclude kernel startup and video display.

The two complete ACT inference bundles contain 14 files. `robot-lab references pack` writes a 115,596,447-byte ZIP archive at `.local-data/bundles/act-references.zip`. `robot-lab references install --archive <path>` checks all 14 hashes against [the reference manifest](../manifests/act_references.yaml) before it writes checkpoint files. A clean `/tmp` workspace restore passed. The [transport record](../artifacts/reference/act/transport.json) holds the archive hash. This archive is local only; a distribution source for new learner machines is still required. It contains no dataset or simulator assets.

## Bash completion

Load completion in the current Bash terminal without a shell startup change:

```bash
source <(_ROBOT_LAB_COMPLETE=source_bash .venv/bin/robot-lab)
source <(just --completions bash)
```

`robot-lab demo` completes task aliases, development state IDs, the local profile, options, and file paths. The installed `just` completion supplies recipe names. The automated Bash check ran without network access. Use quotation marks around file paths with spaces when you enter a command.
