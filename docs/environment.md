# Project environment

The project uses Python 3.12 and one `uv` environment. `uv.lock` pins the Python packages. The LeRobot source is pinned to the Git revision in [the core manifest](../manifests/core.yaml). The local GPU path uses PyTorch 2.11.0 with CUDA 12.8. Video decoding uses the TorchCodec 0.11.0 CPU wheel. This choice keeps GPU policy work separate from video decode and avoids a missing CUDA NPP library in the TorchCodec GPU wheel on this host.

## Local setup

Use Ubuntu 24.04, `uv`, `just`, the NVIDIA driver, and the FFmpeg and EGL shared libraries listed in [the initial setup checklist](initial-setup.md). Keep the data root on a disk with at least 60 GB free before large downloads. The default root is `.local-data/` in this repository. It is ignored by Git.

```bash
UV_CACHE_DIR=.local-data/uv-cache uv sync --locked --extra dev
```

Network access is needed for the first install and for public Hub assets. Later `uv run --locked` commands use the lock. Set `HF_HOME=.local-data/hf`, `LIBERO_CONFIG_PATH=.local-data/libero-config`, `MPLCONFIGDIR=.local-data/mpl`, and `MUJOCO_GL=egl` for simulator work. The LIBERO config must point to the installed benchmark files and the downloaded assets. A course setup command will create this config before the first learner run.

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
| ACT, SmolVLA, and Diffusion Policy | Pending policy probes and trained reference runs |

The restricted shell cannot see the GPU. Use the approved host execution route for GPU and EGL checks. A CPU-only result does not pass the GPU gate. The simulator reports a 20 Hz controller. Dataset timestamps are 10 Hz labels; they do not set the simulator control period.

The initial selected download used 20 MB of action/state Parquet shards, 371 MB of selected task video shards, and 409 MB of pinned simulator assets. The `uv` cache used about 9.5 GB after installation. These are setup measurements, not a full course disk bound. The exact files and source revisions are in the core manifest.
