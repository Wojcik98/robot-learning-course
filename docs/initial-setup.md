# Set up your machine

Run the course on a Linux machine with an NVIDIA GPU. The reference environment uses Ubuntu 24.04, Bash, Python 3.12, and an RTX 3070 with 8 GiB of GPU memory. Allow at least 60 GB of free disk space before large downloads. This is a storage reserve, not the full course size.

The required training experiments target a 48 GB GPU and a maximum of two hours per experiment. The 8 GB local GPU can run small interface checks and policy rollouts. It is not the reference for the full training memory budget. Use the local GPU first, then use a bounded cloud run when memory is insufficient.

## 1. Check host tools and libraries

Install `uv` and `just`, and use an NVIDIA driver that supports the CUDA 12.8 PyTorch wheel selected by the project. Check the tools from your terminal:

```bash
uv --version
just --version
nvidia-smi
df -h .
```

The host needs FFmpeg shared libraries with AV1 decoding and EGL/OpenGL libraries. On Ubuntu 24.04, install the native runtime packages if they are absent:

```bash
sudo apt install ffmpeg libegl1 libgl1
```

The Python environment does not supply the NVIDIA kernel driver or these host libraries. The `ffmpeg` command alone does not check whether TorchCodec can load its shared libraries. Use the live check in section 4 for video decoding and rendering.

## 2. Create the Python environment

From the repository root:

```bash
UV_CACHE_DIR=.local-data/uv-cache uv sync --locked --extra dev
source .venv/bin/activate
python --version
robot-lab --help
just --list
```

`uv` creates `.venv` with Python 3.12 and the locked packages. Use this environment for Python, Jupyter, and the course CLI. You can also use `.venv/bin/python` and `.venv/bin/robot-lab` without activation.

The default data root is `.local-data/`. Git ignores it. See [the environment guide](environment.md) for cache paths and pinned dependencies.

## 3. Get data and checkpoints

```bash
.venv/bin/robot-lab data fetch --scope demo
.venv/bin/robot-lab references install --archive /path/to/act-references.zip
```

Obtain the ACT archive from a machine that has the reference bundles. The course has no public archive download location. The [transfer guide](operations.md#41-restore-act-reference-checkpoints-on-a-clean-workspace) explains how to pack, copy, and install it. The archive contains checkpoints only; `data fetch` supplies the simulator assets.

Lesson 1 also needs the prepared Diffusion checkpoint. Lessons 20 and 22 need the prepared SmolVLA learner checkpoint. Copy these complete directories from the course machine to the same paths in your checkout:

| Checkpoint | Local path |
|---|---|
| Diffusion Policy | `.local-data/runs/diffusion-bowl-plate-reference-2000/checkpoint-002000` |
| SmolVLA learner | `.local-data/runs/smolvla-c2-5000-development/checkpoint-005000` |

Keep each directory complete: weights alone are not enough. Configuration, processor files, manifests, and saved training records identify how the checkpoint is used. The model hashes in `configs/experiments/comparison.yaml` identify the prepared checkpoints. Do not substitute a newly trained model at these paths; keep new training runs in their own directories. These two prepared checkpoints have no public download URL.

Fetch training data before a training experiment. Fetch model files before Lesson 1, which also runs the published SmolVLA policy:

```bash
.venv/bin/robot-lab data fetch --scope training
.venv/bin/robot-lab data fetch-models
```

Network access is needed for the first install and download. Public core assets do not require a token. For an asset that requires authentication, use `hf auth login` and check access with `hf auth whoami`. Keep tokens out of notebooks, tracked files, and logs. Keep `HF_HOME` stable after login; use `HF_HUB_CACHE` and `HF_DATASETS_CACHE` for large cache files.

## 4. Check the runtime and start the course

```bash
.venv/bin/robot-lab doctor --live
.venv/bin/jupyter lab tutorials/index.ipynb
```

Use a terminal with GPU access. The live check tests CUDA, AV1 decoding, checkpoint files, and an EGL simulator reset. Resolve each failed check before you start Lesson 1. A restricted command environment can block GPU access even when your desktop terminal works.

Select the `Python 3 (ipykernel)` kernel from this repository's `.venv`. Start with Lesson 1, then follow the course index. The course uses local compute. See [the cloud guide](cloud.md) if local GPU memory or run time is insufficient.

## 5. Enable Bash completion

Load completion in the current terminal:

```bash
source <(_ROBOT_LAB_COMPLETE=source_bash .venv/bin/robot-lab)
source <(just --completions bash)
```

Use `robot-lab` completion for task aliases, development state IDs, options, and file paths. Use `just` completion for recipe names.
