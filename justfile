set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments := true

# List the available commands.
default:
    @just --list

# Install the locked environment and development tools.
setup:
    UV_CACHE_DIR=.local-data/uv-cache uv sync --locked --extra dev

# Download the simulator assets and demo sample.
data-demo *args:
    .venv/bin/robot-lab data fetch --scope demo "$@"

# Download the selected training data and videos.
data-training *args:
    .venv/bin/robot-lab data fetch --scope training "$@"

# Download the pinned SmolVLA and SmolVLM models.
data-models *args:
    .venv/bin/robot-lab data fetch-models "$@"

# Install and check the ACT reference archive.
references-install archive=".local-data/bundles/act-references.zip":
    .venv/bin/robot-lab references install --archive "$1"

# Check local files and GPU requirements.
doctor *args:
    .venv/bin/robot-lab doctor "$@"

# Check local files, GPU requirements, and live rendering.
doctor-live:
    .venv/bin/robot-lab doctor --live

# Start the course server without a password or token, on localhost.
jupyter notebook="tutorials/index.ipynb":
    .venv/bin/jupyter lab "$1" --ServerApp.root_dir=. --ip=100.120.25.105 --IdentityProvider.token='' --PasswordIdentityProvider.hashed_password='' --PasswordIdentityProvider.password_required=False

# Run one ACT demo. Use a new output directory.
demo *args:
    .venv/bin/robot-lab demo "$@"

# Train and compare the ACT development variants.
act-suite *args:
    .venv/bin/python -m vla_lab.experiments.act "$@"

# Submit the ACT suite as a local job.
act-suite-job:
    .venv/bin/robot-lab jobs submit --config configs/train/act_task.yaml --purpose act-reference --seed 20260930 -- .venv/bin/python -m vla_lab.experiments.act --config configs/train/act_task.yaml --output '{run_dir}/experiment'

# Submit a local job or inspect its status, logs, and artifacts.
jobs *args:
    .venv/bin/robot-lab jobs "$@"

# Evaluate SmolVLA on development states.
eval-smolvla *args:
    .venv/bin/python -m vla_lab.evaluation.smolvla "$@"

# Train or resume a SmolVLA learner.
train-smolvla *args:
    .venv/bin/python -m vla_lab.training.smolvla "$@"

# Download and check the ResNet-18 weights for Diffusion Policy.
diffusion-weights:
    TORCH_HOME=.local-data/torch .venv/bin/python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"

# Train Diffusion Policy with the local Torch cache.
train-diffusion *args:
    TORCH_HOME=.local-data/torch .venv/bin/python -m vla_lab.training.diffusion "$@"

# Record a keyboard demonstration on a development state.
record-keyboard *args:
    .venv/bin/python -m vla_lab.data.keyboard "$@"

# Check accepted recordings and write a recovery manifest.
recovery-manifest *args:
    .venv/bin/python -m vla_lab.experiments.build_recovery_manifest "$@"

# Train and assess an ACT recovery continuation.
act-recovery *args:
    .venv/bin/python -m vla_lab.experiments.recovery "$@"

# Compare runtime modes. The default config uses development states.
runtime-modes *args:
    .venv/bin/python -m vla_lab.experiments.runtime_modes "$@"

# Run the automated checks.
test:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q

# Check Python lint and formatting.
lint:
    .venv/bin/ruff check src tests tools tutorials
    .venv/bin/ruff format --check src tests tools tutorials

# Check shared source and notebook types, and reject invalid policy interfaces.
typecheck:
    .venv/bin/python tools/check_typing.py
