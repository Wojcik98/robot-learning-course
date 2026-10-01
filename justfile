set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

setup:
    uv sync --locked

data-demo:
    .venv/bin/robot-lab data fetch --scope demo

data-training:
    .venv/bin/robot-lab data fetch --scope training

doctor:
    .venv/bin/robot-lab doctor --live

lab0:
    .venv/bin/jupyter lab tutorials/index.ipynb

lab3:
    .venv/bin/jupyter lab tutorials/03-train-act/03-train-act.ipynb

test:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q

lint:
    .venv/bin/ruff check src tests
    .venv/bin/ruff format --check src tests
