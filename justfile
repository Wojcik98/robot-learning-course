set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

setup:
    uv sync --locked

doctor:
    .venv/bin/robot-lab doctor --live

lab0:
    .venv/bin/jupyter lab tutorials/index.ipynb

test:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q

lint:
    .venv/bin/ruff check src tests
    .venv/bin/ruff format --check src tests
