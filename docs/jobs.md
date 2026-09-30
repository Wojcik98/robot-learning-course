# Local job controls

Long training runs use a detached worker. Closing a notebook or terminal does not stop the worker. Its status, heartbeat, log, and artifacts stay under `.local-data/jobs/runs/<run-id>/`.

Run this short local ACT training probe from the repository root on a machine with the pinned data and a GPU:

```bash
.venv/bin/robot-lab jobs submit \
  --config configs/jobs/act-bowl-plate-probe.yaml \
  --purpose train-act-probe --seed 20260930 -- \
  .venv/bin/python -m vla_lab.training.act \
  --task bowl_plate --steps 1 --batch-size 8 \
  --checkpoint-every 1 --seed 20260930 \
  --output '{run_dir}/training'
```

The command prints a `run_id`. The worker replaces `{run_dir}` with the saved run path. It executes an argument list without a shell. The config file, purpose, seed, and command define the run key. Submit the same command again after a notebook restart to reconnect to the same run. Use `--repeat` only when you want a new run after the first one finishes. An active matching run is never duplicated.

Use these commands with the printed ID:

```bash
.venv/bin/robot-lab jobs status RUN_ID
.venv/bin/robot-lab jobs logs RUN_ID
.venv/bin/robot-lab jobs artifacts RUN_ID
.venv/bin/robot-lab jobs artifacts RUN_ID --output .local-data/retrieved/RUN_ID
.venv/bin/robot-lab jobs cancel RUN_ID
```

`status` reads the saved heartbeat and checks the Linux process start time before it reports a live worker. `cancel` sends a stop signal only to that verified worker process group. If a prior worker died before it wrote a final status, a new submit can recover its stale key. Artifact copy uses a size limit of 1 GiB by default and does not overwrite a nonempty destination. Notebook code can use `vla_lab.notebooks.jobs` for the same controls.

These commands run on the local host. They do not start or stop a Brev instance. Remote jobs require the cloud checks in [the plan](plans/vla-tutorial-plan.md#7-cloud-operation-and-budget).

On 2026-10-01, this one-step job completed with exit code 0, finite loss 78.9951, ten saved files, and a copied model file with the same hash. The compact [probe record](../artifacts/reference/jobs/local_act_probe.json) gives its run ID. This verifies job control and one optimizer step; it does not establish a trained policy's task success.
