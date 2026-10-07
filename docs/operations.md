# Operations guide

This guide tells you how to trace, restore, release, and roll back a course checkpoint. It also tells you what to do when a run fails. Run every command from the repository root. Use the locked environment.

## 1. What this guide covers and what it does not

The guide covers local operation on the prepared host: Ubuntu 24.04, Python 3.12, and one RTX 3070. It uses the existing commands only. The repository has no `release` command. A release record is a JSON file that you write.

Read these limits before you use a result:

- A supervised Brev L40S session has run course policy and training checks. Its export was verified and its instance was deleted. The [validation record](../artifacts/reference/course/l40s_validation.json) separates the quote-based cost estimate from an actual invoice.
- Real Brev allocation, SSH, asset transfer, training, and checkpoint reload have been checked. See [the cloud record](cloud.md) for the scope and cost controls.
- A local watchdog depends on this workstation and its network. It is not a provider deadline or a VM watchdog.
- A managed command host can stop a detached worker when its launcher exits. See [job controls](jobs.md). Check that a long job survives launcher exit on your host before unattended use.
- Hardware results come from an RTX 3070 with 8192 MiB and a Brev L40S with 46068 MiB. They do not predict another GPU, another driver, or a physical robot.
- A hash proves that two files are the same bytes. It does not prove that a policy is good.
- Simulator results come from LIBERO-Goal. They are not physical robot results. The source demonstrations may overlap the simulator initial states. This overlap is not resolved.

## 2. Terms

| Term | Meaning |
|---|---|
| Checkpoint | One saved policy directory with weights, processors, and a manifest. |
| Model hash | The SHA-256 value of `model.safetensors`. |
| Release | One checkpoint plus its evidence and limits. |
| Release record | A JSON file that names one release. |
| Release pointer | A small file that names the current release. |
| Rollback | A change of the release pointer to an earlier release. It does not delete anything. |
| Development state | A simulator state ID from 0 to 19. Use it for selection. |
| Final state | A simulator state ID from 20 to 49. Open it only for a frozen recipe. |

## 3. Provenance: where each file comes from

Every checkpoint must trace to its data, code, and evaluation. Use this table.

| Item | Source of truth | Tracked? |
|---|---|---|
| Python packages | `uv.lock` | Yes |
| LeRobot revision | [core manifest](../manifests/core.yaml): `e0d50211ef236143ae867228662b7dfaba554f02` | Yes |
| Dataset revision | Core manifest: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` | Yes |
| Simulator asset revision | Core manifest: `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` | Yes |
| Task states and splits | [`configs/eval/splits.yaml`](../configs/eval/splits.yaml) | Yes |
| ACT reference file hashes | [`manifests/act_references.yaml`](../manifests/act_references.yaml) | Yes |
| ACT checkpoint files | `.local-data/runs/.../checkpoint-000200` | No. Git ignores `.local-data/`. |
| Transfer archive hash | [`artifacts/reference/act/transport.json`](../artifacts/reference/act/transport.json) | Yes |
| Transfer archive | `.local-data/bundles/act-references.zip` | No |
| Evaluation records | `artifacts/reference/**/episodes.jsonl` and `summary.json` | Yes |
| Job run records | `.local-data/jobs/runs/<run-id>/` | No |
| Compact job record | [`artifacts/reference/jobs/local_act_probe.json`](../artifacts/reference/jobs/local_act_probe.json) | Yes |

Each checkpoint `manifest.json` records the dataset revision, split hash, and training statistics hashes. The tracked reference manifest records the hash of each of the seven required files.

### 3.1 Current chain for the `bowl_plate` ACT checkpoint

The model hash is `e14e1ef8866623882b92eeaad8deda1c22acbf798e9bef3d27d700eb64216107`. These tracked files name it:

- `manifests/act_references.yaml`
- `artifacts/reference/comparison/final/three_method_report.json` (ACT entry)
- `artifacts/reference/data_quality/recovery_reference.json` (`base_model_sha256` and `teacher_model_sha256`)

The transfer archive has 14 files and 115,596,447 bytes. Its SHA-256 value is `0e4494cfabdc989cb005254b1521a8536191fed5a6023350ed4ec10318589f88`. It holds both ACT reference checkpoints. It holds no dataset or simulator assets.

### 3.2 Provenance limits

- A distribution source for a new learner machine does not exist yet. The archive is local only.
- The source demonstrations may overlap the simulator initial states. A result on final states is not a test on new object placements.
- The supplied recovery reference data are policy generated. They are not human demonstrations.
- The supplied ACT reference bundles lack saved random state and cannot support exact continuation. ACT training saves model, optimizer, and PyTorch random state for resumable checkpoints.
- A checkpoint file does not record the host or the user that made it. Only the job record and your release record can state this.

### 3.3 Verify provenance

Check pinned data without a network:

```bash
.venv/bin/robot-lab data fetch --scope demo --offline
.venv/bin/robot-lab data fetch --scope training --offline
.venv/bin/robot-lab data fetch-models --offline
```

Check both ACT checkpoints, the data, and the GPU:

```bash
.venv/bin/robot-lab doctor
.venv/bin/robot-lab doctor --live
```

`doctor` reads the hash of each checkpoint file against the reference manifest. `--live` also resets LIBERO with EGL and needs a terminal that can see the GPU. Check one file by hand:

```bash
sha256sum .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200/model.safetensors
```

## 4. Restore

### 4.1 Restore ACT reference checkpoints on a clean workspace

Pack on the source machine:

```bash
.venv/bin/robot-lab references pack --output .local-data/bundles/act-references.zip
```

The command writes a new archive. It does not overwrite one. Copy the archive to the target machine. For SSH transfer, make `.local-data/bundles` in the target checkout, then run this command on the source machine with your own host and checkout path:

```bash
scp .local-data/bundles/act-references.zip USER@TARGET_HOST:/PATH/TO/COURSE/.local-data/bundles/
```

You can also copy the file by removable drive. On the target, compare its hash with `transport.json`:

```bash
sha256sum .local-data/bundles/act-references.zip
```

Install on the target machine:

```bash
.venv/bin/robot-lab references install --archive /path/to/act-references.zip
```

The installer checks all 14 hashes against the manifest before it writes checkpoint files.

Then fetch data and check the result:

```bash
.venv/bin/robot-lab data fetch --scope demo
.venv/bin/robot-lab doctor --live
```

### 4.2 Restore after an interrupted job

A detached job keeps its status, heartbeat, log, and artifacts under `.local-data/jobs/runs/<run-id>/`. Use these steps:

```bash
.venv/bin/robot-lab jobs status RUN_ID
.venv/bin/robot-lab jobs logs RUN_ID
.venv/bin/robot-lab jobs artifacts RUN_ID
```

If the worker is alive, wait. If the status is `interrupted` or the worker is dead, find the last complete checkpoint in `jobs artifacts`. Start a new training run from it:

```bash
.venv/bin/python -m vla_lab.training.act --task bowl_plate --steps 200 \
  --batch-size 8 --checkpoint-every 50 --seed 20260930 \
  --resume-from .local-data/runs/RUN/checkpoint-000100 \
  --output .local-data/runs/RUN-resumed
```

Replace the paths with your own. Use a new output directory. Resume needs a complete checkpoint that has model, optimizer, and random state. The supplied ACT reference bundles lack random state and cannot be resumed in this way.

Resumed training is not guaranteed to reproduce bitwise identical weights. Keep the parent checkpoint hash, seed, and evaluation records with the new run.

Submitting the same `jobs submit` command again reconnects to the same run while it is active. Use `--repeat` only when you want a new run after the first one has finished.

## 5. Release metadata

Write one release record for each checkpoint that you want to keep. Write it in a new file. Do not edit an old record. A record has these fields:

| Field | Value |
|---|---|
| `release_id` | A unique name, for example `bowl_plate-act-200`. |
| `task_alias` | A task from `configs/eval/splits.yaml`. |
| `model_sha256` | SHA-256 of `model.safetensors`. |
| `checkpoint_files_sha256` | Hash of each checkpoint file you can verify. Use `null` for an unknown file. |
| `parent_release` | The release that this one replaces, or `null`. |
| `evaluation` | Partition, state IDs, success count, attempt count, result file, and result file hash. |
| `repository` | Git commit and a dirty flag. |
| `uv_lock_sha256` | SHA-256 of `uv.lock`. |
| `paid_cloud_cost_usd` | Measured cost. Use 0 only when no paid job ran. |
| `limits` | A list of limits in plain words. |

Rules:

1. Use only values that a file proves. Write `null` for the rest. Do not guess.
2. Keep the evaluation partition. A development result selects a release. A final result reports it.
3. Keep every attempt in the denominator. An infrastructure error makes a paired comparison invalid.
4. Keep the record in a tracked location only after you remove paths that name a person or a secret.

The tracked [ACT release record](../artifacts/reference/release/act_bowl_plate_200.json) checks every checkpoint file hash and names its development and final result hashes. It sets the original checkpoint source commit to `null` because the saved checkpoint did not record it.

## 6. Roll back

Use a release pointer. A pointer is a small file with the current `release_id` and `model_sha256`. Keep an append-only history file. A rollback changes the pointer. It deletes no checkpoint and no record.

Decide with a rule that you write before you read the result. Use this rule:

1. Reject the candidate if a hash check or a state-ID check fails.
2. Reject the candidate if any attempt has an error.
3. Promote the candidate only if it has more successes than the current release on the same ordered states.
4. Otherwise keep the current release.

Roll back when a promoted release fails a later check. Do these steps:

1. Read `parent_release` in the current record.
2. Check that the parent record exists and its hash is unchanged.
3. Check the parent model hash against the manifest or the local file.
4. Write the pointer to the parent. Append one line to the history file.
5. Run `robot-lab doctor` and one development evaluation.
6. Keep the bad release record and its checkpoint for study.

To undo a rollback, write the pointer to the later release. The history file shows both changes.

## 7. Failure runbook

| Symptom | Check | Action |
|---|---|---|
| Notebook or terminal closed during training | `robot-lab jobs status RUN_ID` | If alive, wait. Use `jobs logs`. Do not start a second run. |
| Status `interrupted` or a dead worker | `jobs logs RUN_ID`, `jobs artifacts RUN_ID` | Resume from the last complete checkpoint (section 4.2). |
| Same command gives the old run | Run is still active or finished | Use `--repeat` only for a new run. |
| Run must stop | `robot-lab jobs status RUN_ID` | `robot-lab jobs cancel RUN_ID`. It stops only the verified worker group. |
| Artifact copy refused | Destination is not empty, or files exceed 1 GiB | Use a new empty destination. Copy fewer files. |
| Hash mismatch | Compare `sha256sum` with the manifest | Delete the bad copy. Install again from a verified archive. Do not use the file. |
| `data fetch --offline` fails | File is missing or wrong | Run `data fetch` with network access. Check the Hugging Face login. |
| GPU not visible | `nvidia-smi` in your terminal | Use a terminal that can see the GPU. A restricted shell cannot. A CPU result does not verify GPU operation. |
| EGL or reset fails | `robot-lab doctor --live` | Check the EGL libraries in [initial-setup](initial-setup.md). Check the simulator asset hash. |
| Out of GPU memory | `nvidia-smi` | Close other GPU programs. Reduce `--batch-size`. Record the setting you used. |
| Disk space low | `df -h .local-data` | Keep at least 60 GB free before a large download. Remove only files you can recreate. |
| Evaluation shows an infrastructure error | `episodes.jsonl` termination reason | Do not compare policies. Fix the cause. Run the same state IDs again in a new output directory. |
| Final state ID refused | Evaluator message | Use development IDs 0 to 19. Final IDs need a frozen recipe. |
| Candidate looks worse | Section 6 rule | Keep the current release. Roll back if it was promoted. |

### 7.1 Cloud failure (supervised use only)

Do not run unattended remote training. The local watchdog and monitor rules have fake-provider tests only. For a paid session that you supervise:

1. Verify the live balance, charge path, quote, provider stop, independent stop, budget limit, and monitoring. See [the cloud record](cloud.md). Stop before allocation if a check fails.
2. Keep the exact instance name. Never run `brev stop --all`.
3. If a job hangs or the deadline is near, save work if you can. Then stop the instance:

```bash
brev stop INSTANCE_NAME
brev ls
```

4. Check that the state is stopped in `brev ls`. The lifecycle code tries the stop with bounded retries and raises an error if it cannot verify the stop. If the error appears, run `brev stop INSTANCE_NAME` by hand. Check again in the Brev console.
5. A stopped instance can still bill for disk storage. Check the console. Delete only the course instance when you have the data you need.
6. Never put a Brev token in a notebook, tracked file, log, or remote command.

## 8. Limits of cloud and hardware claims

| Claim | Status |
|---|---|
| The course runs locally on an RTX 3070. | Measured on 2026-09-30 and 2026-10-01. One host. |
| Brev can run a course job. | A supervised L40S session ran all three policy, training, and adaptation paths. Its export and provider deletion were verified. |
| A job can stay under US$5 and two hours. | The supervised session ended within two hours with a US$3.62 quote-based estimate. No invoice was obtained; this is not a guarantee for another run. |
| A stop works when the job fails. | Fake-provider and terminal-forwarding tests pass. Terminal-backed deletion was verified; a live watchdog deadline stop was not exercised. |
| An unattended run stops by itself. | Not verified. Do not do it. |
| A local quote is the price. | A quote is a time-specific offer. Check it again before an allocation. |
| Cost of a course run | The L40S record gives elapsed time and a US$3.62 quote-based estimate. Actual billed cost is unknown. |
| Results transfer to a physical robot. | No. These are simulator results. |

## 9. Command list

All commands run from the repository root.

```bash
# Setup and checks
UV_CACHE_DIR=.local-data/uv-cache uv sync --locked --extra dev
.venv/bin/robot-lab data fetch --scope demo
.venv/bin/robot-lab data fetch --scope training
.venv/bin/robot-lab data fetch-models
.venv/bin/robot-lab data fetch --scope demo --offline
.venv/bin/robot-lab doctor
.venv/bin/robot-lab doctor --live

# Checkpoint transfer
.venv/bin/robot-lab references pack --output .local-data/bundles/act-references.zip
.venv/bin/robot-lab references install --archive .local-data/bundles/act-references.zip

# Live runs
.venv/bin/robot-lab demo --output .local-data/runs/demo-1 --task bowl_plate --state-id 0
.venv/bin/robot-lab eval --checkpoint CHECKPOINT_DIR --output .local-data/runs/eval-1 --state-id 0 --state-id 1
.venv/bin/robot-lab compare --a A/episodes.jsonl --b B/episodes.jsonl --output .local-data/runs/pair.json

# Detached jobs
.venv/bin/robot-lab jobs submit --config configs/jobs/act-bowl-plate-probe.yaml \
  --purpose train-act-probe --seed 20260930 -- \
  .venv/bin/python -m vla_lab.training.act --task bowl_plate --steps 1 \
  --batch-size 8 --checkpoint-every 1 --seed 20260930 --output '{run_dir}/training'
.venv/bin/robot-lab jobs status RUN_ID
.venv/bin/robot-lab jobs logs RUN_ID
.venv/bin/robot-lab jobs artifacts RUN_ID --output .local-data/retrieved/RUN_ID
.venv/bin/robot-lab jobs cancel RUN_ID

# Code checks
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
```

`robot-lab eval` and `robot-lab compare` need paths you choose. `robot-lab eval` accepts development state IDs only. Use a new output path each time. The paired report needs the same ordered state IDs in both files.
