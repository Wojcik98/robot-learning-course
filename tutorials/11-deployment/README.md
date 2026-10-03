# Lab 11: Check, release, and roll back an ACT checkpoint

This lab teaches you to trace a checkpoint to its sources, check its hashes, write a release record, and decide whether to keep or roll back a candidate. The rollback is a pointer change that you can undo. You finish with a short career assessment memo.

Start from the repository root:

```bash
just jupyter tutorials/11-deployment/11-deployment.ipynb
```

The notebook runs on CPU. It needs no GPU, no LIBERO assets, and no checkpoint file. Cloud cost is US$0. No paid job ran for this lab.

## What you learn

- One model hash appears in three tracked records. You check that all four values match.
- Large files are in ignored `.local-data/`. The notebook hashes a file when it exists and reports `absent` when it does not. A mismatch fails the notebook.
- A candidate must start from a named parent. The Lab 8 candidate has 5 policy-generated recordings and no human demonstration.
- A release record names one checkpoint, its evidence, its Git commit, and its limits. Unknown values are `null`.
- The decision rule is written before the counts. The candidate had 14 of 15 successes, the same as the current release. The rule keeps the current release.
- A rollback changes `current.json`. It appends to `history.jsonl`. It never edits or deletes a record.
- The career memo asks which work you liked, which repeated work you accept, when a VLA helps, what real data are missing, which robotics skills transfer, and what result would change your choice.

## Files that the lab uses

- `manifests/act_references.yaml`
- `artifacts/reference/act/transport.json`
- `artifacts/reference/comparison/final/three_method_report.json`
- `configs/experiments/comparison.yaml`
- `artifacts/reference/data_quality/recovery_reference.json`
- `artifacts/reference/data_quality/heldout/base/episodes.jsonl`
- `artifacts/reference/data_quality/heldout/tick_aligned/episodes.jsonl`
- `configs/eval/splits.yaml` and `uv.lock`
- `vla_lab.evaluation.paired`: `wilson_interval`, `exact_mcnemar`

## Optional steps

The GPU check is optional. Its flag `RUN_GPU_CHECK` is `False`. Set it to `True` only on the prepared local host. It runs `just doctor-live`.

The notebook has no cloud step. No real Brev allocation has been tested. Read [the cloud record](../../docs/cloud.md) and pass its live cost and stop checks before paid work.

## Limits

- The notebook reads saved results. It runs no policy and no simulator.
- It writes only to a temporary directory. It removes that directory at the end.
- The candidate and the current release have the same success count on 15 development states. This does not show that either is better.
- One training seed was used. Source demonstrations may overlap the simulator initial states.
- No final state is opened. A rollback here does not change a real checkpoint.

Read the [operations guide](../../docs/operations.md) for the exact commands. Read the [exercise](11-deployment.ipynb#exercise) in the notebook. Read the [solution guide](../../solutions/11-deployment.md) after you save your answer.
