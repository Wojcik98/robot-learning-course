# Lab 10: Check the RTC mask and guidance

This lab teaches how real-time chunking (RTC) keeps a new action chunk close to actions that the robot already committed to run. You compute the RTC v2 mask, compare it with the pinned upstream mask, check the guidance term on a toy model, and read a saved report of five runtime modes.

Start from the repository root:

```bash
.venv/bin/jupyter lab tutorials/10-rtc/10-rtc.ipynb
```

The notebook runs on CPU. It needs no GPU, no LIBERO assets, and no model checkpoint. Cloud cost is US$0.

## Reading assignment (20 minutes)

Read [RTC v2](https://arxiv.org/html/2506.07339v2). Read only equations 2–5 and Algorithm 1. Stop after 20 minutes. The notebook gives a time split and four questions.

**Prediction.** Before you run section 2, use `H = 50`, `s = 10`, `d = 2`. State how many mask weights are above 0. State whether the weight at index 2 is below or above 0.9.

## What you learn

- `L = H - s` is the guided overlap. Weights are 1 for `i < d`, soft for `d <= i < L`, and 0 from `L`.
- With `d = 0`, every guided weight is soft. With `d = L`, the soft mask equals the hard mask.
- The course mask, equation 5, and the pinned upstream `RTCProcessor` mask agree.
- The guidance term uses the model derivative. Clipping at `β = 5` keeps it finite.
- The tracked report compares five modes with one SmolVLA checkpoint and `n = 5` denoising steps. The RTC modes use `β = 5`.
- The virtual delay of 100 ms (2 ticks) is a setting. The measured request time was about 175–320 ms (4 to 7 ticks).
- The final config was frozen before the final run. The notebook checks its hashes and reads the [final report](../../artifacts/reference/runtime/final_modes.json) without opening a simulator state.

## Code that the lab uses

- `src/vla_lab/rtc/guidance.py`: `prefix_weights`, `GuidedRTCProcessor`.
- `src/vla_lab/runtime/virtual_scheduler.py`: `latency_to_ticks`.
- `configs/experiments/latency.yaml` and `latency_final.yaml`.
- `artifacts/reference/runtime/development_modes.json`.
- `artifacts/reference/runtime/final_modes.json`.
- Tests in `tests/test_rtc_guidance.py`.

## Optional GPU run

You do not need this step. It needs a local CUDA GPU, LIBERO assets, and the pinned checkpoint under `.local-data`:

```bash
.venv/bin/python -m vla_lab.experiments.runtime_modes \
  --config configs/experiments/latency.yaml \
  --output .local-data/runs/runtime-modes-development
```

Use only `latency.yaml`. Do not use `latency_final.yaml`.

## Limits

- The report has 5 development states and one checkpoint. Success is 2/5 to 4/5. It gives no stable ranking. It does not show that RTC improves success.
- The final report has 10 states and the same checkpoint. Naive asynchronous control and temporal ensembling reached 9/10, synchronous control and hard RTC reached 7/10, and soft RTC reached 4/10. These small paired counts do not give a stable ranking.
- The reported discontinuity is the mean L2 change between all adjacent seven-value, bounded controller actions. It includes changes inside chunks and at chunk boundaries. A lower value does not prove a better task result.
- The virtual delay of 2 ticks is not the measured wall latency. The runs did not use the measured delay.
- Peak memory is PyTorch allocation only. It excludes the renderer.
- The lab does not run a simulator. It reads the saved final report; it does not select a recipe with final states.

Read the [exercise](10-rtc.ipynb#exercise) in the notebook. Read the [solution guide](../../solutions/10-rtc.md) after you save your answer.
