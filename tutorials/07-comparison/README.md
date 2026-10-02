# Lab 7: Compare three policy recipes

This lab compares ACT, learner SmolVLA C2, and Diffusion Policy on the same bowl-plate task. It reads the frozen final-state recipe and every saved attempt. It also shows where the three training recipes differ.

Start Jupyter from the repository root:

```bash
.venv/bin/jupyter lab tutorials/07-comparison/07-comparison.ipynb
```

The notebook uses tracked result files. It does not run training or open simulator states. Run Labs 4–6 first if you want to inspect each policy and its processor.
All six code cells passed from a fresh kernel. The [notebook check](../../artifacts/reference/comparison/lab7_notebook.json) records source and executed file hashes. The executed file is in ignored `.local-data/runs/`.

## Frozen result

The comparison uses final state IDs 20–49 with reset seed `20260930 + state_id`. The three recipes use the same 39 robot training episodes, task, cameras, controller, and timeout. The recipe was committed before the final runs. Every run has 30 attempts and zero policy or infrastructure errors.

| Recipe | Successes | 95% Wilson interval | Robot optimizer steps | Recorded training loop |
| --- | ---: | ---: | ---: | ---: |
| ACT | 30/30 | 0.886–1.000 | 200 | 16.7 s |
| Learner SmolVLA C2 | 18/30 | 0.423–0.754 | 5,000 | 828.8 s |
| Diffusion Policy | 20/30 | 0.488–0.808 | 2,000 | 195.3 s |

The training times come from one local RTX 3070 run. They exclude download, setup, and evaluation. The models use different pretraining, architecture, image processing, action history, and robot training budgets. The ACT result is the best practical recipe for this fixed task and these saved runs. It does not isolate an architecture effect or predict another task or robot.

The report is [three_method_report.json](../../artifacts/reference/comparison/final/three_method_report.json). The raw attempt rows are next to it. The exact config is [comparison.yaml](../../configs/experiments/comparison.yaml). The report checks the config hash, all state IDs, reset seeds, model hashes, and errors. The 30 states can share scene structure with the source demonstrations. The interval is a descriptive binomial interval for this set, not a claim about new scenes.

## Third task: Turn on the stove

The [development results](../../artifacts/reference/comparison/third_task_gate.json) records ACT at 5/20 after 200 steps and 20/20 after 1,000 steps. Learner SmolVLA reached 17/20 after 5,000 steps, and Diffusion Policy reached 19/20 after 2,000 steps. Checkpoint selection requires at least 10/20 development successes. The [stove-on recipe](../../configs/experiments/comparison_stove_on.yaml) fixes the checkpoints before final evaluation.

| Recipe | Stove-on final successes | 95% Wilson interval | Robot optimizer steps | Recorded training loop |
| --- | ---: | ---: | ---: | ---: |
| ACT | 30/30 | 0.886–1.000 | 1,000 | 83.8 s |
| Learner SmolVLA C2 | 28/30 | 0.787–0.982 | 5,000 | 838.0 s |
| Diffusion Policy | 29/30 | 0.833–0.994 | 2,000 | 196.5 s |

All 90 stove-on final attempts completed without policy or infrastructure errors. The [checked stove-on report](../../artifacts/reference/comparison/stove_on_final/three_method_report.json) has every paired result and the raw attempt paths. ACT succeeded on two states where learner SmolVLA failed and one state where Diffusion Policy failed. These few differences do not establish a stable method ranking. The [training resource record](../../artifacts/reference/comparison/stove_on_training_resources.json) shows that ACT needed a 200-step pilot, one development check, an 800-step continuation, and a second development check for this task. This external checkpoint selection took extra work. The optimizer-loop times exclude those evaluations and all setup time.

The published SmolVLA prompt study is a separate one-state development exercise. Its source training data are unknown.

Read the [exercise](07-comparison.ipynb#exercise) in the notebook. Save your answer before you read the [solution guide](../../solutions/07-comparison.md).
