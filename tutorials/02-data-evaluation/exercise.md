# Exercise: Check another complete episode

1. In the notebook, select `train_ids[1]` instead of `train_ids[0]` for the complete episode audit. Keep the first decoded sample and its processor check unchanged.
2. Check that frame indices start at zero and have no gaps. Record the episode length and median stored timestamp step.
3. Explain what happens if a converter treats a relative action as an absolute target. State which schema field should record this convention.
4. Keep development state IDs `[0, 1]` and the saved checkpoint fixed. Compare the new live rows with the saved 20-state reference. Explain why the two-state result has a wider uncertainty.

Use the [solution guide](../../solutions/02-data-evaluation.md) after you finish. Your episode length is data dependent; copy the value from your run.
