# Exercise: Change one demonstration count

1. Copy `configs/train/act_task.yaml` to a new YAML file. Change only `demos10_chunk20.episode_count` from `10` to `20`. Give that variant a new name, such as `demos20_chunk20`. Keep the 39-episode variants, seed, training steps, action chunks, and development state IDs fixed.
2. Run the new YAML with `python -m vla_lab.experiments.act --config YOUR_FILE --output NEW_DIRECTORY`. Use a new output directory. The command trains the variants in order and saves each closed-loop attempt.
3. Compare the 10-, 20-, and 39-episode results from the reference and your run. For each setting, report the selected episode count, frame count, statistics hash, training loss curve, success count, denominator, and infrastructure errors.
4. Explain why a lower final training loss does not prove a better controller. State one reason that a single seed and 20 development states cannot establish a small performance difference.

Use the [solution guide](../../solutions/03-train-act.md) after you finish. Do not open final simulator states to select the exercise setting.
