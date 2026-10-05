# Lab 8: Inspect and repair recording alignment

This guide describes the earlier lab. Its notebook was removed. Use the [course index](../index.ipynb) for current exercises.

This lab uses the seven-value LIBERO relative action contract. It checks source action values, a saved recording, and a labelled one-tick fault. It then guides you through short keyboard recordings and one bounded ACT continuation.

The saved analysis runs on CPU. The [source action audit](../../artifacts/reference/data_quality/source_action_audit.json) covers 39 bowl-plate training episodes and 3,661 actions. It found 0 arm actions at or below 0.001, 11 at or below 0.01, and 61 at or below 0.05, using the largest absolute value among six arm components. Gripper values are excluded. These counts do not prove why near-zero values are rare. Source filtering history is unknown, and dataset timestamps cannot restore removed physical time.

The [synthetic fixture manifest](../../artifacts/reference/data_quality/synthetic_fixture_manifest.json) identifies three small NPZ files: clean, faulty labels, and repaired. These files are from a deterministic synthetic simulator. They are not LIBERO demonstrations. Each has 12 applied actions and 13 observations. The fault shifts action tick labels by one. The repair restores those labels and leaves applied actions unchanged.

## Keyboard recording

Use a desktop terminal with the pinned local LIBERO assets. Only development states are accepted:

```bash
just record-keyboard \
  --task bowl_plate --state 0 \
  --output .local-data/keyboard-recordings/my-first-demo.npz
```

Click the camera window to give it keyboard focus. Press `W` to move the arm along `+X`. Use `S` for `-X`, `A` and `D` for `-Y` and `+Y`, and `R` and `F` for `+Z` and `-Z`. Use `U/O` for roll, `J/L` for pitch, and `I/K` for yaw. A motion key tap sends four controller actions over 200 ms. Hold the key for longer motion. The panel shows the last key and active command.

Press `Space` to switch the gripper command. Press `1` to open (`-1`) or `2` to close (`+1`). The selected gripper command stays active until you change it. The `[` and `]` keys are optional aliases for `1` and `2`. Press `X` to save or `Q` to discard. A save or quit key causes one final zero-motion action before the recorder stops; that action is in the saved file. Use a new output path for each episode. Check the saved action and observation count before training. The default data root is `.local-data`.
The reset seed is `20260930 + state_id` and is saved in the NPZ file.

Record 5–10 short accepted episodes. Put failed or rejected files in a different directory. Build a checked manifest from only the accepted files:

```bash
just recovery-manifest \
  --task bowl_plate \
  --recordings .local-data/keyboard-recordings/accepted/*.npz \
  --output .local-data/keyboard-recordings/manifest.json
```

The command checks file hashes, task IDs, reset seeds, and the development split. It reserves every unused development state for assessment. It cannot judge task success or human control quality; you must select accepted files. The ACT continuation takes this manifest:

```bash
just act-recovery \
  --manifest .local-data/keyboard-recordings/manifest.json \
  --base-checkpoint .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200 \
  --output .local-data/runs/my-act-recovery-200 --steps 200
```

The supplied [reference report](../../artifacts/reference/data_quality/recovery_reference.json) uses five separately labelled ACT-generated demonstrations. The valid tick-aligned continuation reached 14/15 on held-out development states, as did the unchanged base checkpoint; the failure state changed. The diagnostic continuation that discards odd controller ticks reached 0/15. It has invalid timing alignment and must not be used as a reference recipe. The course keeps every recorded 50 ms controller tick. The [teacher manifest](../../artifacts/reference/data_quality/teacher_manifest.json) gives hashes and local paths for the five full NPZ files. They are in ignored `.local-data/` on this prepared machine. Their source is a policy, not a human.

Read the exercise in the notebook. Read the [solution guide](../../solutions/08-data-quality.md) after you save your answer.
