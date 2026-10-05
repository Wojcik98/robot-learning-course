# Record development demonstrations

Use the [data-improvement unit](../tutorials/13-data-quality/13-data-quality.ipynb) to check alignment and test a data change. This guide gives the keyboard commands.

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

The command checks file hashes, task IDs, reset seeds, and the development split. It reserves every unused development state for assessment. It cannot judge task success or human control quality; you must select accepted files. The command below trains on the full 20 Hz recording. Unit 13 supplies its own alignment and window functions. Use the notebook for that experiment.

The ACT continuation command takes this manifest:

```bash
just act-recovery \
  --manifest .local-data/keyboard-recordings/manifest.json \
  --base-checkpoint .local-data/runs/act-bowl-plate-probe-200/checkpoint-000200 \
  --output .local-data/runs/my-act-recovery-200 --steps 200
```
