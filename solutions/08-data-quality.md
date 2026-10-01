# Lab 8 exercise guide

The LIBERO relative controller uses seven values: three translation, three rotation, and one gripper value. The recorder clips each value to the action schema bounds before it calls the simulator. It saves the clipped, applied value. At tick `t`, action `t` belongs to the observation seen before that action. A complete episode with `N` actions has `N + 1` observations. The simulator control period is 50 ms. This period does not show how much wall time a human operator took between inputs.

The source audit covers 39 bowl-plate training episodes and 3,661 action rows. It uses the largest absolute value among the six arm action values. At threshold 0.001 it found 0 near-zero rows; at 0.01 it found 11; at 0.05 it found 61. The gripper is excluded. These are observed values. The source filtering history is unknown. Different episode lengths and timestamps cannot recover missing physical time.

The synthetic fixture has 12 actions and 13 observations. Its clean labels start `[0, 1, 2]`; the faulty labels start `[1, 2, 3]`. The file labels this as a synthetic one-tick action-label offset. The repair subtracts one from each action tick label. It does not change the action values. The repaired labels and values match the clean file. This does not prove that the source dataset has that defect.

The saved replay applies all 12 actions to a deterministic synthetic simulator and matches all 13 saved numeric states. This checks action order and state update for that fixture. It does not establish real LIBERO repeatability or physical timing. A real replay should also record its simulator version, reset seed, and any observation difference.

For ACT recovery samples, use each recorded 50 ms controller tick once. The source dataset's 10 Hz timestamp label does not authorize dropping every second controller action. Keep the 20-action window inside one episode and mark its padded tail.

For keyboard recordings, list each file and development state ID, count actions and observations, and check the declared task. Keep rejected and failed episodes separate. Keep every recorded state ID out of the held-out development assessment. A valid answer for the continuation reports its exact checkpoint, the accepted 5–10 recordings, optimizer steps, held-out state IDs, success counts for both policies, and errors. A negative result is valid evidence. Do not change the checkpoint after looking at final states.

The supplied reference uses five ACT-generated recordings from states 0–4. Its 200-step tick-aligned continuation and unchanged base each succeeded on 14/15 held-out development states 5–19. The base failed on state 8; the continuation failed on state 5. This is no net gain on those states. An invalid adapter that kept only even ticks failed all 15. The real keyboard-path smoke test used injected key events and exact replay. It does not measure human usability.
