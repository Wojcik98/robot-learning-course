# Lab 4 exercise: Change an instruction

Use bowl-plate development state 0 with the published model. Keep the model, initial state, seed, ten-action execution setting, and bowl-plate evaluation goal fixed.

1. Run the canonical prompt `put the bowl on the plate`.
2. Run the paraphrase `place the bowl on the plate`.
3. Run the conflicting prompt `put the plate on the bowl`. Both objects are in this scene.
4. Record the model prompt, evaluation task, success, step count, and clipped action count for each run.
5. State one limit of three attempts on one simulator state.

The bowl-plate success check applies to all three runs. This exercise checks prompt sensitivity. It does not test selection between two evaluation goals. Read the [solution guide](../../solutions/04-smolvla.md) after you save your results.
