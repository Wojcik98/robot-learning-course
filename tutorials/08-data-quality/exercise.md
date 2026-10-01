# Lab 8 exercise: Repair one fault and assess recovery data

1. State the action components and the 50 ms simulator control period. Count saved observations and actions in one episode. Explain which observation belongs to action tick `t`.
2. Use the source-action audit. Report the near-zero threshold and count for six arm components. Explain why this count cannot establish whether source demonstrations were filtered.
3. Load the clean and faulty synthetic fixture. Show the first three action tick labels for each. Repair the labelled offset. Prove that the repaired action values and labels match the clean reference.
4. Replay the saved applied actions and compare all saved numeric states. State what a replay match checks and what it cannot prove about physical timing.
5. Make one ACT sample from every recorded 50 ms controller tick. Do not resample to match the source dataset's 10 Hz timestamp label. Explain how you keep an action window inside one episode.
6. Record 5–10 short keyboard episodes on declared development states. Save rejected or failed episodes separately. State the IDs you used, count applied actions, and keep those IDs out of the development assessment.
7. Run one bounded ACT continuation with only the accepted recovery episodes. Compare its result with the unchanged checkpoint on held-out development states. Report the training steps, data source, success counts, and any policy or infrastructure error. Do not use final states to select a continuation.

The supplied fixture is synthetic. The supplied reference recovery set is policy-generated. Label your own keyboard data and do not describe either supplied set as human demonstrations.
