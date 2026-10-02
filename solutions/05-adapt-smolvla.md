# Lab 5 exercise guide

The 1,000-step checkpoint must have `stage: c2` and `step: 1000`. Its `parent_model_sha256` must equal the saved C1 model hash `d697544e…`. The VLM source must be `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at the pinned revision. The model must keep the 50-action chunk and the ten-action execution setting. The hashes of `model.safetensors`, `config.json`, `optimizer.pt`, and each processor file must equal the manifest values. The processor hashes must include the state and action normalization tensor files.

Saved seeded references use development states 0–19, base seed 20260930, and no final state:

| Checkpoint | All 20 states | States 0–4 |
| --- | --- | --- |
| C1, step 100 | 0/20 | 0/5 |
| C2, step 2,000 | 5/20 | 1/5 |
| C2, step 5,000 | 14/20 | 4/5 |

Compare your 1,000-step result with the 2,000-step and 5,000-step values for states 0–4. Use the same five state IDs and the same base seed. Count all five attempts. Report policy errors and infrastructure errors apart from task failures. Your result can differ if your local environment or stochastic execution differs.

Do not compare any result with the unseeded 500-step check. That check used `reset(seed=None)`, so it is diagnostic only. There is no seeded 500-step score.

A higher step count uses more training compute. A lower training loss does not prove better closed-loop control. Five states and one training seed show only a trend. They do not give a stable success rate. The published model B has unknown training data and a different robot architecture. It is not the parent of C1 or C2.
