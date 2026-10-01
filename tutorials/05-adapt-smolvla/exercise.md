# Lab 5 exercise: Check a 1,000-step continuation

1. Copy `configs/train/smolvla_bowl_plate.yaml` to a new YAML file. Set `c2_steps` to 1000. Keep every other field the same.
2. Continue from the saved course C1 checkpoint at step 100. Use a new output directory and your YAML file with `--config`. Do not start from published model B.
3. Read the new C2 manifest. Make sure `stage` is `c2` and `step` is 1000. Make sure the parent model hash equals the C1 model hash.
4. Verify the model, `config.json`, `optimizer.pt`, and processor files against the manifest hashes. Check the VLM source and the split hash.
5. Evaluate states 0–4 with `.venv/bin/python -m vla_lab.evaluation.smolvla_learner --checkpoint YOUR_CHECKPOINT --state-ids 0 1 2 3 4 --output NEW_DIRECTORY`.
6. Compare your result with the saved seeded results on the same five states. Use the 2,000-step and 5,000-step results in `artifacts/reference/smolvla/learner_c2_2000_dev_all` and `learner_c2_5000_dev_all`. Read their `episodes.jsonl` files and use states 0–4 only.
7. Report every failed attempt and every infrastructure error. Explain why one training seed and five states do not give a stable success rate.

Do not compare your result with the earlier 500-step check. That check used `reset(seed=None)`. It is diagnostic only. This lab has no seeded 500-step score.

The 1,000-step run changes the training compute. Do not use final states 20–49 to select a continuation length.
