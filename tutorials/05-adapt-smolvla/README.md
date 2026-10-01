# Lab 5: Train a SmolVLA robot policy

This lab follows one learner model from source A to the trained checkpoints C1 and C2. It shows which weights come from the pinned SmolVLM source. It shows which weights start new and which weights get gradients. It then checks the saved files, the model lineage, the training loss, and the live development results.

Run from the repository root after Lab 4 and the training data download:

```bash
.venv/bin/robot-lab data fetch --scope training
.venv/bin/robot-lab data fetch-models
.venv/bin/jupyter lab tutorials/05-adapt-smolvla/05-adapt-smolvla.ipynb
```

To train both stages yourself, use two new output directories:

```bash
.venv/bin/python -m vla_lab.training.smolvla \
  --stage c1 --output .local-data/runs/my-smolvla-c1

.venv/bin/python -m vla_lab.training.smolvla \
  --stage c2 \
  --resume-from .local-data/runs/my-smolvla-c1/checkpoint-000100 \
  --output .local-data/runs/my-smolvla-c2
```

The course config uses 39 frozen bowl-plate training episodes, a batch of one, a 50-action prediction chunk, and ten executed actions. C1 is the declared 100-step early checkpoint. C2 continues from C1 to 5,000 total steps. The lineage is C1 (step 100), step 500, step 2,000, and step 5,000. Each checkpoint is the parent of the next one.

## Reference results

Each result below uses the 20 development states 0–19. Each state uses an explicit simulator reset seed, and the base seed is 20260930. No final state is used.

| Checkpoint | Successes | Development states |
| --- | --- | --- |
| C1, step 100 | 0 | 20 |
| C2, step 2,000 | 5 | 20 |
| C2, step 5,000 | 14 | 20 |

You can compare these three results with each other. They use the same states and the same seeds. The 5,000-step checkpoint is the selected C2. The course selected it with development states only.

The earlier 500-step check used `reset(seed=None)`. It is diagnostic only. Do not compare its score with the seeded results. This lab has no seeded 500-step score.

## Cost and limits

The local GPU can run these stages. Keep several GiB free for the VLM source, model checkpoints, optimizer state, and data. The saved optimizer and random states let training continue from the saved step. The checkpoint manifest records these items:

- the VLM source
- the training split
- the train-only statistics
- the hashes of the processor files, `config.json`, and `optimizer.pt`
- the model hash and the parent model hash

The four local training jobs used about 829 seconds in total. The final step-2,000 to step-5,000 job used 493 seconds. No paid cloud job was used. These times exclude data download, setup, and evaluation.

The published model B from Lab 4 is a separate reference with unknown training data. It is not a parent of C1 or C2.

The notebook also reads the first seeded ACT/VLA development comparison. ACT succeeded on 19/20 states, published B on 16/20, and learner C2 on 14/20. These are practical recipes with different models, pretrained data, and robot training steps. The counts do not isolate an effect of pretraining or architecture. Final states stay closed until the course freezes all recipes.

Use the [exercise](exercise.md) after the notebook. Read the [solution guide](../../solutions/05-adapt-smolvla.md) after you save your answer.
