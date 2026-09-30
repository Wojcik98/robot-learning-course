# Lab 1: Understand ACT

This lab traces one training example through the saved ACT processor and model. It shows both camera views, the robot state, a 20-action target chunk, the CVAE loss terms, and a predicted action chunk. It then uses a small action queue to show temporal ensembling.

Run from the repository root after Lab 0 and the training data download:

```bash
.venv/bin/robot-lab data fetch --scope training
.venv/bin/jupyter lab tutorials/01-act/01-act.ipynb
```

The reference path uses one training example and one saved checkpoint. It does not change checkpoint weights. The loss is a diagnostic value for one example, not a validation score. The corresponding live ACT run is `robot-lab demo` from Lab 0.

This lab needs the local GPU and the saved ACT bundle. Its download adds the selected two-camera videos and Parquet shards. The code uses development and training data only. Do not use final simulator states 20–49 to choose a setting.
