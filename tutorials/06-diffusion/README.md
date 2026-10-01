# Lab 6: Train and inspect a Diffusion Policy

This lab checks one trained Diffusion Policy for the bowl-plate task. It covers observation history, action horizons, end padding, train-only state and action statistics, the training loss, checkpoint hashes, seeded development results, and a denoising-step comparison.

Run from the repository root after you fetch the training data and the pinned LIBERO assets:

```bash
.venv/bin/robot-lab data fetch --scope training
.venv/bin/jupyter lab tutorials/06-diffusion/06-diffusion.ipynb
```

To train a new reference, first make sure the pinned ResNet-18 file is in the local Torch cache. The training code checks its hash and stops if the file is missing or has a different hash. Then use a new output directory:

```bash
TORCH_HOME=.local-data/torch .venv/bin/python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"
TORCH_HOME=.local-data/torch .venv/bin/python -m vla_lab.training.diffusion \
  --output .local-data/runs/my-diffusion-reference
```

The notebook reads the supplied 2,000-step reference by default. Do not write a new run into that reference directory.

The reference recipe uses 39 frozen training episodes, a batch size of 8, and 2,000 optimizer steps. It uses two observation frames, a 16-action prediction horizon, and eight executed actions. It uses two 256-pixel camera inputs resized by the policy to 128 pixels. The visual input uses the pinned ResNet-18 ImageNet weights. The policy uses ImageNet mean and standard deviation for the images. It uses state and action extrema from the training split only.

The data loader excludes the last seven frames of each episode as current frames. It marks padded action values, and the loss mask excludes them. The notebook checks a real training window and the true end of an episode.

## Reference results

The 2,000-step run used seed `20260930`. The training loop took 195.3 seconds on the recorded local run. It reached a final recorded noise-prediction loss of 0.2203. Loss is a training diagnostic. It does not measure task success.

The seeded evaluation uses the 20 bowl-plate development states (IDs 0–19) and base seed `20260930`. It recorded **15 successes in 20 attempts**, with 0 policy errors and 0 infrastructure errors. The reference manifest and `artifacts/reference/diffusion/seeded_dev_20step_all/` hold the checkpoint identity and attempt records. No final state was used.

The sampler ablation uses paired development states 0–4. The reset seed and initial noise seed stay fixed for each state. Later noise draws can differ because the samplers consume different numbers of random values. Both settings succeeded on 2/5. The 5-step successes were states 1 and 2. The 20-step successes were states 2 and 3. The weighted mean sampler latency was 29.052 ms for 5 steps and 99.263 ms for 20 steps. The 20-step mean was 3.417 times the 5-step mean. The timing measured CUDA-synchronized `generate_actions` calls on the local RTX 3070. The mean includes the first cold call. It excludes simulator and processor time. The five-state sample is too small to rank behavior. The selected setting stays at 20 steps because the full 20-state result was stronger evidence.

The notebook includes a separate live rollout of development state 2, which succeeded in the saved reference run. It writes a new result under `.local-data/runs`. State 0 is a saved failure to inspect. The notebook uses no final state. Run the live cell after the saved-result cells.

## Limits

One training seed and 20 development states do not establish a general success rate. The denoising-step comparison changes sampling compute. A behavior difference can result from the sampling setting. The course has not used final states to select the checkpoint or sampler setting.

Use the [exercise](exercise.md) after the notebook. Read the [solution guide](../../solutions/06-diffusion.md) after you save your answer.
