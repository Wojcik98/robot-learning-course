# Lab 6 exercise: Compare two sampler settings

1. Read `manifests/diffusion_reference.yaml`. Confirm the checkpoint path, model hash, training step count, training episode count, and base seed.
2. Use the trained step-2,000 checkpoint. Run the evaluator on development states 0–4 with 5 denoising steps. Use base seed `20260930` and a new output directory. Do not use final states.
3. Compare your five attempts with the tracked 5-step and 20-step ablation records. Match each state ID. Report successes, policy errors, infrastructure errors, and action-generation latency.
4. Keep the reset seed and initial noise seed paired by state. Later noise draws can differ because the samplers consume different numbers of random values. Explain why the 5-step setting can change both latency and behavior.
5. Run one additional live rollout on development state 2 with the 20-step sampler. Record the model hash, reset seed, success, termination reason, elapsed time, and sampler-call latency. Inspect the saved state-0 failure separately.
6. Explain why five states and one training seed do not show a stable success rate. Explain why the lower training loss cannot prove better task behavior.

Example command for the new five-state run:

```bash
.venv/bin/python -m vla_lab.evaluation.diffusion \
  --checkpoint .local-data/runs/diffusion-bowl-plate-reference-2000/checkpoint-002000 \
  --state-ids 0 1 2 3 4 \
  --seed 20260930 \
  --inference-steps 5 \
  --output .local-data/runs/my-diffusion-5step-dev-0-4
```

Report all five attempts. Keep task failures separate from policy and infrastructure errors. Do not use final states 20–49 to select the sampler setting.
