# Lab 4: Run and inspect SmolVLA

This lab runs a published SmolVLA robot policy. It shows two camera images, robot state, text, and a predicted action chunk. It checks how often the policy asks for a new chunk. It also keeps the simulator goal visible when you change the model prompt.

Run from the repository root after Labs 0–3:

```bash
just data-training
just data-models
just jupyter tutorials/04-smolvla/04-smolvla.ipynb
```

The published model is `HuggingFaceVLA/smolvla_libero`. Its training data are unknown. The local learner in Lab 5 starts from a different VLM source. Do not treat the published model as the learner's parent.

The prepared development result used ten actions per policy call and succeeded on 16 of 20 bowl-plate states. The model's stored setting uses one action per call. The five-state check succeeded on 3/5 with one action and 4/5 with ten. Each run used an explicit simulator reset seed. These results use development states. They are not a final test.

To run a prompt check on one fixed state, use a new output directory for each prompt:

```bash
just eval-smolvla \
  --model-dir .local-data/models/smolvla-libero \
  --vlm-dir .local-data/models/smolvlm2-500m-video-instruct \
  --task bowl_plate --state-ids 0 --n-action-steps 10 \
  --model-prompt 'place the bowl on the plate' \
  --output .local-data/runs/my-smolvla-paraphrase
```

The output records `evaluation_task` and `model_prompt` for each attempt. A prompt change does not change the bowl-plate success check. Use a scene that has the objects named in the prompt. The [exercise](04-smolvla.ipynb#exercise) and [solution guide](../../solutions/04-smolvla.md) use this rule.

The saved state-0 prompt check kept the model, ten-action setting, reset seed, and bowl-plate success check fixed. The canonical and paraphrased prompts succeeded in 74 and 85 steps. The conflicting prompt failed at the 300-step limit. One state cannot show whether the model followed the words; object motion needs a separate check.
