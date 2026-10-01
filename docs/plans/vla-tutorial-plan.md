# End-to-end robot learning: ACT, VLAs, and Diffusion Policy

**Created:** 2026-09-29

**Revised:** 2026-09-30

**Status:** Claude review applied. Ready for final user review and handoff to Sol. Implementation has not started.

**Learner:** Senior robotics engineer with classical robotics, simulation, and RL experience.

**Goal:** Build a complete simulation course that lets the learner run, train, adapt, inspect, and deploy learned visuomotor policies, then judge whether this work is a good long-term fit.

**Execution:** GPT Sol leads implementation and integration. Luna subagents do bounded tasks. Claude Code helps with selected work and independent reviews. GPT Astra advises on design and difficult findings.

## 1. Recommended approach

Use **LeRobot and LIBERO**, with **ACT → SmolVLA → Diffusion Policy** as the teaching sequence. Introduce one method at a time. Reuse the robot, task data, and evaluation tools when adding the next method.

Start by running a supplied ACT checkpoint on a simple manipulation task. Then inspect and train ACT before introducing the first VLA in code. Explain the broad relationship to VLAs in the early text. Later notebooks run and adapt SmolVLA, introduce Diffusion Policy, and compare all three with practical exercises.

The implementation team must prepare and verify the first ACT checkpoint before delivering Lab 0. Train a small task-specific reference during P0, with recorded provenance. The learner's first session must not depend on completing a training run.

The main interface is JupyterLab. Use `just` for readable command aliases, a Python CLI for detailed control and completion, and `uv` for a locked project environment. Plan for 30–50 learner hours. Use local compute first. The course budget remains €150 per month; implementation debugging has a separate prepaid allowance of approximately US$30.

Build in three releases:

1. **ACT and the common workflow:** Live simulation, the model, demonstrations, training, and evaluation.
2. **VLAs and Diffusion Policy:** Introduce each method, run the same work cycle, then compare data needs, behavior, and cost.
3. **Runtime and research:** Data recovery, timing faults, action chunks, a paper mechanism, and a deployment decision report.

Each release must work on its own. Complete the first working run before building the rest of the course tools.

### Alternatives considered

| Approach | Benefits | Costs | Decision |
|---|---|---|---|
| LeRobot + ACT, SmolVLA, Diffusion Policy + LIBERO | Three policy types with a common demonstration and evaluation workflow | Verify each adapter; prepare the initial ACT checkpoint | Main path |
| OpenPI + a π model + LIBERO | A useful larger-model reference and policy server examples | More VRAM; a second software stack if OpenPI is used directly | Optional cloud comparison; prefer a compatible LeRobot model first |
| Isaac-based simulation + a humanoid model | Useful for later whole-body work | More assets, compute, and integration work before the first useful exercise | Separate extension after the core |

These choices optimize the learning sequence. They are not a ranking of VLA model quality.

Here, end-to-end means a learned mapping from sensor observations to the selected robot action representation. The simulator's controller still converts those actions into robot motion. Each lesson states where the learned policy ends and the controller begins.

## 2. Requirements and decisions

### Confirmed by the user

- Start with manipulation. Physical robots are unavailable.
- Use simulation for all required exercises.
- Local GPU: RTX 3070 with 8 GB VRAM.
- Use local compute when it fits; switch the same commands to cloud profiles when needed.
- Course planning limit: **€150 per month**. The user separately authorizes up to **approximately US$30 of prepaid credit for implementation debugging** once funded. Use the lower of the confirmed funded project balance and US$30 as the implementation cap. No automatic top-ups.
- Monitor cloud jobs and stop them if stalled or much slower than expected.
- Core course: **30–50 learner hours**, with optional advanced labs.
- **Jupyter notebooks are the main learner interface.**
- Webcam control and humanoids are later extensions.
- No cloud account exists yet. Include account setup in the implementation instructions.
- The user can provide another local disk or free at least 60 GB.
- Cover ACT, a compact VLA, and Diffusion Policy through runnable exercises. Introduce each in sequence.
- Cover pretrained inference, supervised training/adaptation with public data, data formats, observation and action spaces, research, and deployment issues.
- Discuss other approaches in text where useful. Core coding exercises use the demonstration-to-policy workflow.
- Provide readable CLI help, command discovery, and shell completion. Prefer `just` aliases.
- Choose environment isolation from measured compatibility. Host packages are acceptable when setup is simple.
- Keep extension points small and useful for later course parts.
- Keep planning changes uncommitted during review. After plan approval, commit implementation directly to `main` with Conventional Commits and preserve existing edits.
- Keep only course, setup, source, tests, and useful evidence files. Do not store review prompts or duplicate handoff documents.
- Collect user setup and standing permissions in one initial stage. Verify them before unattended implementation starts.
- Include the routine work: data inspection, failed experiments, evaluation, regression checks, and reports.
- Prepare the plan for review before implementation.
- Write lessons, documentation, and comments in ASD-STE100 Simplified Technical English. Define technical terms when needed.
- Check the Codex 5-hour usage window. Above 85%, pause current agents and tasks and arrange a resume after reset. Do not start new work while paused.
- Use Luna for Codex subagents. Select effort for each task.

### Remaining implementation assumptions

- Learner-hour estimates exclude downloads and unattended GPU jobs.
- Use notebook text, diagrams, controls, plots, and small code exercises, with separate reference solutions.
- Support both local storage and cloud storage. The exact local data directory will be selected at setup.
- Store large assets under a configurable local data root. Recheck available space before each large download.

The main course preferences are confirmed. Exact runtime versions and resource profiles depend on the feasibility gate.

### Current workspace observations

Checks on 2026-09-30 found Ubuntu 24.04, about 31 GiB RAM, and **166 GB free** on the workspace filesystem. Recheck space during setup. `uv 0.9.28` and `just 1.36.0` are installed.

The GPU is available through approved host execution. Outside the restricted sandbox, `nvidia-smi` reports **RTX 3070, 8192 MiB, driver 580.178.04**. The CUDA driver API also passed `cuInit`, device enumeration, and device-name lookup. The restricted command still cannot access the GPU. This is an execution-access issue in this session; driver replacement is not required by these findings.

PyTorch is not installed in the inspected default Python environment. A tensor operation, simulator rendering, and a policy rollout still need verification in the project environment. See [initial setup and GPU tests](../initial-setup.md#gpu-access-and-tests).

Git is initialized on `main`; the latest commit is `33b9969`. The user removed the earlier planning commit. Leave this revision uncommitted. After approval, use direct Conventional Commits such as `docs(plan): expand the policy course` and `feat(eval): add paired policy reports`. Check the working tree first and stage only files owned by the current change.

### Initial setup before implementation

Complete **S0** using [the initial setup checklist](../initial-setup.md). Collect storage, account access, the cloud spending envelope, and external worker access in one batch. Sol prepares the exact commands and checks the result. The user performs account login, billing, and any required system approval together.

After S0, routine installation, downloads, training, cloud lifecycle actions, and tests proceed within the recorded scope. Ask again only when a required action exceeds that scope, credentials stop working, or the tool permission system requires it. Do not repeat approval questions for actions already authorized.

## 3. Technical basis and feasibility checks

Primary sources were rechecked for the review on 2026-09-30. Full runtime compatibility remained a P0 check at that time. The original Claude review notes are not in this repository; this plan contains the accepted implementation decisions.

### Method map

ACT is a visual imitation policy. Standard ACT uses images and robot state to predict action chunks; it has no language input. It is therefore not normally called a VLA. Action chunking alone does not make a policy a VLA. The [ACT project and paper](https://tonyzhaozh.github.io/aloha/) describe its conditional variational autoencoder and sequence prediction. The course covers the latent variable, reconstruction and KL losses, inference behavior, and temporal ensembling.

[LeRobot recommends ACT as a starting policy](https://huggingface.co/docs/lerobot/act). Use that implementation for the first labs. The original ACT work used joint control; our LIBERO exercise will train it on the selected dataset's state and Cartesian action representation. Verify this configuration explicitly. This is an application of ACT, not a reproduction of its original robot experiment.

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) is the third practical method. It learns action sequences by conditional denoising and executes a receding horizon. It fits the same demonstration, training, and rollout workflow. Use [LeRobot's implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/diffusion) if P0 verifies it on the chosen task. Teach observation history, prediction horizon, execution horizon, and denoising steps separately.

| Method | Conditioning in this course | Training | Main comparison question |
|---|---|---|---|
| ACT | Images and robot state; one task per policy | Supervised imitation; optional pretrained image encoder | How far can a compact task-specific policy go? |
| SmolVLA | Images, state, and instruction | Adapt a pretrained model | What does pretraining and language add for this task and data budget? |
| Diffusion Policy | Images and state history; one task per policy | Supervised imitation with a denoising objective | Does iterative action generation change robustness or latency? |

These are credible methods for comparison, not a claim of equal performance. Results depend on task, data, controller, pretraining, and compute. Include a short text survey of recurrent behavior cloning, action tokenization, flow matching, and RL adaptation. The learner implements the three main methods through existing backbones and libraries.

### Main policy and simulator

The first demo uses a verified ACT checkpoint with its training recipe, split, and processor. Train two separate task checkpoints if the two-task usability gate needs them. Standard ACT receives no text prompt, and the first notebook must not display a language-control field.

The VLA candidate for the later labs is [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero). This is a published SmolVLA checkpoint adapted to LIBERO. Its model card has generic examples that do not form a reliable simulation recipe. Implementation must use the checkpoint files, the pinned library code, and an actual rollout to establish the recipe.

Its [configuration](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/main/config.json) currently declares two RGB inputs, an 8-element state, and 7 action elements. It also declares a chunk size of 50, execution of one action per selection, mean/std state and action normalization, and image processing that includes 512 × 512 padding/resizing. These are facts to check at the selected revision. Preserve the bundled processor and run the stored `n_action_steps=1` recipe first. Also test `n_action_steps=10` on development states; select and record the reference setting there. The current documentation uses 10 for π₀.₅, so this is a SmolVLA ablation, not a claimed published SmolVLA recipe.

[LeRobot's LIBERO adapter](https://huggingface.co/docs/lerobot/libero) supplies a manipulation environment with a main camera, wrist camera, and end-effector control. It provides a Linux path and headless rendering options. Use the task's native controller first. Record its frames, scaling, control period, and reset rules in the experiment manifest.

The [base config](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json) declares 6-element state/action, three camera keys, 16 VLM layers, and expert width 0.75. The published LIBERO config declares 8/7 state/action, two cameras, all VLM layers, and width 0.5. Do not treat these as interchangeable checkpoints. The published LIBERO model has unknown training-data provenance. Section 5 defines the learner initialization separately.

### Tasks and data

Use **LIBERO-Goal**. Start with `put_the_bowl_on_the_plate` and `put_the_bowl_on_the_stove`; retain `turn_on_the_stove` as the third task for the final comparison. These are task-name suffixes, not invented library IDs. P0 resolves exact IDs from the pinned task registry and checks episode counts, camera views, initial states, and success predicates. The shared scene supports a later task-selection exercise. Train the first ACT reference on the bowl/plate task, then the second reference for the demo gate.

Use [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), the compact AV1 video dataset, with a pinned revision. Confirm the actual selected episode counts rather than assuming 50 demonstrations per task. Decode actual frames during P0. The large image-based copy is not an automatic fallback; first repair the decoder or select the supported alternative backend.

The dataset records 10 Hz metadata, while [LeRobot's environment config](https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/configs.py) specifies a 20 Hz controller. Set and verify **one simulator control tick = 50 ms**. Use this clock for delays and deployment metrics. Use dataset indices to select training frames; treat the dataset timestamps as labels, not proof of elapsed physical time. Do not multiply or resample actions merely to reconcile the metadata rates.

Claude observed very few no-op actions and varying episode counts. Filtering is a plausible explanation, but its exact history remains unverified. Record it as a provenance limit. Lab 8 measures near-zero actions and studies an explicit synthetic timing defect. It must not claim to reconstruct missing physical time.

Use [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) metadata to resolve episode boundaries. Teach HDF5 and [RLDS/Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) from small samples. Save state/action schemas, units, transforms, data hashes, and conversion assumptions.

### Compute

- **8 GB local:** Test ACT and Diffusion Policy inference and small training batches, then SmolVLA inference, with one environment. Start at batch size one for the probes. Measure total device memory with rendering active. Aim to leave at least 1 GB free for the desktop and transient allocation. Report measured latency and memory. Do not claim a guaranteed fit.
- **24–48 GB cloud:** Initial target range for small-model training and larger-model inference. Run a brief memory and throughput probe before assigning a full job. A 24 GB GPU is a candidate, not a minimum supported specification.
- **48–80 GB cloud:** Optional large-model training if measured need and budget permit it.
- **CPU:** Support data inspection, replay, contract tests, and small simulator checks. A recorded video must be labelled as replay. It does not count as a live policy rollout.

Probe small-model training locally before cloud use. Include ACT/DP training, the SmolVLA action expert, and RTC guidance separately: guided sampling uses gradients and may need more VRAM than ordinary inference. The course must retain a cloud fallback for a failed memory or practical runtime gate. For context, [OpenPI](https://github.com/Physical-Intelligence/openpi) lists more than 8 GB for inference, more than 22.5 GB for LoRA, and more than 70 GB for full fine-tuning for its stated configurations. These figures do not apply to every implementation or policy.

### Feasibility gate before full course construction

1. Resolve and record exact model, data, simulator, and library revisions.
2. Decode one real AV1 file. Validate camera order, image transforms, state layout, action convention, normalization, and the 20 Hz controller from actual data and simulator steps.
3. Run a rendered simulator reset and inference for ACT, SmolVLA, and Diffusion Policy. Check tensor shape, finite output, state reset, and correct action timing for each.
4. Inventory initial states and freeze development/test IDs before model selection. Complete the two ACT demo tasks on development states only. Save every result, including failures.
5. Course target: success on at least half of all development states on each task (normally at least 10 of 20). This is a first-demo usability gate, not a benchmark claim.
6. Use the section 5 initialization for SmolVLA. Audit loaded/reinitialized parameters and probe `n_action_steps` 1 and 10. Run a short training probe for each method: finite loss and gradients, intended trainable parameters, saved checkpoint, reload, and rollout. Confirm resource use. Prepare the full ACT reference for the first demo and show nonzero task success. Short gradient probes alone do not establish learned behavior for the later models.
7. Record total local disk use, peak VRAM, cold and warm inference times, and run cost.

If local inference fails the resource gate, run the same small-model experiment in the cloud. If the ACT reference fails the task gate, diagnose its data, controller mapping, and recipe, then retry within the allocated budget. Stop P0 with a concrete alternative if it remains unusable. Later SmolVLA and Diffusion Policy labs must each have a verified working reference before release. Do not silently change the teaching order, simulator, or method to hide a failed gate.

## 4. Curriculum

Use a short introduction to connect learned visuomotor control to perception, planning, and control. Early text mentions the other methods, but code exercises complete one method's workflow before adding the next.

| Lab | Work and key question | Learner output | Time |
|---|---|---|---|
| 0. Run ACT | Load the supplied checkpoint, run a manipulation task, reset, and inspect success and failure. | Live rollout, video, and run manifest | 1–2 h |
| 1. Understand ACT | Trace image/state inputs, the CVAE, action chunks, losses, and temporal ensembling. Explain how a VLA differs in text. | Tensor trace and chunk/ensemble exercise | 2–3 h |
| 2. Data and evaluation | Inspect complete demonstrations, timing, frames, actions, and normalization. Establish episode splits and fixed evaluation states. | Dataset audit and baseline report | 3–4 h |
| 3. Train ACT | Train a task policy on public demonstrations, reload it, and evaluate it. Compare demonstration counts and one chunk setting. | Trained policy, curves, and held-out results | 3–5 h |
| 4. Run and understand SmolVLA | Introduce images, text, state, pretrained backbone, and flow action expert. Run the published adapted checkpoint and test instructions. | VLA rollout and language-behavior analysis | 2–3 h |
| 5. Train and adapt SmolVLA | Initialize a compatible action expert on a pretrained VLM, then continue from an early robot checkpoint. Compare with ACT and keep the published VLA reference separate. | Adapted policy and first ACT/VLA comparison | 4–5 h |
| 6. Diffusion Policy | Introduce denoising and action horizons, then train/evaluate a policy using the same data. Inspect noise prediction and the latency effect of sampling steps. | Third policy and sampler experiment | 3–4 h |
| 7. Compare the methods | Compare success, data use, shift sensitivity, compute, and runtime. Separate single-task control from language-based task selection. | Three-method report and a justified method choice | 3–4 h |
| 8. Data quality and recovery | Repair a timing or action error and record 5–10 short recovery demonstrations with a supplied keyboard tool. Apply one bounded data change and training continuation. | Curated dataset and paired result | 3–4 h |
| 9. Runtime failures | Inject delay, jitter, dropped frames, stale replies, and restarts. Compare chunk execution and ACT temporal ensembling. | Success/latency plots and incident report | 3–5 h |
| 10. Paper mechanism | Implement an RTC component for SmolVLA and compare with the reference. Use same-model execution baselines. | Tested mechanism and ablation report | 4–6 h |
| 11. Deployment and career assessment | Package a selected policy, run regressions, recover from a bad release, and estimate a new task's cost. | Release, runbook, and decision memo | 2–3 h |

**Total:** 33–48 guided hours, including the timeboxed readings below. Downloads and unattended training are additional elapsed time. Keep the core to one or a few declared tasks. Broader task suites, large sweeps, and multiple training seeds are extensions. Supply completed reference runs so a learner can inspect the next lesson while their job runs.

### Lesson design

Every lab includes:

1. A learning goal and a connection to classical robotics.
2. Required inputs, compute class, estimated disk use, run time, and cost.
3. One command that runs the reference case.
4. A short exercise that changes one factor.
5. Expected observations and failure symptoms. Use measured reference artifacts when available.
6. Questions that require interpretation, not only a screenshot of success.
7. A reference solution stored separately from the exercise.
8. A completion check and a note on what the result does not establish.

Use Jupyter notebooks for every lab. Include task controls, inline rollout video, data inspection, plots, and exercise cells. Keep reusable training and rollout logic in tested Python modules, so notebook state cannot change the result silently. Show the effective configuration and the equivalent CLI command for each run.

Long training and evaluation jobs run as detached processes with a run directory, status/heartbeat file, and an atomic lock keyed by resolved config hash plus run purpose/seed. Use an explicit repeat flag for a new intentional repetition. Check process identity before recovering a stale lock. A persistent job service is unnecessary. A notebook submits a job and reads its status and logs. Kernel restart or browser closure must not terminate the job or submit it twice. Provide cancel, reconnect, and artifact retrieval controls. Run each package's reference notebooks with nbclient from a fresh kernel at that package's exit; do not defer this check to the final release. Exercise cells must be clearly marked and separate from the fully runnable reference path.

### Task controls and language

Lab 0 selects a scene and its matching ACT checkpoint. Its task label is experiment metadata. Standard ACT does not read that label as a command.

Add instruction editing when introducing SmolVLA in Lab 4. Explain that the scene must contain the required objects and that changing text does not change the success predicate. Save both the evaluation task and the model prompt. Include canonical, paraphrased, and conflicting prompts. Free-form prompts are experiments with measured outcomes.

## 5. Training and evaluation design

### SmolVLA initialization and comparison

- **A: Starting weights for the learner model.** Construct SmolVLA with the selected compatible LIBERO inputs/actions, `policy.type=smolvla`, and `policy.load_vlm_weights=true`. Load the pinned `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` VLM; initialize the action expert and task projections for this configuration. Start with 16 VLM layers and expert width 0.75. Log every loaded, frozen, trainable, missing, and reinitialized parameter group. Reject unexplained missing keys. Report A as **not evaluable as a trained robot policy**; do not spend budget on a mapped zero-shot rollout of `smolvla_base`.
- **B: Published `HuggingFaceVLA/smolvla_libero`.** Use as a separate practical inference reference with its own architecture and bundled normalization. Its training data are unknown. Do not infer that it descends from A or that a B–C gap measures the effect of the learner's data.
- **C: Learner model trained from A.** Freeze the pretrained vision/language backbone initially; train the action expert and required state/action projections. P0 verifies the actual parameter groups and upstream flag behavior. Save an early robot checkpoint C1, then continue to C2 on the declared training split. This provides a genuine checkpoint adaptation exercise. Compare C1/C2 and ACT on identical final states only after selection is complete.

The [documented LIBERO training recipe](https://huggingface.co/docs/lerobot/libero) loads VLM weights. It does not establish a direct fine-tune chain from `smolvla_base` to the published LIBERO model. Describe initial C training as learning a robot action expert on a pretrained backbone, and C1→C2 as continued robot-policy training.

Keep C1/C2 architecture and processor fixed. Choose C1 from a predeclared early step and C2 from development checks, not the final test. Continued training changes training compute; report that difference. LoRA is a later comparison after the core recipe works.

A learner run need not improve. Provide measured reference runs and preserve negative results. A poor result triggers a bounded diagnosis, not hidden task or model selection on final-test states.

### Splits and claims

- Split by complete episode before making frame windows. Never randomly split adjacent frames across training and validation.
- Use group splits for object layouts or task families when making a generalization claim.
- Fit normalization and data filters on training data only. Materialize an episode split manifest before window sampling; use a fixed 80/20 training/offline-validation split within each task, with deterministic rounding and at least one validation episode.
- Recompute state/action statistics from only those training episodes. Do not assume selecting episodes changes `dataset.meta.stats`: the [trainer](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py) passes metadata statistics to processors. Inject the derived statistics, save their hashes in the checkpoint, and verify the actual tensors used. Use the selected pretrained image encoder's expected ImageNet normalization for ACT/DP, not statistics computed from the full dataset. Preserve B's own processor.
- Add a test where changing validation-only values cannot change training statistics, plus normalized-action round-trip and checkpoint-reload tests.
- Use validation to select hyperparameters and checkpoints. Keep the final test set fixed and unopened for model selection.
- Record what the source checkpoint already saw. A new episode split cannot remove earlier benchmark exposure.
- Label tests as familiar-task performance, new initial states, visual shift, language shift, or held-out task performance.
- Record failures, timeouts, controller limit events, and infrastructure errors separately. Show denominators; do not discard failed episodes to improve a score.

### Evaluation states and uncertainty

There are two independent partitions: **demonstration episodes** for training/offline validation loss, and **simulator initial states** for development/final closed-loop evaluation. One does not imply the other is disjoint.

P0 counts and hashes the available initial states for every selected task. With 50 states, use IDs **0–19 for smoke, demo gate, tuning, and checkpoint selection**; reserve **20–49 for the final 30-state test**. Freeze explicit lists in `configs/eval/splits.yaml`. Resolve and log the actual IDs used; reject modulo wrapping and unrequested repeated states. If a task has a different count, use the first floor(0.4N) states for development and the remainder for test, and revise the denominator in every report. Do not generate extra states for the core.

Use 2–3 development states for smoke tests, all 20 for a development comparison, and all 30 final states per selected task for the final comparison. Run sequentially or in small batches on 8 GB. Test that batching preserves the explicit state sequence.

The final states are held out from the course's tuning. They may overlap source demonstration start states or earlier model training. Audit accessible simulation-state provenance; if overlap cannot be resolved, state this limit. Do not claim unseen-state generalization. A strict new-placement benchmark is a later extension.

Report per-state paired outcomes, counts, 95% binomial intervals, and paired differences. Use an exact McNemar test where applicable, or bootstrap paired state outcomes; resample state clusters when stochastic repetitions are included. Repeating a deterministic state does not add an independent sample. Twenty states mainly expose large differences; a rough 25–30 percentage-point difference can be visible, but power depends on discordant pairs and no universal detection threshold is claimed. Small differences remain inconclusive.

One trained checkpoint does not estimate training-seed variance. Multiple training seeds and larger placement sets are optional research extensions. Keep final-test states out of tuning, including repeated timing ablations; design those on development states first.

### Fair comparisons across methods

ACT and Diffusion Policy are required practical baselines. Use identical source demonstrations, episode splits, simulator initial states, cameras, state information, controller conventions, success predicates, and rollout timeouts for the single-task comparison. Record model-specific resizing, normalization, history, augmentation, and horizon choices. Use each model's correct processor rather than forcing incompatible identical tensors.

Keep two separate questions:

1. **Single-task control:** Compare a task-specific ACT, task-specific Diffusion Policy, and adapted SmolVLA on the same task. The VLA gets a constant task instruction. Match the new demonstration budget and disclose all pretrained data and encoders. This compares practical recipes; it cannot isolate architecture from pretraining.
2. **Language-based task selection:** Use a scene with multiple possible goals and fixed scene states. Change the VLA instruction and the corresponding evaluation goal deliberately. ACT and Diffusion Policy need an externally selected task-specific checkpoint in this course. Label that selection as an oracle, measure its storage/training cost, and do not count it as language understanding.

Give each method a declared tuning budget and keep the test set fixed. Show both an equal-compute view and a practical tuned-recipe view where affordable. Equal steps alone do not match training cost. Report total pretraining exposure as known or unknown, adaptation time, examples seen, peak memory, inference latency, and success. Do not infer language generalization from a single-task score or claim that similar published scores transfer to this setup.

Initially compare deterministic settings where supported, then report sampling variation for stochastic inference separately. Document ACT latent handling and ensembling, Diffusion Policy noise seeds, and VLA sampling settings. A scripted controller can serve as a simulator sanity check; disclose privileged inputs.

### Useful metrics

Record task success, completion steps and time, action limit events, recovery attempts, and failure type. For runtime labs, add observation age, request and response times, queue depth, discarded actions, missed deadlines, p50/p95/p99 latency, and peak VRAM. Keep cold startup separate from warmed inference. Report action discontinuity with its units and representation.

## 6. Infrastructure to build

### Keep the learner interface small

The main interface is JupyterLab with a course index notebook. Use **`just` + `uv` + a Typer CLI**, with distinct responsibilities:

- `just`: Short documented recipes, command listing, and shell completion for recipe names.
- `uv`: Python version, dependency lock, project environment, and command execution.
- `robot-lab`: A typed CLI for detailed options, validation, completion, and structured output. Implement with Typer; notebooks call the same Python functions.

Examples of planned commands:

```text
just                         # List the available recipes
just doctor
just notebook
just demo-act
just train-act
just demo-vla
just train-vla
just train-diffusion
just compare
just cloud-status
just cloud-stop

uv run --locked robot-lab train --policy act --config configs/train/act_task.yaml
uv run --locked robot-lab eval --policy smolvla --profile cloud
uv run --locked robot-lab --help
```

After activating the project environment, the same CLI is available as `robot-lab`. Provide completion setup for Bash, Zsh, and Fish, and test Bash on this host. Follow the [just completion instructions](https://just.systems/man/en/shell-completion-scripts.html) and [Typer option completion](https://typer.tiangolo.com/tutorial/options-autocompletion/).

Test completion for `just` recipe names and, in the activated environment, `robot-lab` subcommands, options, policy names, profiles, task IDs, and config paths. Do not assume `just` completes arbitrary forwarded arguments or that completion works through `uv run` automatically. Detailed completion uses `robot-lab` directly; document this distinction.

Show grouped help, defaults, units, valid choices, short examples, and a clear error that suggests a correction. Provide readable summaries in a terminal and plain machine-readable JSON when requested. Completion must work offline without loading weights, opening a GPU context, or contacting a cloud service. Keep recipe logic thin and quote file paths correctly.

The visible task alias resolves to an exact benchmark task in the saved config. A profile selects compute and storage paths. `robot-lab train`, `eval`, `demo`, and timing experiments accept `--profile local|cloud` with the same experiment config; only resource and path settings change. Notebook controls expose the same choice. Record profile/environment differences. Do not silently rent compute on a local failure; switch only under the standing funded cloud scope.

### Execution profiles

1. **Local, default:** Notebook, simulator, renderer, and policy run on the workstation once the resource checks pass. Prefer this for inference, timing labs, and training that fits.
2. **Cloud:** Prefer batch jobs with results viewed in the local notebook. Where a full interactive cloud lab is needed, colocate simulator, policy, and Jupyter kernel on the VM and use authenticated SSH forwarding. Record every interactive VM hour as billed time.
3. **Split, later:** Local simulator with remote policy server. Use it to study network latency after the colocated reference works.
4. **Offline inspection:** Bundled small data samples, measured reference results, and replay. Make this path available when a cloud VM is stopped.

### Environment selection

Start with a **project-local `uv` environment and `uv.lock`** on this Ubuntu host. Start with Python 3.12, consistent with the checked [LeRobot requirement](https://github.com/huggingface/lerobot/blob/main/pyproject.toml). Sol pins Python, PyTorch/CUDA wheel, MuJoCo, and LeRobot together during P0. Use `uv sync --locked` and `uv run --locked` after the lock is verified. [uv documents the project environment](https://docs.astral.sh/uv/concepts/projects/layout/) and [locked synchronization](https://docs.astral.sh/uv/concepts/projects/sync/).

A Python environment isolates Python packages. It does not supply an NVIDIA kernel driver or every native graphics/video library. Include FFmpeg shared libraries with AV1 decode support and EGL/OpenGL runtime libraries in the S0 native dependency check. On Ubuntu, inspect `ffmpeg`, `libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libegl1`, and `libgl1` providers/versions before installing. Verify torchcodec can load the shared libraries, not just that the ffmpeg executable exists. Decode a real dataset file and render an actual frame in P0. Follow the selected [LeRobot installation requirements](https://huggingface.co/docs/lerobot/installation). A working GPU wheel does not require an arbitrary host CUDA toolkit installation.

Use one environment if ACT, SmolVLA, and Diffusion Policy resolve and pass tests together. If real dependency conflicts require separation, add one named runtime environment with its own lock and Jupyter kernel. Install kernels only for environments that exist. Record the kernel identity in notebook outputs.

Use Docker when P0 finds a concrete distro/native-library conflict, or when a cloud image provides a simpler repeatable setup. Pin its base digest and use the same Python lock where compatible. Record the reason in `docs/environment.md`. Build and test only the selected paths; do not maintain a container and several virtual environments without a demonstrated need. GPU containers still require host GPU access and the container runtime integration.

`doctor` checks GPU visibility on the selected machine, container GPU access when applicable, renderer output, video decoding, RAM, disk headroom, cache paths, and required model access. It reports a specific fix or selects an explicitly labelled inspection-only mode. It must not silently fall back to slow CPU inference and claim that the GPU profile passed.

### Storage and downloads

- Keep weights, datasets, and checkpoints outside source control.
- Support a configurable data root on another disk or the cloud VM.
- Preflight download, extraction, container, temporary, and checkpoint space together.
- Planning allowance: roughly 60 GB free for a comfortable local core setup; measure and publish the actual footprint in the first release. This is a reserve estimate, not a measured minimum.
- Current disk capacity is sufficient for the planned reserve; recheck before downloads. Prefer the compressed dataset and selected shards.
- Download only selected dataset files where the source layout permits it. Do not assume episode selection prevents full shard downloads.
- Resume interrupted transfers and check content hashes. Set cache quotas. Show a cleanup preview before deleting artifacts.
- Keep best and latest checkpoints, plus the base revision reference. Make additional retention explicit.

### Reproducibility

Lock Python packages, source commits, model and dataset revisions, simulator assets, and image digests where containers are used, after the feasibility gate. Save seeds, initial-state IDs, controller and renderer settings, full resolved configs, transformations, normalization, trainable parameter lists, and environment versions with each run.

Treat a checkpoint as weights plus its processor, statistics, action convention, tokenizer, and config. Test a saved bundle in a new process. Set external tracking and model uploads off by default; local JSON/CSV, videos, and an HTML report are sufficient for the course.

### Core interfaces

Use a narrow adapter around upstream APIs. Avoid building a second general robotics framework.

| Interface | Required content and behavior |
|---|---|
| Observation packet | Episode ID, reset counter, sequence number, observation tick, capture timestamp, camera identities and arrays, state schema, task ID, and optional instruction |
| Policy request | Observation packet, policy/config identity, declared horizon settings, and request ID |
| Action chunk | Originating episode/reset/request, observation tick, action array, action schema, controller period, and availability tick |
| Action schema | Frame, units, rotation convention, absolute/delta meaning, scaling, gripper meaning, bounds, and padding mask |
| Episode result | Initial-state ID, policy revision, success predicate version, termination reason, metrics, and artifact paths |

Use one schema version and reject incompatible inputs. Policy capabilities state whether language, observation history, temporal ensembling, or sampler guidance are supported. Hide unsupported controls and reject unsupported options. A model checkpoint owns its trained prediction horizon; a request cannot arbitrarily change it. Reset model state and clear pending actions on episode reset or task change. Validate finite values, shapes, ranges, and camera names before control.

### Extension rules

The common boundaries are justified by current use: three policies share evaluation, three entry points share commands, and simulation/recording share an observation-action contract. Use small concrete adapters and a lightweight protocol for `reset` and action prediction. Keep losses, optimizers, encoders, and training loops in the upstream method implementations.

Add a short `docs/extending.md` recipe for a new lesson: declare prerequisites and compute needs, add a config and notebook, reuse the evaluator, add a small fixture, and update the course index. A simple course index lists lesson IDs, prerequisites, outputs, and estimated time. It must be easy to append later parts without renumbering existing lessons.

A webcam extension can supply operator commands to the existing recording path. A later learning algorithm can consume recorded episodes or produce the same action contract. Add a new interface only when an actual second implementation needs it. Prefer a small explicit policy mapping over a plugin discovery system. Keep model-specific options visible instead of reducing everything to a least-common feature set.

### Proposed repository layout

```text
README.md                         Start here and first-run commands
justfile                          Thin documented command recipes
pyproject.toml / uv.lock           Project dependencies and lock
docs/environment.md               Selected runtime and measured host requirements
configs/                          Runtime, policy, data, training, and evaluation configs
manifests/                        Locked upstream revisions and asset hashes
src/vla_lab/contracts.py           Observation, action, and result schemas
src/vla_lab/cli.py                 Typer CLI and completion
src/vla_lab/sim/                   LIBERO adapter and task registry
src/vla_lab/policy/                LeRobot adapter and processor loading
src/vla_lab/runtime/               Action queue, virtual ticks, and fault injection
src/vla_lab/notebooks/             Notebook controls and detached-job client
src/vla_lab/data/                  Inspection, validation, conversion, and recording
src/vla_lab/training/              Thin training entry point and resume checks
src/vla_lab/evaluation/            Episode runner, metrics, and reports
src/vla_lab/cloud/                 Provider connection and job lifecycle
tutorials/00-first-run/ ...        Ordered notebook lessons and exercises
solutions/                        Reference exercise solutions
tests/                            Contract, CPU integration, and GPU smoke tests
docs/                             Setup, architecture, operations, evidence, and decisions
artifacts/sample/                 Small reference outputs and provenance
configs/eval/splits.yaml           Explicit development and final state IDs
```

## 7. Cloud operation and budget

Use local compute by default. NVIDIA Brev is the fallback once account access and a suitable quote are verified. Keep commands portable to an ordinary GPU VM through the same project environment. Provide one Brev connection/lifecycle adapter, not a multi-provider framework.

### Implementation allowance: approximately US$30

The user permits this prepaid balance for implementation debugging. It is separate from the future €150 monthly course budget. In S0, verify the actual funded amount, currency, taxes, billing granularity, and other resources sharing the balance. Set the project cap to the lesser of **US$30 and the confirmed amount allocated to this project**. Disable auto recharge and check that no postpaid overage route defeats the cap. If the provider cannot establish that boundary, finish local work and report the issue before using paid compute.

Use at most one stoppable VM. Conservative defaults: at most **US$3/hour**, **US$5 per job**, and **2 hours per job**, with the lowest applicable limit controlling the run. Reserve at least US$5 for storage, transfer, tax/billing uncertainty, and shutdown time. Allocate up to US$5 to access/runtime probes, US$15 to model debugging, and US$5 to interactive troubleshooting. These are ceilings, not targets. Never replenish credit automatically.

The [Brev billing guide](https://docs.nvidia.com/brev/guides/console-reference#billing--usage) documents shared credits and optional auto recharge. Verify these settings in the account; do not claim that a local cost counter alone enforces a provider bill. Export important data before stopping or credit exhaustion. [Stopped instances retain storage charges and may be unavailable on restart](https://docs.nvidia.com/brev/concepts/gpu-instances).

### Future course allocation

| Use | Monthly planning allowance |
|---|---:|
| Setup and compatibility | €10 |
| ACT, SmolVLA, and Diffusion Policy batch work | €60 |
| Runtime/research batch experiments | €20 |
| Interactive GPU VM time | €20 |
| Storage, transfer, taxes, and reserve | €40 |
| **Total** | **€150** |

At an illustrative €2/hour, the interactive allowance buys 10 hours, not the entire course. Most notebook reading, inference that fits, plotting, and timing exercises run locally. Use short VM sessions when necessary. Actual quotes control the allowed hours. This table does not authorize spending the future monthly allowance during implementation.

### Monitoring and stop rules

Before a remote job, save its quote, budget balance, config hash, expected work units, and deadline. Measure 100–200 warmed training steps or several evaluation episodes to estimate duration. Store progress as completed steps/episodes and phase; GPU utilization alone is not progress.

- A detached monitor checks heartbeat and progress every **60 seconds**. It runs independently of the notebook and assistant turn. Record completed steps/episodes at least every 30 seconds when work progresses.
- Declare startup/compile allowance before launch, initially at most **15 minutes**. After startup, stop on no work-unit progress for `max(5 minutes, 5 × measured p95 work-unit time)`, subject to the hard deadline.
- Stop when projected duration exceeds **2× the initial estimate**, once the overrun is at least 15 minutes. Do not extend a deadline automatically.
- Hard deadline is the minimum of **2 hours**, the approved cost-derived duration, and **2 × estimated duration + 10 minutes**. Reserve startup/export/stop charges in the cost calculation.
- On timeout, stall, budget exhaustion, or cancellation: request a checkpoint with a **60-second grace**, terminate the job, export the latest available result with a **120-second bound**, and issue provider stop. Export failure must not keep the GPU billing indefinitely.
- Verify provider state. Retry a failed stop after 15, 30, and 60 seconds, then notify the user immediately with the instance ID and stop command. Continue bounded reconciliation while access is available.
- Test an independent shutdown path before unattended remote jobs: a provider deadline or VM watchdog that can issue the actual provider stop operation, plus an external monitor. Killing Python or requesting guest OS shutdown is not proof that billing stopped. If this path cannot be verified, use supervised cloud runs only.
- Checkpoint/export during long jobs, not only at the end. When the assistant pauses for usage, the monitor and deadlines continue; stop paid compute before waiting for the reset.

Core fault tests use a fake provider for stalls, lost heartbeat, delayed billing data, stop failures, and monitor restart. S0/P1 verifies one short real allocation and shutdown within the US$30 cap. The user has already authorized monitoring and stopping overlong jobs; do not ask again for those routine actions. Public publishing and destructive cleanup of unrelated resources remain outside this scope.

## 8. Runtime and research labs

### Clocks and the primary timing instrument

Use a **deterministic virtual-tick scheduler** as the primary experiment. Confirm 20 Hz control in P0: delays `d = 0, 1, 2, 4` ticks correspond to 0, 50, 100, and 200 ms. Store delays in ticks; derive milliseconds from the verified controller period. Dataset timestamps never set this clock.

At observation tick `t`, obtain a chunk and assign availability tick `t+d`. Computing it may block wall time, but only virtual availability controls the simulated experiment. Advance physics through each intervening tick using the previous valid chunk or the declared hold. At tick `k`, a chunk based on tick `t` contributes index `k-t`; never restart a delayed chunk at index zero.

Key requests by `(episode_id, reset_counter, observation_tick, request_id)`. Reject old episode/reset IDs, duplicates, out-of-order replies that would replace newer accepted state, and exhausted chunks. Retain valid remaining actions until replacement. Queue underflow uses zero Cartesian motion with the current gripper command for the selected relative controller; verify that hold behavior in simulation. Reset queues, history, and policy state together.

Provide benchmark stepping for ordinary evaluation and wall-clock timing as a separate demonstration. Record renderer/inference contention and transfer cost. Feed measured latency distributions into virtual ticks with a stated quantization rule: ceiling to the next tick for conservative measured delays; paper-aligned tests specify integer `d` directly. Do not subtract unrelated clocks across machines.

Model process loss and connection loss in the core as scheduler events: unavailable policy for `k` ticks and optional policy-state reset. Test camera loss, stale observations, reordered/duplicate responses, underflow, and restart events. The real network server is a later split-deployment extension.

### Horizon names

| Symbol | Meaning in the course | LeRobot mapping |
|---|---|---|
| H | Predicted future action count | `chunk_size` for ACT/SmolVLA; diffusion `horizon` needs explicit history alignment |
| s | Control ticks between successive prediction starts | Scheduler setting; ordinary synchronous selection uses `n_action_steps` |
| d | Observation-to-available delay in control ticks | RTC `inference_delay`; scheduler availability offset |
| L | Guided overlap length, `H-s` | RTC `execution_horizon`, after checking the pinned implementation |
| n | Denoising integration steps | Model-specific sampler setting |

Use valid RTC settings `d <= s <= H-d`. Hold `H` fixed for each trained model. Sweep ordinary execution settings 1/5/10 where valid; an `s=1, d=4` RTC case is invalid and must fail configuration validation. ACT's upstream temporal ensembling uses synchronous one-action selection. Teach it that way in Lab 1/9. Delay-aware ensembling in our virtual scheduler is new adapter code and must have its own tests.

### Paper mechanism and controlled comparison

Use [RTC v2, equations 2–5 and Algorithm 1](https://arxiv.org/html/2506.07339v2). Keep one SmolVLA checkpoint, its processor, noise seeds, task states, and scheduler fixed. Compare synchronous execution, naive asynchronous execution, temporal ensembling of that same model's chunks, and RTC. Ablate RTC hard-prefix and soft masks. Keep ACT's ensembling in Lab 9; an ACT/SmolVLA score gap cannot isolate RTC.

Test the mask numerically against equation 5, including `d=0`, `d=L`, and invalid horizons. The hard region covers `i<d`; the soft region ends at `L`; later weights are zero. Test guidance clipping and finite gradients at the first denoising step. Start the paper check with `n=5`, `beta=5`; also report the checkpoint's native sampling setting as a separate condition. Guidance uses a vector-Jacobian product, so measure its VRAM separately. A guided prefix is not guaranteed to match exactly; actions already committed to the controller must remain unchanged, and generated-prefix error is a measured quantity.

Primary outcomes are boundary discontinuity, queue underflow, and timing, with task success also reported. LIBERO may show little or no success benefit. This is a controlled mechanism study, not reproduction of the paper's dynamic tasks. Batching and inference cost must be measured for every mode. Do not report reduced virtual delay as a measured GPU speedup.

### Further research options

- Add limited observation history under occlusion. State which model inputs and training change; compare with a no-history control.
- Study training data selection or recovery data with a fixed pretrained backbone.
- Study inference denoising steps or precision and plot success against latency. Check closed-loop behavior after each numerical change.
- Compare with OpenVLA-OFT on cloud compute. Its [paper](https://arxiv.org/abs/2502.19645) and [published resource table](https://github.com/openvla-oft/openvla-oft.github.io/blob/main/index.html) support a later action-representation and efficiency exercise.
- Add VLA RL adaptation as an advanced project after supervised adaptation is stable. Bound simulator throughput, reward design, algorithm compatibility, and compute first.

Select one main mechanism. The course should teach experimental discipline before expanding the number of papers.

### Reading sequence and time limits

The lab estimates include **six 20-minute reading blocks (2 hours total)**: ACT architecture/loss in Lab 1; Open X-Embodiment data schema in Lab 2; SmolVLA architecture and initialization in Lab 4; Diffusion Policy formulation/horizons in Lab 6; RT-2 action-token discussion in Lab 7; RTC equations 2–5 and Algorithm 1 in Lab 10. Point to these sections, not whole-paper reading assignments. Each ends with one testable prediction or a limit of the exercise.

Reference links: [ACT](https://tonyzhaozh.github.io/aloha/), [Open X-Embodiment](https://arxiv.org/abs/2310.08864), [SmolVLA](https://arxiv.org/abs/2506.01844), [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), [RT-2](https://arxiv.org/abs/2307.15818), and [RTC](https://arxiv.org/abs/2506.07339). Lab 8 uses a supplied keyboard tool and 5–10 short demonstrations, followed by one bounded retraining job. Larger collection sessions are extensions.

## 9. Optional projects

### Webcam control and custom demonstrations

Keep this as a separate mini-project unless the user selects it for the core. The core data-collection path uses keyboard control in simulation, so custom data does not require a camera project.

Recommended sequence:

1. Attach a rigid ArUco board or marker to a hand-held object. Calibrate webcam intrinsics and use known marker dimensions. See [OpenCV's marker pose guide](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html).
2. Map relative marker motion from a reset pose to robot end-effector motion. Define the camera-to-simulation frame transform, motion scale, rotation convention, and workspace bounds.
3. Add a clutch, pose reset, filtering, velocity limits, and hold-on-tracking-loss. Begin with a button for the gripper; hand opening/closing can follow.
4. Use the simulator controller or IK to apply motion. Record synchronized **simulated camera images, robot state, and applied robot actions**. Webcam frames and marker poses are teleoperation inputs; they are not the robot's training observations by default.
5. Replay the recorded actions and inspect timing before training. Save rejected and failed episodes separately.

Bare-hand monocular tracking can be a second option. It introduces depth and scale ambiguity and occlusion. The marker path gives a simpler metric pose reference. Estimated extra learner effort: 8–15 hours after the core tools work.

### Future RL comparison

A later requested module can compare RL adaptation with ACT, Diffusion Policy, and a VLA on the same task. Reuse task definitions, recording, evaluation splits, and reporting where their semantics match. Define reward access, interaction budget, demonstrations, and pretrained data separately. Extend the training layer only when that module is specified.

### Humanoids

Treat humanoid manipulation as a new feasibility study. Select a model, embodiment, simulator, and checkpoint that are compatible. A locomotion controller must own balance and contact behavior unless the selected policy explicitly covers them. Arm end-effector actions cannot be assumed to map directly to whole-body actions. The main course makes no whole-body or physical robot readiness claim.

## 10. Implementation work packages

Sol starts after final user approval of this plan. Read the full plan before assigning work. Each package has one integration owner, bounded worker file scopes, and an exit check. Sol owns dependency resolution and shared contracts throughout. A later package extends the earlier implementation; it must not create a second version of the same tool.

| Release | Required packages | Delivered labs |
|---|---|---|
| 1: ACT workflow | S0, P0, P1, P2, P3a | 0–3 |
| 2: VLA and Diffusion Policy | P3b, P3c, P3d | 4–7 |
| 3: Data, runtime, and research | P4, P5, P6 | 8–11 |

Run each package's reference notebooks with nbclient from a fresh kernel at its exit. Save the resolved config, environment identity, exit status, and artifact paths. Long reference jobs can reconnect to verified saved runs; at least one short live path must execute. Mark exercise cells separately. Do not count replay as live inference.

### S0. Complete the initial user setup

**Owner:** Sol coordinates; the user supplies login, prepaid credit, and any required administrator actions in one batch.

**Files:** Update `docs/initial-setup.md`; add `.gitignore` and ignored `config.local.toml`. Keep the redacted setup results in the checklist, rather than adding another setup document.

- [ ] Use Bash and `<workspace>/.local-data` by default. Recheck free space and a file read/write round trip. Collect any storage or region overrides with the account steps.
- [ ] Inspect host libraries and prepare one exact bootstrap/package list. Include Python 3.12, FFmpeg shared libraries with AV1 support, and EGL/OpenGL dependencies. Group administrator actions.
- [ ] Enumerate required Hub assets. Verify metadata and small-file access; collect any gated terms or login steps now.
- [ ] Verify Brev login, SSH access, billing currency, prepaid balance, auto recharge disabled, shared-balance scope, and a current quote within section 7. The US$30 debugging allowance is already authorized; do not ask to authorize it again.
- [ ] Verify a short supervised allocation, export, and provider stop within that allowance. Record cost and remaining storage charges. Test restart only if the chosen workflow needs it.
- [ ] Check Claude Code login and usage access. Establish the Astra advice route and Luna worker assignments.
- [ ] Verify host GPU access and record the tool permission route. Prepare project dependency resolution for P0.

**Exit:** Known user setup actions are complete or listed together with a concrete fallback. Local work can proceed if cloud access is unavailable; cloud-dependent work stays blocked. Do not leave an unpaid account, missing credential, or required administrator action hidden in a later package. Exact runtime success remains a P0 check.

### P0. Establish the working reference

**Owner:** Sol; consult Astra for compatibility decisions.

**Files:** `pyproject.toml`, `uv.lock`, `docs/environment.md`, `docs/feasibility.md`, `manifests/core.yaml`, `configs/policy/`, `configs/data/`, `configs/eval/splits.yaml`, `src/vla_lab/data/splits.py`, `src/vla_lab/data/stats.py`, `tests/test_episode_splits.py`, `tests/test_training_stats.py`.

- [ ] Resolve and lock the environment. Test a CUDA tensor operation, actual EGL rendering, and decoding of a real AV1 dataset file. Use host execution where required. Record a concrete reason before adding a container or second environment.
- [ ] Pin exact source, model, processor, and dataset revisions. Record licenses and download/storage requirements.
- [ ] Resolve exact task IDs. Freeze demonstration partitions and explicit simulator-state lists before any reference training or selection. Test that validation-only changes cannot affect training statistics. Inject the verified training statistics into the actual processors.
- [ ] Run every feasibility check in section 3. Verify state/action semantics and the controller period. Train the initial ACT references on the training episodes; select only on development states.
- [ ] Probe the other models, the SmolVLA loaded/trainable parameter groups, and guided RTC memory locally. Record which jobs need cloud compute and why.
- [ ] Keep any P0 cloud job supervised and bounded by section 7. P1 must pass before unattended remote work. A notebook or assistant pause must not leave a supervised VM running.
- [ ] Save the full ACT reference bundle, every development result, resource measurements, and a short rollout video. Keep large artifacts in ignored storage; retain small evidence and retrieval manifests in the repository.

**Exit:** Both ACT demo tasks meet the development usability gate, and all three methods pass shape, finite-gradient, save/reload, and resource probes. Report a concrete blocker if this cannot be achieved within available compute. Do not consume the final evaluation states to pass P0.

### P1. Package installation, jobs, and the first lab

**Owner:** Luna for bounded CLI/notebook tasks; Sol integrates and retains ownership of locks and cloud controls.

**Files:** `justfile`, `src/vla_lab/cli.py`, `src/vla_lab/notebooks/`, `src/vla_lab/cloud/`, `tests/test_doctor.py`, `tests/test_notebook_jobs.py`, `tests/test_cli_completion.py`, `tests/test_job_lifecycle.py`, `tutorials/index.ipynb`, `tutorials/00-first-run/`, `README.md`. Extend the existing setup/environment docs.

- [ ] Wrap the P0 recipe with shared local/cloud profiles, cache paths, and `doctor`. Do not solve dependencies again without a demonstrated conflict.
- [ ] Implement detached jobs, an atomic config/seed/purpose lock, status, reconnect, cancel, and artifact retrieval. Test kernel restart, duplicate submission, stale-lock recovery, and cancellation.
- [ ] Implement all section 7 cost checks, monitoring, export bounds, and verified provider stop before unattended jobs. Test failure cases with a fake provider and verify the independent stop path on a short funded allocation. Reuse S0 evidence where it covers the same operation.
- [ ] Add readable help and offline completion. Test recipe names and CLI options, policies, profiles, task IDs, and paths with spaces.
- [ ] Test missing GPU access, failed rendering/decoding, low disk space, and interrupted download behavior.
- [ ] Deliver the first ACT notebook: select task/checkpoint, run, reset, and show inline video. Execute it with nbclient and test both a clean setup and cached launch.

**Exit:** The README leads to a live ACT rollout. Target 15 minutes after prerequisites and downloads; report measured time. Long jobs survive kernel restart, and unattended cloud work has verified stop controls.

### P2. Define contracts, data inspection, and evaluation

**Owner:** Sol owns interfaces; Luna can implement a bounded adapter or report.

**Files:** `src/vla_lab/contracts.py`, `src/vla_lab/sim/`, `src/vla_lab/policy/`, `src/vla_lab/runtime/action_queue.py`, `src/vla_lab/data/inspection.py`, `src/vla_lab/evaluation/`, corresponding contract/reset/queue/metric tests, `tutorials/01-act/`, `tutorials/02-data-evaluation/`.

- [ ] Implement section 6 packets and policy capabilities. Extend P0 data functions for the Lab 2 audit; retain their tested split and statistics behavior.
- [ ] Implement the basic action queue now. Test ACT chunk indexing, synchronous temporal ensembling, history alignment, valid horizons, and reset behavior with known sequences. P5 adds virtual scheduling to this queue.
- [ ] Reject swapped/missing cameras, malformed state, nonfinite actions, and invalid ranges. Test pose conventions and gripper commands with known outcomes.
- [ ] Implement explicit state-ID evaluation, per-episode manifests, paired reports, confidence intervals, and failure videos. Reject modulo wrapping and unintended duplicate states.
- [ ] Distinguish policy failure from infrastructure failure. Verify that failed attempts remain visible in denominators and that batching preserves state order.
- [ ] Execute Labs 1–2 with nbclient, including data decoding and one short live development evaluation.

**Exit:** Data, policy, and evaluator contracts are tested. A reader can trace one demonstration through its processor and compare runs on the same states.

### P3a. Train ACT and close Release 1

**Owner:** Sol owns training integration; Luna can implement bounded notebook and report tasks.

**Files:** `src/vla_lab/training/`, `configs/train/act_task.yaml`, checkpoint/resume tests, `tutorials/03-train-act/`, matching `solutions/` and reference manifests.

- [ ] Wrap upstream ACT training with the P0 split and processor. Verify finite loss, checkpoint reload, resume, and closed-loop behavior.
- [ ] Compare nested demonstration subsets and one chunk setting on development states. Record the training and tuning budget.
- [ ] Deliver the measured reference, a failure diagnosis, and the notebook exercise. Execute Lab 3 and the Release 1 reference path with nbclient.

**Exit:** Labs 0–3 form a complete ACT run/inspect/train/evaluate workflow. Reserve the cross-method final comparison for P3d; any final ACT report uses the predeclared frozen recipe and cannot feed later tuning.

### P3b. Introduce and adapt SmolVLA

**Owner:** Sol integrates; Claude Code reviews initialization and processor loading; Luna can own lesson files.

**Files:** SmolVLA policy/training configs and adapter, parameter-loading tests, `tutorials/04-smolvla/`, `tutorials/05-adapt-smolvla/`, solutions and reference manifests.

- [ ] Run B with its own architecture and bundled processor. Test stored one-action execution and the development-only ten-action ablation.
- [ ] Implement A→C1→C2 exactly as section 5 states. Audit loaded, frozen, and reinitialized parameters and persist processor hashes.
- [ ] Demonstrate instruction changes with the task success predicate visible. Keep task selection separate from motor-control comparisons.
- [ ] Produce a working C reference, C1/C2 development results, and the first ACT/VLA comparison. Report negative results and cost. Execute Labs 4–5 with nbclient.

**Exit:** The learner can run a published VLA and continue training a compatible robot checkpoint. The report cannot misread B as the parent of C or interpret their score gap as a data effect.

### P3c. Introduce Diffusion Policy

**Owner:** Sol integrates; Luna can own a bounded adapter or lesson.

**Files:** Diffusion policy/training configs and adapter, history/horizon/noise-seed tests, `tutorials/06-diffusion/`, solutions and reference manifests.

- [ ] Verify frame/action window alignment, padding, pretrained visual normalization, history resets, and valid prediction/execution horizons.
- [ ] Train and reload a working reference on the same demonstration split. Measure denoising-step effects on latency and behavior.
- [ ] Execute Lab 6 with nbclient and record resource use and failure cases.

**Exit:** The learner completes the third run/train/evaluate workflow with a verified reference and declared sampling settings.

### P3d. Compare methods and close Release 2

**Owner:** Sol; Astra reviews claims; Claude Code may independently audit evaluation/split logic.

**Files:** `configs/experiments/comparison.yaml`, comparison reports/tests, `tutorials/07-comparison/`, solutions and reference manifests.

- [ ] Freeze the selected recipes and evaluate all methods on the reserved final state lists. Show paired results, uncertainty, cost, and failures.
- [ ] Include the third task only after its task-specific baselines meet development gates. Report the cost of externally selecting ACT/DP checkpoints.
- [ ] Disclose pretraining, architecture, history, image processing, and compute differences. Provide the practical recipe comparison; add an equal-compute view only if the measured budget permits it.
- [ ] Execute Lab 7 with nbclient. Give a result-based method choice and list inconclusive questions.

**Exit:** Labs 4–7 work in sequence. The comparison supports practical decisions without claiming that pretraining or architecture effects have been isolated.

### P4. Add data curation and recovery

**Owner:** Luna with a separate file scope; Sol integrates recording semantics.

**Files:** `src/vla_lab/data/recording.py`, `configs/experiments/data_quality.yaml`, `tests/test_recording_alignment.py`, `tutorials/08-data-quality/`, solutions and small fixtures.

- [ ] Supply keyboard control through the action contract. Record applied actions and synchronized simulated observations; replay a saved episode.
- [ ] Add a labelled synthetic timing/action defect and a repaired reference. Keep source filtering history labelled as uncertain.
- [ ] Limit the core exercise to 5–10 short recovery demonstrations and one bounded training continuation. Select and assess the change on development states; any final comparison must be frozen in advance.
- [ ] Execute the reference notebook with saved demonstrations. Test the human keyboard path separately and record that result.

**Exit:** The learner can find a data defect, record usable demonstrations, and measure a bounded data change within the stated lesson time.

### P5. Add timing and RTC experiments

**Owner:** Sol owns scheduling semantics; Luna handles bounded fault tests; Claude Code reviews timing; Astra advises on paper claims.

**Files:** Extend `src/vla_lab/runtime/` and its P2 queue, virtual-clock and RTC mask/guidance tests, `configs/experiments/latency.yaml`, `tutorials/09-runtime/`, `tutorials/10-rtc/`, solutions and reference manifests.

- [ ] Implement section 8 virtual availability and action indexing. Test exact expected actions across delays, jitter, duplicates, reordering, underflow, reset, and policy-unavailable events.
- [ ] Provide a separate wall-clock demonstration that measures contention and missed deadlines. Do not require a real network server for core fault tests.
- [ ] Test RTC equation 5 masks, horizon constraints, finite/clipped guidance, and committed-action preservation. Compare the exercise with the pinned upstream reference.
- [ ] Use one SmolVLA checkpoint for synchronous, naive asynchronous, temporal-ensemble, and RTC hard/soft conditions. Record sampler settings and extra gradient memory.
- [ ] Select experiments on development states. Freeze any final ablation matrix before test evaluation. Report discontinuity, underflow, timing, success, and regressions.
- [ ] Execute Labs 9–10 with nbclient. State the limits of a static LIBERO experiment.

**Exit:** Deterministic tests prove the delay semantics; measured results support the timing and mechanism claims. No action from an earlier reset can reach the current controller.

### P6. Complete operations and Release 3

**Owner:** Sol; Luna can write docs after command behavior is fixed.

**Files:** `docs/cloud.md`, `docs/operations.md`, `docs/architecture.md`, `docs/extending.md`, `docs/evidence.md`, `tutorials/11-deployment/`, `README.md`. Extend P1 lifecycle tests only where integration adds new failure cases.

- [ ] Verify interrupted training recovery and restoration on a clean runtime. Reuse the existing cloud monitor and stop path; do not defer basic lifecycle controls to this package.
- [ ] Complete checkpoint/data provenance, release metadata, rollback, failure runbook, and a short lesson-extension guide.
- [ ] Prove that an added lesson can reuse an existing policy and evaluator with its notebook, config, fixture, and index entry.
- [ ] Execute Lab 11, then the complete reference course from fresh kernels. Run CLI equivalents on supported profiles. Identify reused long-job artifacts and live checks clearly.
- [ ] Remove credentials, personal account data, temporary logs, and large outputs from tracked files. Keep compact, useful evidence and artifact retrieval manifests.
- [ ] Review accuracy, STE language, learner effort, all user setup steps, and measured cost. List optional untested paths.

**Exit:** Deliver runnable lessons, solutions, user/backend docs, and measured evidence. All core reference workflows pass. Resource limits and unresolved provenance limits remain visible.

## 11. Agent coordination and review

- Sol owns the shared contracts, lock files, integration, and acceptance evidence.
- Keep this plan revision uncommitted during review. After approval, commit verified implementation changes directly to `main` with Conventional Commit messages. Preserve unrelated edits; workers return bounded changes and Sol commits each integrated unit. Use `docs`, `feat`, `fix`, `test`, or `chore` commit types as appropriate.
- Use Luna with medium effort for bounded code/docs work and high effort for a bounded adapter or diagnostic task. Each assignment has an input contract, exact file scope, check command, and stop condition.
- Assign concurrent writers disjoint files. A worker must not silently change a shared schema or launch another external worker.
- Use Claude Code for concrete reviews or a bounded implementation task. First check that it is available and that its usage can be inspected. If not, record the limitation instead of assuming it ran.
- Consult Astra after P0, after the first training comparison, and before accepting the research conclusions. Provide short evidence packets and a specific question.
- Check Codex usage before dispatch and at work boundaries. If the 5-hour window exceeds 85%, stop new dispatches, interrupt active workers, and save a continuation note. Pause active work at a recoverable point.
- If a paid GPU job is running when work must pause, checkpoint and stop it; do not leave it billing until the assistant resets.
- Set one resume timer for the returned session reset time. On wake, check usage and current user instructions before resuming. If the user cancels the timer, require a new request to resume.
- Ask the user to resolve scope or spending changes with the concrete result and cost visible. Routine implementation decisions stay with Sol.

The original Claude review notes were not retained in this repository. This plan is the implementation authority after user approval. Do not infer a review finding from a missing source document.

## 12. Completion criteria and final review

The project is complete when:

- A clean supported setup can run a live pretrained policy in simulation.
- The learner can select ACT tasks, then vary VLA instructions in the appropriate lesson, inspect observations/actions, and save results.
- ACT and Diffusion Policy are taught and trained as core methods. A practical report compares them with SmolVLA using declared data, information, and compute budgets.
- Just recipes and the detailed CLI have tested completion, clear help, and one shared implementation.
- The selected environment is reproducible and its GPU tests run with verified host access.
- New lessons can reuse the existing data, policy, and evaluation boundaries without changing unrelated methods.
- A small public dataset can be audited, split, and used with training-only statistics. Final simulator states are excluded from course tuning; unresolved source-data overlap is stated.
- The course includes measured adaptation results, routine failure analysis, and custom simulated demonstrations.
- Timing faults and action chunks have explicit tested semantics.
- One paper mechanism has a working exercise, reference solution, and honest ablation report.
- Local and cloud resource requirements come from measurement. Implementation stays within the funded US$30 cap, with tested monitoring and stop controls. The future €150 monthly plan includes interactive time and storage; its spending authority is separate.
- Initial setup is recorded, and routine work fits the user's standing permissions and compute envelope.
- User docs, backend docs, solutions, offline samples, and reference artifacts are complete.
- Simulation evidence and pending physical deployment work are stated separately.

For the career assessment, finish with a short memo: Which work was interesting? Which repeated tasks were acceptable? Was the VLA better than a narrow imitation policy for this use case? What data would the next real deployment require? Which integration tasks used your existing robotics skills? What result would change your decision?

**Final review:** The task suite is LIBERO-Goal; the default state split is 20 development / 30 final states when 50 exist. Use compatible VLM initialization for the learner SmolVLA, with the published LIBERO checkpoint as a separate reference. Run locally first and use the prepaid implementation allowance only when needed. After user approval, Sol starts S0, then P0, and follows the release/package dependencies above. Exact pins, model success, memory use, and account billing controls remain measured acceptance checks, not facts established by this plan.
