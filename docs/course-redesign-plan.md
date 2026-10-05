# Approved course design: implement a working robot learning system

The approved course has 14 numbered units and 17 learner/reference notebook pairs. Learners start with working policies, then implement one function at a time and use each function in the same unit. Each pair has matching basenames in the same folder. See [the course index](../tutorials/index.ipynb) for the exact notebook paths.

## Learner model

The course assumes Python and NumPy. It explains PyTorch inline when a function needs it. CPU checks are the default. Real GPU experiments are explicit, bounded steps that use local assets. No notebook automatically downloads data or starts long training. Short function exercises are about 15 minutes; larger experiments take longer. These are learning targets, not measured run times.

Each learner notebook contains the full exercise prompt and a clear `NotImplementedError` stub. Plain hints come first; more revealing hints can be collapsed. The matching reference notebook contains full answers. A learner function must pass a behavioral check and must drive the unit's later experiment. Later units may import tested package functions for concepts already taught.

## Approved order

| Unit | Notebooks | Main work |
|---|---|---|
| 1 | `01-playground` | Run supplied ACT, diffusion, and VLA policies; inspect observations, actions, and reset behavior. |
| 2 | `02-run-policy` | Implement the episode loop and connect learner actions to the simulator. |
| 3 | `03-act-training` | Implement ACT losses, an optimizer update, and a bounded training loop. |
| 4 | `04-adapters` | Convert camera, state, and action values at the policy/controller boundary. |
| 5 | `05-data` | Read episode records, build action windows, split by episode, and compute training-only statistics. |
| 6 | `06-cvae` | Implement CVAE sampling and latent loss operations. |
| 7 | `07-attention` | Implement and check attention operations with explicit shapes and masks. |
| 8A | `08a-act-decoder` | Implement ACT decoder operations. |
| 8B | `08b-act-assembly` | Connect ACT components and check fixed-weight parity. |
| 9 | `09-diffusion-training` | Implement the diffusion objective and connect it to a real update. |
| 10A | `10a-denoiser-blocks` | Implement denoiser building blocks. |
| 10B | `10b-unet` | Assemble and check the U-Net path. |
| 11 | `11-sampling` | Implement sampling steps and the executed action slice. |
| 12A | `12a-vla-inputs` | Prepare images, state, tokens, and masks for a VLA. |
| 12B | `12b-flow-matching` | Implement flow training pairs and loss; verify frozen weights. |
| 13 | `13-data-quality` | Improve episode alignment and data quality; preserve episode boundaries. |
| 14 | `14-comparison` | Pair results on fixed task/state/seed cases and compare methods. |

The matching reference notebook is in each unit folder and has the learner notebook's basename plus `-reference`. Keep the exact links in the index current when paths change.

## Data and evidence contracts

Use whole episodes for data splits. Compute statistics from training data only. The state partitions are: 0–9 exploration, 10–14 method checks, 15–19 data-intervention checks, and 20–49 final evaluation. Use final states only after Unit 13 and recipe freeze. Keep failed attempts in every denominator.

Record task, partition, reset seeds, data and checkpoint provenance, and timing with every experiment. Keep source data frequency, controller frequency, observation history, prediction horizon, and number of executed actions separate. Check model conversions against the pinned implementation on fixed weights and inputs, then test the learner-trained path separately. For VLA training, check that the intended parameters update and the frozen backbone does not.

A code check, a model update, and simulator success are different evidence. Do not infer one from another. Do not claim resource use without a measured run and its machine/settings. Keep assets local; report missing assets and their setup instructions before an experiment.

## Implementation acceptance checklist

- [ ] Each of the 17 learner notebooks has a matching reference notebook in the same folder.
- [ ] Exercise contracts state shapes, units, masks, and outputs. Checks call learner functions.
- [ ] The next experiment uses those functions in its real model, optimizer, or simulator path.
- [ ] Model and data conversions match the pinned source on fixed inputs and weights where applicable.
- [ ] Noise, time, and sampling inputs are explicit. Frozen parameters remain frozen.
- [ ] CPU checks work without downloading assets or starting a long job.
- [ ] GPU experiments have explicit bounds and output locations.
- [ ] Reference notebooks run from fresh kernels. Learner notebooks fail clearly at unfinished stubs.
- [ ] Notebooks have valid format, empty outputs, unset execution counts, and repository-root discovery.
- [ ] Index links, setup steps, and earlier command-guide links are checked after integration.
- [ ] Results preserve partition, seeds, provenance, timing, and failures.

## Preserved learning material

Keep the original Labs 0–11 and their solutions and learner answers. Link Labs 9–11 as focused runtime, RTC, and deployment extensions. The separate playground preview remains optional. Do not describe historical reports as new experiments.
