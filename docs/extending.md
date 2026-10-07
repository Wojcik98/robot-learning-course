# Add or change a course lesson

Use this guide to add a lesson to the 22-lesson course. Keep the learner notebook and complete reference notebook in the same lesson folder. Put the exercise in the learner notebook as Markdown and code cells. Do not put required exercise text in a separate file.

## Course sequence

Use the order in [the course index](../tutorials/index.ipynb): 01-playground, 02-act-execution, 03-act-data, 04-act-training, 05-act-cvae, 06-act-attention, 07-act-decoder, 08-act-assembly, 09-act-adaptation, 10-diffusion-execution, 11-diffusion-training, 12-diffusion-blocks, 13-diffusion-unet, 14-diffusion-sampling, 15-diffusion-adaptation, 16-smolvla-execution, 17-smolvla-training, 18-smolvla-conditioning, 19-smolvla-flow, 20-smolvla-adaptation, 21-data-quality, 22-comparison.

## Lesson rules

- Assume Python and NumPy. Explain each PyTorch operation when learners first need it.
- Show a working result before the first exercise. Label cells **Supplied**, **Implement**, **Check**, and **Use your function**.
- Each short exercise implements one named function. State input keys, shapes, units, masks, and return values beside the stub. Use `NotImplementedError` so an unfinished function fails clearly.
- Give plain-text hints first. Put more revealing hints in a collapsed section. Put full answers in the reference notebook with the same basename and folder.
- Test the learner function itself. The same function must drive the later unit experiment. After a concept is taught, later units can import its tested package function.
- Use real `torch.nn.Module` objects and explicit callbacks in behavioral checks. Inject random noise and time values. Do not patch globals or hide a reference implementation behind an import in the learner notebook.
- Use real data for the unit's use case and synthetic data for focused boundary checks. Test that learner output reaches the real optimizer, model, or environment path.
- State the data partition, seeds, model and data provenance, training statistics, and timing. Keep failures in the result denominator. Keep validation and final data out of training statistics and tuning.
- Mark real experiments with cell metadata, a step limit, and a new output path. Each cell runs its stated operation. The quick automation profile skips named experiment groups and reports them; the full profile runs selected experiments and their prerequisites. Do not add execution flags or skip branches.
- Keep notebook outputs empty and code execution counts unset. Find the repository root by locating `manifests/act_references.yaml`; do not assume `Path.cwd()` is the root.
- Link the method source where the exercise appears. Check technical details against the pinned implementation.

## Author and check a lesson

Run these commands from the repository root. The first command runs static checks. The second checks and executes one reference notebook in a fresh local kernel. A file or directory path is accepted. Check one notebook at a time to limit resource use.

```bash
.venv/bin/python tools/check_course.py tutorials --report .local-data/course-runs/checks.json

.venv/bin/python tools/check_course.py tutorials/06-act-attention/06-act-attention-reference.ipynb --execute --profile quick --report .local-data/course-runs/checks.json
```

The report records one result per notebook. The execution check expects each learner notebook to stop at its first `NotImplementedError` exercise. It runs reference notebooks and the playground through their default quick path. The quick profile skips experiment cells; a full profile runs them. Both profiles use fresh local Python kernels with model and data downloads disabled. The quick profile hides CUDA. Put each real GPU or simulator experiment in separate cells with a named `metadata.course.experiment` value. Declare its prerequisite groups in `requires`. Use the full profile to execute the selected group and its prerequisites; use the quick profile for the short checks. Do not add execution flags.

The [course validation record](../artifacts/reference/course/l40s_validation.json) lists bounded runs that passed on the stated machine. It includes kernel startup in wall time and CPU notebook setup in execution time. Its measurements show code execution and resource use; they do not show policy quality or successful learning.

## Measure one GPU experiment

Use `tools/measure_course.py NOTEBOOK --experiment NAME` for one named GPU experiment. The experiment name must be present in cell metadata; it lists prerequisite slugs and provides a unique identifier for the run. The tool requires a CUDA device and a notebook that marks the experiment with `metadata.course.experiment`. It executes unmarked cells, the selected experiment, and its prerequisites in a fresh local kernel. Keep all other long jobs behind experiment metadata. The tool keeps the source notebook unchanged, disables network downloads, and writes a JSON report with the notebook hash, workload description, execution result, wall time, and CUDA memory peaks. The CUDA peak covers PyTorch allocations after the measurement reset, across all selected cells, including phases that reset the counters; it excludes non-PyTorch GPU allocations and other processes. The run includes CPU notebook setup after the reset. It does not measure policy quality.

For example, compare the learner flow objective and Euler sampler with the real SmolVLA model:

```bash
.venv/bin/python tools/measure_course.py tutorials/19-smolvla-flow/19-smolvla-flow-reference.ipynb \
  --experiment flow \
  --workload 'Lesson 19 prepared SmolVLA: flow objective and cached Euler sampler on CUDA' \
  --timeout 1800 \
  --report .local-data/course-runs/lesson19-flow-gpu.json
```

This job needs the prepared SmolVLA, VLM, and local LIBERO data. Its notebook must mark the experiment in cell metadata; the measurement tool requires a CUDA device to proceed. The timeout limits each cell. Training cells must also bound their optimizer loops. The command names the experiment; it does not select assets or create a checkpoint. Check the report before you describe the run. Record its workload and machine with any resource result. Do not use the CUDA memory peak as evidence of training quality or simulator performance.

1. Read the prior and next lesson contracts. Reuse only concepts already taught; do not make one notebook kernel depend on another.
2. Define the learner function, its exact contract, and the output that will show its effect.
3. Add a behavioral check with meaningful inputs, including a boundary or failure case. For model changes, compare fixed-weight outputs with the pinned implementation and separately test the learner-trained path.
4. Connect the learner function to the real lesson path. Confirm that replacing it changes the result or triggers the expected failure.
5. Write the reference notebook beside the learner notebook. Keep it in sync with the exercised interface.
6. Run CPU checks and execute the reference notebook from a fresh kernel. Check the learner notebook fails at the first unfinished stub with its stated message.
7. Check notebook format, root discovery, relative links, empty outputs, and unset execution counts. Record what ran and what did not.

## Data, timing, and model rules

Keep complete episodes in one split. Compute normalization statistics from training episodes only. The course uses states 0–9 for exploration, 10–14 for method checks, 15–19 for data-intervention checks, and 20–49 for final evaluation after Lesson 21 and recipe freeze. Do not report a result without its task, partition, seeds, checkpoint/data source, and timing.

Keep source data frequency, controller frequency, observation history, prediction horizon, and executed action count as separate values. State any resampling rule. The LIBERO controller executes relative actions at 20 Hz. Dataset timestamps record storage frequency (10 Hz labels); this is not proof of physical recording frequency. Distinguish storage timing, simulator execution time, and wallclock inference time. For ACT and diffusion conversion exercises, compare the learner path and the pinned implementation using fixed weights and inputs. For VLA work, record the frozen backbone state and confirm that only intended parameters update.

A code check, a model update, and simulator success are separate results. Report each only when measured. A local resource measurement applies only to its stated machine, model, batch, and run settings.
