# Add or change a course lesson

Use this guide to add a lesson to the 14-unit course. Keep the learner notebook and complete reference notebook in the same unit folder. Put the exercise in the learner notebook as Markdown and code cells. Do not put required exercise text in a separate file.

## Course sequence

Use the order in [the course index](../tutorials/index.ipynb): playground, execution, ACT training, adapters, data, CVAE, attention, ACT decoder, ACT assembly, diffusion training, denoiser blocks, U-Net, sampling, VLA inputs, flow matching, data improvement, and comparison.

## Lesson rules

- Assume Python and NumPy. Explain each PyTorch operation when learners first need it.
- Show a working result before the first exercise. Label cells **Supplied**, **Implement**, **Check**, and **Use your function**.
- Each short exercise implements one named function. State input keys, shapes, units, masks, and return values beside the stub. Use `NotImplementedError` so an unfinished function fails clearly.
- Give plain-text hints first. Put more revealing hints in a collapsed section. Put full answers in the reference notebook with the same basename and folder.
- Test the learner function itself. The same function must drive the later unit experiment. After a concept is taught, later units can import its tested package function.
- Use real `torch.nn.Module` objects and explicit callbacks in behavioral checks. Inject random noise and time values. Do not patch globals or hide a reference implementation behind an import in the learner notebook.
- Use real data for the unit's use case and synthetic data for focused boundary checks. Test that learner output reaches the real optimizer, model, or environment path.
- State the data partition, seeds, model and data provenance, training statistics, and timing. Keep failures in the result denominator. Keep validation and final data out of training statistics and tuning.
- CPU checks are the default. Mark real GPU experiments as opt-in, with a step limit and output path. Do not download assets or start long jobs automatically. A disabled run flag must not make a required learning goal optional.
- Keep notebook outputs empty and code execution counts unset. Find the repository root by locating `manifests/act_references.yaml`; do not assume `Path.cwd()` is the root.
- Link the method source where the exercise appears. Check technical details against the pinned implementation.

## Author and check a unit

Run these commands from the repository root. The first command runs static checks. The second checks and executes one reference notebook in a fresh local kernel. A file or directory path is accepted. Check one notebook at a time to limit resource use.

```bash
.venv/bin/python tools/check_course.py tutorials --report .local-data/course-redesign/notebook-validation.json

.venv/bin/python tools/check_course.py tutorials/07-attention/07-attention-reference.ipynb --execute --report .local-data/course-redesign/notebook-validation.json
```

The report records one result per notebook. The execution check expects each learner notebook to stop at its first `NotImplementedError` exercise. It runs reference notebooks and the playground through their default quick path. It uses fresh local Python kernels, hides CUDA, and sets model/data downloads to offline mode. Keep real GPU or robot experiments behind explicit flags and run them as separate, measured experiments. Do not enable them for the notebook check.

The [local resource record](../artifacts/reference/course/local_resources.json) lists bounded runs that passed on the stated machine. It includes kernel startup in wall time and CPU notebook setup in execution time. Its measurements show code execution and resource use; they do not show policy quality or successful learning.

## Measure one GPU experiment

Use `tools/measure_course.py` for one opt-in GPU job. It requires a CUDA device and a notebook flag that is set to `False` in one code cell. It enables that flag in memory in a fresh local kernel and executes the full notebook. Keep all other long jobs behind disabled flags. You can use `--replace OLD NEW` to bound a subset or set the device. The tool keeps the source notebook unchanged, disables network downloads, and writes a JSON report with the notebook hash, workload description, execution result, wall time, and CUDA memory peaks. The CUDA peak covers PyTorch allocations after the measurement reset, across the full notebook run; it excludes non-PyTorch GPU allocations and other processes. The run includes CPU notebook setup after the reset. It does not measure policy quality.

For example, compare the Unit 12B loss with the prepared 450M learner and make one optimizer update on CUDA:

```bash
.venv/bin/python tools/measure_course.py tutorials/12b-flow/12b-flow-reference.ipynb \
  --flag RUN_REAL_FLOW_LOSS \
  --assertion 'assert DEVICE == "cuda" and REAL_UPDATES == 1' \
  --workload 'Unit 12B prepared learner: one flow-loss comparison and one optimizer update; auto-select CUDA' \
  --report .local-data/course-redesign/unit12b-real-flow-gpu.json
```

This job needs the prepared learner, VLM and local LIBERO data. Its notebook selects CUDA when it is available; the measurement tool stops before running cells if no CUDA device is present. The assertion checks the selected device and one-update limit. The command enables only the named notebook flag; it does not select assets or create a checkpoint. Check the report before you describe the run. Record its workload and machine with any resource result. Do not use the CUDA memory peak as evidence of training quality or simulator performance.

1. Read the prior and next unit contracts. Reuse only concepts already taught; do not make one notebook kernel depend on another.
2. Define the learner function, its exact contract, and the output that will show its effect.
3. Add a behavioral check with meaningful inputs, including a boundary or failure case. For model changes, compare fixed-weight outputs with the pinned implementation and separately test the learner-trained path.
4. Connect the learner function to the real unit path. Confirm that replacing it changes the result or triggers the expected failure.
5. Write the reference notebook beside the learner notebook. Keep it in sync with the exercised interface.
6. Run CPU checks and execute the reference notebook from a fresh kernel. Check the learner notebook fails at the first unfinished stub with its stated message.
7. Check notebook format, root discovery, relative links, empty outputs, and unset execution counts. Record what ran and what did not.

## Data, timing, and model rules

Keep complete episodes in one split. Compute normalization statistics from training episodes only. The course uses states 0–9 for exploration, 10–14 for method checks, 15–19 for data-intervention checks, and 20–49 for final evaluation after Unit 13 and recipe freeze. Do not report a result without its task, partition, seeds, checkpoint/data source, and timing.

Keep source data frequency, controller frequency, observation history, prediction horizon, and executed action count as separate values. State any resampling rule. For ACT and diffusion conversion exercises, compare the learner path and the pinned implementation using fixed weights and inputs. For VLA work, record the frozen backbone state and confirm that only intended parameters update.

A code check, a model update, and simulator success are separate results. Report each only when measured. A local resource measurement applies only to its stated machine, model, batch, and run settings.
