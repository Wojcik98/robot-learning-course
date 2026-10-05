"""Build learner/reference notebook pairs for course units 4, 5, 13, and 14."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorials"


def source_function(module: str, name: str) -> str:
    """Copy one package reference function into the solution notebook."""
    path = ROOT / "src" / "vla_lab" / "teaching" / f"{module}.py"
    text = path.read_text()
    tree = ast.parse(text)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    lines = text.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build(
    slug: str,
    title: str,
    intro: str,
    imports: str,
    exercises: list[dict[str, str]],
    experiment: tuple[str, ...],
) -> None:
    for reference in (False, True):
        setup = """from pathlib import Path
import os, sys
WORKSPACE = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'manifests/act_references.yaml').is_file()), None)
if WORKSPACE is None:
    raise FileNotFoundError('Run this notebook inside the VLA course workspace')
sys.path.insert(0, str(WORKSPACE / 'src'))
HF_ROOT = WORKSPACE / '.local-data' / 'hf'
os.environ.setdefault('HF_HUB_CACHE', str(HF_ROOT / 'hub'))
os.environ.setdefault('HF_DATASETS_CACHE', str(HF_ROOT / 'datasets'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
print('Workspace:', WORKSPACE)
"""
        cells = [
            md(
                f"# {title}\n\n{intro}\n\n**Notebook:** {'reference solutions' if reference else 'learner copy'}. Work through the cells in order."
            ),
            code(setup),
            code(imports),
        ]
        for index, ex in enumerate(exercises, 1):
            cells.append(
                md(
                    f"## Exercise {index}: {ex['title']}\n\n**Purpose:** {ex['purpose']}\n\n"
                    f"**Contract:** `{ex['signature']}`\n\n{ex['contract']}\n\n"
                    f"**Method:** {ex['method']}\n\n**Pointers:** {ex['pointers']}\n\n"
                    f"<details><summary>Detailed hint</summary>\n\n{ex['hint']}\n\n</details>\n\n"
                    f"Estimated time: about 15 minutes."
                )
            )
            if reference:
                cells.append(code(ex["solution"]))
            else:
                cells.append(code(ex["starter"]))
            cells.append(md("### Check"))
            cells.append(code(ex["check"]))
            cells.append(md("### Use your function"))
            cells.append(code(ex["use"]))
        cells.append(md(f"## Larger experiment\n\n{experiment[0]}\n\n{experiment[1]}"))
        cells.append(
            code(
                experiment[2]
                if len(experiment) > 2
                else "print('Follow the bounded experiment steps above; save every attempt.')"
            )
        )
        cells.append(
            md(
                "## Record your result\n\nKeep the inputs, every attempt, and the result table with the notebook. A failed run is a result and stays in the denominator."
            )
        )
        for index, cell in enumerate(cells):
            cell["id"] = f"{slug.replace('-', '')[:4]}{index:04d}"
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        target = OUT / slug / f"{slug}{'-reference' if reference else ''}.ipynb"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, indent=1) + "\n")


def ex(
    module: str,
    name: str,
    *,
    title: str,
    purpose: str,
    signature: str,
    contract: str,
    method: str,
    pointers: str,
    hint: str,
    starter: str,
    check: str,
    use: str,
) -> dict[str, str]:
    return locals() | {"solution": source_function(module, name)}


build(
    "04-adapters",
    "Unit 4: Adapt observations and actions",
    "Convert camera arrays and robot state to the model contract. Then convert model output to a controller action. The starter images are small fixtures; the optional local rollout uses the saved policy and simulator.",
    """from __future__ import annotations
import hashlib, json
import numpy as np
from vla_lab.teaching.adapters import camera_hwc_to_chw, assemble_libero_state, pack_observation, normalize_features, denormalize_features, controller_action
""",
    [
        ex(
            "adapters",
            "camera_hwc_to_chw",
            title="Convert one camera frame",
            purpose="Prepare pixels for the model.",
            signature="camera_hwc_to_chw(image, *, channel_order='rgb') -> (3,H,W) float32",
            contract="Input is uint8 `(H,W,3)`. Output is contiguous RGB CHW float32 in `[0,1]`, rotated by 180 degrees as required by the pinned LIBERO processor.",
            method="Flip both image axes, transpose HWC to CHW, and divide by 255. Reverse channels only when input is BGR.",
            pointers="Check one red pixel after conversion; it must move to the opposite corner.",
            hint="NumPy slicing with `[::-1, ::-1]` rotates the image by 180 degrees.",
            starter="def camera_hwc_to_chw(image, *, channel_order='rgb'):\n    raise NotImplementedError('Exercise 1: convert camera layout, orientation, and scale')",
            check="rgb = np.zeros((2, 3, 3), dtype=np.uint8)\nrgb[0, 0] = [255, 0, 0]\nconverted = camera_hwc_to_chw(rgb)\nassert converted.shape == (3, 2, 3), f'Expected CHW, got {converted.shape}'\nassert np.allclose(converted[:, -1, -1], [1, 0, 0]), 'Pixel channels, 180-degree rotation, or scale are wrong'",
            use="frame = np.zeros((64, 80, 3), dtype=np.uint8)\nprint('Model image:', camera_hwc_to_chw(frame).shape)",
        ),
        ex(
            "adapters",
            "assemble_libero_state",
            title="Assemble raw LIBERO state",
            purpose="Convert nested robot readings into the eight values expected by ACT.",
            signature="assemble_libero_state(robot_state) -> (8,) float32",
            contract="Read `eef.pos` (3), `eef.quat` in XYZW order (4), and `gripper.qpos` (2). Return position, axis-angle rotation (3), and gripper qpos (2). Do not normalize these physical state values here.",
            method="Normalize the quaternion. Convert its vector part to an axis-angle vector, then join the three fields.",
            pointers="Identity rotation `[0,0,0,1]` becomes `[0,0,0]`.",
            hint="For quaternion `(x,y,z,w)`, use angle `2*atan2(norm([x,y,z]), w)` and multiply the unit axis by that angle.",
            starter="def assemble_libero_state(robot_state):\n    raise NotImplementedError('Exercise 2: map raw LIBERO robot fields to ACT state')",
            check="robot = {'eef': {'pos': [1, 2, 3], 'quat': [0, 0, 0, 1]}, 'gripper': {'qpos': [-0.1, 0.1]}}\nstate = assemble_libero_state(robot)\nassert state.shape == (8,) and np.allclose(state, [1, 2, 3, 0, 0, 0, -0.1, 0.1]), f'Wrong raw state mapping: {state}'",
            use="print('Raw state:', state)",
        ),
        ex(
            "adapters",
            "pack_observation",
            title="Assemble named inputs",
            purpose="Keep camera order and state shape explicit.",
            signature="pack_observation(*, cameras, state, camera_order, channel_order='rgb') -> dict",
            contract="Each camera is uint8 HWC. State is finite float32 `(8,)`. Output keys match pinned ACT: `observation.images.image` and `observation.images.image2`.",
            method="Convert each camera with Exercise 1; use the raw state from Exercise 2.",
            pointers="Reject missing cameras instead of silently changing order.",
            hint="The first name in `camera_order` maps to `image`; the second maps to `image2`.",
            starter="def pack_observation(*, cameras, state, camera_order, channel_order='rgb'):\n    raise NotImplementedError('Exercise 3: assemble model inputs')",
            check="frame = np.zeros((2, 3, 3), dtype=np.uint8)\nbatch = pack_observation(cameras={'front': frame, 'wrist': frame}, state=np.arange(8), camera_order=('front', 'wrist'))\nassert batch['observation.images.image'].shape == (3, 2, 3), 'Pinned ACT image key or camera order is wrong'\nassert batch['observation.state'].shape == (8,), 'State width must be eight'",
            use="print('Input keys:', sorted(batch))",
        ),
        ex(
            "adapters",
            "normalize_features",
            title="Normalize with saved statistics",
            purpose="Put state or action features on the scale used for training.",
            signature="normalize_features(values, mean, std) -> array",
            contract="The last axis is the feature axis. `mean` and positive `std` are one-dimensional and match that axis.",
            method="Compute `(values - mean) / std` with NumPy broadcasting.",
            pointers="Use statistics fitted from training episodes only. The raw state from Exercise 2 has not been normalized yet.",
            hint="A `(2,8)` batch broadcasts against an `(8,)` mean.",
            starter="def normalize_features(values, mean, std):\n    raise NotImplementedError('Exercise 4: apply training statistics')",
            check="normalized = normalize_features([[2, 4]], [1, 2], [1, 2])\nassert np.allclose(normalized, [[1, 1]]), f'Unexpected normalized values: {normalized}'",
            use="print('Normalized state:', normalize_features(np.arange(8), np.zeros(8), np.ones(8)))",
        ),
        ex(
            "adapters",
            "denormalize_features",
            title="Return predicted actions to physical units",
            purpose="Undo the training action normalization before the controller receives a prediction.",
            signature="denormalize_features(values, mean, std) -> physical values",
            contract="The last axis is seven action features. Apply `values * std + mean` with saved action statistics.",
            method="ACT predicts normalized actions. Convert them to controller units, then apply controller limits in Exercise 6.",
            pointers="Clipping normalized predictions directly changes the model output scale.",
            hint="Use multiplication by standard deviation followed by addition of the training mean.",
            starter="def denormalize_features(values, mean, std):\n    raise NotImplementedError('Exercise 5: restore physical action scale')",
            check="physical = denormalize_features([[1, -1]], [0.2, 0.0], [0.5, 2.0])\nassert np.allclose(physical, [[0.7, -2.0]]), f'Wrong physical scale: {physical}'",
            use="print('Physical action:', denormalize_features(np.zeros(7), np.zeros(7), np.ones(7)))",
        ),
        ex(
            "adapters",
            "controller_action",
            title="Bound an action in controller units",
            purpose="Send one finite action to the simulator controller.",
            signature="controller_action(physical_action, *, lower=-1, upper=1) -> (7,) float32",
            contract="Input is a seven-value physical controller action. Validate shape and values, then clip within controller limits.",
            method="Call only after denormalizing the model output with the saved training statistics.",
            pointers="Clipping count is useful evidence; it does not repair a wrong scale.",
            hint="`np.clip` accepts scalar bounds and preserves the action shape.",
            starter="def controller_action(physical_action, *, lower=-1.0, upper=1.0):\n    raise NotImplementedError('Exercise 6: validate and bound a controller action')",
            check="physical = denormalize_features([2, -2, 0, 0, 0, 0, 0], np.zeros(7), np.ones(7))\naction = controller_action(physical)\nassert action.shape == (7,) and np.allclose(action[:2], [1, -1]), 'Expected a bounded seven-value action'",
            use="print('Controller action:', controller_action(np.zeros(7)))",
        ),
    ],
    (
        "Use the same local ACT policy, simulator state, and reset seed. Diagnose three supplied faults: camera order swapped, uint8 left unscaled, and normalized actions clipped before inverse normalization. Fix one fault at a time and record tensor shape/range, clipped count, success, and policy calls. Explore only states 0–9.",
        "Set `RUN_LOCAL=True` only when the prepared local ACT checkpoint and LIBERO assets already exist. This cell loads the pinned observation processor, then uses your `pack_observation`, `normalize_features`, `denormalize_features`, and `controller_action` functions in callbacks passed to the shared real environment loop. It does not download files. The pinned ACT keys are `observation.images.image` and `observation.images.image2`. Keep recorded data at 10 Hz and simulator control at 20 Hz.",
        """# Small supplied faults run before any local model or simulator load.
fault_frames = {'front': np.zeros((2, 3, 3), dtype=np.uint8),
                'wrist': np.full((2, 3, 3), 255, dtype=np.uint8)}
swapped = pack_observation(cameras=fault_frames, state=np.zeros(8),
                           camera_order=('wrist', 'front'))
assert swapped['observation.images.image'].mean() == 1.0, 'Fault fixture did not swap camera order'
unscaled = fault_frames['wrist'].astype(np.float32)
assert unscaled.max() == 255.0, 'Fault fixture did not expose uint8 scale'
normalized = np.array([0.5], dtype=np.float32)
wrong_scale_action = controller_action(np.pad(normalized, (0, 6)))
correct_action = controller_action(denormalize_features(
    np.pad(normalized, (0, 6)), np.array([0.2] + [0.0] * 6), np.array([0.2] + [1.0] * 6)))
print('Faults: swapped front mean=', swapped['observation.images.image'].mean(),
      'unscaled max=', unscaled.max(), 'wrong-scale controller value=', wrong_scale_action[0],
      'denormalized controller value=', correct_action[0])

RUN_LOCAL = False
if RUN_LOCAL:
    import dataclasses
    import json
    import torch
    from vla_lab.teaching.live import find_workspace, load_prepared_policy
    from vla_lab.teaching.runtime import run_chunked_episode, prepare_libero_observation
    from vla_lab.sim.episode import open_libero_task

    root = find_workspace()
    policy, preprocessor, postprocessor, recipe = load_prepared_policy('act', root)
    policy.eval()
    device = next(policy.parameters()).device
    stats_file = json.loads((root / 'configs/data/stats/bowl_plate.json').read_text())
    state_stats = stats_file['modalities']['observation.state']
    action_stats = stats_file['modalities']['action']
    environment = open_libero_task('bowl_plate', 0, root / 'configs', root / '.local-data')
    learner_clipped_values = [0]
    def prepare(raw):
        batch = prepare_libero_observation(raw, preprocessor)
        raw_state = assemble_libero_state(raw['robot_state'])
        adapted = pack_observation(
            cameras={'front': raw['pixels']['image'], 'wrist': raw['pixels']['image2']},
            state=raw_state,
            camera_order=('front', 'wrist'),
        )
        batch['observation.images.image'] = torch.from_numpy(adapted['observation.images.image'][None]).to(device)
        batch['observation.images.image2'] = torch.from_numpy(adapted['observation.images.image2'][None]).to(device)
        batch['observation.state'] = torch.from_numpy(normalize_features(adapted['observation.state'], state_stats['mean'], state_stats['std'])[None]).to(device)
        return batch
    def predict(policy, batch):
        return policy.predict_action_chunk(batch)
    def to_controller(action):
        normalized = action.detach().cpu().numpy() if hasattr(action, 'detach') else action
        physical = denormalize_features(normalized, action_stats['mean'], action_stats['std'])
        learner_clipped_values[0] += int(np.count_nonzero((physical < -1.0) | (physical > 1.0)))
        return controller_action(physical)
    try:
        initial_observation, _info = environment.reset(seed=20260930)
        reference_batch = prepare_libero_observation(initial_observation, preprocessor)
        learner_batch = prepare(initial_observation)
        for key in ('observation.images.image', 'observation.images.image2', 'observation.state'):
            torch.testing.assert_close(learner_batch[key], reference_batch[key], atol=1e-5, rtol=1e-4,
                                       msg=lambda message: f'{key} adapter differs from pinned processor: {message}')
        with torch.no_grad():
            reference_chunk = policy.predict_action_chunk(reference_batch)
            learner_chunk = policy.predict_action_chunk(learner_batch)
            torch.testing.assert_close(learner_chunk, reference_chunk, atol=1e-5, rtol=1e-4,
                                       msg='Learner adapters changed ACT predictions')
            reference_physical = postprocessor(reference_chunk)
        learner_physical = denormalize_features(learner_chunk.detach().cpu().numpy(),
                                                 action_stats['mean'], action_stats['std'])
        np.testing.assert_allclose(learner_physical, reference_physical.detach().cpu().numpy(),
                                   atol=1e-5, rtol=1e-4)
        policy.reset()
        result = run_chunked_episode(environment, policy, prepare, predict, to_controller,
                                     max_steps=40, execute_steps=10, seed=20260930)
        result = dataclasses.replace(result, clipped_values=learner_clipped_values[0])
        print(result, 'learner action values clipped at physical-unit conversion=', learner_clipped_values[0])
    finally:
        environment.close()""",
    ),
)


build(
    "05-data",
    "Unit 5: Read data and build batches",
    "Read episode rows in order. Build action windows without crossing an episode end. Split whole episodes and fit statistics on training data only.",
    """from __future__ import annotations
import hashlib, json
import numpy as np
from vla_lab.data.stats import TrainingStats
from vla_lab.teaching.data import order_episode_rows, extract_episode_arrays, action_window, split_episode_ids, fit_training_statistics, build_act_batches
""",
    [
        ex(
            "data",
            "order_episode_rows",
            title="Order recorded rows",
            purpose="Read rows in stable episode and frame order.",
            signature="order_episode_rows(rows) -> list[rows]",
            contract="Rows contain nonnegative integer `episode_index` and `frame_index`; duplicate pairs are invalid.",
            method="Sort by the two-part key `(episode_index, frame_index)`.",
            pointers="A frame number alone is not unique across episodes.",
            hint="Keep the original row dictionaries so other dataset columns remain available.",
            starter="def order_episode_rows(rows):\n    raise NotImplementedError('Exercise 1: order rows by episode and frame')",
            check="rows = [{'episode_index': 1, 'frame_index': 1}, {'episode_index': 0, 'frame_index': 2}]\nordered = order_episode_rows(rows)\nassert [(r['episode_index'], r['frame_index']) for r in ordered] == [(0, 2), (1, 1)], 'Rows are not ordered by episode then frame'",
            use="print('First ordered key:', ordered[0]['episode_index'], ordered[0]['frame_index'])",
        ),
        ex(
            "data",
            "extract_episode_arrays",
            title="Read model columns from raw rows",
            purpose="Turn real recorded rows into action and state arrays for one full episode.",
            signature="extract_episode_arrays(rows) -> {episode_index, frame_indices, actions, states}",
            contract="Each row has `episode_index`, `frame_index`, `action` (7), and `observation.state` (8). Sort with Exercise 1 and reject mixed episode IDs.",
            method="Read the stored action column directly. Do not create a delta action here; make chunks in Exercise 3.",
            pointers="A frame row contains one saved action. Preserve that action and its pre-action state.",
            hint="Stack each ordered row's `action` and `observation.state` values into a matrix.",
            starter="def extract_episode_arrays(rows):\n    raise NotImplementedError('Exercise 2: extract raw action and state columns')",
            check="rows = [{'episode_index': 3, 'frame_index': 1, 'action': np.ones(7), 'observation.state': np.ones(8)},\n        {'episode_index': 3, 'frame_index': 0, 'action': np.zeros(7), 'observation.state': np.zeros(8)}]\npart = extract_episode_arrays(rows)\nassert part['frame_indices'].tolist() == [0, 1], 'Raw rows are not ordered'\nassert np.allclose(part['actions'][0], 0) and np.allclose(part['states'][1], 1), 'Wrong raw columns were read'",
            use="print('Episode rows:', len(part['actions']), 'action/state:', part['actions'].shape, part['states'].shape)",
        ),
        ex(
            "data",
            "action_window",
            title="Build a padded action window",
            purpose="Create a fixed-size chunk and mark its padded tail.",
            signature="action_window(actions, *, start, horizon, action_dim=None) -> (window, mask)",
            contract="Input has shape `(episode_steps, action_dim)`. Output has `(horizon, action_dim)` and bool mask `(horizon,)`; `True` means padding.",
            method="Copy only rows from this episode. Fill beyond its end with zeros and mark them padded.",
            pointers="Do not concatenate episodes before window creation.",
            hint="Initialize zeros and an all-True mask, then overwrite the valid prefix.",
            starter="def action_window(episode_actions, *, start, horizon, action_dim=None):\n    raise NotImplementedError('Exercise 2: create an episode-safe action window')",
            check="window, mask = action_window([[1, 2], [3, 4]], start=1, horizon=3)\nassert np.array_equal(window, [[3, 4], [0, 0], [0, 0]]), 'Window crossed episode end or used wrong padding'\nassert np.array_equal(mask, [False, True, True]), 'Mask must mark only the padded tail'",
            use="print('window shape:', window.shape, 'valid steps:', int((~mask).sum()))",
        ),
        ex(
            "data",
            "split_episode_ids",
            title="Split by complete episodes",
            purpose="Keep validation episodes out of the training rows.",
            signature="split_episode_ids(ids, *, validation_fraction=0.2) -> (train_ids, val_ids)",
            contract="Input IDs are unique and nonnegative. Preserve input order and return nonempty disjoint partitions.",
            method="Choose a validation suffix based on the requested fraction.",
            pointers="A random row split leaks nearby frames into validation.",
            hint="Round the validation count, then clamp it to `[1, len(ids)-1]`.",
            starter="def split_episode_ids(episode_ids, *, validation_fraction=0.2):\n    raise NotImplementedError('Exercise 3: split whole episodes')",
            check="train_ids, val_ids = split_episode_ids([0, 1, 2, 3, 4], validation_fraction=0.2)\nassert set(train_ids).isdisjoint(val_ids) and train_ids + val_ids == [0, 1, 2, 3, 4], 'Episodes must be split without overlap'\nassert train_ids and val_ids, 'Both partitions must contain an episode'",
            use="print('train episodes:', train_ids, 'validation episodes:', val_ids)",
        ),
        ex(
            "data",
            "fit_training_statistics",
            title="Fit training-only statistics",
            purpose="Normalize model features without validation leakage.",
            signature="fit_training_statistics(train, *, validation_episodes=None) -> TrainingStats",
            contract="Mappings map episode IDs to arrays with the feature axis last. Return population mean/std and digest.",
            method="Concatenate training episode samples only. Replace zero std with one for safe normalization.",
            pointers="Pass validation to the contract so your check can prove it has no effect.",
            hint="Validation values can be extreme; the fitted result must stay unchanged.",
            starter="def fit_training_statistics(training_episodes, *, validation_episodes=None):\n    raise NotImplementedError('Exercise 4: fit statistics from training data only')",
            check="stats = fit_training_statistics({0: [[0.0], [2.0]]}, validation_episodes={9: [[1000.0]]})\nassert np.allclose(stats.mean, [1]) and np.allclose(stats.std, [1]), 'Validation rows affected training statistics'",
            use="print('training mean/std:', stats.mean, stats.std)",
        ),
    ],
    (
        "Use the local dataset only when its pinned files already exist. Compare two whole-episode training subsets (2 versus 8 episodes) with the same validation episodes, fresh ACT initialization, chunk horizon, optimizer, seed, and 20-update budget. Your row ordering, action-window function, and train-only statistics function build the batches that enter the Unit 3 loss callback and optimizer loop. Compare validation error after converting both predictions and targets to physical action units.",
        "Set `RUN_LOCAL_DATA=False` for the notebook default. The public reader loads every action/state row from at most eight frozen episodes, and decodes only four image frames per episode. The rows use individual 10 Hz actions; your window function creates the action chunks. No data download occurs. Training uses CUDA when available and CPU otherwise. Each subset records `completed` or `failed`; keep both rows. Both trials start from the same newly initialized pinned ACT weights, with no robot-trained checkpoint or pretrained vision backbone. This short run teaches data and batch effects, not policy success. It does not change the runner's one-action-per-controller-tick behavior.",
        """RUN_LOCAL_DATA = False
if RUN_LOCAL_DATA:
    import copy
    import yaml
    import torch
    from vla_lab.teaching.live import find_workspace, make_act_raw_model
    from vla_lab.teaching.data import read_local_episode_rows, build_act_batches
    from vla_lab.teaching.training import act_loss_fn, train_loop
    from lerobot.policies.act import ACTPolicy
    from vla_lab.policy.act import make_act_config
    root = find_workspace()
    task = yaml.safe_load((root / 'configs/data/episodes.yaml').read_text())['tasks']['bowl_plate']
    small_ids, large_ids = task['train_episode_ids'][:2], task['train_episode_ids'][:8]
    validation_ids = task['validation_episode_ids'][:2]
    results = {}
    loaded = {}

    def order_samples(episodes):
        keyed = [{'episode_index': eid, 'frame_index': sample['start'], '_episode': eid, '_sample': sample}
                 for eid, episode in episodes.items() for sample in episode['samples']]
        ordered = order_episode_rows(keyed)
        for eid, episode in episodes.items():
            episode['samples'] = [row['_sample'] for row in ordered if row['_episode'] == eid]
        return episodes

    try:
        small_raw = read_local_episode_rows('bowl_plate', small_ids, workspace=root,
                                            row_fn=extract_episode_arrays)
        large_raw = read_local_episode_rows('bowl_plate', large_ids, workspace=root,
                                            row_fn=extract_episode_arrays)
        validation_raw = read_local_episode_rows('bowl_plate', validation_ids, workspace=root,
                                                 row_fn=extract_episode_arrays)
        small_data = order_samples(small_raw)
        large_data = order_samples(large_raw)
        validation_data = order_samples(validation_raw)
        batches_small = build_act_batches(small_data, validation_data, horizon=20, batch_size=2,
                                          window_fn=action_window, stats_fn=fit_training_statistics)
        batches_large = build_act_batches(large_data, validation_data, horizon=20, batch_size=2,
                                          window_fn=action_window, stats_fn=fit_training_statistics)
        for built in (batches_small, batches_large):
            assert tuple(built['train'][0]['action'].shape) == (2, 20, 7), 'Learner windows did not reach ACT batches'
            assert tuple(built['train'][0]['observation.state'].shape) == (2, 8), 'Learner statistics did not reach ACT batches'
            assert built['train'][0]['action_is_pad'].dtype == torch.bool, 'Padding mask must reach the loss callback'
        # The pinned config disables pretrained backbone weights. Both trials
        # copy this exact fresh initialization; no prepared robot checkpoint is loaded.
        torch.manual_seed(17)
        base_policy = ACTPolicy(make_act_config(device='cpu', chunk_size=20)).cpu()
        initial_digest = hashlib.sha256()
        for name, value in sorted(base_policy.state_dict().items()):
            initial_digest.update(name.encode())
            initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        initial_weights_sha256 = initial_digest.hexdigest()
        loaded = {'two episodes': batches_small, 'eight episodes': batches_large}
    except Exception as error:
        message = f'{type(error).__name__}: {error}'
        results = {
            'two episodes': {'status': 'failed', 'error': message},
            'eight episodes': {'status': 'failed', 'error': message},
        }

    if loaded:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def move_batches(built):
            return {**built,
                'train': [{key: value.to(device) for key, value in batch.items()} for batch in built['train']],
                'validation': [{key: value.to(device) for key, value in batch.items()} for batch in built['validation']]}
        loaded = {label: move_batches(built) for label, built in loaded.items()}
        for label, built in loaded.items():
            policy = model = optimizer = physical_validation_l1 = None
            try:
                torch.manual_seed(17)
                policy = copy.deepcopy(base_policy).to(device)
                model = make_act_raw_model(policy).to(device)
                optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-6)
                def physical_validation_l1(batch):
                    prediction = model(batch)[0]
                    mean = torch.as_tensor(built['action_stats'].mean, dtype=prediction.dtype, device=device)
                    std = torch.as_tensor(built['action_stats'].std, dtype=prediction.dtype, device=device)
                    prediction = prediction * std + mean
                    target = batch['action'] * std + mean
                    valid = (~batch['action_is_pad'].bool()).unsqueeze(-1).to(prediction.dtype)
                    return ((prediction - target).abs() * valid).sum() / (valid.sum() * prediction.shape[-1]).clamp_min(1)
                model.eval()
                with torch.no_grad():
                    initial_validation_l1 = [float(physical_validation_l1(batch).cpu()) for batch in built['validation']]
                history = train_loop(model, built['train'], optimizer, act_loss_fn, steps=20)
                model.eval()
                with torch.no_grad():
                    validation_l1 = [float(physical_validation_l1(batch).cpu()) for batch in built['validation']]
                results[label] = {'status': 'completed', 'device': str(device),
                    'train_episode_ids': built['training_episode_ids'],
                    'validation_episode_ids': built['validation_episode_ids'], 'updates': len(history),
                    'initial_weights_sha256': initial_weights_sha256,
                    'last_train_loss_normalized': history[-1]['loss'],
                    'initial_validation_action_l1_physical': sum(initial_validation_l1) / len(initial_validation_l1),
                    'validation_action_l1_physical': sum(validation_l1) / len(validation_l1)}
            except Exception as error:
                results[label] = {'status': 'failed', 'device': str(device),
                    'train_episode_ids': built['training_episode_ids'],
                    'validation_episode_ids': built['validation_episode_ids'],
                    'initial_weights_sha256': initial_weights_sha256,
                    'error': f'{type(error).__name__}: {error}'}
            finally:
                policy = model = optimizer = physical_validation_l1 = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    assert set(results) == {'two episodes', 'eight episodes'}, 'Keep both subset attempts in the result table'
    for label, trial in results.items():
        if trial['status'] == 'completed':
            assert trial['updates'] == 20, f'{label} did not use the fixed 20-update budget'
    print(results)""",
    ),
)


build(
    "13-data-quality",
    "Unit 13: Diagnose and improve demonstrations",
    "Inspect applied actions and their pre-action observations. Find a timing defect before building windows. Use states 15–19 for a bounded data intervention.",
    """from __future__ import annotations
import numpy as np
from vla_lab.teaching.recording import transition_alignment, aligned_transition_indices, aligned_transition_rows, resample_aligned_transitions_10hz, dataset_tick_indices, intervention_effect
from vla_lab.teaching.data import action_window
""",
    [
        ex(
            "recording",
            "transition_alignment",
            title="Detect a timing defect",
            purpose="Check that action i follows observation i on the same simulator tick.",
            signature="transition_alignment(observation_ticks, action_ticks) -> report",
            contract="There are N actions and N+1 observations. Observation ticks are contiguous from zero. An offset is reported in 20 Hz control ticks.",
            method="Compare action tick i with observation tick i; do not shift the saved actions to hide an error.",
            pointers="LIBERO controls at 20 Hz. The reference demonstration rows are 10 Hz.",
            hint="Subtract the expected pre-action tick sequence from the saved action tick sequence.",
            starter="def transition_alignment(observation_ticks, action_ticks):\n    raise NotImplementedError('Exercise 1: detect transition misalignment')",
            check="report = transition_alignment([0, 1, 2, 3], [1, 2, 3])\nassert report['aligned'] is False and report['offset_ticks'] == [1], f'Expected a one-tick defect, got {report}'",
            use="print('Detected action offset (20 Hz ticks):', report['offset_ticks'])",
        ),
        ex(
            "recording",
            "aligned_transition_indices",
            title="Select aligned transitions",
            purpose="Reject faulty row pairings before training windows are built.",
            signature="aligned_transition_indices(observation_ticks, action_ticks) -> integer indices",
            contract="Return indices only when all N actions align with the first N observations. Raise a clear error on a defect.",
            method="Call the diagnostic first, then return one index for each valid action.",
            pointers="Do not make a one-tick defect look clean by shifting every row.",
            hint="The alignment report is a guard; it does not repair the recording.",
            starter="def aligned_transition_indices(observation_ticks, action_ticks):\n    raise NotImplementedError('Exercise 2: select only correctly aligned rows')",
            check="assert np.array_equal(aligned_transition_indices([0, 1, 2], [0, 1]), [0, 1]), 'Expected one index per aligned action'\ntry:\n    aligned_transition_indices([0, 1, 2], [1, 2])\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Misaligned rows must be rejected')",
            use="print('Aligned transitions:', aligned_transition_indices([0, 1, 2, 3], [0, 1, 2]))",
        ),
        ex(
            "recording",
            "aligned_transition_rows",
            title="Build aligned state-action rows",
            purpose="Create the exact observations and actions used by later action windows.",
            signature="aligned_transition_rows(observations, actions, observation_ticks, action_ticks) -> rows",
            contract="Input has N actions, N+1 observations and matching ticks. Output row i contains observation i and action i.",
            method="Reject a timing defect before pairing arrays.",
            pointers="The final observation has no following action and does not become a training row.",
            hint="Reuse the checked indices from Exercise 2.",
            starter="def aligned_transition_rows(observations, actions, observation_ticks, action_ticks):\n    raise NotImplementedError('Exercise 3: construct aligned transition rows')",
            check="rows = aligned_transition_rows([{'s': 0}, {'s': 1}, {'s': 2}], [[4], [5]], [0, 1, 2], [0, 1])\nassert rows[1]['observation'] == {'s': 1} and np.array_equal(rows[1]['action'], [5]), 'Observation/action pairing is wrong'",
            use="print('Second aligned transition:', rows[1])",
        ),
        ex(
            "recording",
            "resample_aligned_transitions_10hz",
            title="Convert new recordings to the dataset rate",
            purpose="Build ACT windows at the pinned 10 Hz dataset cadence from aligned 20 Hz controller transitions.",
            signature="resample_aligned_transitions_10hz(rows) -> selected rows",
            contract="Input rows have contiguous 20 Hz tick IDs starting at zero. Keep rows at ticks 0, 2, 4, and so on. Preserve each observation/action pair. This exercise changes only new intervention data; it does not change the historical runner.",
            method="Select one aligned transition for each 100 ms interval. Do not average or repeat actions.",
            pointers="Run this after alignment checks and before making ACT action windows.",
            hint="Use a stride of `control_hz // dataset_hz`, then select source indices at that stride.",
            starter="def resample_aligned_transitions_10hz(rows):\n    raise NotImplementedError('Exercise 4: select aligned 10 Hz transitions')",
            check="rows = [{'tick': i, 'observation': i, 'action': [i]} for i in range(5)]\nsampled = resample_aligned_transitions_10hz(rows)\nassert [row['tick'] for row in sampled] == [0, 2, 4], 'Expected aligned 10 Hz rows from the 20 Hz stream'\nassert [row['action'] for row in sampled] == [[0], [2], [4]], 'Resampling must preserve applied actions'",
            use="print('Saved data ticks:', [row['tick'] for row in sampled])",
        ),
        ex(
            "recording",
            "dataset_tick_indices",
            title="Map simulator ticks to data cadence",
            purpose="Keep the saved 10 Hz row timing explicit.",
            signature="dataset_tick_indices(control_ticks) -> 10 Hz row indices",
            contract="Input is increasing even-valued 20 Hz timestamps. Output is the corresponding 10 Hz row index. It does not change how long the simulator executes each action.",
            method="One timestamp interval in the dataset equals two simulator control ticks.",
            pointers="Record the current mapping; do not change the pipeline timing.",
            hint="Integer-divide even control ticks by two.",
            starter="def dataset_tick_indices(control_ticks):\n    raise NotImplementedError('Exercise 5: map 20 Hz ticks to 10 Hz rows')",
            check="assert np.array_equal(dataset_tick_indices([0, 2, 4]), [0, 1, 2]), '10 Hz mapping is wrong'",
            use="print('dataset indices:', dataset_tick_indices([0, 2, 4, 6]))",
        ),
        ex(
            "recording",
            "intervention_effect",
            title="Compare paired intervention outcomes",
            purpose="Measure gains and losses without dropping failed attempts.",
            signature="intervention_effect(baseline, changed_data, *, case_ids) -> summary",
            contract="Each row has a unique `case_id` for one state and reset seed, plus `state_id`, `reset_seed`, `task_id`, and `success`. Keep the task and reset identity paired across methods. The denominator includes every attempt.",
            method="Join by unique case ID, then check the state, seed, and task before counting gains and losses.",
            pointers="Use intervention states 15–19. Keep baseline data, evaluation conditions, and budget fixed.",
            hint="A case key can include both state and seed, for example `state-15-seed-20260945`. Reject duplicate keys before making a lookup map.",
            starter="def intervention_effect(baseline, changed_data, *, case_ids):\n    raise NotImplementedError('Exercise 6: compare matched intervention results')",
            check="case = 'state-15-seed-20260945'\nbase = {'case_id': case, 'state_id': 15, 'reset_seed': 20260945, 'task_id': 'bowl_plate', 'success': False}\nchanged = {**base, 'success': True}\nresult = intervention_effect([base], [changed], case_ids=[case])\nassert result['gains'] == [case] and result['failed_attempts_included'], 'Paired gain or failure accounting is wrong'",
            use="print('Paired intervention summary:', result)",
        ),
    ],
    (
        "Test whether improved demonstrations help. Record and review accepted demonstrations, then compare the frozen baseline with a bounded ACT continuation. Use held-out development states from 15–19; keep final states 20–49 closed. Record hashes, training steps, all attempts, and paired gains/losses.",
        "Use the keyboard recorder and checked recovery pipeline. Record five to ten accepted demonstrations on development states only. This cell uses your alignment function to build state-action rows, resamples aligned 20 Hz transitions to 10 Hz, then makes 20-step ACT windows. The controller still applies one action per 20 Hz tick. The continuation uses a new optimizer. This cell stays off by default and never fetches weights or data. See `tutorials/08-data-quality/README.md` for the keyboard recorder commands and key map.",
        """RUN_RECOVERY = False
if RUN_RECOVERY:
    from vla_lab.experiments.build_recovery_manifest import build_keyboard_manifest
    from vla_lab.experiments.recovery import continue_act_with_recovery
    from vla_lab.evaluation.states import evaluate_act_states
    from vla_lab.teaching.live import find_workspace
    from vla_lab.data.recording import RecordedEpisode
    root = find_workspace()
    recording_dir = root / '.local-data/course-results/recovery/recordings'
    recording_files = sorted(recording_dir.glob('*.npz'))
    if not 5 <= len(recording_files) <= 10:
        raise ValueError('Record and review five to ten accepted .npz episodes first')
    for path in recording_files:
        episode = RecordedEpisode.load(path)
        transitions = aligned_transition_rows(episode.observations, episode.actions,
                                               episode.observation_ticks, episode.action_ticks)
        transitions_10hz = resample_aligned_transitions_10hz(transitions)
        sampled_actions = np.stack([row['action'] for row in transitions_10hz])
        window, padding = action_window(sampled_actions, start=0, horizon=20)
        assert len(transitions) == len(episode.actions)
        assert len(transitions_10hz) == (len(episode.actions) + 1) // 2
        assert window.shape == (20, 7) and padding.shape == (20,)
    manifest_path = root / '.local-data/course-results/recovery/manifest.json'
    manifest = build_keyboard_manifest(recording_files, task_alias='bowl_plate',
                                       configs_root=root / 'configs', output=manifest_path)
    intervention_cases = list(range(15, 20))
    heldout = manifest['heldout_development_state_ids']
    if not set(intervention_cases) <= set(heldout):
        raise ValueError('Record demonstrations only on states 0–9; states 15–19 are reserved before training')
    base_checkpoint = root / '.local-data/runs/act-bowl-plate-probe-200/checkpoint-000200'
    intervention = continue_act_with_recovery(
        manifest_path=manifest_path, base_checkpoint=base_checkpoint,
        configs_root=root / 'configs', data_root=root / '.local-data',
        output_dir=root / '.local-data/course-results/recovery/continued', steps=100,
        transition_rows_fn=aligned_transition_rows,
        resample_rows_fn=resample_aligned_transitions_10hz,
        training_transition_hz=10,
        window_fn=action_window)
    heldout = intervention_cases
    baseline_dir = root / '.local-data/course-results/recovery/eval-baseline'
    changed_dir = root / '.local-data/course-results/recovery/eval-changed'
    baseline_summary = evaluate_act_states(checkpoint=base_checkpoint, configs_root=root / 'configs',
        data_root=root / '.local-data', output_dir=baseline_dir, state_ids=heldout)
    changed_summary = evaluate_act_states(checkpoint=intervention['checkpoint'], configs_root=root / 'configs',
        data_root=root / '.local-data', output_dir=changed_dir, state_ids=heldout)
    def intervention_rows(folder):
        import json
        records = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
        converted = []
        for row in records:
            state_id = int(row['initial_state_id'])
            # The evaluator's frozen rule is reset_seed = 20260930 + state_id.
            # Error rows can have empty metrics; keep them with that declared seed.
            expected_seed = 20260930 + state_id
            saved_seed = row.get('metrics', {}).get('reset_seed')
            if saved_seed is not None and int(saved_seed) != expected_seed:
                raise ValueError(f"Saved reset seed differs for state {state_id}")
            converted.append({'case_id': f"state-{state_id}-seed-{expected_seed}",
                 'state_id': state_id, 'reset_seed': expected_seed,
                 'task_id': row['task_id'], 'success': row['success'] is True})
        return converted
    expected_cases = [f'state-{state}-seed-{20260930 + state}' for state in heldout]
    result = intervention_effect(intervention_rows(baseline_dir), intervention_rows(changed_dir),
                                 case_ids=expected_cases)
    (root / '.local-data/course-results/recovery/paired-summary.json').write_text(json.dumps(result, indent=2) + '\\n')
    print('New checkpoint:', intervention['checkpoint'], 'paired result:', result)
    print('Baseline attempts/successes:', baseline_summary['attempts'], baseline_summary['successes'])
    print('Changed attempts/successes:', changed_summary['attempts'], changed_summary['successes'])""",
    ),
)


build(
    "14-comparison",
    "Unit 14: Compare and select policies",
    "Build an exact evaluation case list, join paired outcomes, and include every attempt. Use final states only after Unit 13 and after recipes are frozen.",
    """from __future__ import annotations
from vla_lab.teaching.evaluation import development_cases, final_cases, validate_case_results, paired_summary, evaluate_frozen_method_timed
""",
    [
        ex(
            "evaluation",
            "development_cases",
            title="Select development cases by purpose",
            purpose="Use separate states for exploration, method checks, and data intervention.",
            signature="development_cases(stage) -> list[int]",
            contract="`exploration` returns 0–9; `method` returns 10–14; `intervention` returns 15–19.",
            method="Keep these groups fixed so experiments do not use the final set.",
            pointers="Try the state IDs once and inspect the exact returned list.",
            hint="A dictionary maps each stage name to one range.",
            starter="def development_cases(stage):\n    raise NotImplementedError('Exercise 1: select the declared development group')",
            check="assert development_cases('method') == [10, 11, 12, 13, 14], 'Method states are wrong'",
            use="print('Method check states:', development_cases('method'))",
        ),
        ex(
            "evaluation",
            "validate_case_results",
            title="Check complete evaluation records",
            purpose="Catch missing or duplicate cases before reporting a score.",
            signature="validate_case_results(rows, *, expected_case_ids) -> ordered rows",
            contract="Each row has a unique `case_id`. Require exact coverage and return rows in expected order.",
            method="Compare actual IDs with the frozen expected set, then order records by ID.",
            pointers="Never silently drop a failed or missing run.",
            hint="Build a map only after checking duplicate IDs.",
            starter="def validate_case_results(results, *, expected_case_ids):\n    raise NotImplementedError('Exercise 2: validate exact case coverage')",
            check="ordered = validate_case_results([{'case_id': 1, 'success': False}, {'case_id': 0, 'success': True}], expected_case_ids=[0, 1])\nassert [row['case_id'] for row in ordered] == [0, 1], 'Rows are not in frozen case order'",
            use="print('Included case IDs:', [row['case_id'] for row in ordered])",
        ),
        ex(
            "evaluation",
            "paired_summary",
            title="Summarize paired attempts",
            purpose="Compare two recipes on the same cases and report disagreements.",
            signature="paired_summary(left, right, *, expected_case_ids) -> summary",
            contract="Pair by `case_id = state_id + reset_seed`. Each row also carries task, partition, prompt, action length, policy revision, and success. State, seed, task, partition, and prompt must match. Keep each policy's action length and revision fixed across its rows.",
            method="Pair records before computing rates and disagreements. Count only explicit `success=True` as success; retain all failed attempts in the denominator.",
            pointers="Small paired samples can be inconclusive; show disagreement counts.",
            hint="Count `left_success and not right_success` separately from the reverse. Check state ID and reset seed as well as the display case key.",
            starter="def paired_summary(left, right, *, expected_case_ids):\n    raise NotImplementedError('Exercise 3: aggregate paired attempts')",
            check="base = {'state_id': 0, 'reset_seed': 4, 'task_id': 'bowl_plate', 'partition': 'method', 'prompt': 'place bowl', 'action_steps': 10}\nleft = [{**base, 'case_id': 'state-0-seed-4', 'policy_revision': 'rev-a', 'success': True}]\nright = [{**base, 'case_id': 'state-0-seed-4', 'policy_revision': 'rev-b', 'success': False}]\nsummary = paired_summary(left, right, expected_case_ids=['state-0-seed-4'])\nassert summary['cases'] == 1 and summary['left_only'] == 1 and summary['right_only'] == 0, 'Paired disagreement counts are wrong'",
            use="print('Paired difference:', summary['paired_difference'], 'disagreements:', summary['left_only'], summary['right_only'])",
        ),
        ex(
            "evaluation",
            "final_cases",
            title="Open the final evaluation set",
            purpose="Protect the final states from earlier model and recipe choices.",
            signature="final_cases(*, data_unit_complete, recipes_frozen) -> list[int]",
            contract="Return states 20–49 only when Unit 13 data work is complete and all recipes are frozen.",
            method="Require both gates before returning the final IDs.",
            pointers="A valid final comparison can be inconclusive. Make no changes after seeing these outcomes.",
            hint="Check both booleans before creating the range.",
            starter="def final_cases(*, data_unit_complete, recipes_frozen):\n    raise NotImplementedError('Exercise 4: enforce the final-set gate')",
            check="try:\n    final_cases(data_unit_complete=False, recipes_frozen=True)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Final states opened before the data unit was complete')",
            use="print('Final set size:', len(final_cases(data_unit_complete=True, recipes_frozen=True)))",
        ),
    ],
    (
        "Freeze the task, checkpoint hashes, prompt, action length, reset seed rule, and time budget before final evaluation. Copy the pinned comparison recipe to a new output folder; verify absolute checkpoint paths and on-disk hashes before evaluation. Use states 20–49 once for each recipe. The evaluator records every policy or infrastructure failure. Pair by `(state_id, reset_seed)` and report all attempts and disagreements. It records both action-selection API latency (which includes queued actions) and action-generation latency. Report mean, p95, maximum, and every attempt; missing timing stays `None`. This is a comparison of prepared practical recipes: training budgets and pretraining differ. Report those recipe fields so the table does not imply a from-scratch or equal-training comparison.",
        "Set `RUN_FINAL=False` by default because the full comparison runs 90 simulator episodes. The runner uses local checkpoints and does not download. Keep outputs in a new folder. Do not report historical results as new runs. The code resolves checkpoint paths against the workspace, checks hashes before the first episode, writes every attempt, a table, and a plot. The table includes checkpoint hash, robot training steps, and pretraining provenance from the frozen recipe.",
        """RUN_FINAL = False
if RUN_FINAL:
    import json
    import hashlib
    import yaml
    from pathlib import Path
    import matplotlib.pyplot as plt
    from vla_lab.evaluation.comparison import METHODS, load_comparison_config
    from vla_lab.teaching.live import find_workspace
    root = find_workspace()
    output = root / '.local-data/course-results/final-bowl-plate-course-01'
    output.mkdir(parents=True, exist_ok=False)
    frozen_config = output / 'comparison.yaml'
    recipe = yaml.safe_load((root / 'configs/experiments/comparison.yaml').read_text())
    for method, item in recipe['policies'].items():
        checkpoint = Path(item['checkpoint'])
        item['checkpoint'] = str((root / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve())
        if 'vlm_dir' in item:
            vlm_dir = Path(item['vlm_dir'])
            item['vlm_dir'] = str((root / vlm_dir).resolve() if not vlm_dir.is_absolute() else vlm_dir.resolve())
    frozen_config.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config = load_comparison_config(frozen_config, root / 'configs')
    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    for method, item in config['policies'].items():
        checkpoint = Path(item['checkpoint'])
        weights = checkpoint / 'model.safetensors'
        if not checkpoint.is_absolute() or not weights.is_file():
            raise FileNotFoundError(f'{method} checkpoint is missing or not absolute: {checkpoint}')
        if sha256_file(weights) != item['model_sha256']:
            raise ValueError(f'{method} checkpoint hash differs from the frozen recipe')
        if 'vlm_dir' in item and not Path(item['vlm_dir']).is_dir():
            raise FileNotFoundError(f"SmolVLA VLM directory is missing: {item['vlm_dir']}")
    # This gate runs only after the frozen recipe and each local model hash pass.
    recipes_frozen = True
    recovery = root / '.local-data/course-results/recovery'
    paired_path = recovery / 'paired-summary.json'
    provenance_path = recovery / 'continued/recovery_provenance.json'
    baseline_summary_path = recovery / 'eval-baseline/summary.json'
    changed_summary_path = recovery / 'eval-changed/summary.json'
    data_complete = all(path.is_file() for path in (
        paired_path, provenance_path, baseline_summary_path, changed_summary_path))
    if data_complete:
        paired_data = json.loads(paired_path.read_text())
        baseline_eval = json.loads(baseline_summary_path.read_text())
        changed_eval = json.loads(changed_summary_path.read_text())
        data_complete = (
            paired_data.get('cases') == 5
            and paired_data.get('failed_attempts_included') is True
            and baseline_eval.get('attempts') == 5
            and changed_eval.get('attempts') == 5
        )
    final_ids = final_cases(data_unit_complete=data_complete, recipes_frozen=True)
    assert final_ids == config['state_ids']
    result_dirs = {method: output / method for method in METHODS}
    timing_by_method = {}
    for method in METHODS:
        timed_result = evaluate_frozen_method_timed(method=method, config_path=str(frozen_config),
            configs_root=root / 'configs', data_root=root / '.local-data',
            output_dir=result_dirs[method])
        method_summary = timed_result['summary']
        timing_by_method[method] = timed_result['timing']
        print(method, 'attempts:', method_summary['attempts'], 'successes:', method_summary['successes'],
              'policy/infrastructure errors:', method_summary['policy_errors'], method_summary['infrastructure_errors'])
    rows_by_method = {}
    table = []
    for method in METHODS:
        saved = [json.loads(line) for line in (result_dirs[method] / 'episodes.jsonl').read_text().splitlines()]
        recipe = config['policies'][method]
        converted = []
        for row in saved:
            seed = int(row['metrics']['seed'])
            state_id = int(row['initial_state_id'])
            converted.append({'case_id': f'state-{state_id}-seed-{seed}', 'state_id': state_id,
                'reset_seed': seed, 'task_id': row['task_id'], 'partition': row['partition'],
                'prompt': config['task_prompt'], 'action_steps': recipe['n_action_steps'],
                'policy_revision': row['policy_revision'], 'success': row['success'] is True})
        rows_by_method[method] = converted
        successes = sum(row['success'] for row in converted)
        elapsed = [float(row['metrics']['elapsed_s']) for row in saved if row['metrics'].get('elapsed_s') is not None]
        action_api = timing_by_method[method]['action_selection_api']
        action_generation = timing_by_method[method]['action_generation']
        table.append({'method': method, 'successes': successes, 'attempts': len(converted),
            'success_rate': successes / len(converted),
            'recipe_file': str(frozen_config), 'checkpoint_sha256': recipe['model_sha256'],
            'robot_training_steps': recipe.get('robot_training_steps'),
            'pretraining': recipe.get('pretraining'),
            'mean_episode_seconds': (sum(elapsed) / len(elapsed)) if elapsed else None,
            'mean_action_api_seconds': action_api['mean_s'], 'p95_action_api_seconds': action_api['p95_s'],
            'max_action_api_seconds': action_api['max_s'], 'action_api_calls': action_api['count'],
            'mean_action_generation_seconds': action_generation['mean_s'],
            'p95_action_generation_seconds': action_generation['p95_s'],
            'max_action_generation_seconds': action_generation['max_s'],
            'action_generation_calls': action_generation['count'],
            'timing_scope': 'episode wall time includes simulator and processing; action API includes queue reads; action generation is the model chunk hook',
            'policy_errors': sum(row['termination_reason'] == 'policy_error' for row in saved),
            'infrastructure_errors': sum(row['termination_reason'] == 'infrastructure_error' for row in saved)})
    expected = [f"state-{state}-seed-{config['base_seed'] + state}" for state in final_ids]
    for left_name, right_name in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
        pair = paired_summary(rows_by_method[left_name], rows_by_method[right_name], expected_case_ids=expected)
        print(left_name, 'vs', right_name, pair)
    (output / 'course-table.json').write_text(json.dumps(table, indent=2) + '\\n')
    plt.bar([item['method'] for item in table], [item['success_rate'] for item in table])
    plt.ylim(0, 1)
    plt.ylabel('Success rate over all attempts')
    plt.tight_layout()
    plt.savefig(output / 'success-rates.png', dpi=160)
    print('Saved recipe, all attempts, table, and plot under', output)""",
    ),
)
