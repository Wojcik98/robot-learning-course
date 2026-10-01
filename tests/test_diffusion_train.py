"""Check the Diffusion Policy training data windows, settings, and checkpoint bundle on the CPU."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import datasets.config as datasets_config
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion import DiffusionPolicy
from torch.utils.data import Dataset

from vla_lab.policy.diffusion import make_diffusion_config, make_diffusion_processors
from vla_lab.training.diffusion import (
    DROP_N_LAST_FRAMES,
    RESNET18_IMAGENET_SHA256,
    diffusion_delta_timestamps,
    eligible_frame_indices,
    executed_action_slice,
    load_diffusion_bundle,
    load_diffusion_train_config,
    save_diffusion_bundle,
    train_diffusion_reference,
    validate_horizons,
    validate_window_batch,
    verify_backbone_weights,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TRAIN_CONFIG = CONFIGS / "train" / "diffusion_bowl_plate.yaml"
LOCAL_DATA = ROOT / ".local-data"


def _small_policy() -> DiffusionPolicy:
    """Build the real recipe without the ResNet download; weights are random."""
    config = make_diffusion_config(device="cpu")
    config.pretrained_backbone_weights = None
    torch.manual_seed(0)
    return DiffusionPolicy(config)


class SyntheticWindows(Dataset):
    """Return windows with the shapes that the real dataset gives."""

    def __init__(self, length: int = 4) -> None:
        self.length = length
        self.requested: list[int] = []

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.requested.append(index)
        scale = (index + 1) / (self.length + 1)
        return {
            "observation.images.image": torch.full((2, 3, 256, 256), scale),
            "observation.images.image2": torch.full((2, 3, 256, 256), scale),
            "observation.state": torch.zeros(2, 8),
            "action": torch.zeros(16, 7),
            "action_is_pad": torch.zeros(16, dtype=torch.bool),
        }


def _batch(size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "observation.images.image": torch.zeros(size, 2, 3, 256, 256),
        "observation.images.image2": torch.zeros(size, 2, 3, 256, 256),
        "observation.state": torch.zeros(size, 2, 8),
        "action": torch.zeros(size, 16, 7),
        "action_is_pad": torch.zeros(size, 16, dtype=torch.bool),
    }


# --- Settings ---------------------------------------------------------------------------------


def test_train_config_matches_the_fixed_recipe() -> None:
    config = load_diffusion_train_config(TRAIN_CONFIG, CONFIGS)
    policy_config = make_diffusion_config(device="cpu")

    assert config.task_alias == "bowl_plate"
    assert len(config.train_episode_ids) == 39
    assert (config.n_obs_steps, config.horizon, config.n_action_steps) == (2, 16, 8)
    assert config.pretrained_backbone_weights == policy_config.pretrained_backbone_weights
    assert config.batch_size <= 8
    assert len(config.config_sha256) == 64
    assert config.state_ids == tuple(range(20))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("horizon", 15, "horizon"),
        ("n_action_steps", 9, "n_action_steps"),
        ("n_obs_steps", 1, "n_obs_steps"),
        ("pretrained_backbone_weights", None, "pretrained_backbone_weights"),
        ("train_episode_count", 10, "train_episode_count"),
        ("policy", "act", "diffusion"),
        ("profile", "cloud", "local"),
        ("steps", 0, "positive"),
        ("batch_size", 0, "positive"),
        ("warmup_steps", -1, "warmup"),
    ],
)
def test_train_config_rejects_changed_recipe(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    payload = yaml.safe_load(TRAIN_CONFIG.read_text())
    payload[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=message):
        load_diffusion_train_config(path, CONFIGS)


def test_train_config_rejects_final_simulator_states(tmp_path: Path) -> None:
    splits = yaml.safe_load((CONFIGS / "eval" / "splits.yaml").read_text())
    final_state = splits["tasks"]["bowl_plate"]["final_state_ids"][0]
    payload = yaml.safe_load(TRAIN_CONFIG.read_text())
    payload["development_state_ids"] = [final_state]
    path = tmp_path / "final.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="development"):
        load_diffusion_train_config(path, CONFIGS)


def test_train_config_hash_changes_with_file_content(tmp_path: Path) -> None:
    payload = yaml.safe_load(TRAIN_CONFIG.read_text())
    payload["steps"] += 1
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(payload))

    assert (
        load_diffusion_train_config(path, CONFIGS).config_sha256
        != load_diffusion_train_config(TRAIN_CONFIG, CONFIGS).config_sha256
    )


# --- Window alignment and padding -------------------------------------------------------------


def test_delta_timestamps_start_one_step_before_the_current_frame() -> None:
    deltas = diffusion_delta_timestamps(make_diffusion_config(device="cpu"), fps=10)

    assert deltas["observation.images.image"] == pytest.approx([-0.1, 0.0])
    assert deltas["observation.images.image2"] == pytest.approx([-0.1, 0.0])
    assert deltas["observation.state"] == pytest.approx([-0.1, 0.0])
    assert deltas["action"] == pytest.approx([(index - 1) / 10 for index in range(16)])


def test_executed_actions_are_the_current_frame_and_seven_later_frames() -> None:
    config = make_diffusion_config(device="cpu")
    start, end = executed_action_slice(config)
    window_offsets = [index - 1 for index in range(config.horizon)]  # action_delta_indices

    assert (start, end) == (1, 9)
    assert window_offsets[start:end] == list(range(8))
    # The same slice must be the one that LeRobot executes at run time.
    assert config.action_delta_indices == window_offsets


def test_policy_executes_the_slice_named_by_the_helper() -> None:
    policy = _small_policy()
    horizon = policy.config.horizon
    marked = torch.arange(horizon, dtype=torch.float32).reshape(1, horizon, 1).expand(1, horizon, 7)
    policy.diffusion.conditional_sample = lambda *args, **kwargs: marked
    policy.diffusion._prepare_global_conditioning = lambda batch: torch.zeros(1, 1)
    batch = {"observation.state": torch.zeros(1, 2, 8)}

    executed = policy.diffusion.generate_actions(batch)

    start, end = executed_action_slice(policy.config)
    assert executed[0, :, 0].tolist() == list(range(start, end))


def test_horizon_check_rejects_unusable_prediction_and_execution_lengths() -> None:
    good = SimpleNamespace(
        n_obs_steps=2, horizon=16, n_action_steps=8, drop_n_last_frames=7, down_dims=(128, 256, 512)
    )
    validate_horizons(good)

    for change, message in [
        ({"n_action_steps": 16}, "n_action_steps"),
        ({"horizon": 12}, "multiple"),
        ({"drop_n_last_frames": 0}, "drop_n_last_frames"),
        ({"n_obs_steps": 0}, "n_obs_steps"),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_horizons(SimpleNamespace(**{**vars(good), **change}))
    assert DROP_N_LAST_FRAMES == 7


def test_eligible_frames_drop_the_last_seven_frames_of_each_episode() -> None:
    episode = [5] * 10 + [9] * 8 + [2] * 7
    frame = list(range(10)) + list(range(8)) + list(range(7))

    kept = eligible_frame_indices(episode, frame, drop_n_last_frames=7)

    assert kept == [0, 1, 2, 10]  # 10 - 7 = 3, 8 - 7 = 1, and an episode of 7 frames gives none


def test_eligible_frames_reject_a_dataset_without_one_usable_frame() -> None:
    with pytest.raises(ValueError, match="No frame"):
        eligible_frame_indices([0] * 5, list(range(5)), drop_n_last_frames=7)


def test_eligible_frames_reject_rows_that_disagree() -> None:
    with pytest.raises(ValueError, match="same length"):
        eligible_frame_indices([0, 0], [0], drop_n_last_frames=1)


def _write_tiny_dataset(root: Path) -> list[int]:
    """Write two short episodes. State, action, and pixels all carry the frame code."""
    features = {
        "observation.images.image": {"dtype": "image", "shape": (8, 8, 3), "names": None},
        "observation.images.image2": {"dtype": "image", "shape": (8, 8, 3), "names": None},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    dataset = LeRobotDataset.create(
        "local/tiny", fps=10, features=features, root=root, use_videos=False
    )
    lengths = [12, 9]
    for episode, length in enumerate(lengths):
        for frame in range(length):
            code = 100 * episode + frame
            dataset.add_frame(
                {
                    "observation.images.image": np.full((8, 8, 3), code, dtype=np.uint8),
                    "observation.images.image2": np.full((8, 8, 3), code, dtype=np.uint8),
                    "observation.state": np.full(8, code, dtype=np.float32),
                    "action": np.full(7, code + 0.5, dtype=np.float32),
                    "task": "tiny",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return lengths


@pytest.fixture(scope="module")
def tiny_windows(tmp_path_factory: pytest.TempPathFactory) -> LeRobotDataset:
    root = tmp_path_factory.mktemp("tiny") / "dataset"
    cache = root.parent / "cache"
    cache.mkdir()
    patch = pytest.MonkeyPatch()
    patch.setattr(datasets_config, "HF_DATASETS_CACHE", str(cache))
    try:
        _write_tiny_dataset(root)
        deltas = diffusion_delta_timestamps(make_diffusion_config(device="cpu"), fps=10)
        yield LeRobotDataset("local/tiny", root=root, delta_timestamps=deltas)
    finally:
        patch.undo()


def test_real_dataset_window_starts_one_frame_before_a_middle_frame(
    tiny_windows: LeRobotDataset,
) -> None:
    sample = tiny_windows[5]  # episode 0, frame 5

    assert sample["observation.state"][:, 0].tolist() == [4.0, 5.0]
    assert sample["action"][:, 0].tolist() == [code + 0.5 for code in range(4, 12)] + [11.5] * 8
    assert sample["action_is_pad"].tolist() == [False] * 8 + [True] * 8
    assert not sample["observation.state_is_pad"].any()


def test_real_dataset_pads_the_first_observation_and_action_at_an_episode_start(
    tiny_windows: LeRobotDataset,
) -> None:
    sample = tiny_windows[0]

    assert sample["observation.state"][:, 0].tolist() == [0.0, 0.0]
    assert sample["observation.state_is_pad"].tolist() == [True, False]
    assert sample["action"][0, 0].item() == pytest.approx(0.5)
    assert sample["action_is_pad"].tolist() == [True] + [False] * 12 + [True] * 3
    # Image pixels follow the same indices as the state values.
    pixels = (sample["observation.images.image"][:, 0, 0, 0] * 255).round()
    assert pixels.tolist() == [0.0, 0.0]


def test_real_dataset_never_reads_across_an_episode_boundary(
    tiny_windows: LeRobotDataset,
) -> None:
    first_of_second = tiny_windows[12]  # episode 1, frame 0
    last_of_first = tiny_windows[11]

    assert first_of_second["observation.state"][:, 0].tolist() == [100.0, 100.0]
    assert first_of_second["observation.state_is_pad"].tolist() == [True, False]
    assert last_of_first["action"][:, 0].tolist() == [10.5, 11.5] + [11.5] * 14
    assert last_of_first["action_is_pad"].tolist() == [False, False] + [True] * 14


def test_eligible_frames_from_the_real_dataset_stay_inside_the_episode(
    tiny_windows: LeRobotDataset,
) -> None:
    episode = tiny_windows.hf_dataset["episode_index"]
    frame = tiny_windows.hf_dataset["frame_index"]
    kept = eligible_frame_indices(
        [int(value) for value in episode], [int(value) for value in frame], DROP_N_LAST_FRAMES
    )

    assert kept == [0, 1, 2, 3, 4, 12, 13]
    # The last kept frame pads at most seven future actions (horizon - n_obs_steps - n_action_steps + 1 = 7).
    padded = tiny_windows[4]["action_is_pad"].sum().item()
    assert padded == 16 - (12 - 4 + 1)  # frames 3..11 exist for the 16-step window
    assert padded <= 7


def test_window_batch_check_accepts_the_real_shapes_and_rejects_bad_ones() -> None:
    config = make_diffusion_config(device="cpu")
    validate_window_batch(_batch(), config)

    bad_obs = _batch()
    bad_obs["observation.state"] = torch.zeros(2, 1, 8)
    bad_action = _batch()
    bad_action["action"] = torch.zeros(2, 8, 7)
    bad_pad = _batch()
    bad_pad["action_is_pad"] = torch.zeros(2, 16)
    missing = _batch()
    del missing["observation.images.image2"]
    swapped = _batch()
    swapped["observation.images.image"] = torch.zeros(2, 2, 256, 256, 3)
    nonfinite = _batch()
    nonfinite["action"][0, 0, 0] = float("nan")
    for batch in (bad_obs, bad_action, bad_pad, missing, swapped, nonfinite):
        with pytest.raises(ValueError):
            validate_window_batch(batch, config)


# --- Processors and weights -------------------------------------------------------------------


def test_processors_use_only_training_extrema() -> None:
    config = make_diffusion_config(device="cpu")
    preprocessor, postprocessor = make_diffusion_processors(config, "bowl_plate", CONFIGS)
    stats = json.loads((CONFIGS / "data" / "stats" / "bowl_plate.json").read_text())
    train_ids = yaml.safe_load((CONFIGS / "data" / "episodes.yaml").read_text())["tasks"][
        "bowl_plate"
    ]["train_episode_ids"]
    action_min = torch.tensor(stats["modalities"]["action"]["min"])

    assert len(train_ids) == 39
    assert torch.allclose(postprocessor(-torch.ones(1, 7)), action_min.unsqueeze(0), atol=1e-5)
    assert preprocessor is not None


def test_pinned_backbone_weights_hash_is_checked(tmp_path: Path) -> None:
    checkpoints = tmp_path / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "resnet18-f37072fd.pth").write_bytes(b"not the pinned weights")

    with pytest.raises(ValueError, match="hash"):
        verify_backbone_weights(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_backbone_weights(tmp_path / "missing")
    assert RESNET18_IMAGENET_SHA256.startswith("f37072fd")


@pytest.mark.skipif(
    not (LOCAL_DATA / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth").is_file(),
    reason="The pinned ResNet-18 file is not in the local cache",
)
def test_cached_backbone_weights_match_the_pinned_hash() -> None:
    assert verify_backbone_weights(LOCAL_DATA / "torch") == RESNET18_IMAGENET_SHA256


def test_model_stays_small_enough_for_an_eight_gb_probe() -> None:
    policy = _small_policy()
    parameters = sum(parameter.numel() for parameter in policy.parameters())

    assert 40_000_000 < parameters < 42_000_000
    assert load_diffusion_train_config(TRAIN_CONFIG, CONFIGS).batch_size <= 8


@pytest.mark.skipif(
    not (LOCAL_DATA / "hub" / "lerobot-libero" / "meta" / "episodes").is_dir(),
    reason="The local LIBERO metadata is not present",
)
def test_local_metadata_gives_the_frozen_windows_without_decoding_video() -> None:
    meta = LOCAL_DATA / "hub" / "lerobot-libero" / "meta"
    table = pq.read_table(next((meta / "episodes").rglob("*.parquet"))).to_pandas()
    ids = yaml.safe_load((CONFIGS / "data" / "episodes.yaml").read_text())["tasks"]["bowl_plate"][
        "train_episode_ids"
    ]
    rows = table[table["episode_index"].isin(ids)]
    info = json.loads((meta / "info.json").read_text())

    assert info["fps"] == 10
    assert len(rows) == 39
    assert (rows["length"] > DROP_N_LAST_FRAMES).all()
    assert int((rows["length"] - DROP_N_LAST_FRAMES).sum()) < int(rows["length"].sum())


# --- Training and bundle ----------------------------------------------------------------------


def test_one_training_step_changes_weights_and_saves_a_reloadable_bundle(tmp_path: Path) -> None:
    policy = _small_policy()
    original = {name: value.detach().clone() for name, value in policy.named_parameters()}

    result = train_diffusion_reference(
        task_alias="bowl_plate",
        configs_root=CONFIGS,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "run",
        steps=1,
        batch_size=2,
        device="cpu",
        dataset=SyntheticWindows(),
        policy=policy,
        warmup_steps=0,
    )

    assert result.steps == 1
    assert result.last_loss > 0
    assert any(
        not torch.equal(original[name], value.detach()) for name, value in policy.named_parameters()
    )
    checkpoint = tmp_path / "run" / "checkpoint-000001"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    split = yaml.safe_load((CONFIGS / "data" / "episodes.yaml").read_text())["tasks"]["bowl_plate"]
    assert manifest["policy"] == "diffusion"
    assert manifest["split_sha256"] == split["split_sha256"]
    assert (manifest["n_obs_steps"], manifest["horizon"], manifest["n_action_steps"]) == (2, 16, 8)
    assert run["train_episode_ids"] == split["train_episode_ids"]
    assert run["sample_count"] == 4
    assert (tmp_path / "run" / "metrics.jsonl").is_file()

    reloaded, preprocessor, postprocessor, loaded_manifest = load_diffusion_bundle(checkpoint)
    assert loaded_manifest == manifest
    for name, value in policy.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name]), name
    assert reloaded.config.horizon == 16
    assert preprocessor is not None and postprocessor is not None


def test_reloaded_bundle_gives_the_same_action_for_the_same_noise(tmp_path: Path) -> None:
    policy = _small_policy().eval()
    config = policy.config
    preprocessor, postprocessor = make_diffusion_processors(config, "bowl_plate", CONFIGS)
    save_diffusion_bundle(
        policy, preprocessor, postprocessor, tmp_path / "bundle", "bowl_plate", CONFIGS
    )
    reloaded, _, _, _ = load_diffusion_bundle(tmp_path / "bundle")
    reloaded.eval()
    batch = {
        "observation.state": torch.zeros(1, 2, 8),
        "observation.images": torch.zeros(1, 2, 2, 3, 128, 128),
    }
    noise = torch.randn(1, 16, 7, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        torch.manual_seed(42)
        expected = policy.diffusion.generate_actions(batch, noise=noise)
        torch.manual_seed(42)
        actual = reloaded.diffusion.generate_actions(batch, noise=noise)

    assert expected.shape == (1, 8, 7)
    assert torch.isfinite(actual).all()
    assert torch.allclose(expected, actual, atol=1e-6)


def test_bundle_rejects_a_nonempty_destination_and_a_changed_manifest(tmp_path: Path) -> None:
    policy = _small_policy()
    preprocessor, postprocessor = make_diffusion_processors(policy.config, "bowl_plate", CONFIGS)
    target = tmp_path / "bundle"
    save_diffusion_bundle(policy, preprocessor, postprocessor, target, "bowl_plate", CONFIGS)

    with pytest.raises(FileExistsError):
        save_diffusion_bundle(policy, preprocessor, postprocessor, target, "bowl_plate", CONFIGS)
    config_path = target / "config.json"
    saved_config = config_path.read_bytes()
    config_path.write_bytes(saved_config + b" ")
    with pytest.raises(ValueError, match="config.json"):
        load_diffusion_bundle(target)
    config_path.write_bytes(saved_config)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["horizon"] = 64
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="horizon"):
        load_diffusion_bundle(target)
    (target / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_diffusion_bundle(target)


def test_bundle_rejects_stats_hashes_that_differ_from_the_saved_processors(
    tmp_path: Path,
) -> None:
    policy = _small_policy()
    preprocessor, postprocessor = make_diffusion_processors(policy.config, "bowl_plate", CONFIGS)
    target = tmp_path / "bundle"
    save_diffusion_bundle(policy, preprocessor, postprocessor, target, "bowl_plate", CONFIGS)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["training_stats_sha256"]["action"] = hashlib.sha256(b"other").hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="statistics"):
        load_diffusion_bundle(target)


def test_training_uses_only_eligible_samples(tmp_path: Path) -> None:
    dataset = SyntheticWindows(length=6)

    train_diffusion_reference(
        task_alias="bowl_plate",
        configs_root=CONFIGS,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "run",
        steps=2,
        batch_size=2,
        device="cpu",
        dataset=dataset,
        sample_indices=[1, 4],
        policy=_small_policy(),
        warmup_steps=0,
        checkpoint_every=10,
    )

    assert set(dataset.requested) == {1, 4}
    assert len(dataset.requested) == 4


def test_training_rejects_unusable_settings(tmp_path: Path) -> None:
    common = {
        "task_alias": "bowl_plate",
        "configs_root": CONFIGS,
        "data_root": tmp_path / "data",
        "device": "cpu",
        "dataset": SyntheticWindows(),
        "policy": _small_policy(),
    }
    with pytest.raises(ValueError, match="positive"):
        train_diffusion_reference(output_dir=tmp_path / "a", steps=0, batch_size=1, **common)
    with pytest.raises(ValueError, match="fewer"):
        train_diffusion_reference(output_dir=tmp_path / "b", steps=1, batch_size=9, **common)
    with pytest.raises(ValueError, match="sample"):
        train_diffusion_reference(
            output_dir=tmp_path / "c", steps=1, batch_size=1, sample_indices=[99], **common
        )
    with pytest.raises(ValueError, match="Unknown task"):
        train_diffusion_reference(
            output_dir=tmp_path / "d", steps=1, batch_size=1, **{**common, "task_alias": "none"}
        )
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "old").write_text("x")
    with pytest.raises(FileExistsError):
        train_diffusion_reference(output_dir=tmp_path / "e", steps=1, batch_size=1, **common)


def test_training_rejects_nonfinite_input(tmp_path: Path) -> None:
    class BadFrames(SyntheticWindows):
        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            sample = super().__getitem__(index)
            sample["observation.state"] = torch.full((2, 8), float("nan"))
            return sample

    with pytest.raises(ValueError, match="finite"):
        train_diffusion_reference(
            task_alias="bowl_plate",
            configs_root=CONFIGS,
            data_root=tmp_path / "data",
            output_dir=tmp_path / "run",
            steps=1,
            batch_size=1,
            device="cpu",
            dataset=BadFrames(),
            policy=_small_policy(),
        )


def test_policy_config_drops_the_same_frames_as_the_training_module() -> None:
    assert make_diffusion_config(device="cpu").drop_n_last_frames == DROP_N_LAST_FRAMES
