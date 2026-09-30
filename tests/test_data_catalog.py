from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vla_lab.data.catalog import catalog_episodes


def write_fixture(root: Path, *, mixed_task=False, incomplete=False):
    root.mkdir(parents=True)
    tasks_path = root / "tasks.parquet"
    episodes_path = root / "episodes.parquet"
    shard_path = root / "data" / "chunk-000" / "file-000.parquet"
    shard_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"task_index": [0, 1], "__index_level_0__": ["move cup", "open drawer"]}),
        tasks_path,
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [3, 8],
                "length": [2, 2],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 4],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
            }
        ),
        episodes_path,
    )
    rows = [
        {"episode_index": 3, "frame_index": 0, "index": 0, "task_index": 0},
        {"episode_index": 3, "frame_index": 1, "index": 1, "task_index": 1 if mixed_task else 0},
        {"episode_index": 8, "frame_index": 0, "index": 2, "task_index": 1},
        {"episode_index": 8, "frame_index": 1, "index": 3, "task_index": 1},
    ]
    if incomplete:
        rows.pop()
    pq.write_table(pa.Table.from_pylist(rows), shard_path)
    return tasks_path, [episodes_path], [shard_path]


def test_catalog_maps_task_indices_to_names_and_complete_episode_ids(tmp_path):
    tasks, episode_shards, data_shards = write_fixture(tmp_path / "fixture")

    catalog = catalog_episodes(data_shards, episode_shards, tasks)

    assert catalog[0].task_name == "move cup"
    assert catalog[0].episode_ids == (3,)
    assert catalog[1].task_name == "open drawer"
    assert catalog[1].episode_ids == (8,)


def test_catalog_rejects_incomplete_episode_boundary(tmp_path):
    tasks, episode_shards, data_shards = write_fixture(tmp_path / "fixture", incomplete=True)

    with pytest.raises(ValueError, match="incomplete"):
        catalog_episodes(data_shards, episode_shards, tasks)


def test_catalog_rejects_episode_with_multiple_task_indices(tmp_path):
    tasks, episode_shards, data_shards = write_fixture(tmp_path / "fixture", mixed_task=True)

    with pytest.raises(ValueError, match="task_index"):
        catalog_episodes(data_shards, episode_shards, tasks)


def test_catalog_rejects_non_contiguous_frame_indices(tmp_path):
    tasks, episode_shards, data_shards = write_fixture(tmp_path / "fixture")
    table = pq.read_table(data_shards[0]).to_pydict()
    table["frame_index"][2] = 2
    pq.write_table(pa.Table.from_pydict(table), data_shards[0])

    with pytest.raises(ValueError, match="frame_index"):
        catalog_episodes(data_shards, episode_shards, tasks)
