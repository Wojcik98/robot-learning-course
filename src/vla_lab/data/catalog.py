"""Read a local LeRobot v3 episode catalogue."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pyarrow import parquet


@dataclass(frozen=True)
class TaskEpisodes:
    """A task name and its complete episode IDs."""

    task_index: int
    task_name: str
    episode_ids: tuple[int, ...]


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        pass
    except ImportError as error:
        raise RuntimeError("PyArrow is required to read LeRobot parquet files") from error
    return parquet.read_table(path).to_pylist()


def _shard_key(path: Path) -> tuple[int, int]:
    match = re.search(r"chunk-(\d+)[/\\]file-(\d+)\.parquet$", str(path))
    if not match:
        raise ValueError(f"data shard path must end in chunk-NNN/file-NNN.parquet: {path}")
    return int(match.group(1)), int(match.group(2))


def catalog_episodes(
    data_shard_paths: Iterable[str | Path],
    episode_metadata_paths: Iterable[str | Path],
    tasks_path: str | Path,
) -> dict[int, TaskEpisodes]:
    """Map task indices to complete episodes found in the supplied data shards.

    Episode metadata defines each episode's row interval. The function checks the
    frame and global row indices, episode length, and one task index per episode.
    """
    data_paths = tuple(Path(path) for path in data_shard_paths)
    metadata_paths = tuple(Path(path) for path in episode_metadata_paths)
    if not data_paths or not metadata_paths:
        raise ValueError("data and episode metadata shard paths are required")
    task_names: dict[int, str] = {}
    for row in _read_rows(Path(tasks_path)):
        task_index = int(row["task_index"])
        name_value = next(
            (row[key] for key in ("task", "task_name", "__index_level_0__") if key in row),
            None,
        )
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError(f"task {task_index} has no task name")
        if task_index in task_names:
            raise ValueError(f"duplicate task_index {task_index}")
        task_names[task_index] = name_value

    episodes: dict[int, dict[str, int]] = {}
    for metadata_path in metadata_paths:
        for row in _read_rows(metadata_path):
            episode_id = int(row["episode_index"])
            if episode_id in episodes:
                raise ValueError(f"duplicate episode metadata for episode {episode_id}")
            start, stop, length = (
                int(row["dataset_from_index"]),
                int(row["dataset_to_index"]),
                int(row["length"]),
            )
            if start < 0 or stop <= start or stop - start != length:
                raise ValueError(f"invalid episode boundary for episode {episode_id}")
            episodes[episode_id] = {
                "start": start,
                "stop": stop,
                "length": length,
                "chunk": int(row["data/chunk_index"]),
                "file": int(row["data/file_index"]),
            }

    rows_by_episode: dict[int, list[dict[str, object]]] = {}
    for data_path in data_paths:
        shard_chunk, shard_file = _shard_key(data_path)
        expected = {
            episode_id
            for episode_id, boundary in episodes.items()
            if (boundary["chunk"], boundary["file"]) == (shard_chunk, shard_file)
        }
        rows = _read_rows(data_path)
        actual = {int(row["episode_index"]) for row in rows}
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            if missing:
                raise ValueError(f"incomplete episode data in {data_path}: {missing}")
            raise ValueError(f"data shard has episodes absent from its metadata: {unknown}")
        for row in rows:
            episode_id = int(row["episode_index"])
            rows_by_episode.setdefault(episode_id, []).append(row)

    task_episodes: dict[int, list[int]] = {}
    for episode_id, rows in rows_by_episode.items():
        boundary = episodes[episode_id]
        if len(rows) != boundary["length"]:
            raise ValueError(f"incomplete episode {episode_id}: row count does not match metadata")
        rows.sort(key=lambda row: int(row["frame_index"]))
        frame_indices = [int(row["frame_index"]) for row in rows]
        if frame_indices != list(range(boundary["length"])):
            raise ValueError(f"invalid frame_index sequence for episode {episode_id}")
        global_indices = [int(row["index"]) for row in rows]
        if global_indices != list(range(boundary["start"], boundary["stop"])):
            raise ValueError(f"invalid global index boundary for episode {episode_id}")
        task_indices = {int(row["task_index"]) for row in rows}
        if len(task_indices) != 1:
            raise ValueError(f"episode {episode_id} has multiple task_index values")
        task_index = task_indices.pop()
        if task_index not in task_names:
            raise ValueError(f"episode {episode_id} uses unknown task_index {task_index}")
        task_episodes.setdefault(task_index, []).append(episode_id)

    return {
        task_index: TaskEpisodes(task_index, task_names[task_index], tuple(sorted(episode_ids)))
        for task_index, episode_ids in sorted(task_episodes.items())
    }
