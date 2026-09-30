import hashlib
import json

from vla_lab.data.splits import split_episodes


def test_split_is_deterministic_and_keeps_episodes_disjoint():
    first = split_episodes(range(10))
    second = split_episodes(range(10))

    assert first == second
    assert len(first.train_ids) == 8
    assert len(first.validation_ids) == 2
    assert set(first.train_ids).isdisjoint(first.validation_ids)
    assert set(first.train_ids) | set(first.validation_ids) == set(range(10))


def test_split_keeps_at_least_one_validation_episode():
    split = split_episodes(["only"])

    assert split.train_ids == ()
    assert split.validation_ids == ("only",)


def test_manifest_hash_covers_split_membership():
    split = split_episodes(["a", "b", "c", "d"])
    payload = json.dumps(
        {"train_ids": list(split.train_ids), "validation_ids": list(split.validation_ids)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert split.sha256 == hashlib.sha256(payload).hexdigest()


def test_split_rejects_duplicate_episode_ids():
    try:
        split_episodes([1, 1])
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("duplicate episode IDs must fail")
