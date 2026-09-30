import json

import numpy as np

from vla_lab.data.stats import TrainingStats


def test_validation_only_values_do_not_change_training_statistics():
    train = {"a": np.array([[1.0, 2.0], [3.0, 4.0]])}
    first = TrainingStats.fit(train)
    changed_validation = np.array([[10000.0, -10000.0]])
    second = TrainingStats.fit(train, validation_episodes={"v": changed_validation})

    np.testing.assert_allclose(first.mean, second.mean)
    np.testing.assert_allclose(first.std, second.std)
    assert first.sha256 == second.sha256


def test_action_normalization_round_trips():
    stats = TrainingStats.fit({"episode": np.array([[0.0, 10.0], [2.0, 14.0]])})
    actions = np.array([[1.0, 12.0], [-1.0, 20.0]])

    np.testing.assert_allclose(stats.denormalize(stats.normalize(actions)), actions)


def test_stats_reload_from_checkpoint_payload():
    stats = TrainingStats.fit({"episode": np.array([[1.0, 2.0], [3.0, 6.0]])})
    payload = json.loads(json.dumps(stats.to_dict()))
    restored = TrainingStats.from_dict(payload)

    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.std, stats.std)
    assert restored.sha256 == stats.sha256
