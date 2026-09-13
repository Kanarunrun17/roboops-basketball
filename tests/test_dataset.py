from pathlib import Path

import numpy as np

from roboops.data import DatasetBundle, action_chunks, load_npz_dataset
from validate_dataset import validate

DATASET_PATH = Path("data/processed/pusht_keypoints.npz")


def load_workshop_dataset():
    assert DATASET_PATH.exists(), "Run `python src/download_dataset.py` before pytest"
    return load_npz_dataset(DATASET_PATH)


def test_dataset_can_be_loaded_and_has_required_features() -> None:
    dataset = load_workshop_dataset()
    assert dataset.frame_count > 0
    assert dataset.agent_state.shape[1] == 2
    assert dataset.environment_state.shape[1] == 16
    assert dataset.action.shape[1] == 2


def test_dataset_has_no_nan_or_inf() -> None:
    dataset = load_workshop_dataset()
    assert np.isfinite(dataset.state).all()
    assert np.isfinite(dataset.action).all()


def test_action_chunks_do_not_cross_episode_boundaries() -> None:
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
    bundle = DatasetBundle(
        agent_state=np.zeros((6, 2), dtype=np.float32),
        environment_state=np.zeros((6, 16), dtype=np.float32),
        action=actions,
        episode_index=np.asarray([0, 0, 0, 1, 1, 1]),
        frame_index=np.asarray([0, 1, 2, 0, 1, 2]),
    )
    chunks, padding = action_chunks(bundle, np.arange(6), chunk_size=4)
    np.testing.assert_array_equal(chunks[1, :2], actions[1:3])
    assert padding[1].tolist() == [False, False, True, True]


def test_validator_rejects_wrong_shapes() -> None:
    values = {
        "observation.state": np.zeros((5, 9)),
        "observation.environment_state": np.zeros((5, 1)),
        "action": np.zeros((5, 3)),
        "episode_index": np.zeros(5, dtype=int),
    }
    checks = validate(values)
    shape_check = next(check for check in checks if check.label == "state / action shapes match schema")
    assert not shape_check.passed
