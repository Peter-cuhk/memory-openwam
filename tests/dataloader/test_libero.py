from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.libero import ACTION_MODE, ACTION_STATS_KEY, STATE_STATS_KEY, LiberoDataset
from openwam.dataloader.registry import DATASET_REGISTRY
from openwam.dataloader.utils.stats_computation.libero_stats_computation import compute_libero_stats

HEAD = "observation.images.image"
WRIST = "observation.images.wrist_image"


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    for camera in (HEAD, WRIST):
        directory = tmp_path / "videos" / camera / "chunk-000"
        directory.mkdir(parents=True)
        (directory / "file-000.mp4").touch()
    info = {
        "fps": 20,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            **{camera: {"dtype": "video", "shape": [512, 512, 3], "info": {"video.fps": 20}}
               for camera in (HEAD, WRIST)},
        },
        # Train is the SECOND episode, so offsets must be computed before filtering.
        "splits": {"train": "1:2", "val": "0:1"},
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    episodes = []
    for index in range(2):
        episodes.append({
            "episode_index": index, "length": 6, "dataset_from_index": index * 6,
            "data/chunk_index": 0, "data/file_index": 0,
            **{f"videos/{camera}/{key}": value for camera in (HEAD, WRIST)
               for key, value in (("chunk_index", 0), ("file_index", 0), ("from_timestamp", index * 0.3))},
        })
    pd.DataFrame(episodes).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick the cup"], name="task")).to_parquet(
        tmp_path / "meta/tasks.parquet"
    )
    action = np.arange(42, dtype=np.float32).reshape(6, 7) / 100
    action[:, 6] = [0, 1, 0, 1, 0, 1]
    state = np.arange(48, dtype=np.float32).reshape(6, 8) / 10
    state[:, 3] += 3.5  # Do not wrap rotation coordinates or pin them to identity.
    pd.DataFrame({
        "action": list(np.concatenate((action + 100, action))),
        "observation.state": list(np.concatenate((state + 100, state))),
        "task_index": np.zeros(12, dtype=np.int64),
    }).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    decoded = []

    def decode(path, frame_indices, height, width):
        decoded.append((str(path), list(frame_indices)))
        color = (255, 0, 0) if HEAD in str(path) else (0, 255, 0)
        return [Image.new("RGB", (width, height), color) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    return tmp_path, action, state, decoded


def reader(path, **kwargs):
    return LiberoDataset(
        str(path), num_frames=5, video_stride=1, height=384, width=320,
        multiview=True, **kwargs,
    )


def test_raw_reader_alignment_masks_and_both_cameras(bucket):
    path, action, state, decoded = bucket
    dataset = reader(path, normalize_mode=None, target_camera=HEAD)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["action"], action[:4])
    np.testing.assert_array_equal(sample["proprio"], state[:1])
    assert sample["action_mask"].shape == (4, 7)
    assert sample["proprio_mask"].shape == (1, 8)
    assert sample["action_mask"].all() and sample["proprio_mask"].all()
    assert sample["prompt"] == "pick the cup"
    assert len(sample["video"]) == 5
    assert sample["video"][0].size == (320, 384)
    assert sample["video"][0].getpixel((20, 300)) == (0, 255, 0)
    assert sample["video"][0].getpixel((200, 300)) == (0, 0, 0)
    assert decoded[0][1] == list(range(6, 11))
    terminal = dataset[len(dataset) - 1]
    np.testing.assert_array_equal(terminal["action"][0], action[-1])
    assert terminal["action_mask"][0].all()
    assert not terminal["action_mask"][1:].any()


@pytest.mark.parametrize("mode", ["min-max", "z-score", "quantile"])
def test_separate_training_stats_and_deployment_roundtrip(bucket, mode):
    from openwam.deploy.model_loader import _build_normalizer

    path, action, state, _ = bucket
    stats = compute_libero_stats(path)
    np.testing.assert_allclose(stats[ACTION_STATS_KEY]["max"], action.max(0))
    np.testing.assert_allclose(stats[STATE_STATS_KEY]["max"], state.max(0))
    assert stats[ACTION_STATS_KEY]["num_timesteps"] == 6
    stats_path = path / "normalization_stats.npy"
    np.save(stats_path, stats)
    dataset = reader(path, normalize_mode=mode, normalization_stats_path=str(stats_path))
    sample = dataset[1]  # interior values avoid quantile clipping on roundtrip
    cfg = OmegaConf.create({"dataloader": {
        "normalize_mode": mode, "action_mode": ACTION_MODE, "unify_action": False,
    }})
    normalizer = _build_normalizer(cfg, str(path))
    np.testing.assert_allclose(normalizer.normalize(state[1:2]), sample["proprio"], atol=1e-6)
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), action[1:5], atol=1e-6)
    if mode == "min-max":
        assert sample["action"][0, 6] == 1
        assert sample["action"][1, 6] == -1
        np.testing.assert_allclose(sample["proprio"], np.full((1, 8), -0.6), atol=1e-6)


def test_rejects_wrong_schema(bucket):
    path, *_ = bucket
    info_path = path / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["shape"] = [10]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="shape \\[8\\]"):
        reader(path, normalize_mode=None)
    with pytest.raises(ValueError, match="requires explicit unify_action_map"):
        reader(path, unify_action=True)


def test_libero_config_and_registry():
    cfg = OmegaConf.load("configs/dataloader/libero.yaml")
    assert cfg.action_mode == ACTION_MODE
    assert cfg.unify_action is False
    assert cfg.video_stride == 4
    assert DATASET_REGISTRY["libero"] is LiberoDataset


@pytest.mark.parametrize("mode", [None, "min-max"])
def test_unify_independent_action_state_maps_and_deployment(bucket, mode):
    from openwam.dataloader.utils.unify_action import UNIFY_DIM
    from openwam.deploy.model_loader import _build_normalizer

    path, action, state, _ = bucket
    stats_path = path / "normalization_stats.npy"
    np.save(stats_path, compute_libero_stats(path))
    options = {"normalize_mode": mode, "normalization_stats_path": str(stats_path)}
    # Different, non-contiguous destinations expose accidental map sharing.
    action_slots = [0, 1, 2, 4, 5, 6, 9]
    state_slots = [20, 21, 22, 24, 25, 26, 28, 29]
    mapping = {
        "unify_action": True,
        "unify_action_map": action_slots,
        "unify_state_map": state_slots,
    }
    raw = reader(path, **options)[1]
    dataset = reader(path, **options, **mapping)
    sample = dataset[1]
    assert dataset.action_dim == UNIFY_DIM
    assert sample["action"].shape == (4, UNIFY_DIM)
    assert sample["proprio"].shape == (1, UNIFY_DIM)
    np.testing.assert_array_equal(sample["action"][:, action_slots], raw["action"])
    np.testing.assert_array_equal(sample["proprio"][:, state_slots], raw["proprio"])
    action_mask = np.zeros((4, UNIFY_DIM), dtype=bool)
    action_mask[:, action_slots] = True
    state_mask = np.zeros((1, UNIFY_DIM), dtype=bool)
    state_mask[:, state_slots] = True
    np.testing.assert_array_equal(sample["action_mask"], action_mask)
    np.testing.assert_array_equal(sample["proprio_mask"], state_mask)
    assert not sample["action"][~action_mask].any()
    assert not sample["proprio"][~state_mask].any()
    empty, mask = dataset._finalize_proprio(None)
    assert empty.shape == mask.shape == (1, UNIFY_DIM)
    assert not empty.any() and not mask.any()
    terminal = dataset[len(dataset) - 1]
    np.testing.assert_array_equal(terminal["action_mask"][0], action_mask[0])
    assert not terminal["action_mask"][1:].any()

    cfg = OmegaConf.create({"dataloader": {
        "normalize_mode": mode, "action_mode": ACTION_MODE, **mapping,
    }})
    normalizer = _build_normalizer(cfg, str(path))
    np.testing.assert_allclose(normalizer.normalize(state[1:2]), sample["proprio"], atol=1e-6)
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), action[1:5], atol=1e-6)


def test_unify_validates_action_and_state_width_independently(bucket):
    path, *_ = bucket
    with pytest.raises(ValueError, match="8-D state"):
        reader(path, unify_action=True, unify_action_map=["0-6"], unify_state_map=["0-6"])
    with pytest.raises(ValueError, match="7-D action"):
        reader(path, unify_action=True, unify_action_map=["0-7"], unify_state_map=["0-7"])
