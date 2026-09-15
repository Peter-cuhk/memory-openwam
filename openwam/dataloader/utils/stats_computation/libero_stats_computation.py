"""Training-split statistics for LIBERO action7/state8.

Action and state are measured independently; rotation vectors are normalized
like other continuous dimensions, without changing their representation.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.libero import (
    ACTION_DIM,
    ACTION_STATS_KEY,
    GRIPPER_CONVENTION,
    NORMALIZATION_STATS_FILENAME,
    OUTPUT_REPRESENTATION,
    STATE_DIM,
    STATE_STATS_KEY,
)
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    compute_file_local_offsets,
    load_episodes_parquet,
    parse_info_json,
)

ACTION_COLUMN = "action"
STATE_COLUMN = "observation.state"

def _feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"statistics require a non-empty 2-D array, got {values.shape}")
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _load_columns(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    info = parse_info_json(dataset_dir)
    episodes = load_episodes_parquet(dataset_dir)
    # Compute offsets BEFORE filtering: validation episodes may share a shard.
    episodes["_file_offset"] = compute_file_local_offsets(episodes, "data/chunk_index", "data/file_index")
    episodes = apply_info_splits(episodes, "train", info.get("splits", {}), source_name="LIBERO stats")
    if episodes.empty:
        raise ValueError("LIBERO statistics require a non-empty train split")
    actions, states = [], []
    for (chunk, file), group in episodes.groupby(["data/chunk_index", "data/file_index"]):
        path = dataset_dir / info["data_path"].format(chunk_index=int(chunk), file_index=int(file))
        table = pq.read_table(path, columns=[ACTION_COLUMN, STATE_COLUMN])
        for offset, length in zip(group["_file_offset"], group["length"]):
            window = table.slice(int(offset), int(length))
            for column, dim, output in ((ACTION_COLUMN, ACTION_DIM, actions), (STATE_COLUMN, STATE_DIM, states)):
                values = np.asarray(window.column(column).to_pylist(), dtype=np.float32)
                if values.shape != (int(length), dim) or not np.isfinite(values).all():
                    raise ValueError(f"{path}:{column} requires {length} rows of {dim} finite values")
                output.append(values)
    return np.concatenate(actions, axis=0), np.concatenate(states, axis=0)


def compute_libero_stats(dataset_dir: str | Path) -> dict:
    """Scan training episodes only, using independent action and state statistics."""
    action_all, state_all = _load_columns(Path(dataset_dir))
    action_stats = _feature_stats(action_all)
    state_stats = _feature_stats(state_all)
    action_stats.update(
        {
            "num_timesteps": int(action_all.shape[0]),
            "pool": "action_only",
            "split": "train",
            "gripper_convention": GRIPPER_CONVENTION,
            "representation": OUTPUT_REPRESENTATION,
            "action_semantics": "native normalized LIBERO delta command",
        }
    )
    state_stats.update(
        {
            "num_timesteps": int(state_all.shape[0]),
            "pool": "state_only",
            "split": "train",
            "gripper_convention": "two_finger_joint_positions_meters",
            "representation": OUTPUT_REPRESENTATION,
            "state_semantics": "EEF xyz, axis-angle radians, two gripper joint positions",
        }
    )
    return {ACTION_STATS_KEY: action_stats, STATE_STATS_KEY: state_stats}


def build_and_save_libero_stats(dataset_dir: str | Path, output: str | Path | None = None) -> Path:
    """Compute and atomically write the stats payload; returns the output path."""
    dataset_dir = Path(dataset_dir)
    out_path = Path(output) if output else dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_libero_stats(dataset_dir)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_name, out_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="LIBERO LeRobot v3 dataset root")
    parser.add_argument(
        "--output",
        default=None,
        help=f".npy path (default <dataset_dir>/meta/{NORMALIZATION_STATS_FILENAME})",
    )
    args = parser.parse_args()
    out_path = build_and_save_libero_stats(args.dataset_dir, args.output)
    payload = np.load(out_path, allow_pickle=True).item()
    print(f"wrote {out_path}")
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        print(f"  {key}: num_timesteps={block['num_timesteps']}, mean[:3]={np.round(block['mean'][:3], 5).tolist()}")


if __name__ == "__main__":
    main()
