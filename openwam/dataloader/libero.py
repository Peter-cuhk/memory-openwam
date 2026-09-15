"""LeRobot v3 LIBERO: raw 7-D actions and 8-D achieved states.

Rotation coordinates remain axis-angle. Optional unify maps scatter normalized
action and state into independently selected slots.
Action and state use separate, per-dimension training statistics.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import build_proprio_mask_2d
from openwam.dataloader.utils.normalization import (
    apply_normalization,
    materialize_eef_stats,
)

logger = logging.getLogger(__name__)

ACTION_MODE = "eef"
ACTION_STATS_KEY = ACTION_MODE
STATE_STATS_KEY = f"{ACTION_MODE}_state"
OUTPUT_REPRESENTATION = "libero_action7_state8"
ACTION_DIM = 7
STATE_DIM = 8
GRIPPER_CONVENTION = "zero_closed_one_open"
NORMALIZATION_STATS_FILENAME = "libero_normalization_stats.npy"


class LiberoDataset(LeRobotV3Reader):
    """Read action[t] alongside image[t] and state[t], without shifting rows."""

    DATASET_NAME = "LIBERO"
    ACTION_DIM = ACTION_DIM
    STATE_DIM = STATE_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEPLOY_ACTION_MODE = ACTION_MODE

    # Fixed camera layout (head, wrist, unused); None slots are rendered black
    # in multiview mode. Override via ``camera_layout``.
    DEFAULT_CAMERA_LAYOUT: ClassVar[Tuple[Optional[str], ...]] = (
        "observation.images.image",
        "observation.images.wrist_image",
        None,
    )
    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "normalization_stats_path",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = ACTION_MODE,
        normalization_stats_path: Optional[str] = None,
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        unify_state_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != ACTION_MODE:
            raise ValueError(f"LIBERO supports only action_mode={ACTION_MODE!r}, got {action_mode!r}.")
        if unify_action and (unify_action_map is None or unify_state_map is None):
            raise ValueError(
                "LIBERO unify_action=true requires explicit unify_action_map (7 source dims) "
                "and unify_state_map (8 source dims)."
            )

        self.action_mode = mode
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._state_normalization_stats: Optional[dict] = None
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=bool(unify_action),
            unify_action_map=unify_action_map,
            unify_state_map=unify_state_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        if not self._multiview and self._target_camera is not None:
            return self._target_camera, None, None
        layout = list(self._camera_layout_param or self.DEFAULT_CAMERA_LAYOUT)
        layout += [None] * (3 - len(layout))
        return tuple(str(cam) if cam else None for cam in layout[:3])

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"LIBERO {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        for column, dim in (("observation.state", STATE_DIM), ("action", ACTION_DIM)):
            shape = tuple(features.get(column, {}).get("shape", ()))
            if shape != (dim,):
                raise ValueError(f"LIBERO {column} feature must have shape [{dim}], got {shape}")
        for camera in self._resolve_cameras(info):
            if camera is None:
                continue
            feature = features.get(camera, {})
            if feature.get("dtype") != "video":
                raise ValueError(f"LIBERO camera {camera!r} is not a video feature")

    def _build_stats_rank0(self, path: Path) -> None:
        """Auto-build the pooled stats file: rank 0 scans, other ranks wait."""
        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        if dist_ready:
            rank = dist.get_rank()
        else:
            # torchrun sets RANK before init_process_group; honor it so
            # pre-init constructions still elect a single builder.
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            from openwam.dataloader.utils.stats_computation.libero_stats_computation import (
                build_and_save_libero_stats,
            )

            logger.info("No LIBERO stats at %s — scanning the dataset (rank 0; other ranks wait)", path)
            build_and_save_libero_stats(self._dataset_dir, output=path)
        else:
            deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
            poll_interval_s = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
            while not path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank 0 to build LIBERO stats: {path}")
                time.sleep(poll_interval_s)

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            self._state_normalization_stats = None
            return None
        if self._normalize_mode not in ("min-max", "z-score", "quantile"):
            raise ValueError(f"Unsupported LIBERO normalize_mode={self._normalize_mode!r}")
        stats_path = (
            Path(self._source_stats_path)
            if self._source_stats_path
            else self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
        )
        if not stats_path.is_file():
            self._build_stats_rank0(stats_path)

        raw = np.load(stats_path, allow_pickle=True).item()
        if not isinstance(raw, dict):
            raise ValueError(f"{stats_path} must contain a dictionary payload")
        if ACTION_STATS_KEY not in raw or STATE_STATS_KEY not in raw:
            raise KeyError(f"{stats_path} must contain {ACTION_STATS_KEY!r} and {STATE_STATS_KEY!r} blocks")
        action_raw = raw[ACTION_STATS_KEY]
        state_raw = raw[STATE_STATS_KEY]

        def materialize(block: dict, key: str, dim: int) -> dict:
            if block.get("representation") != OUTPUT_REPRESENTATION:
                raise ValueError(f"{stats_path}:{key} has incompatible representation; regenerate LIBERO stats")
            stats = materialize_eef_stats(
                block, self._normalize_mode, dim=dim, strict_minmax=False,
                source_hint=f"{stats_path}:{key}",
            )
            for name, values in stats.items():
                if values.shape != (dim,) or not np.isfinite(values).all():
                    raise ValueError(f"{stats_path}:{key}:{name} must contain {dim} finite values")
            return stats

        action_stats = materialize(action_raw, ACTION_STATS_KEY, ACTION_DIM)
        self._state_normalization_stats = materialize(state_raw, STATE_STATS_KEY, STATE_DIM)
        self.normalization_stats_path = str(stats_path)
        return action_stats

    @staticmethod
    def _read_column(win: pd.DataFrame, column: str, dim: int) -> np.ndarray:
        values = np.stack(win[column].values).astype(np.float32)
        if values.ndim != 2 or values.shape[1] != dim:
            raise ValueError(f"LIBERO {column} must be (T, {dim}), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"LIBERO {column} contains NaN or infinity")
        return values

    # These hook names are inherited from LeRobotV3Reader; outputs are 7-D / 8-D.
    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        return apply_normalization(
            self._read_column(win, "action", ACTION_DIM),
            self._normalization_stats, self._normalize_mode,
        )

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        return apply_normalization(
            self._read_column(win.iloc[:1], "observation.state", STATE_DIM),
            self._state_normalization_stats, self._normalize_mode,
        )

    def _finalize_proprio(self, proprio: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self._unify:
            return super()._finalize_proprio(proprio)
        # Raw state has 8 dimensions, independently of the 7-D action.
        mask = build_proprio_mask_2d(action_dim=STATE_DIM, enabled=proprio is not None)
        if proprio is None:
            proprio = np.zeros((1, STATE_DIM), dtype=np.float32)
        return np.asarray(proprio, dtype=np.float32), mask
