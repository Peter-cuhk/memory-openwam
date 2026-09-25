"""MemMimic (Gated Memory Policy) ``push_cube`` reader with episode memory.

Source: the GMP zarr export (``episode_data.zarr`` with ``episode_<i>``
groups). Per step it stores the third-person RGB frame, the TCP pose as
``xyz + quaternion(wxyz)`` and the gripper width, for both the observed robot
state and the commanded action.

Representation: 10-D ``[xyz(3), rot6d(6), gripper_width(1)]`` for action and
state, mapped onto the OpenWAM-α slots ``0:10`` (left arm: xyz 0:3, rot6d 3:9,
gripper 9) through ``unify_action_map``.

Windows follow the α temporal contract (``num_frames=33`` source steps,
``video_stride=4`` -> 9 video frames, 32 actions, proprio at the window
start). With memory enabled a window may only start on a latent boundary
(``offset = 16 c`` at ``video_stride=4``): the current frame ``s_{4c}`` is then
both OpenWAM's clean conditioning frame and the last frame of history chunk
``c``, which is exactly the streaming write cadence at deployment
(one gist every 4 sampled frames = 16 control steps).

Each sample additionally returns:
  memory_video          list[PIL]  the episode's stride-sampled frames s_0..s_{4c}
                                   (or ``memory_latents`` from the latent cache)
  memory_times          list[int]  absolute latent times of the memory slots
                                   kept (anchor 0 first; truncated to
                                   ``memory_max_latents``)
  memory_context_index  int        c
  memory_actions        (n_slots, 16, 10) normalized commands executed during each memory slot
                                   (only with ``memory_action_history``; anchor slot = zeros)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.utils.eef import build_action_mask_2d, build_proprio_mask_2d, quat_wxyz_to_rot6d
from openwam.dataloader.utils.normalization import apply_normalization, pin_rot6d_identity
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec

logger = logging.getLogger(__name__)

ACTION_DIM = 10
STATE_DIM = 10
ROT6D_DIMS = tuple(range(3, 9))
REPRESENTATION = "memmimic_pose10"
# action_repr="delta": xyz as per-step increments (speed), rot6d / gripper absolute.
DELTA_REPRESENTATION = "memmimic_delta10"
DELTA_STATS_FILENAME = "openwam_normalization_stats_delta.npy"
DEFAULT_PROMPT = "push the blue cube so that it stops inside the target box"
STATS_FILENAME = "openwam_normalization_stats.npy"
VAE_TEMPORAL_FACTOR = 4
# Dims whose training range is (numerically) constant are widened to a unit
# range so min-max normalization maps them to ~0 instead of dividing by zero.
CONSTANT_RANGE_EPS = 1e-6

_EPISODE_RE = re.compile(r"^episode_(\d+)$")


def pose10_from_xyz_wxyz(xyz_wxyz: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """``(T, 7) xyz+wxyz`` and ``(T, 1)`` gripper -> ``(T, 10)`` xyz+rot6d+gripper."""
    xyz_wxyz = np.asarray(xyz_wxyz, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(xyz_wxyz.shape[0], 1)
    rot6d = quat_wxyz_to_rot6d(xyz_wxyz[:, 3:7])
    return np.concatenate([xyz_wxyz[:, :3], rot6d, gripper], axis=-1).astype(np.float32)


def pose10_to_delta(actions: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
    """``(T, 10)`` absolute commands -> xyz as per-step increments, rot6d / gripper unchanged.

    The first increment is relative to ``ref_state`` (the observed pose at the window start,
    i.e. the deploy-time proprio), the others to the previous command."""
    actions = np.asarray(actions, dtype=np.float32)
    out = actions.copy()
    prev = np.concatenate([np.asarray(ref_state, dtype=np.float32)[None, :3], actions[:-1, :3]], axis=0)
    out[:, :3] = actions[:, :3] - prev
    return out


def delta_to_pose10(deltas: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pose10_to_delta` (deploy: integrate predicted increments from the proprio)."""
    deltas = np.asarray(deltas, dtype=np.float32)
    out = deltas.copy()
    out[:, :3] = np.asarray(ref_state, dtype=np.float32)[:3] + np.cumsum(deltas[:, :3], axis=0)
    return out


def memory_action_history(actions: np.ndarray, times: Sequence[int], steps_per_latent: int) -> np.ndarray:
    """Per-slot action history for the memory: ``(len(times), steps_per_latent, D)``.

    ``actions`` are the episode's normalized actions from step 0 on (training: the
    recorded ``action0`` commands; deploy: the executed chunks the server sent,
    concatenated). Slot ``t >= 1`` gets the commands executed while latent ``t`` was
    being observed, i.e. steps ``[steps_per_latent*(t-1), steps_per_latent*t)``; the
    anchor slot (``t = 0``) has no past actions and gets zeros.
    """
    actions = np.asarray(actions, dtype=np.float32)
    out = np.zeros((len(times), int(steps_per_latent), actions.shape[-1]), dtype=np.float32)
    for i, t in enumerate(times):
        t = int(t)
        if t == 0:
            continue
        lo, hi = steps_per_latent * (t - 1), steps_per_latent * t
        if hi > actions.shape[0]:
            raise ValueError(f"memory slot {t} needs actions up to step {hi}, only {actions.shape[0]} available.")
        out[i] = actions[lo:hi]
    return out


def compute_pose10_stats(arrays: Sequence[np.ndarray]) -> dict:
    """min/max/mean/std over ``(T_i, 10)`` arrays; rot6d pinned to identity;
    constant dims widened (see ``CONSTANT_RANGE_EPS``)."""
    stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
    mn = stacked.min(axis=0)
    mx = stacked.max(axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    q01 = np.quantile(stacked, 0.01, axis=0)
    q99 = np.quantile(stacked, 0.99, axis=0)
    constant = (mx - mn) < CONSTANT_RANGE_EPS
    for d in np.nonzero(constant)[0]:
        if d in ROT6D_DIMS:
            continue
        center = float(mean[d])
        mn[d], mx[d], q01[d], q99[d] = center - 0.5, center + 0.5, center - 0.5, center + 0.5
        std[d] = 1.0
    stats = {
        "min": mn.astype(np.float32),
        "max": mx.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": np.maximum(std, 1e-6).astype(np.float32),
        "q01": q01.astype(np.float32),
        "q99": q99.astype(np.float32),
    }
    pin_rot6d_identity(stats, ROT6D_DIMS)
    stats["constant_dims"] = [int(d) for d in np.nonzero(constant)[0] if d not in ROT6D_DIMS]
    return stats


class MemMimicDataset(BaseDataset):
    DATASET_NAME = "MemMimic"
    ACTION_DIM = ACTION_DIM
    STATE_DIM = STATE_DIM

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "num_frames",
        "video_stride",
        "window_stride",
        "height",
        "width",
        "normalize_mode",
        "normalization_stats_path",
        "unify_action",
        "unify_action_map",
        "unify_state_map",
        "prompt",
        "memory_enabled",
        "memory_max_latents",
        "memory_latent_cache_dir",
        "memory_action_history",
        "critical_step_weight",
        "action_repr",
        "traj_num",
        "val_episodes",
        "min_window_offset",
        "image_key",
        "zarr_name",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 16,
        height: int = 256,
        width: int = 320,
        normalize_mode: Optional[str] = "min-max",
        normalization_stats_path: Optional[str] = None,
        unify_action: bool = True,
        unify_action_map: Any = ("0-9",),
        unify_state_map: Any = None,
        prompt: str = DEFAULT_PROMPT,
        memory_enabled: bool = True,
        memory_max_latents: int = 40,
        memory_latent_cache_dir: Optional[str] = None,
        memory_action_history: bool = False,
        critical_step_weight: float = 1.0,
        action_repr: str = "absolute",
        traj_num: Optional[int] = None,
        val_episodes: int = 0,
        min_window_offset: int = 0,
        image_key: str = "third_person_camera",
        zarr_name: str = "episode_data.zarr",
    ):
        super().__init__()
        self._dataset_dir = Path(dataset_dir)
        self._zarr_path = self._dataset_dir if self._dataset_dir.suffix == ".zarr" else self._dataset_dir / zarr_name
        if not self._zarr_path.exists():
            raise FileNotFoundError(f"MemMimic zarr not found: {self._zarr_path}")
        self._split = str(split)
        self._num_frames = int(num_frames)
        self._video_stride = max(1, int(video_stride))
        self._window_stride = max(1, int(window_stride))
        self._height, self._width = int(height), int(width)
        self._normalize_mode = None if normalize_mode in (None, "none", "null") else str(normalize_mode)
        self._prompt = str(prompt)
        self._image_key = str(image_key)
        self._memory_enabled = bool(memory_enabled)
        self._memory_max_latents = int(memory_max_latents)
        self._latent_stride = self._video_stride * VAE_TEMPORAL_FACTOR  # source steps per latent
        self._cache_dir = Path(memory_latent_cache_dir) if memory_latent_cache_dir else None
        self._memory_action_history = bool(memory_action_history)
        # Loss weight of the "critical" steps (GMP action0_is_critical = the push acceleration phase,
        # where the push speed -- the task's only control knob -- is decided). 1.0 = uniform.
        self._critical_step_weight = float(critical_step_weight)
        if action_repr not in ("absolute", "delta"):
            raise ValueError(f"action_repr must be 'absolute' or 'delta', got {action_repr!r}")
        self._action_repr = action_repr
        self._representation = DELTA_REPRESENTATION if action_repr == "delta" else REPRESENTATION
        if self._memory_enabled and self._window_stride % self._latent_stride != 0:
            raise ValueError(
                f"memory_enabled requires window_stride to be a multiple of video_stride*{VAE_TEMPORAL_FACTOR}="
                f"{self._latent_stride} (window starts must sit on latent boundaries), got {self._window_stride}."
            )
        if self._memory_max_latents <= 0:
            raise ValueError("memory_max_latents must be positive.")

        self._unify = bool(unify_action)
        if self._unify:
            if unify_action_map is None:
                raise ValueError("unify_action=true requires unify_action_map (e.g. ['0-9']).")
            self._action_dst = parse_unify_spec(unify_action_map, UNIFY_DIM)
            self._state_dst = parse_unify_spec(unify_state_map or unify_action_map, UNIFY_DIM)
            if len(self._action_dst) != ACTION_DIM or len(self._state_dst) != STATE_DIM:
                raise ValueError("MemMimic unify maps must cover exactly 10 source dims.")
            self._out_action_dim = UNIFY_DIM
            self._out_state_dim = UNIFY_DIM
        else:
            self._action_dst = self._state_dst = None
            self._out_action_dim = ACTION_DIM
            self._out_state_dim = STATE_DIM

        self._root_cache: Dict[int, Any] = {}
        root = self._root()
        episodes = []
        for name in root.group_keys():
            m = _EPISODE_RE.match(name)
            if m:
                episodes.append(int(m.group(1)))
        episodes.sort()
        if not episodes:
            raise ValueError(f"No episode_<i> groups in {self._zarr_path}")
        val_n = int(val_episodes)
        train_eps = episodes[: len(episodes) - val_n] if val_n > 0 else episodes
        val_eps = episodes[len(episodes) - val_n :] if val_n > 0 else []
        if traj_num is not None:
            train_eps = train_eps[: int(traj_num)]
        self._train_episodes = train_eps
        self._episodes = train_eps if self._split == "train" else val_eps
        self._lengths = {e: int(root[f"episode_{e}"]["action0_tcp_xyz_wxyz"].shape[0]) for e in self._episodes}

        default_stats = DELTA_STATS_FILENAME if action_repr == "delta" else STATS_FILENAME
        stats_path = Path(normalization_stats_path) if normalization_stats_path else self._dataset_dir / default_stats
        self.normalization_stats_path = str(stats_path)
        self._action_stats, self._state_stats = self._load_or_build_stats(stats_path, train_eps)
        # Memory action history is always absolute commands: normalize with an absolute-pose block
        # (the deploy server mirrors this choice, see robotmq_policy_server.py).
        self._memory_action_stats = self._state_stats if action_repr == "delta" else self._action_stats

        # Window index: (episode, offset) pairs.
        self._index: List[Tuple[int, int]] = []
        for e in self._episodes:
            T = self._lengths[e]
            last = T - self._num_frames
            if last < 0:
                continue
            # min_window_offset > 0 keeps only late windows (longest memory): used to probe peak GPU memory.
            for offset in range(0, last + 1, self._window_stride):
                if offset >= int(min_window_offset):
                    self._index.append((e, offset))
        logger.info(
            "MemMimic(%s): %d episodes, %d windows, window_stride=%d, memory=%s (max_latents=%d), cache=%s",
            self._split,
            len(self._episodes),
            len(self._index),
            self._window_stride,
            self._memory_enabled,
            self._memory_max_latents,
            self._cache_dir,
        )

    # ------------------------------------------------------------------ io
    def _root(self):
        import zarr

        pid = os.getpid()
        root = self._root_cache.get(pid)
        if root is None:
            self._root_cache.clear()
            root = zarr.open(str(self._zarr_path), mode="r")
            self._root_cache[pid] = root
        return root

    def _episode(self, e: int):
        return self._root()[f"episode_{e}"]

    def _load_or_build_stats(self, path: Path, train_eps: Sequence[int]) -> Tuple[Optional[dict], Optional[dict]]:
        if self._normalize_mode is None:
            return None, None
        if not path.is_file():
            logger.info("MemMimic: building normalization stats over %d train episodes -> %s", len(train_eps), path)
            root = self._root()
            actions, states = [], []
            for e in train_eps:
                g = root[f"episode_{e}"]
                a = pose10_from_xyz_wxyz(g["action0_tcp_xyz_wxyz"][:], g["action0_gripper_width"][:])
                st = pose10_from_xyz_wxyz(g["robot0_tcp_xyz_wxyz"][:], g["robot0_gripper_width"][:])
                if self._action_repr == "delta":
                    # Both increment kinds a window can start with: command-to-command and command-to-state.
                    d = pose10_to_delta(a, st[0])
                    d0 = a.copy()
                    d0[:, :3] = a[:, :3] - st[: len(a), :3]
                    actions.extend([d, d0])
                else:
                    actions.append(a)
                states.append(st)
            payload = self._with_deploy_keys(
                {
                    "representation": self._representation,
                    "action": compute_pose10_stats(actions),
                    "state": compute_pose10_stats(states),
                    "rot6d_dims": list(ROT6D_DIMS),
                    "train_episodes": [int(e) for e in train_eps],
                }
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            with open(tmp, "wb") as f:  # np.save would append ".npy" to a bare path
                np.save(f, payload, allow_pickle=True)
            os.replace(tmp, path)
        raw = np.load(path, allow_pickle=True).item()
        if raw.get("representation") != self._representation:
            raise ValueError(f"{path}: incompatible representation {raw.get('representation')!r}")
        if self._representation not in raw or f"{self._representation}_state" not in raw:
            # Older files predate the deploy schema: upgrade in place (additive).
            raw = self._with_deploy_keys(raw)
            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            with open(tmp, "wb") as f:
                np.save(f, raw, allow_pickle=True)
            os.replace(tmp, path)
        return raw["action"], raw["state"]

    def _with_deploy_keys(self, payload: dict) -> dict:
        """Add the ``<action_mode>`` / ``<action_mode>_state`` blocks the deploy loader reads.

        ``openwam.deploy.model_loader`` selects ``stats[dataloader.action_mode]`` for
        action unnormalization and ``stats[f"{action_mode}_state"]`` for proprio
        normalization (``load_mode_stats``); ``configs/dataloader/memmimic.yaml`` sets
        ``action_mode: memmimic_pose10`` so both directions use the reader's stats.
        """
        out = dict(payload)
        rep = getattr(self, "_representation", REPRESENTATION)
        out[rep] = payload["action"]
        out[f"{rep}_state"] = payload["state"]
        return out

    # --------------------------------------------------------------- helpers
    def _frame(self, g, idx: int) -> Image.Image:
        arr = np.asarray(g[self._image_key][idx])
        img = Image.fromarray(arr)
        if img.size != (self._width, self._height):
            img = img.resize((self._width, self._height), Image.BILINEAR)
        return img

    def _memory_times(self, c: int) -> List[int]:
        if c + 1 <= self._memory_max_latents:
            return list(range(c + 1))
        return [0] + list(range(c - self._memory_max_latents + 2, c + 1))

    def _memory_payload(self, e: int, g, c: int) -> dict:
        times = self._memory_times(c)
        out = {"memory_times": times, "memory_context_index": int(c)}
        if self._memory_action_history:
            n = self._latent_stride * c
            past = pose10_from_xyz_wxyz(g["action0_tcp_xyz_wxyz"][:n], g["action0_gripper_width"][:n]) if n else np.zeros((0, ACTION_DIM), np.float32)
            past = apply_normalization(past, self._memory_action_stats, self._normalize_mode) if n else past
            out["memory_actions"] = torch.from_numpy(memory_action_history(past, times, self._latent_stride))
        if self._cache_dir is not None:
            cache_file = self._cache_dir / f"episode_{e}.pt"
            if cache_file.is_file():
                lat = torch.load(cache_file, map_location="cpu", weights_only=True)
                if lat.ndim != 4 or lat.shape[1] <= c:
                    raise ValueError(f"{cache_file}: expected (z, T_lat>{c}, h, w), got {tuple(lat.shape)}")
                out["memory_latents"] = lat[:, : c + 1].clone()
                return out
        # Contiguous stride-sampled history s_0 .. s_{4c}; the architecture
        # VAE-encodes it causally and keeps the slots in ``memory_times``.
        n_frames = c * VAE_TEMPORAL_FACTOR + 1
        out["memory_video"] = [self._frame(g, j * self._video_stride) for j in range(n_frames)]
        return out

    # ---------------------------------------------------------------- dataset
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        e, offset = self._index[idx]
        g = self._episode(e)
        frame_ids = [offset + k for k in range(0, self._num_frames, self._video_stride)]
        video = [self._frame(g, i) for i in frame_ids]

        T_action = self._num_frames - 1
        action = pose10_from_xyz_wxyz(
            g["action0_tcp_xyz_wxyz"][offset : offset + T_action],
            g["action0_gripper_width"][offset : offset + T_action],
        )
        proprio = pose10_from_xyz_wxyz(
            g["robot0_tcp_xyz_wxyz"][offset : offset + 1],
            g["robot0_gripper_width"][offset : offset + 1],
        )
        if self._action_repr == "delta":
            action = pose10_to_delta(action, proprio[0])
        action = apply_normalization(action, self._action_stats, self._normalize_mode)
        proprio = apply_normalization(proprio, self._state_stats, self._normalize_mode)
        if self._unify:
            action, action_dim_mask = map_to_unify(action, self._action_dst, UNIFY_DIM)
            proprio, state_dim_mask = map_to_unify(proprio, self._state_dst, UNIFY_DIM)
        else:
            action_dim_mask = state_dim_mask = None
        action_mask = build_action_mask_2d(T_action, self._out_action_dim, T_action, dim_mask=action_dim_mask)
        proprio_mask = build_proprio_mask_2d(self._out_state_dim, enabled=True, dim_mask=state_dim_mask)

        sample = {
            "video": video,
            "prompt": self._prompt,
            "action": torch.from_numpy(np.ascontiguousarray(action, dtype=np.float32)),
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(np.ascontiguousarray(proprio, dtype=np.float32)),
            "proprio_mask": torch.from_numpy(proprio_mask),
            "video_mask": torch.ones(len(video), dtype=torch.bool),
            "episode_index": int(e),
            "offset": int(offset),
        }
        if self._critical_step_weight != 1.0:
            crit = np.asarray(g["action0_is_critical"][offset : offset + T_action], dtype=np.float32).reshape(-1)
            sample["action_step_weight"] = torch.from_numpy(1.0 + (self._critical_step_weight - 1.0) * crit)
        if self._memory_enabled:
            c = offset // self._latent_stride
            sample.update(self._memory_payload(e, g, c))
        return sample

    # ------------------------------------------------------------ metadata
    @property
    def action_dim(self) -> int:
        return self._out_action_dim

    @property
    def normalization_stats(self) -> Optional[dict]:
        return self._action_stats

    @property
    def episodes(self) -> List[int]:
        return list(self._episodes)

    def episode_length(self, e: int) -> int:
        return self._lengths[e]

    def episode_history_frames(self, e: int) -> List[Image.Image]:
        """All stride-sampled frames of an episode (for the latent cache script)."""
        g = self._episode(e)
        T = self._lengths[e]
        n = (T - 1) // self._video_stride + 1
        return [self._frame(g, j * self._video_stride) for j in range(n)]

    @classmethod
    def from_config(cls, config, split: str = "train") -> "MemMimicDataset":
        from openwam.dataloader.utils import get_cfg

        dataset_dir = get_cfg(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError("memmimic: missing dataset_dir")
        kwargs: Dict[str, Any] = {"split": split}
        for key in cls.CONFIG_KEYS:
            v = get_cfg(config, key, None)
            if v is None and key != "normalize_mode":
                continue
            if key == "normalize_mode" and not hasattr(config, key) and (not hasattr(config, "get") or key not in config):
                continue
            kwargs[key] = v
        return cls(dataset_dir=str(dataset_dir), **kwargs)


__all__ = ["MemMimicDataset", "pose10_from_xyz_wxyz", "compute_pose10_stats", "ACTION_DIM", "STATE_DIM"]
