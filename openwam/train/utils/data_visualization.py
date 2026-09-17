"""Interactive inspection of samples emitted by the training DataLoader."""

from __future__ import annotations

import json
import os
import queue
import random
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_VIS_KEY = "_openwam_vis_data"
_STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def vis_data_enabled() -> bool:
    """Return whether interactive data inspection is explicitly enabled."""
    return os.environ.get("VIS_DATA", "0").strip().lower() in {"1", "true", "yes", "on"}


class VisualizationDataset(torch.utils.data.Dataset):
    """Attach serializable source metadata without changing normal training samples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        leaf, leaf_index = _resolve_leaf_dataset(self.dataset, int(index))
        sample[_VIS_KEY] = _reader_metadata(leaf, leaf_index, sample)
        return sample


def _resolve_leaf_dataset(dataset, index: int):
    """Resolve aggregate/mixture indices to the reader that emitted the sample."""
    current = dataset
    local_index = int(index)
    while True:
        if hasattr(current, "_index_map") and hasattr(current, "_datasets"):
            dataset_index, child_index = current._index_map[local_index]
            current = current._datasets[int(dataset_index)]
            local_index = int(child_index)
            continue

        children = getattr(current, "_buckets", None)
        cumulative = getattr(current, "_cum_lens", None)
        if children is not None and cumulative is not None:
            child_index = int(np.searchsorted(cumulative, local_index, side="right") - 1)
            local_index -= int(cumulative[child_index])
            current = children[child_index]
            continue

        children = getattr(current, "_sub_datasets", None)
        cumulative = getattr(current, "_cumulative_lengths", None)
        if children is not None and cumulative is not None:
            child_index = int(np.searchsorted(np.asarray(cumulative), local_index, side="right"))
            previous = 0 if child_index == 0 else int(cumulative[child_index - 1])
            local_index -= previous
            current = children[child_index]
            continue

        children = getattr(current, "_datasets", None)
        cumulative = getattr(current, "_cum", None)
        if children is not None and cumulative is not None:
            child_index = int(np.searchsorted(cumulative, local_index, side="right") - 1)
            local_index -= int(cumulative[child_index])
            current = children[child_index]
            continue

        return current, local_index


def _first_attr(obj, names, default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _normalizer_details(reader, modality: str) -> dict[str, Any]:
    normalizer = getattr(reader, "_normalizer", None)
    mode = _first_attr(reader, ("_normalize_mode", "normalize_mode"))
    if mode is None and normalizer is not None:
        mode = getattr(normalizer, "mode", None)
    if hasattr(mode, "value"):
        mode = mode.value

    if modality == "action":
        stats = _first_attr(
            reader,
            ("_action_stats", "_action_norm_stats", "_normalization_stats", "_mode_stats"),
        )
        dst_index = _first_attr(reader, ("_action_dst_index", "_unify_dst_index"))
        if stats is None and normalizer is not None:
            stats = getattr(normalizer, "stats", None)
    else:
        stats = _first_attr(
            reader,
            ("_state_stats", "_state_normalization_stats", "_proprio_norm_stats"),
        )
        dst_index = _first_attr(reader, ("_state_dst_index", "_unify_state_dst_index"))
        if stats is None and normalizer is not None:
            stats = getattr(normalizer, "normalize_stats", None)
        if stats is None:
            stats = _first_attr(
                reader,
                ("_action_stats", "_action_norm_stats", "_normalization_stats", "_mode_stats"),
            )
        if dst_index is None:
            dst_index = _first_attr(reader, ("_action_dst_index", "_unify_dst_index"))

    clean_stats = {}
    if isinstance(stats, dict):
        for key in _STAT_KEYS:
            if key in stats:
                clean_stats[key] = np.asarray(stats[key], dtype=np.float32)

    return {
        "mode": None if mode is None else str(mode),
        "stats": clean_stats,
        "dst_index": None if dst_index is None else np.asarray(dst_index, dtype=np.int64),
    }


def _reader_metadata(reader, index: int, sample: dict) -> dict[str, Any]:
    fps = _first_attr(reader, ("_fps", "fps"))
    video_stride = _first_attr(reader, ("_video_stride", "video_stride"), 1)
    dataset_name = sample.get("_dataset_name") or getattr(reader, "DATASET_NAME", type(reader).__name__)
    dataset_id = _first_attr(reader, ("_dataset_id", "task_name"))
    return {
        "reader_class": type(reader).__name__,
        "dataset_name": str(dataset_name),
        "dataset_id": None if dataset_id is None else str(dataset_id),
        "reader_index": int(index),
        "fps": None if fps is None else float(fps),
        "video_stride": int(video_stride),
        "action": _normalizer_details(reader, "action"),
        "proprio": _normalizer_details(reader, "proprio"),
    }


def _as_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _broadcast_mask(mask, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    result = _as_numpy(mask).astype(bool, copy=False)
    if result.ndim == len(shape) - 1:
        result = np.expand_dims(result, axis=-1)
    if (
        result.ndim == len(shape)
        and result.shape[:-1] == shape[:-1]
        and result.shape[-1] < shape[-1]
    ):
        padding = [(0, 0)] * result.ndim
        padding[-1] = (0, shape[-1] - result.shape[-1])
        result = np.pad(result, padding, constant_values=False)
    try:
        return np.broadcast_to(result, shape)
    except ValueError as exc:
        raise ValueError(f"mask shape {result.shape} cannot broadcast to data shape {shape}") from exc


def _stats_width(stats: dict[str, np.ndarray]) -> int | None:
    for value in stats.values():
        array = np.asarray(value)
        if array.ndim == 1:
            return int(array.shape[0])
    return None


def _inverse_normalize(values: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    mode = details.get("mode")
    stats = details.get("stats") or {}
    if mode is None or mode.lower() in {"none", "null"} or not stats:
        return values.astype(np.float32, copy=True)

    mode = mode.lower().replace("_", "-")
    if mode in {"q99", "quantile"}:
        return ((values + 1.0) * 0.5 * (stats["q99"] - stats["q01"]) + stats["q01"]).astype(np.float32)
    if mode in {"min-max", "minmax"}:
        return ((values + 1.0) * 0.5 * (stats["max"] - stats["min"]) + stats["min"]).astype(np.float32)
    if mode in {"mean-std", "z-score", "zscore"}:
        return (values * stats["std"] + stats["mean"]).astype(np.float32)
    if mode == "scale":
        scale = np.maximum(np.abs(stats["min"]), np.abs(stats["max"]))
        return (values * scale).astype(np.float32)
    if mode == "binary":
        return values.astype(np.float32, copy=True)
    raise ValueError(f"unsupported normalization mode {details.get('mode')!r}")


def denormalize_with_layout(array, mask, details: dict[str, Any]) -> np.ndarray:
    """Invert normalization while preserving the model input layout."""
    normalized = _as_numpy(array).astype(np.float32, copy=False)
    if normalized.ndim == 1:
        normalized = normalized[None, :]
    valid = _broadcast_mask(mask, normalized.shape)
    valid_dims = np.any(valid, axis=tuple(range(normalized.ndim - 1)))
    output = np.full(normalized.shape, np.nan, dtype=np.float32)

    stats = details.get("stats") or {}
    width = _stats_width(stats)
    dst_index = details.get("dst_index")
    candidate = np.arange(normalized.shape[-1], dtype=np.int64)
    if dst_index is not None:
        candidate = np.asarray(dst_index, dtype=np.int64)
        candidate = candidate[(candidate >= 0) & (candidate < normalized.shape[-1])]

    if width is None:
        positions = np.flatnonzero(valid_dims)
        output[..., positions] = normalized[..., positions]
    elif width == len(candidate):
        positions = candidate
        output[..., positions] = _inverse_normalize(normalized[..., positions], details)
    else:
        valid_candidate = candidate[valid_dims[candidate]]
        if width != len(valid_candidate):
            all_valid = np.flatnonzero(valid_dims)
            if width != len(all_valid):
                raise ValueError(
                    f"normalization stats width {width} does not match layout width {normalized.shape[-1]} "
                    f"or valid dimensions {len(all_valid)}"
                )
            positions = all_valid
        else:
            positions = valid_candidate
        output[..., positions] = _inverse_normalize(normalized[..., positions], details)

    output[~valid] = np.nan
    return output


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _save_csv(path: Path, array: np.ndarray) -> None:
    values = np.asarray(array)
    if values.ndim == 1:
        values = values[None, :]
    values = values.reshape(-1, values.shape[-1])
    header = ",".join(f"dim_{index:03d}" for index in range(values.shape[-1]))
    np.savetxt(path, values, delimiter=",", header=header, comments="", fmt="%.9g")


def _save_video(path: Path, frames, fps: float) -> int:
    import imageio.v2 as imageio

    arrays = []
    for frame in frames if frames is not None else []:
        if hasattr(frame, "convert"):
            frame = frame.convert("RGB")
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"video frame must be HWC RGB/RGBA, got {array.shape}")
        arrays.append(array[..., :3].astype(np.uint8, copy=False))
    if not arrays:
        raise ValueError("sample contains no video frames")
    imageio.mimwrite(path, arrays, fps=fps, codec="libx264", macro_block_size=1)
    return len(arrays)


class InteractiveDataVisualizer:
    """Export one rank-0 sample per micro-step and wait interactively."""

    def __init__(self, output_path: str, accelerator):
        self.accelerator = accelerator
        self.is_main = bool(accelerator.is_main_process)
        seed = int.from_bytes(os.urandom(16), byteorder="big")
        self._rng = random.Random(seed)
        launch_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.output_dir = Path(output_path) / "vis_data" / launch_id
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=False)

        self._control_group = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                self._control_group = torch.distributed.new_group(backend="gloo")

    def inspect(self, batch: list[dict], *, global_step: int, epoch: int) -> None:
        error: BaseException | None = None
        sample_path: Path | None = None
        if self.is_main:
            try:
                sample_index = self._rng.randrange(len(batch))
                sample_path = self._export(batch[sample_index], sample_index, global_step, epoch)
            except BaseException as exc:
                error = exc

        for sample in batch:
            sample.pop(_VIS_KEY, None)

        if self.is_main and error is None:
            print(f"\n[VIS_DATA] saved {sample_path}", flush=True)
            print("[VIS_DATA] press Enter to train this step...", flush=True)

        if self._control_group is None:
            if error is not None:
                raise RuntimeError(f"VIS_DATA failed: {error}") from error
            self._wait_for_enter()
            return

        self._distributed_wait(error)

    def _distributed_wait(self, initial_error: BaseException | None) -> None:
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        if self.is_main and initial_error is None:
            threading.Thread(target=self._read_enter, args=(result_queue,), daemon=True).start()

        status = torch.zeros(1, dtype=torch.int32)
        error = initial_error
        while True:
            if self.is_main:
                if error is not None:
                    status.fill_(-1)
                else:
                    try:
                        kind, payload = result_queue.get(timeout=1.0)
                        if kind == "ok":
                            status.fill_(1)
                        else:
                            error = payload
                            status.fill_(-1)
                    except queue.Empty:
                        status.fill_(0)
                    except BaseException as exc:
                        error = exc
                        status.fill_(-1)
            torch.distributed.broadcast(status, src=0, group=self._control_group)
            state = int(status.item())
            if state != 0:
                break

        if state < 0:
            message = str(error) if self.is_main and error is not None else "rank 0 failed or interrupted"
            payload = [message]
            torch.distributed.broadcast_object_list(payload, src=0, group=self._control_group)
            raise RuntimeError(f"VIS_DATA failed: {payload[0]}") from error

    @staticmethod
    def _read_enter(result_queue: queue.Queue) -> None:
        try:
            InteractiveDataVisualizer._wait_for_enter()
            result_queue.put(("ok", None))
        except BaseException as exc:
            result_queue.put(("error", exc))

    @staticmethod
    def _wait_for_enter() -> None:
        if sys.stdin is not None and not sys.stdin.closed:
            line = sys.stdin.readline()
            if line != "":
                return
        try:
            with open("/dev/tty", "r", encoding="utf-8") as terminal:
                if terminal.readline() != "":
                    return
        except OSError as exc:
            raise RuntimeError("interactive stdin is unavailable and /dev/tty cannot be opened") from exc
        raise EOFError("interactive input reached EOF")

    def _export(self, sample: dict, sample_index: int, global_step: int, epoch: int) -> Path:
        vis_meta = sample.get(_VIS_KEY)
        if not isinstance(vis_meta, dict):
            raise ValueError("sample is missing VIS_DATA reader metadata")

        sample_dir = self.output_dir / f"step_{global_step:06d}_sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=False)
        prompt = str(sample.get("prompt", ""))
        (sample_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        action = _as_numpy(sample.get("action")).astype(np.float32, copy=False)
        proprio = _as_numpy(sample.get("proprio")).astype(np.float32, copy=False)
        if action.ndim == 1:
            action = action[None, :]
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        action_mask = _broadcast_mask(sample.get("action_mask"), action.shape)
        proprio_mask = _broadcast_mask(sample.get("proprio_mask"), proprio.shape)
        video_mask = _as_numpy(sample.get("video_mask")).astype(bool, copy=False)
        action_denormalized = denormalize_with_layout(action, action_mask, vis_meta["action"])
        proprio_denormalized = denormalize_with_layout(proprio, proprio_mask, vis_meta["proprio"])

        _save_csv(sample_dir / "action_normalized.csv", action)
        _save_csv(sample_dir / "action_denormalized.csv", action_denormalized)
        _save_csv(sample_dir / "proprio_normalized.csv", proprio)
        _save_csv(sample_dir / "proprio_denormalized.csv", proprio_denormalized)
        np.savez_compressed(
            sample_dir / "arrays.npz",
            action_normalized=action,
            action_denormalized=action_denormalized,
            proprio_normalized=proprio,
            proprio_denormalized=proprio_denormalized,
            action_mask=action_mask,
            proprio_mask=proprio_mask,
            video_mask=video_mask,
        )

        source_fps = vis_meta.get("fps")
        video_stride = max(1, int(vis_meta.get("video_stride", 1)))
        fps_fallback = source_fps is None or float(source_fps) <= 0
        video_fps = 10.0 if fps_fallback else float(source_fps) / video_stride
        frame_count = _save_video(sample_dir / "video.mp4", sample.get("video"), video_fps)

        excluded = {"video", "vace_video", "first_frame_image", "action", "proprio", _VIS_KEY}
        sample_metadata = {
            key: value
            for key, value in sample.items()
            if key not in excluded and not key.endswith("_mask")
        }
        metadata = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_sample_index": int(sample_index),
            "prompt": prompt,
            "reader": vis_meta,
            "sample_metadata": sample_metadata,
            "action_shape": list(action.shape),
            "proprio_shape": list(proprio.shape),
            "video_frame_count": frame_count,
            "video_fps": video_fps,
            "video_fps_fallback": fps_fallback,
            "denormalized_semantics": "inverse of the training input normalization; clipped source values are not recoverable",
            "created_at": datetime.now().astimezone().isoformat(),
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(_json_value(metadata), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return sample_dir


__all__ = [
    "InteractiveDataVisualizer",
    "VisualizationDataset",
    "denormalize_with_layout",
    "vis_data_enabled",
]
