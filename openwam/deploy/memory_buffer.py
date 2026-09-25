"""Memory-OpenWAM deploy side: per-episode observation buffer.

Mirrors the training layout of :mod:`openwam.dataloader.memmimic` and the
streaming semantics in ``memory-openwam/DESIGN.md`` §5.1: every
``video_stride``-th control-step frame is kept; four kept frames form one
history chunk (= one VAE latent = one gist); the anchor is the episode's first
frame. Only real observations enter the buffer (predicted video never does),
and the buffer is reset at every episode start.

At a replan on control step ``t`` the memory exposed to the model is the
prefix of *complete* chunks, ``s_0 .. s_{4c}`` with ``c = floor((n-1)/4)`` where
``n`` is the number of kept frames so far. When ``inference_horizon`` is a
multiple of ``video_stride * 4`` (16 at stride 4) every replan lands exactly
on a chunk boundary and the current frame is ``s_{4c}``, bit-for-bit the
training layout; otherwise the memory lags the current frame by up to one
chunk (logged once).
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

FRAMES_PER_LATENT = 4


def memory_times_for(context_index: int, max_latents: int) -> List[int]:
    """Slots kept for context ``c``: all of ``0..c`` or anchor + newest."""
    c = int(context_index)
    if c + 1 <= max_latents:
        return list(range(c + 1))
    return [0] + list(range(c - max_latents + 2, c + 1))


class EpisodeMemoryBuffer:
    def __init__(self, *, video_stride: int, max_latents: int, frames_per_latent: int = FRAMES_PER_LATENT):
        if video_stride <= 0 or max_latents <= 0 or frames_per_latent <= 0:
            raise ValueError("video_stride, max_latents and frames_per_latent must be positive.")
        self.video_stride = int(video_stride)
        self.max_latents = int(max_latents)
        self.frames_per_latent = int(frames_per_latent)
        self._frames: list = []
        self._step = 0
        self._warned_misaligned = False

    def reset(self) -> None:
        self._frames = []
        self._step = 0

    @property
    def control_step(self) -> int:
        return self._step

    @property
    def num_kept_frames(self) -> int:
        return len(self._frames)

    def observe(self, image) -> None:
        """Register the observation of the current control step (call every step)."""
        if self._step % self.video_stride == 0:
            self._frames.append(image)
        self._step += 1

    def context_index(self) -> int:
        n = len(self._frames)
        if n == 0:
            raise RuntimeError("EpisodeMemoryBuffer.snapshot before any observation.")
        return (n - 1) // self.frames_per_latent

    def snapshot(self) -> dict:
        """Memory inputs for a replan at the current step (see module docstring)."""
        c = self.context_index()
        n_frames = c * self.frames_per_latent + 1
        if n_frames != len(self._frames) and not self._warned_misaligned:
            self._warned_misaligned = True
            logger.warning(
                "EpisodeMemoryBuffer: replan at control step %d is not on a chunk boundary "
                "(%d kept frames, using %d); set inference_horizon to a multiple of %d for exact "
                "train/deploy alignment.",
                self._step - 1,
                len(self._frames),
                n_frames,
                self.video_stride * self.frames_per_latent,
            )
        return {
            "memory_video": list(self._frames[:n_frames]),
            "memory_times": memory_times_for(c, self.max_latents),
            "memory_context_index": c,
        }


__all__ = ["EpisodeMemoryBuffer", "memory_times_for", "FRAMES_PER_LATENT"]
