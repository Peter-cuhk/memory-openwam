"""Memory-OpenWAM Phase 2: deploy-time K/V cache of the memory prefix (one episode, batch 1).

The training layout already has streaming semantics (``memory-openwam/DESIGN.md`` §4-5): memory
rows ``A / F_k / G_k`` never read the window, the action stream or proprio, carry ``t = 0`` and use
absolute-time RoPE. Their per-layer K/V therefore depend only on the history latents, so they can be
computed once and reused:

* ``prefix_once`` — every request computes the memory-row K/V once (memory rows as queries, the
  training mask restricted to them) and reuses them for all denoising steps. Exact for any layout,
  including the truncated one (``c + 1 > max_latents``).
* ``streaming`` — K/V persist across requests; each new latent writes only ``[F_k | G_k]``
  (keys: anchor, ``F_{k-1}``, ``G_{<k}`` and itself, all visible — no mask). Once the layout is
  truncated the earliest kept gist would have to "forget" its dropped predecessors, which a stream
  cannot do, so those requests fall back to ``prefix_once`` (still exact).

Memory rows do read the text prompt through cross-attention (not the trailing proprio token), so a
stream is valid for one prompt: reset it at every episode start; the backbone raises if the text
context changes mid-episode.

The backbone (``WanVideoBackbone._attach_memory_stream``) does the writes; this class only holds
state. Readers see the anchor frame tokens, every gist and (``recent_kv``) ``F_c``; the cache
stores exactly those columns, in :func:`memory_reader_key_index` order.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

MEMORY_INFERENCE_MODES = ("recompute", "prefix_once", "streaming")


class MemoryKVStream:
    def __init__(self, mode: str = "streaming") -> None:
        if mode not in ("prefix_once", "streaming"):
            raise ValueError(f"MemoryKVStream mode must be prefix_once or streaming, got {mode!r}.")
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        # Per absolute latent time: (1, z, 1, H, W) history latents (anchor first).
        self.latents: list[Tensor] = []
        # streaming state, one entry per layer: (k, v) with k/v shaped (1, tokens, H*D)
        self.anchor_kv: Optional[list[tuple[Tensor, Tensor]]] = None
        self.gist_kv: Optional[list[list[tuple[Tensor, Tensor]]]] = None
        self.frame_kv: Optional[list[tuple[Tensor, Tensor]]] = None  # F_{n_written-1}
        self.n_written = 0
        # current request (set by the backbone, reused across its denoising steps)
        self.request_key: Optional[tuple] = None
        self.read_kv: Optional[list[tuple[Tensor, Tensor]]] = None
        self.layout = None
        self.key_index: Optional[Tensor] = None
        self.window_freqs: Optional[Tensor] = None
        self.context_ref = None  # (text context, mask) the cached K/V were written under
        self.reader_masks: dict = {}
        self.stats = {"slots_written": 0, "prefix_rebuilds": 0, "requests": 0}

    def append_latent(self, latent: Tensor) -> None:
        """Register the next history latent ``(1, z, 1, H, W)`` (absolute time ``len(latents)``)."""
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 1:
            raise ValueError(f"expected a (1, z, 1, H, W) latent, got {tuple(latent.shape)}")
        self.latents.append(latent)

    def latents_at(self, times) -> Tensor:
        """``(1, z, len(times), H, W)`` history latents at the given absolute times."""
        missing = [t for t in times if t >= len(self.latents)]
        if missing:
            raise ValueError(f"memory times {missing} are not encoded yet ({len(self.latents)} latents).")
        return torch.cat([self.latents[t] for t in times], dim=2)

    def layer_kv(self, layer_id: int) -> tuple[Tensor, Tensor]:
        if self.read_kv is None:
            raise RuntimeError("MemoryKVStream has no K/V for the current request.")
        return self.read_kv[layer_id]


__all__ = ["MemoryKVStream", "MEMORY_INFERENCE_MODES"]
