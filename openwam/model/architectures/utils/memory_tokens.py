"""Memory-OpenWAM: config + the (tiny) set of new parameters.

Everything else in the memory path reuses pretrained OpenWAM weights. The only
new parameters are the learnable gist token embeddings (shared across memory
slots; RoPE tells them apart in time) and, optionally, one scalar per video
layer that gates how much the readers (noisy future / action tokens) attend to
the gists. The gate starts strongly negative so the joint forward is
numerically the vanilla OpenWAM forward at step 0 and the model opens the
memory path gradually during finetuning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn


@dataclass
class MemoryConfig:
    enabled: bool = False
    gist_tokens: int = 8
    max_latents: int = 40  # memory slots kept (anchor + newest history), train == deploy
    recent_kv: bool = False  # expose F_c (last history chunk) KV to the readers
    f_cur_reads_memory: bool = False  # let OpenWAM's clean frame read the memory
    use_gist_gate: bool = True
    gate_init: float = -4.0
    memory_dropout: float = 0.0  # p(hide gists/recent from readers) per sample, train only
    clean_noise_prob: float = 0.0  # CoCo clean-context noise on history latents (anchor stays clean)
    clean_noise_scale: float = 0.0
    # Action history (GMP ``include_action_history``): the commands executed during each
    # memory slot are embedded and added to that slot's gists (zero-init -> no-op at step 0).
    action_history: bool = False
    action_history_steps: int = 16  # control steps per latent (video_stride * 4)
    action_history_dim: int = 10  # raw normalized action width fed by the reader / server
    # Train-time perturbation of the action history (anti-copycat, 2026-09-26): per slot a constant offset
    # ~N(0, offset) and a linear drift ~N(0, drift) over the slot's steps (a speed error), in normalized units;
    # plus dropping a sample's whole action history with probability `action_history_dropout`.
    action_history_noise_offset: float = 0.0
    action_history_noise_drift: float = 0.0
    action_history_dropout: float = 0.0
    # Experiment B (2026-09-27): keep every slot, its time (RoPE) and its action history, but replace each history
    # slot's (t > 0) video latent by the anchor latent (detached), so the only image the memory ever sees is the
    # first frame. Applied in the backbone right before the memory rows are built (train, recompute, prefix_once,
    # streaming alike); see ``blind_history_latents``.
    history_video_blind: bool = False

    def __post_init__(self) -> None:
        if self.gist_tokens <= 0:
            raise ValueError("memory.gist_tokens must be positive.")
        if self.max_latents <= 0:
            raise ValueError("memory.max_latents must be positive.")
        if not 0.0 <= self.memory_dropout <= 1.0:
            raise ValueError("memory.memory_dropout must be in [0, 1].")
        if not 0.0 <= self.clean_noise_prob <= 1.0:
            raise ValueError("memory.clean_noise_prob must be in [0, 1].")
        if self.clean_noise_scale < 0.0:
            raise ValueError("memory.clean_noise_scale cannot be negative.")
        if min(self.action_history_noise_offset, self.action_history_noise_drift) < 0.0:
            raise ValueError("memory.action_history_noise_* cannot be negative.")
        if not 0.0 <= self.action_history_dropout <= 1.0:
            raise ValueError("memory.action_history_dropout must be in [0, 1].")

    def mask_cfg(self) -> dict:
        return {"recent_kv": bool(self.recent_kv), "f_cur_reads_memory": bool(self.f_cur_reads_memory)}


_FIELDS = tuple(MemoryConfig.__dataclass_fields__)


def parse_memory_config(raw: Any) -> MemoryConfig:
    """``None`` / missing -> disabled; dict or DictConfig -> MemoryConfig."""
    if raw is None:
        return MemoryConfig()
    if isinstance(raw, MemoryConfig):
        return raw
    if hasattr(raw, "items"):
        items = dict(raw.items())
    else:
        items = {k: getattr(raw, k) for k in _FIELDS if hasattr(raw, k)}
    unknown = sorted(k for k in items if k not in _FIELDS)
    if unknown:
        raise ValueError(f"Unknown memory config keys: {unknown}. Known: {list(_FIELDS)}")
    kwargs = {}
    for k, v in items.items():
        if v is None:
            continue
        kwargs[k] = v
    return MemoryConfig(**kwargs)


def blind_history_latents(latents: torch.Tensor, memory_times) -> torch.Tensor:
    """``memory.history_video_blind``: every real history slot (time > 0) takes the anchor slot's latent.

    ``latents`` is ``(B, z, S, H, W)`` with slot 0 = the anchor (time 0); ``memory_times[b]`` lists sample ``b``'s
    valid slots (a shorter list = zero padding, left as is). The anchor is copied detached; slot count, times and
    action history are untouched, so only the history *images* are removed from the memory.
    """
    if latents.ndim != 5:
        raise ValueError(f"memory latents must be (B, z, S, H, W), got {tuple(latents.shape)}")
    B, _, S = latents.shape[:3]
    if len(memory_times) != B:
        raise ValueError(f"memory_times has {len(memory_times)} samples for a batch of {B}.")
    hist = torch.zeros(B, S, dtype=torch.bool, device=latents.device)
    for b, times in enumerate(memory_times):
        times = [int(t) for t in times]
        if not times or times[0] != 0 or len(times) > S:
            raise ValueError(f"memory_times[{b}]={times}: slot 0 must be the anchor (time 0), at most {S} slots.")
        hist[b, 1 : len(times)] = True
    anchor = latents[:, :, :1].detach().expand_as(latents)
    return torch.where(hist.view(B, 1, S, 1, 1), anchor, latents)


class MemoryTokens(nn.Module):
    """Learnable gist tokens (+ optional per-layer read gate, + optional action-history embedding)."""

    def __init__(self, cfg: MemoryConfig, *, dim: int, num_layers: int):
        super().__init__()
        self.cfg = cfg
        self.gist_tokens = nn.Parameter(torch.randn(1, cfg.gist_tokens, dim) / math.sqrt(dim))
        if cfg.use_gist_gate:
            self.gist_gate = nn.Parameter(torch.full((num_layers,), float(cfg.gate_init)))
        else:
            self.gist_gate = None
        if cfg.action_history:
            in_dim = 2 * cfg.action_history_steps * cfg.action_history_dim  # actions + scaled per-step deltas
            self.action_history_mlp = nn.Sequential(nn.Linear(in_dim, 1024), nn.SiLU(), nn.Linear(1024, dim))
            nn.init.zeros_(self.action_history_mlp[-1].weight)
            nn.init.zeros_(self.action_history_mlp[-1].bias)
        else:
            self.action_history_mlp = None

    def action_embedding(self, actions: torch.Tensor) -> torch.Tensor:
        """``(B, S, steps, D)`` normalized actions per memory slot -> ``(B, S, dim)`` gist offsets.

        Per-step deltas are scaled by 10 so that push speed (the quantity the policy must
        read back) is not drowned by the absolute pose."""
        if self.action_history_mlp is None:
            raise RuntimeError("memory.action_history is disabled; no action embedding.")
        w = self.action_history_mlp[0].weight
        a = actions.to(dtype=w.dtype, device=w.device)
        delta = torch.diff(a, dim=2, prepend=a[:, :, :1])
        feat = torch.cat([a, 10.0 * delta], dim=-1).flatten(2)
        return self.action_history_mlp(feat)

    @property
    def gate(self) -> Optional[torch.Tensor]:
        return self.gist_gate


__all__ = ["MemoryConfig", "MemoryTokens", "blind_history_latents", "parse_memory_config"]
