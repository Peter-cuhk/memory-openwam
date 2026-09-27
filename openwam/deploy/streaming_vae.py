"""Streaming Wan2.2 VAE encoder for the Memory-OpenWAM history (Phase 2, memory-openwam/docs/07).

``VideoVAE38_.encode`` already encodes a clip as the first frame alone, then 4-frame chunks, threading
the causal-convolution cache between them; ``conv1`` is a 1x1x1 conv. Running the same loop across
requests — one call per newly completed chunk, with a per-episode cache — gives the latents of the
whole history without re-encoding it. Adapted from CoCo ``memory_causal_wan/models/streaming_vae.py``
(the cache lives on this object, not on the shared VAE module, so interleaved episodes cannot clash).
"""

from __future__ import annotations

import torch
from torch import Tensor

from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.models.vae import VideoVAE38_, count_conv3d, patchify


class StreamingVAEEncoder:
    """One episode: ``append([s_0])`` -> anchor latent, then ``append([s_{4k-3}..s_{4k}])`` -> ``F_k``."""

    def __init__(self, video_backbone) -> None:
        if getattr(video_backbone, "video_encoder", None) is not None:
            raise NotImplementedError("Streaming VAE supports the native Wan VAE only.")
        vae = getattr(video_backbone, "vae", None)
        if vae is None or not isinstance(getattr(vae, "model", None), VideoVAE38_):
            raise NotImplementedError("Streaming VAE supports the Wan2.2 (48-channel) VAE only.")
        self.vae = vae
        self.model = vae.model
        self.dtype = video_backbone.dtype
        self.device = video_backbone.device
        self._n_conv = count_conv3d(self.model.encoder)
        self.reset()

    def reset(self) -> None:
        self._feat_map = [None] * self._n_conv
        self.n_latents = 0

    @torch.no_grad()
    def append(self, frames: list) -> Tensor:
        """Encode the next chunk (list of PIL frames) -> ``(1, z, 1, H, W)`` latent in the backbone dtype."""
        expected = 1 if self.n_latents == 0 else 4
        if len(frames) != expected:
            raise ValueError(f"latent {self.n_latents} needs {expected} frame(s), got {len(frames)}.")
        pixels = wan_encode.preprocess_video(frames, encoder=None, dtype=self.dtype, device=self.device)
        x = patchify(pixels.to(self.device), patch_size=2)
        out, self._feat_map, _ = self.model.encoder(x, feat_cache=self._feat_map, feat_idx=[0])
        mu, _ = self.model.conv1(out).chunk(2, dim=1)
        if mu.shape[2] != 1:
            raise RuntimeError(f"a streaming VAE chunk must yield one latent, got {mu.shape[2]}.")
        z = self.model.z_dim
        mean, inv_std = (s.to(dtype=mu.dtype, device=mu.device) for s in self.vae.scale)
        mu = (mu - mean.view(1, z, 1, 1, 1)) * inv_std.view(1, z, 1, 1, 1)
        self.n_latents += 1
        return mu.to(dtype=self.dtype, device=self.device)


__all__ = ["StreamingVAEEncoder"]
