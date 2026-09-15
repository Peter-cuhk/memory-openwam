"""Wan VAE / text-encoder IO with native-VAE ↔ external-encoder routing.

Free functions — ``vae`` / ``encoder`` / ``tokenizer`` / ``text_encoder`` passed
explicitly so the backbone class holds no IO logic. ``encoder is not None``
selects the external-encoder path; otherwise the native Wan VAE is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video

TEXT_ENCODING_SETTINGS = {"max_length": 512, "clean": "whitespace", "dtype": "bfloat16"}


def text_cache_metadata(model_path: str) -> dict:
    return {"model_path": str(Path(model_path).resolve()), **TEXT_ENCODING_SETTINGS}


def load_text_cache(path: str, model_path: str) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache["metadata"] != text_cache_metadata(model_path):
        raise ValueError(f"T5 cache {path} does not match the model path or encoding settings; regenerate it.")
    for prompt, (context, seq_lens) in cache["embeddings"].items():
        if context.shape != (1, 512, 4096) or context.dtype != torch.bfloat16 or seq_lens.shape != (1,):
            raise ValueError(f"Invalid T5 cache entry for {prompt!r}")
    return cache["embeddings"]


def lookup_text_cache(prompts: list, cache: dict, *, device, dtype) -> Tuple[Tensor, Tensor]:
    try:
        entries = [cache[prompt] for prompt in prompts]
    except KeyError as exc:
        raise KeyError(f"Prompt missing from T5 cache: {exc.args[0]!r}; regenerate the cache.") from exc
    return (
        torch.cat([entry[0] for entry in entries]).to(device=device, dtype=dtype),
        torch.cat([entry[1] for entry in entries]).to(device=device),
    )


def encode_text(prompts: list, *, tokenizer, text_encoder, device) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=TEXT_ENCODING_SETTINGS["max_length"],
        padding="max_length",
        truncation=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


def encode_text_for_inference(
    prompt, *, vace_cache, prompt_embed_cache, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    """Deploy ``(context, seq_lens)``, reusing a prompt-keyed cached embed."""
    if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt:
        return vace_cache["context"], vace_cache["seq_lens"]
    if prompt_embed_cache is not None and prompt in prompt_embed_cache:
        return prompt_embed_cache[prompt]
    context, seq_lens = encode_text([prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device)
    if prompt_embed_cache is not None:
        prompt_embed_cache[prompt] = (context, seq_lens)
    return context, seq_lens


def preprocess_video(frames, *, encoder=None, dtype, device) -> Tensor:
    if encoder is not None:
        return encoder.preprocess_video(frames)
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def encode_video(video_tensor: Tensor, *, vae, encoder=None) -> Tensor:
    if encoder is not None:
        return encoder.batch_encode(video_tensor)
    return vae.batch_encode(video_tensor, device=video_tensor.device)


def encode_video_for_vace(
    pixels: Tensor,
    *,
    vae,
    encoder=None,
    device,
    tiled: bool,
    tile_size: tuple,
    tile_stride: tuple,
) -> Tensor:
    """Tiled-aware VAE encode for the VACE pixel→latent helper, returning
    ``(B, z_dim, T_lat, H_lat, W_lat)``. ``tiled=False`` (training) batches in
    one call; ``tiled=True`` (deploy) loops per-sample with bounded peak memory,
    mirroring the vendored unit so large-frame deploy does not OOM.
    """
    if not tiled:
        return encode_video(pixels, vae=vae, encoder=encoder).to(dtype=pixels.dtype, device=pixels.device)
    if encoder is not None:
        # No generic tiled-encode contract; fall back to batch_encode.
        # Unreachable today (VACE + external_encoder is fail-fast).
        return encoder.batch_encode(pixels).to(dtype=pixels.dtype, device=pixels.device)
    # Native Wan VAE: per-sample tiled encode (deploy is B=1).
    outs = []
    for i in range(pixels.shape[0]):
        lat = vae.encode(
            [pixels[i]],
            device=device,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        outs.append(lat)
    return torch.cat(outs, dim=0).to(dtype=pixels.dtype, device=pixels.device)


def decode_latents(latents: Tensor, *, vae, encoder=None, device, tiled: bool = True) -> Tensor:
    if encoder is not None:
        return encoder.decode(latents.to(device), tiled=tiled)
    return vae.decode(latents.to(device), device=device, tiled=tiled)


def latents_to_frames(video_tensor: Tensor, *, encoder=None) -> list:
    if encoder is not None:
        return encoder.to_frames(video_tensor)
    return vae_output_to_video(video_tensor)
