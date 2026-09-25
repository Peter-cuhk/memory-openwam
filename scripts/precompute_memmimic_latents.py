"""Precompute per-episode Wan-VAE latents of the MemMimic history stream.

For every episode, the stride-sampled frames ``s_0, s_1, ..., s_K`` are
encoded ONCE as a single causal clip. Latent ``k`` therefore corresponds to
history chunk ``k`` (frames ``s_{4k-3..4k}``; latent 0 = ``s_0`` alone), which
is exactly what ``MemMimicDataset`` slices as ``memory_latents[:, :c+1]`` and
what a streaming VAE would produce at deployment.

Output: ``<output_dir>/episode_<i>.pt`` holding a ``(z, T_lat, h, w)`` bf16
tensor, plus ``manifest.json``. Point ``dataloader.memory_latent_cache_dir``
at ``<output_dir>`` to skip the online VAE encode during training.

Usage (inside the OpenWAM env):
    python scripts/precompute_memmimic_latents.py \
        --dataset_dir /mnt/cpfs/shared/datasets/memmimic/v1/push_cube \
        --output_dir  /mnt/cpfs/workspace/memory-openwam/cache/push_cube_latents \
        --model_path  /mnt/cpfs/shared/models/base/Wan2.2-TI2V-5B
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True, help="Wan2.2-TI2V-5B directory (VAE weights)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--video_stride", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=None, help="limit (debug)")
    parser.add_argument("--chunk_frames", type=int, default=129, help="frames per VAE call (4k+1); >= episode = one call")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from openwam.dataloader.memmimic import MemMimicDataset
    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.model.video_backbone.wan.loader import build_holder_from_model_path

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = MemMimicDataset(
        args.dataset_dir,
        video_stride=args.video_stride,
        height=args.height,
        width=args.width,
        memory_enabled=False,
        window_stride=1,
    )
    holder = build_holder_from_model_path(args.model_path, device="cpu", skip_text_encoder=True)
    vae = holder.vae.to(device=args.device, dtype=torch.bfloat16)
    device = torch.device(args.device)

    episodes = ds.episodes if args.episodes is None else ds.episodes[: args.episodes]
    manifest = {}
    t0 = time.time()
    for n, e in enumerate(episodes):
        target = out / f"episode_{e}.pt"
        frames = ds.episode_history_frames(e)
        if target.is_file():
            lat = torch.load(target, map_location="cpu", weights_only=True)
            manifest[e] = {"frames": len(frames), "latents": int(lat.shape[1])}
            continue
        with torch.no_grad():
            pixels = wan_encode.preprocess_video(frames, dtype=torch.bfloat16, device=device)  # (1, C, T, H, W)
            # The Wan VAE is causal with a standalone first frame, so a single
            # call over the whole clip is the exact streaming semantics. Very
            # long clips can be split at 4k+1 boundaries only by a streaming
            # VAE; here we always encode in one call.
            lat = wan_encode.encode_video(pixels, vae=vae)[0].to(torch.bfloat16).cpu()  # (z, T_lat, h, w)
        expected = (len(frames) - 1) // 4 + 1
        if lat.shape[1] != expected:
            raise RuntimeError(f"episode {e}: {len(frames)} frames -> {lat.shape[1]} latents, expected {expected}")
        tmp = target.with_name(target.name + ".tmp")
        torch.save(lat, tmp)
        tmp.replace(target)
        manifest[e] = {"frames": len(frames), "latents": int(lat.shape[1])}
        if n % 10 == 0:
            print(f"[{n + 1}/{len(episodes)}] episode {e}: {len(frames)} frames -> {tuple(lat.shape)} ({time.time() - t0:.0f}s)", flush=True)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir),
                "video_stride": args.video_stride,
                "height": args.height,
                "width": args.width,
                "model_path": str(args.model_path),
                "episodes": {str(k): v for k, v in manifest.items()},
            },
            indent=2,
        )
    )
    print(f"done: {len(manifest)} episodes in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
