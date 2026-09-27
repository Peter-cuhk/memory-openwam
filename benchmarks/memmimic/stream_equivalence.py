#!/usr/bin/env python
"""Phase 2 check (memory-openwam/docs/07 V0-V2): replay MemMimic training episodes through the policy
server's request path once per memory-inference mode, with the same per-request noise seeds, and
compare every mode's predicted chunks with the original ``recompute`` path.

Modes: ``recompute`` (original: re-encode the history, memory prefix at every denoising step),
``recompute_vae`` (same, streaming VAE latents), ``prefix_once`` and ``streaming`` (KV cache). The
``recompute`` pass runs twice (run-to-run determinism). References for the size of the differences:
``recompute_math_sdpa`` (original path, mathematically equivalent attention kernel = its own bf16
noise) and ``recompute_seed1`` (another noise seed = the policy's sampling spread).

  python benchmarks/memmimic/stream_equivalence.py --ckpt-dir <training_dir> --ckpt-name checkpoint_step_N.safetensors \\
      --episodes 0 250 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# name -> (memory inference mode, stream VAE, SDPA backend override, noise-seed offset)
MODES = {
    "recompute": ("recompute", False, None, 0),
    "recompute_vae": ("recompute", True, None, 0),
    "prefix_once": ("prefix_once", True, None, 0),
    "streaming": ("streaming", True, None, 0),
    # references: the original path under a mathematically equivalent attention kernel (bf16 noise
    # floor of the original computation), and under another noise seed (the policy's own sampling spread)
    "recompute_math_sdpa": ("recompute", False, "math", 0),
    "recompute_seed1": ("recompute", False, None, 1),
}
DEFAULT_MODES = ["recompute", "recompute_vae", "prefix_once", "streaming"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--dataset-dir", default="/mnt/cpfs/shared/datasets/memmimic/v1/push_cube")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=list(MODES))
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:18998")
    ap.add_argument("--out", required=True, help="directory for records.jsonl and summary.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import torch
    import zarr

    from replay_check import _rot_rows_from_wxyz
    from robotmq_policy_server import MemMimicPolicyServer, gmp10_to_openwam10

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    server = MemMimicPolicyServer(
        ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name, device=args.device, endpoint=args.endpoint,
        deploy_cfg_path=str(PROJECT_ROOT / "configs" / "deploy.yaml"), denoise_steps=None, compile_enabled=False, seed=0,
    )
    root = zarr.open(f"{args.dataset_dir}/episode_data.zarr", mode="r")
    H = server.exec_horizon
    runs = ["recompute", *args.modes] if "recompute" in args.modes else list(args.modes)  # recompute twice
    records = []
    preds: dict = {}
    for e in args.episodes:
        g = root[f"episode_{e}"]
        cam = g["third_person_camera"]
        tcp = np.asarray(g["robot0_tcp_xyz_wxyz"])
        grip = np.asarray(g["robot0_gripper_width"])
        T = tcp.shape[0]
        starts = list(range(0, T - server.action_len, H))[: args.max_chunks]
        requests = []
        for t in starts:
            ids = [0] if t == 0 else [t - 12, t - 8, t - 4, t]  # what the simulator renders per request
            frames = server._to_pil(np.stack([np.asarray(cam[i]) for i in ids]))
            pose = np.concatenate([tcp[t, :3], _rot_rows_from_wxyz(tcp[t, 3:7]), grip[t]]).astype(np.float32)
            requests.append((frames, pose))
        seen: dict = {}
        for name in runs:
            run_name = name if name not in seen else f"{name}#2"
            seen[name] = True
            mode, stream_vae, sdpa, seed_offset = MODES[name]
            server.configure_memory_inference(mode, stream_vae=stream_vae)
            server.seed = seed_offset
            chunk_preds = []
            for i, (frames, pose) in enumerate(requests):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                if sdpa == "math":
                    from torch.nn.attention import SDPBackend, sdpa_kernel

                    with sdpa_kernel([SDPBackend.MATH]):
                        pred = gmp10_to_openwam10(server._infer_one(e, frames, pose))
                else:
                    pred = gmp10_to_openwam10(server._infer_one(e, frames, pose))
                torch.cuda.synchronize()
                dt = time.time() - t0
                chunk_preds.append(pred)
                records.append({
                    "episode": e, "run": run_name, "request": i, "ctx": (len(server.episodes[e]["frames"]) - 1) // 4,
                    "t": dt, "t_vae_stream": float(server.episodes[e].get("t_vae", 0.0)),
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                })
            preds[(e, run_name)] = np.stack(chunk_preds)
            kv = server.episodes[e].get("kv")
            print(f"episode {e} {run_name}: {len(requests)} requests, {sum(r['t'] for r in records if r['episode']==e and r['run']==run_name):.1f}s"
                  + (f", stream stats {kv.stats}" if kv is not None and mode != "recompute" else ""), flush=True)
            server.episodes.clear()
            server.seed = 0
    with open(out_dir / "records.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    # --- summary: action differences vs the first recompute run, and time by context bucket ---
    summary = {"ckpt": f"{args.ckpt_dir}/{args.ckpt_name}", "episodes": args.episodes, "diff_vs_recompute_mm": {}, "time": {}}
    for e in args.episodes:
        base = preds[(e, "recompute")]
        for (ee, run_name), p in preds.items():
            if ee != e or run_name == "recompute":
                continue
            d = np.abs(p - base)
            xyz_mm = d[..., :3] * 1000.0
            per_req = xyz_mm.max(axis=(1, 2))
            summary["diff_vs_recompute_mm"].setdefault(run_name, []).append({
                "episode": e,
                "xyz_max_mm": float(xyz_mm.max()),
                "xyz_mean_mm": float(xyz_mm.mean()),
                "y_step_speed_max": float(np.abs(np.diff(p[:, :, 1], axis=1) - np.diff(base[:, :, 1], axis=1)).max()),
                "per_request_xyz_max_mm": [round(float(x), 4) for x in per_req],
                "all_dims_max": float(d.max()),
            })
    for run_name in sorted({r["run"] for r in records}):
        rs = [r for r in records if r["run"] == run_name and r["request"] > 0]  # request 0 includes warm-up
        buckets = {}
        for r in rs:
            buckets.setdefault(r["ctx"] // 10 * 10, []).append(r)
        summary["time"][run_name] = {
            "total_s": sum(r["t"] for r in rs),
            "peak_mem_gb_max": max(r["peak_mem_gb"] for r in rs),
            "by_ctx": {
                f"{k}-{k + 9}": {"n": len(v), "median_s": float(np.median([r["t"] for r in v])),
                                 "peak_mem_gb": float(np.max([r["peak_mem_gb"] for r in v]))}
                for k, v in sorted(buckets.items())
            },
        }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("diff_vs_recompute_mm",)}, indent=1)[:4000])
    for run_name, tinfo in summary["time"].items():
        print(run_name, f"total {tinfo['total_s']:.1f}s peak {tinfo['peak_mem_gb_max']:.1f} GB",
              {k: round(v["median_s"], 3) for k, v in tinfo["by_ctx"].items()})


if __name__ == "__main__":
    main()
