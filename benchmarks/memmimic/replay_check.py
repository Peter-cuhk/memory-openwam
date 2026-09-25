#!/usr/bin/env python
"""Offline bridge check: replay a MemMimic training episode through the policy
server's inference path (same image resize, rot6d conversion and memory prefix
as the live rollout) and compare the predicted chunks with the recorded actions.

A checkpoint that fits the training set must reproduce the recorded actions here;
if it does not, the bridge (not the policy) is wrong.

  python benchmarks/memmimic/replay_check.py --ckpt-dir <training_dir> --episodes 0 250
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _rot_rows_from_wxyz(q: np.ndarray) -> np.ndarray:
    """GMP-convention rot6d (first two rows of R) from (..., 4) wxyz."""
    w, x, y, z = (q[..., i] for i in range(4))
    r0 = np.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y], -1)
    r1 = np.stack([2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x], -1)
    return np.concatenate([r0, r1], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--dataset-dir", default="/mnt/cpfs/shared/datasets/memmimic/v1/push_cube")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-chunks", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import zarr

    from openwam.dataloader.memmimic import pose10_from_xyz_wxyz
    from robotmq_policy_server import MemMimicPolicyServer, gmp10_to_openwam10

    server = MemMimicPolicyServer(
        ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name, device=args.device,
        endpoint="tcp://127.0.0.1:18999", deploy_cfg_path=str(PROJECT_ROOT / "configs" / "deploy.yaml"),
        denoise_steps=None, compile_enabled=False, seed=0,
    )
    root = zarr.open(f"{args.dataset_dir}/episode_data.zarr", mode="r")
    H = server.exec_horizon
    for e in args.episodes:
        g = root[f"episode_{e}"]
        cam = g["third_person_camera"]
        tcp = np.asarray(g["robot0_tcp_xyz_wxyz"]); grip = np.asarray(g["robot0_gripper_width"])
        act = pose10_from_xyz_wxyz(np.asarray(g["action0_tcp_xyz_wxyz"]), np.asarray(g["action0_gripper_width"]))
        T = tcp.shape[0]
        server.episodes.clear()
        errs, errs_y, t0 = [], [], time.time()
        n_chunks = 0
        for t in range(0, T - server.action_len, H):
            # frames the simulator would have rendered: s_0 at t=0, then s_{4c+1..4c+4}
            ids = [0] if t == 0 else [t - 12, t - 8, t - 4, t]
            frames = server._to_pil(np.stack([np.asarray(cam[i]) for i in ids]))
            pose_gmp = np.concatenate([tcp[t, :3], _rot_rows_from_wxyz(tcp[t, 3:7]), grip[t]]).astype(np.float32)
            pred_gmp = server._infer_one(e, frames, pose_gmp)  # (32, 10) GMP convention
            pred = gmp10_to_openwam10(pred_gmp)
            gt = act[t : t + server.action_len]
            errs.append(np.abs(pred - gt).mean(0)); errs_y.append(np.abs(pred[:, 1] - gt[:, 1]).mean())
            n_chunks += 1
            if args.max_chunks and n_chunks >= args.max_chunks:
                break
        errs = np.stack(errs)
        print(
            f"episode {e}: {n_chunks} chunks in {time.time() - t0:.0f}s | MAE per dim (xyz rot6d grip):",
            np.round(errs.mean(0), 4),
            f"| y MAE {np.mean(errs_y):.4f} m (gt y range {act[:,1].min():.3f}..{act[:,1].max():.3f}, "
            f"gt |dy| per chunk {np.abs(np.diff(act[:,1])).sum()/ (T/32):.3f})",
        )


if __name__ == "__main__":
    main()
