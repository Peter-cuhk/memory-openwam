#!/usr/bin/env python
"""Does the policy read its memory? Counterfactual history swap on push_cube training episodes.

For episode A and push k (k >= 2) we query the policy at the replan step right before the push
starts, with A's current frame and proprio, under three histories:

  own    A's own stride-4 history up to that step (what the policy saw in training)
  swap   the history of episode B up to B's k-th push (B has a very different friction,
         so its previous pushes landed elsewhere and the expert pushed at another speed)
  anchor no history (c = 0, only the current frame as s_0)

and compare the peak y speed of the predicted chunk with the recorded peak speeds of A and B.
A policy that uses memory follows the history: swap moves the prediction from A's speed to B's.

  python benchmarks/memmimic/memory_swap_probe.py --ckpt-dir <training_dir> [--ckpt-name ...] [--n-pairs 8]

Summary line: slope of (pred_swap - pred_own) on (gt_B - gt_A); 1 = history fully used, 0 = ignored.
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

DT = 0.1


def push_starts(tcp_y: np.ndarray, min_peak: float = 0.15):
    """[(start_idx, peak_speed)] of forward pushes (same rule as analyze_push_cube.split_pushes)."""
    v = np.gradient(tcp_y, DT)
    fwd = v > 0.05
    out, i, n = [], 0, len(v)
    while i < n:
        if fwd[i]:
            j = i
            while j < n and fwd[j]:
                j += 1
            if v[i:j].max() >= min_peak:
                out.append((i, float(v[i:j].max())))
            i = j
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--dataset-dir", default="/mnt/cpfs/shared/datasets/memmimic/v1/push_cube")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--pushes", type=int, nargs="+", default=[2, 4, 6], help="1-based push indices to probe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--denoise-steps", type=int, default=None)
    ap.add_argument("--shift", type=float, default=None, help="override the action-stream alpha-shift of the denoise schedule")
    ap.add_argument("--frame-offset-npy", default=None, help="(H, W, 3) float offset added to every dataset frame (rendering-shift test)")
    ap.add_argument("--n-seeds", type=int, default=1, help="repeat 'own' with this many seeds: sampling spread of the push speed")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import zarr

    from openwam.dataloader.memmimic import memory_action_history, pose10_from_xyz_wxyz
    from openwam.dataloader.utils.normalization import apply_normalization
    from openwam.deploy.memory_buffer import memory_times_for
    from robotmq_policy_server import MemMimicPolicyServer

    server = MemMimicPolicyServer(
        ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name, device=args.device,
        endpoint="tcp://127.0.0.1:18998", deploy_cfg_path=str(PROJECT_ROOT / "configs" / "deploy.yaml"),
        denoise_steps=args.denoise_steps, compile_enabled=False, seed=0,
    )
    if not server.memory_enabled:
        raise SystemExit("checkpoint has no memory; nothing to probe")
    H, F = server.exec_horizon, server.frames_per_latent
    stride = H // F
    root = zarr.open(f"{args.dataset_dir}/episode_data.zarr", mode="r")

    # Pair the lowest-friction episodes with the highest-friction ones (largest speed contrast).
    fr = []
    for e in range(len([k for k in root.keys() if k.startswith("episode_")])):
        cfg = dict(root[f"episode_{e}"].attrs).get("episode_config", {}) or {}
        fr.append((cfg.get("sliding_friction", np.nan), e))
    fr = sorted(x for x in fr if np.isfinite(x[0]))
    lows, highs = [e for _, e in fr[: args.n_pairs]], [e for _, e in fr[-args.n_pairs :]][::-1]
    pairs = [(a, b) for a, b in zip(lows, highs)] + [(b, a) for a, b in zip(lows, highs)]

    cache: dict[int, dict] = {}

    def ep(e):
        if e not in cache:
            g = root[f"episode_{e}"]
            tcp = np.asarray(g["robot0_tcp_xyz_wxyz"])
            act = pose10_from_xyz_wxyz(np.asarray(g["action0_tcp_xyz_wxyz"]), np.asarray(g["action0_gripper_width"]))
            if server.action_history:
                act = apply_normalization(act, server._action_stats, server._norm_mode).astype(np.float32)
            cache[e] = {
                "g": g, "tcp": tcp, "grip": np.asarray(g["robot0_gripper_width"]), "act_norm": act,
                "pushes": push_starts(tcp[:, 1]),
                "friction": (dict(g.attrs).get("episode_config", {}) or {}).get("sliding_friction"),
            }
        return cache[e]

    frame_offset = np.load(args.frame_offset_npy).astype(np.float32) if args.frame_offset_npy else None

    def history(e, t):
        cam = ep(e)["g"]["third_person_camera"]
        frames = np.stack([np.asarray(cam[i]) for i in range(0, t + 1, stride)])
        if frame_offset is not None:
            frames = np.clip(frames.astype(np.float32) + frame_offset, 0, 255).astype(np.uint8)
        return server._to_pil(frames)

    def peak(pred):
        return float(np.gradient(pred[:, 1], DT).max())

    rows = []
    t0 = time.time()
    for a, b in pairs:
        A, B = ep(a), ep(b)
        for k in args.pushes:
            if k > len(A["pushes"]) or k > len(B["pushes"]):
                continue
            tA = (A["pushes"][k - 1][0] // H) * H
            tB = (B["pushes"][k - 1][0] // H) * H
            histA, histB = history(a, tA), history(b, tB)
            cur = histA[-1]
            proprio = pose10_from_xyz_wxyz(A["tcp"][tA : tA + 1], A["grip"][tA : tA + 1])[0].astype(np.float32)
            preds = {}

            def run(mem, seed, src):
                c = (len(mem) - 1) // F
                times = memory_times_for(c, server.max_latents)
                cond = {
                    "prompt": server.prompt, "first_frame_image": [cur], "proprio": proprio, "seed": seed,
                    "memory_video": list(mem[: c * F + 1]), "memory_times": times, "memory_context_index": c,
                }
                if args.shift is not None:
                    cond["shift"] = float(args.shift)
                if server.action_history:  # the history's own executed commands travel with its frames
                    cond["memory_actions"] = memory_action_history(src["act_norm"][: H * c], times, H)
                raw = np.asarray(server.engine.generate(cond)["actions"], dtype=np.float32)
                return peak(server.to_absolute(raw, proprio))

            for name, mem, src in (("own", histA, A), ("swap", histB, B), ("anchor", [cur], A)):
                preds[name] = run(mem, 1234, src)
            extra = [run(histA, 1234 + 7 * s, A) for s in range(1, args.n_seeds)]
            preds["own_std"] = float(np.std([preds["own"]] + extra)) if extra else 0.0
            preds["own_mean"] = float(np.mean([preds["own"]] + extra))  # K-sample average (server --action-samples K)
            row = dict(a=a, b=b, k=k, fa=A["friction"], fb=B["friction"], gtA=A["pushes"][k - 1][1],
                       gtB=B["pushes"][k - 1][1], **preds)
            rows.append(row)
            print(
                f"A={a:3d}(f={row['fa']:.4f}) B={b:3d}(f={row['fb']:.4f}) push {k}: gtA {row['gtA']:.3f} gtB {row['gtB']:.3f} | "
                f"own {row['own']:.3f} swap {row['swap']:.3f} anchor {row['anchor']:.3f}  [{time.time()-t0:.0f}s]",
                flush=True,
            )

    own = np.array([r["own"] for r in rows]); swap = np.array([r["swap"] for r in rows])
    anc = np.array([r["anchor"] for r in rows])
    gA = np.array([r["gtA"] for r in rows]); gB = np.array([r["gtB"] for r in rows])
    d_gt, d_pred = gB - gA, swap - own
    slope = float(np.dot(d_gt, d_pred) / max(np.dot(d_gt, d_gt), 1e-9))
    seed_std = float(np.mean([r["own_std"] for r in rows]))
    own_mean = np.array([r["own_mean"] for r in rows])
    print(
        f"\n{len(rows)} probes | MAE own-vs-gtA {np.abs(own-gA).mean():.3f} | anchor-vs-gtA {np.abs(anc-gA).mean():.3f} | "
        f"swap-vs-gtA {np.abs(swap-gA).mean():.3f} | swap-vs-gtB {np.abs(swap-gB).mean():.3f} | "
        f"mean |gtB-gtA| {np.abs(d_gt).mean():.3f} | history-following slope {slope:.2f} "
        f"| corr(own, gtA) {np.corrcoef(own, gA)[0,1]:.2f} | bias own-gtA {np.mean(own-gA):+.3f} | seed std {seed_std:.4f} "
        f"| MAE {args.n_seeds}-sample-mean vs gtA {np.abs(own_mean-gA).mean():.4f} (single {np.abs(own-gA).mean():.4f})"
    )


if __name__ == "__main__":
    main()
