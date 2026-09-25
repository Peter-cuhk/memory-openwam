#!/usr/bin/env python
"""Per-push analysis of MemMimic push_cube rollouts (or the training zarr).

For every episode: hidden sliding friction, the peak robot speed of each push
(the policy's only control knob) and where the cube stopped (target y = +0.15,
tolerance ±0.05, success = last 3 pushes inside). Shows whether a policy adapts
its push speed across trials — the whole point of the task's hidden friction.

  python benchmarks/memmimic/analyze_push_cube.py <rollout_dir_or_zarr> [--episodes 0 1 2] [--all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TARGET_Y, TOL = 0.15, 0.05


def split_pushes(tcp_y: np.ndarray, cube_y: np.ndarray, dt: float = 0.1, min_peak: float = 0.15):
    """Forward (+y) robot motion segments with peak speed >= min_peak = pushes.
    Returns [(peak_speed, cube_landing_y)] with landing read 6 s after the push ends."""
    v = np.gradient(tcp_y, dt)
    fwd = v > 0.05
    out, i, n = [], 0, len(v)
    while i < n:
        if fwd[i]:
            j = i
            while j < n and fwd[j]:
                j += 1
            peak = float(v[i:j].max())
            if peak >= min_peak:
                out.append((peak, float(cube_y[min(n - 1, j + 60)])))
            i = j
        else:
            i += 1
    return out


def episode_rows(root, eps):
    rows = []
    for e in eps:
        g = root[f"episode_{e}"]
        tcp = np.asarray(g["robot0_tcp_xyz_wxyz"])[:, 1]
        cube = np.asarray(g["obj0_object_pose_xyz_wxyz"])[:, 1]
        m = min(len(tcp), len(cube))
        attrs = dict(g.attrs)
        cfg = attrs.get("episode_config", {}) or {}
        rows.append(
            {
                "episode": e,
                "friction": cfg.get("sliding_friction"),
                "success": bool(attrs.get("is_successful", attrs.get("final_reward", 0) >= 1.0)),
                "pushes": split_pushes(tcp[:m], cube[:m]),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="eval run dir, its rollout/ dir, or an episode_data.zarr")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="print every episode (default: first 10)")
    args = ap.parse_args()
    import zarr

    p = Path(args.path)
    for cand in (p, p / "episode_data.zarr", p / "rollout" / "episode_data.zarr"):
        if cand.name.endswith(".zarr") and cand.exists():
            p = cand
            break
    root = zarr.open(str(p), mode="r")
    eps = sorted(int(k.split("_")[1]) for k in root.keys() if k.startswith("episode_"))
    if args.episodes:
        eps = args.episodes
    rows = episode_rows(root, eps)
    shown = rows if args.all else rows[:10]
    for r in shown:
        fr = "?" if r["friction"] is None else f"{r['friction']:.4f}"
        print(
            f"ep{r['episode']:4d} friction={fr} {'OK ' if r['success'] else 'FAIL'} "
            f"speed=" + " ".join(f"{s:.2f}" for s, _ in r["pushes"]) + " | landing=" + " ".join(f"{y:+.2f}" for _, y in r["pushes"])
        )
    n = len(rows)
    succ = sum(r["success"] for r in rows)
    err_first = [abs(r["pushes"][0][1] - TARGET_Y) for r in rows if r["pushes"]]
    err_last3 = [abs(y - TARGET_Y) for r in rows for _, y in r["pushes"][-3:]]
    inside_last3 = [abs(y - TARGET_Y) <= TOL for r in rows for _, y in r["pushes"][-3:]]
    print(
        f"\n{n} episodes: success {succ}/{n} = {succ / max(n, 1):.2f} | first-push |err| {np.mean(err_first):.3f} m"
        f" | last-3 |err| {np.mean(err_last3):.3f} m | last-3 inside tol {np.mean(inside_last3):.2f}"
    )


if __name__ == "__main__":
    main()
