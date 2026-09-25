# MemMimic (Gated Memory Policy simulator) bridge

Evaluates OpenWAM / Memory-OpenWAM checkpoints trained with `dataloader=memmimic`
on the MemMimic tasks of [real-stanford/gated-memory-policy](https://github.com/real-stanford/gated-memory-policy)
(`mujoco-env`), first of all Iterative Pushing (`push_cube`).

The simulator talks robotmq to a policy server. `robotmq_policy_server.py` answers
that protocol with an OpenWAM `JointInferenceEngine` behind it (no WebSocket hop):

| topic | what the server does |
| --- | --- |
| `policy_config` | tells the rollout script how many past frames to send (`image_indices`), proprio layout, absolute pose |
| `policy_inference` | per env: rebuild the memory prefix from the stride-4 frames, run `engine.generate`, return the 32-step 10-D chunk |
| `policy_reset` / `done_rollout` / `export_recorded_data` | clear per-episode state / acknowledge |

10-D pose = xyz + rot6d + gripper on both sides; GMP's rot6d is the first two **rows**
of R, OpenWAM's the first two **columns** — converted both ways in the server
(`tests/test_memmimic_bridge.py`).

## Run

```bash
# 100 episodes (seeds 10005..), 6 parallel envs, OSMesa rendering, policy on cuda:0 / PPU
bash benchmarks/memmimic/run_push_cube_eval.sh <ckpt_dir> <run_name> [episodes=100] [env_num=6] [hydra overrides]
# outputs: /mnt/cpfs/workspace/memory-openwam/outputs/eval/<run_name>/{policy_server.log,rollout.log,rollout/}
```

Env knobs: `PORT`, `DEVICE`, `CKPT_NAME`, `START_SEED`, `OUT_ROOT`, `MUJOCO_GL` (egl on NVIDIA),
`SERVER_ARGS` (e.g. `--denoise-steps 5`), `POLICY_INFERENCE_TIMEOUT_S`.

Requirements: `robotmq` in the OpenWAM venv (`uv pip install robotmq`), GMP's `mujoco-env`
conda env (`/mnt/cpfs/workspace/tools/miniforge3/envs/mujoco-env`), and the GMP repo
patched so `POLICY_INFERENCE_TIMEOUT_S` overrides the 5 s request timeout
(`mujoco-env/env/modules/agents/manipulation_policy_parallel_agent.py`).

## Train/deploy alignment (Memory-OpenWAM)

`action_execution_horizon=16` = 4 stride-4 frames = one VAE latent = one gist, and
`task.render_image_indices=[-13,-9,-5,-1]` makes the simulator render exactly those
frames, so each request appends `s_{4c+1..4c+4}` and the server-side history is the
training layout `s_0 .. s_{4c}`. Plain OpenWAM checkpoints (no memory) use the same
launcher; the server then asks for the latest frame only.

## Offline bridge check

```bash
python benchmarks/memmimic/replay_check.py --ckpt-dir <ckpt_dir> --episodes 0 250
```

Replays training episodes through the exact live inference path and prints the
per-dim MAE between predicted chunks and the recorded actions. A checkpoint that
fits its training set must reproduce them (push_cube: y MAE ≈ 5 mm for a 25 cm
per-chunk motion); if it does not, suspect the bridge before the policy.
