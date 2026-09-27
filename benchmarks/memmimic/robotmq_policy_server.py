#!/usr/bin/env python
"""robotmq policy server bridging OpenWAM / Memory-OpenWAM checkpoints to the
MemMimic simulator (real-stanford/gated-memory-policy ``mujoco-env``).

The simulator's ``ManipulationPolicyParallelAgent`` speaks the protocol of GMP's
``imitation_learning/envs/policy_server.py``; this server answers the same topics
with an OpenWAM engine behind them:

  policy_config           -> dict. ``scripts/rollout_policy_parallel.py`` reads
                             workspace.model.{image_length, image_indices,
                             proprio_length, proprio_indices} and
                             workspace.train_dataset.{use_relative_pose, robot_num}.
  policy_inference        -> {"third_person_camera": (B, n_img, 3, H, W) uint8,
                              "robot0_10d": (B, 1, 10) float, "episode_idx": [B]}
                             reply {"action0_10d": (B, T_action, 10) float32}
  policy_reset            -> reply True
  done_rollout /
  export_recorded_data    -> acknowledged (nothing is recorded server-side)

10-D pose = xyz + rot6d + gripper on both sides, but the rot6d conventions differ:
GMP stores the first two ROWS of the rotation matrix, OpenWAM's memmimic reader the
first two COLUMNS (``quat_xyzw_to_rot6d``). Both directions are converted here.

Memory-OpenWAM: the memory prefix is rebuilt at every replan from the frames the
simulator rendered at control steps 4k. With ``action_execution_horizon=16`` and
``task.render_image_indices=[-13,-9,-5,-1]`` every request carries the 4 new
stride-4 frames, so the server-side history is exactly the training layout
``s_0 .. s_{4c}`` (one VAE latent / gist per 4 frames). The first request of an
episode (all-identical initial observations) contributes only the anchor ``s_0``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

logger = logging.getLogger("memmimic_policy_server")

POSE10_DIM = 10
ROT6D = slice(3, 9)


# ----------------------------------------------------------------- rot6d bridge
def _gram_schmidt(a1: np.ndarray, a2: np.ndarray):
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return b1, b2, b3


def gmp_rot6d_to_openwam(r6: np.ndarray) -> np.ndarray:
    """GMP ``[R[0,:], R[1,:]]`` -> OpenWAM ``[R[:,0], R[:,1]]``."""
    r0, r1, r2 = _gram_schmidt(r6[..., 0:3], r6[..., 3:6])
    rot = np.stack([r0, r1, r2], axis=-2)  # rot[..., i, :] = row i
    return np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)


def openwam_rot6d_to_gmp(r6: np.ndarray) -> np.ndarray:
    """OpenWAM ``[R[:,0], R[:,1]]`` -> GMP ``[R[0,:], R[1,:]]``."""
    c0, c1, c2 = _gram_schmidt(r6[..., 0:3], r6[..., 3:6])
    rot = np.stack([c0, c1, c2], axis=-1)  # rot[..., :, j] = column j
    return np.concatenate([rot[..., 0, :], rot[..., 1, :]], axis=-1)


def gmp10_to_openwam10(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.concatenate([p[..., :3], gmp_rot6d_to_openwam(p[..., ROT6D]), p[..., 9:10]], axis=-1).astype(np.float32)


def openwam10_to_gmp10(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.concatenate([p[..., :3], openwam_rot6d_to_gmp(p[..., ROT6D]), p[..., 9:10]], axis=-1).astype(np.float32)


# --------------------------------------------------------------------- server
class MemMimicPolicyServer:
    def __init__(
        self,
        *,
        ckpt_dir: str,
        ckpt_name: str | None,
        device: str,
        endpoint: str,
        deploy_cfg_path: str | None,
        denoise_steps: int | None,
        compile_enabled: bool,
        seed: int,
    ):
        import robotmq
        import torch
        from omegaconf import OmegaConf

        from openwam.deploy.engine import JointInferenceEngine
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.server import merge_deploy_cfg

        self._robotmq = robotmq
        self._torch = torch
        training_cfg, arch = load_from_checkpoint_dir(ckpt_dir, device=device, ckpt_name=ckpt_name)
        deploy_cfg = OmegaConf.load(deploy_cfg_path) if deploy_cfg_path else OmegaConf.create({})
        OmegaConf.update(deploy_cfg, "optimization.compile.enabled", bool(compile_enabled), merge=False)
        OmegaConf.update(deploy_cfg, "optimization.decode_video", False, merge=False)
        if denoise_steps is not None:
            OmegaConf.update(deploy_cfg, "inference.denoise_steps", int(denoise_steps), merge=False)
        self.cfg = merge_deploy_cfg(training_cfg, deploy_cfg)
        self.engine = JointInferenceEngine(cfg=self.cfg, architecture=arch)

        dl = training_cfg.dataloader
        self.prompt = str(OmegaConf.select(dl, "prompt", default="") or "")
        self.height = int(OmegaConf.select(dl, "height", default=256))
        self.width = int(OmegaConf.select(dl, "width", default=320))
        self.video_stride = int(OmegaConf.select(dl, "video_stride", default=4) or 4)
        self.num_frames = int(OmegaConf.select(dl, "num_frames", default=33))
        self.action_len = self.num_frames - 1
        self.memory_enabled = bool(getattr(arch, "memory_enabled", False))
        self.max_latents = int(arch.memory_cfg.max_latents) if self.memory_enabled else 0
        # Action history (memory.action_history): normalize the executed commands with the
        # training reader's action stats (the checkpoint's normalization_stats.npy "action" block).
        self.action_history = bool(self.memory_enabled and getattr(arch.memory_cfg, "action_history", False))
        self._action_stats = None
        self._norm_mode = OmegaConf.select(dl, "normalize_mode", default="min-max")
        # action_repr=delta: the model emits xyz increments; integrate them from the proprio.
        self.action_repr = str(OmegaConf.select(dl, "action_repr", default="absolute") or "absolute")
        if self.action_history:
            import os

            stats = np.load(os.path.join(ckpt_dir, "normalization_stats.npy"), allow_pickle=True).item()
            # Same absolute-pose block as the reader's _memory_action_stats.
            self._action_stats = stats["state"] if self.action_repr == "delta" else stats["action"]
        from openwam.deploy.memory_buffer import FRAMES_PER_LATENT

        self.frames_per_latent = int(FRAMES_PER_LATENT)
        # One replan per latent boundary keeps train/deploy memory layout identical.
        self.exec_horizon = self.video_stride * self.frames_per_latent
        self.denoise_steps = int(self.cfg.inference.denoise_steps)
        self.seed = int(seed)
        self.ckpt_dir = str(ckpt_dir)
        self.ckpt_name = ckpt_name or "latest"
        self.endpoint = endpoint
        self.episodes: dict[int, dict] = {}
        self.request_count = 0
        self.memory_ablation = "none"
        self.action_samples = 1
        self.dump_dir = None
        # dim -> value written into the action history instead of the model's raw output (see --history-fixed-dims)
        self.history_fixed_dims: dict[int, float] = {}
        # simulator action-smoothing weight replayed on the xyz written into the action history (see --history-smooth)
        self.history_smooth = 0.0
        # Phase 2 (memory-openwam/docs/07): "recompute" re-encodes the history and re-runs the memory
        # prefix at every denoising step (the original path); "prefix_once" runs it once per request;
        # "streaming" keeps per-episode K/V and writes one latent per request. stream_vae encodes only
        # the newly completed chunk (always on for the two KV modes).
        self.memory_inference = "recompute"
        self.stream_vae = False
        self.timing_path = None

        self.server = robotmq.RMQServer("policy_server", endpoint)
        for topic in ("new_checkpoint_loaded", "eval_config"):
            self.server.add_topic(topic, 3600)
        for topic in ("policy_config", "policy_reset", "policy_inference", "done_rollout", "export_recorded_data"):
            self.server.add_topic(topic, 60)
        logger.info(
            "MemMimic policy server on %s: ckpt=%s/%s memory=%s max_latents=%d prompt=%r frame=%dx%d "
            "chunk=%d exec_horizon=%d denoise_steps=%d",
            endpoint, ckpt_dir, self.ckpt_name, self.memory_enabled, self.max_latents, self.prompt,
            self.width, self.height, self.action_len, self.exec_horizon, self.denoise_steps,
        )

    # ------------------------------------------------------------ protocol
    def policy_config(self) -> dict:
        if self.memory_enabled:
            image_length = self.exec_horizon
            image_indices = [-(k * self.video_stride) for k in reversed(range(self.frames_per_latent))]
        else:
            image_length, image_indices = 1, [0]
        return {
            "policy_name": "openwam_memory" if self.memory_enabled else "openwam",
            "project_name": "memory-openwam",
            "task_name": "push_cube",
            "run_name": Path(self.ckpt_dir).parent.name,
            "ckpt_path": f"{self.ckpt_dir}/{self.ckpt_name}",
            "epoch": 0,
            "workspace": {
                "model": {
                    "image_length": int(image_length),
                    "image_indices": [int(i) for i in image_indices],
                    "proprio_length": 1,
                    "proprio_indices": [0],
                    "action_prediction_horizon": int(self.action_len),
                    "action_execution_horizon": int(self.exec_horizon),
                },
                "train_dataset": {
                    "use_relative_pose": False,
                    "robot_num": 1,
                    "action_indices": list(range(self.action_len)),
                },
            },
            "openwam": {
                "prompt": self.prompt,
                "height": self.height,
                "width": self.width,
                "video_stride": self.video_stride,
                "num_frames": self.num_frames,
                "memory_enabled": self.memory_enabled,
                "max_latents": self.max_latents,
                "denoise_steps": self.denoise_steps,
                "rot6d_convention": "openwam=columns, gmp=rows (converted in server)",
            },
        }

    def configure_memory_inference(self, mode: str, *, stream_vae: bool = False, timing_path: str | None = None) -> None:
        if mode == "auto":
            # Evaluation default (2026-09-27): the validated streaming path whenever it applies
            # (docs/07 V1-V3: E12 84 vs 85/100, 0.43 s/request), otherwise the original recompute
            # path exactly as before (no memory, eval-time ablation, or memory.action_history).
            if self.memory_enabled and self.memory_ablation == "none" and not self.action_history:
                mode = "streaming"
            else:
                mode, stream_vae = "recompute", False
        if mode not in ("recompute", "prefix_once", "streaming"):
            raise ValueError(f"unknown memory inference mode {mode!r}")
        if mode != "recompute" or stream_vae:
            if not self.memory_enabled:
                raise ValueError("--memory-inference / --stream-vae need a memory checkpoint.")
            if self.memory_ablation != "none":
                raise ValueError("memory ablations run on the recompute path only (they reorder raw frames).")
        if mode != "recompute" and self.action_history:
            raise NotImplementedError("the KV modes do not support memory.action_history.")
        self.memory_inference = mode
        self.stream_vae = bool(stream_vae or mode != "recompute")
        self.timing_path = timing_path
        self.episodes.clear()

    def _to_pil(self, frames: np.ndarray) -> list:
        """``(n, 3, H, W)`` or ``(n, H, W, 3)`` uint8/float -> list of resized RGB PIL."""
        frames = np.asarray(frames)
        if frames.ndim != 4:
            raise ValueError(f"expected (n, 3, H, W) camera frames, got {frames.shape}")
        if frames.shape[1] == 3 and frames.shape[-1] != 3:
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.dtype != np.uint8:
            frames = np.clip(frames * (255.0 if frames.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        out = []
        for f in frames:
            img = Image.fromarray(np.ascontiguousarray(f), mode="RGB")
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), Image.BILINEAR)  # same as MemMimicDataset._frame
            out.append(img)
        return out

    def _episode_state(self, episode_idx: int) -> dict:
        st = self.episodes.get(episode_idx)
        if st is None:
            st = {"frames": [], "n_req": 0}
            self.episodes[episode_idx] = st
        return st

    def _encode_history(self, st: dict, c: int):
        """Streaming VAE: encode the chunks completed since the last request (anchor first) into the
        episode's MemoryKVStream, which also holds the latents for the recompute + stream_vae path."""
        from openwam.deploy.streaming_vae import StreamingVAEEncoder
        from openwam.model.architectures.utils.memory_stream import MemoryKVStream

        if "kv" not in st:
            mode = "prefix_once" if self.memory_inference == "recompute" else self.memory_inference
            st["kv"] = MemoryKVStream(mode)
            st["vae"] = StreamingVAEEncoder(self.engine.architecture.video_backbone)
        kv, vae, F = st["kv"], st["vae"], self.frames_per_latent
        t0 = time.time()
        while len(kv.latents) <= c:
            k = len(kv.latents)
            chunk = st["frames"][:1] if k == 0 else st["frames"][1 + F * (k - 1) : 1 + F * k]
            kv.append_latent(vae.append(chunk))
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        st["t_vae"] = time.time() - t0
        return kv

    def _infer_one(self, episode_idx: int, frames: list, pose10_gmp: np.ndarray) -> np.ndarray:
        t_start = time.time()
        if self.timing_path and self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        st = self._episode_state(episode_idx)
        st["t_vae"] = 0.0
        if st["n_req"] == 0:
            st["frames"] = [frames[-1]]  # initial observations are copies of s_0
        else:
            new = frames[-self.frames_per_latent :]
            if len(new) != self.frames_per_latent:
                logger.warning(
                    "episode %d request %d carries %d rendered frames (expected %d); memory may lag.",
                    episode_idx, st["n_req"], len(new), self.frames_per_latent,
                )
            st["frames"].extend(new)
        st["n_req"] += 1
        cur = st["frames"][-1]

        conditions = {
            "prompt": self.prompt,
            "first_frame_image": [cur],
            "proprio": gmp10_to_openwam10(pose10_gmp),
            "seed": self.seed + int(episode_idx) * 1000 + st["n_req"],
        }
        if self.memory_enabled:
            from openwam.deploy.memory_buffer import memory_times_for

            F = self.frames_per_latent
            c = (len(st["frames"]) - 1) // F
            times = memory_times_for(c, self.max_latents)
            mem_frames = list(st["frames"][: c * F + 1])
            executed = list(st.get("executed", []))  # one (exec_horizon, 10) normalized chunk per past request
            # Eval-time ablations (DESIGN §8): what does the policy lose without (ordered) history?
            if self.memory_ablation == "anchor_only":
                mem_frames, times = mem_frames[:1], [0]
            elif self.memory_ablation == "shuffle" and c > 1:
                perm = np.random.default_rng(self.seed + int(episode_idx) * 1000 + st["n_req"]).permutation(c)
                chunks = [mem_frames[1 + F * k : 1 + F * (k + 1)] for k in range(c)]
                mem_frames = mem_frames[:1] + [f for k in perm for f in chunks[k]]
                if executed:
                    executed = [executed[k] for k in perm]
            if self.memory_inference == "recompute" and not self.stream_vae:
                conditions.update(memory_video=mem_frames, memory_times=times, memory_context_index=c)
            else:
                kv = self._encode_history(st, c)
                if self.memory_inference == "recompute":
                    conditions.update(
                        memory_latents=kv.latents_at(list(range(c + 1))), memory_times=times, memory_context_index=c
                    )
                else:
                    conditions.update(memory_stream=kv, memory_times=times, memory_context_index=c)
            if self.action_history:
                from openwam.dataloader.memmimic import memory_action_history

                if len(executed) != c:
                    raise RuntimeError(f"episode {episode_idx}: {len(executed)} executed chunks for context {c}.")
                past = np.concatenate(executed, 0) if executed else np.zeros((0, POSE10_DIM), np.float32)
                acts = memory_action_history(past, times, self.exec_horizon)
                if self.memory_ablation == "zero_actions":
                    acts = np.zeros_like(acts)
                conditions["memory_actions"] = acts
        if self.action_samples > 1:
            # Average K flow samples (different seeds) of the same request: shrinks the sampling
            # spread of the push speed by sqrt(K) at K x the inference cost.
            outs = []
            for j in range(self.action_samples):
                cond = dict(conditions, seed=int(conditions["seed"]) + 7919 * j)
                outs.append(np.asarray(self.engine.generate(cond)["actions"], dtype=np.float32))
            actions = np.mean(outs, axis=0)
        else:
            out = self.engine.generate(conditions)
            actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != POSE10_DIM:
            raise RuntimeError(f"expected ({self.action_len}, 10) raw actions, got {actions.shape}")
        actions = self.to_absolute(actions, conditions["proprio"])
        if self.dump_dir and int(episode_idx) < 3 and st["n_req"] <= 12:
            # Debug: the exact inputs/outputs of this request, to diff against the training data.
            import os

            os.makedirs(self.dump_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.dump_dir, f"ep{int(episode_idx)}_req{st['n_req']:02d}.npz"),
                frames=np.stack([np.asarray(f) for f in conditions.get("memory_video", [cur])]),
                cur=np.asarray(cur),
                proprio=np.asarray(conditions["proprio"], dtype=np.float32),
                pose10_gmp=np.asarray(pose10_gmp, dtype=np.float32),
                memory_times=np.asarray(conditions.get("memory_times", [0])),
                context_index=int(conditions.get("memory_context_index", 0)),
                seed=int(conditions["seed"]),
                actions=actions,
            )
        if self.action_history:
            from openwam.dataloader.utils.normalization import apply_normalization

            # The simulator executes the first exec_horizon commands of this chunk before the next request.
            executed = actions[: self.exec_horizon].copy()
            if self.history_smooth > 0:
                # What the simulator actually runs (GMP ManipulationPolicyParallelAgent): s_i = (1 - 2w) a_i +
                # w a_{i-1} + w a_{i+1}, with a_{-1} = the last command it executed; the last command of the
                # prediction horizon is left as is.
                w, n = self.history_smooth, len(executed)
                prev = st.get("last_exec_xyz", np.asarray(conditions["proprio"], np.float32).reshape(-1, POSE10_DIM)[-1, :3])
                before = np.concatenate([np.asarray(prev, np.float32)[None], actions[: n - 1, :3]], 0)
                after = actions[1 : n + 1, :3]
                m = len(after)
                executed[:m, :3] = (1 - 2 * w) * actions[:m, :3] + w * before[:m] + w * after
                st["last_exec_xyz"] = executed[-1, :3].copy()
            for d, v in self.history_fixed_dims.items():
                executed[:, d] = v
            st.setdefault("executed", []).append(
                apply_normalization(executed, self._action_stats, self._norm_mode).astype(np.float32)
            )
        if self.timing_path:
            import json

            rec = {
                "episode": int(episode_idx),
                "request": int(st["n_req"]),
                "ctx": int(conditions.get("memory_context_index", 0)),
                "mode": self.memory_inference,
                "stream_vae": bool(self.stream_vae or self.memory_inference != "recompute"),
                "t_total": time.time() - t_start,
                "t_vae_stream": float(st.get("t_vae", 0.0)),
            }
            if self._torch.cuda.is_available():
                rec["peak_mem_gb"] = self._torch.cuda.max_memory_allocated() / 2**30
            with open(self.timing_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return openwam10_to_gmp10(actions)

    def to_absolute(self, actions: np.ndarray, proprio_openwam10: np.ndarray) -> np.ndarray:
        """Model output (raw units) -> absolute OpenWAM-convention 10-D commands."""
        if self.action_repr != "delta":
            return actions
        from openwam.dataloader.memmimic import delta_to_pose10

        return delta_to_pose10(actions, np.asarray(proprio_openwam10, dtype=np.float32))

    def policy_inference(self, raw: dict) -> dict:
        ep = raw["episode_idx"]
        squeeze = False
        if isinstance(ep, (int, np.integer)) or (isinstance(ep, np.ndarray) and ep.shape == ()):
            squeeze = True
            ep = [int(ep)]
        ep = [int(e) for e in np.asarray(ep).reshape(-1)]
        imgs = np.asarray(raw["third_person_camera"])
        pose = np.asarray(raw["robot0_10d"], dtype=np.float32)
        if squeeze:
            imgs, pose = imgs[None], pose[None]
        if imgs.shape[0] != len(ep) or pose.shape[0] != len(ep):
            raise ValueError(f"batch mismatch: episodes={len(ep)} images={imgs.shape} pose={pose.shape}")
        acts = []
        t0 = time.time()
        for b, e in enumerate(ep):
            acts.append(self._infer_one(e, self._to_pil(imgs[b]), pose[b, -1]))
        # Every running episode is in every request: drop states of finished ones.
        for e in list(self.episodes):
            if e not in ep:
                self.episodes.pop(e, None)
        self.request_count += 1
        logger.info(
            "request %d: %d env(s) in %.2fs (%.2fs/env); episodes=%s ctx=%s",
            self.request_count, len(ep), time.time() - t0, (time.time() - t0) / max(1, len(ep)), ep,
            [(len(self.episodes[e]["frames"]) - 1) // self.frames_per_latent for e in ep],
        )
        out = {"action0_10d": np.stack(acts).astype(np.float32)}
        if squeeze:
            out = {k: v[0] for k, v in out.items()}
        return out

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        rmq = self._robotmq
        logger.info("Checkpoint loaded. Waiting for environment requests")
        print("Checkpoint loaded. Waiting for environment requests", flush=True)
        while True:
            data, topic = self.server.wait_for_request(timeout_s=0.1)
            if not topic:
                continue
            try:
                if topic == "policy_inference":
                    reply = self.policy_inference(rmq.deserialize(data))
                elif topic == "policy_config":
                    reply = self.policy_config()
                elif topic == "policy_reset":
                    self.episodes.clear()
                    reply = True
                elif topic == "done_rollout":
                    self.episodes.clear()
                    reply = True
                elif topic == "export_recorded_data":
                    reply = "nothing recorded server-side"
                else:
                    reply = f"unknown topic {topic}"
            except Exception:  # noqa: BLE001 - report to the client like GMP does
                import traceback

                reply = traceback.format_exc()
                logger.error("error handling %s:\n%s", topic, reply)
            self.server.reply_request(topic=topic, data=rmq.serialize(reply))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--ckpt-name", default=None, help="checkpoint_step_N.safetensors (default: latest)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--endpoint", default="tcp://0.0.0.0:18765")
    ap.add_argument("--deploy-config", default=str(PROJECT_ROOT / "configs" / "deploy.yaml"))
    ap.add_argument("--denoise-steps", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="enable torch.compile fast paths (off by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-dir", default=None, help="debug: save inputs/outputs of the first requests of episodes 0-2")
    ap.add_argument("--action-samples", type=int, default=1, help="average this many sampled chunks per request")
    ap.add_argument(
        "--memory-inference",
        default="recompute",
        choices=["recompute", "prefix_once", "streaming", "auto"],
        help="memory prefix at inference (docs/07): recompute every denoising step (original), once per request, "
        "a per-episode KV stream, or auto (= streaming unless the checkpoint has no memory / uses action history / "
        "an eval-time ablation is requested, in which case recompute)",
    )
    ap.add_argument(
        "--history-fixed-dims",
        default="",
        help="action history: write these OpenWAM-10 dims as fixed values instead of the model's raw output, e.g. "
        "'0=0.1,2=0.365,9=0.08' for push_cube (x / z / gripper are constant in every recorded command and the sim "
        "only moves y). Delta models normalize the history with the state stats, whose x / z / gripper ranges are "
        "sub-mm, so raw output jitter would otherwise be clipped to +-1 there.",
    )
    ap.add_argument(
        "--history-smooth",
        type=float,
        default=0.0,
        help="action history: record the xyz the simulator actually executes after its action smoothing "
        "(GMP push_cube task.agent.smoothen_action_weight = 0.2) instead of the raw commands. The expert demos were "
        "collected without smoothing, so in training the recorded commands are exactly what moved the cube.",
    )
    ap.add_argument("--stream-vae", action="store_true", help="encode only new history chunks (implied by the KV modes)")
    ap.add_argument("--timing-jsonl", default=None, help="append per-request timing / peak memory records here")
    ap.add_argument(
        "--memory-ablation",
        default="none",
        choices=["none", "anchor_only", "shuffle", "zero_actions"],
        help="eval-time memory ablation: drop the history (anchor only), shuffle its chunk order, or zero the action history",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = MemMimicPolicyServer(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        device=args.device,
        endpoint=args.endpoint,
        deploy_cfg_path=args.deploy_config,
        denoise_steps=args.denoise_steps,
        compile_enabled=args.compile,
        seed=args.seed,
    )
    server.memory_ablation = args.memory_ablation
    server.action_samples = max(1, int(args.action_samples))
    server.dump_dir = args.dump_dir
    for item in filter(None, (x.strip() for x in args.history_fixed_dims.split(","))):
        d, v = item.split("=")
        server.history_fixed_dims[int(d)] = float(v)
    if not 0.0 <= args.history_smooth < 0.5:
        raise ValueError("--history-smooth must be in [0, 0.5)")
    server.history_smooth = float(args.history_smooth)
    server.configure_memory_inference(args.memory_inference, stream_vae=args.stream_vae, timing_path=args.timing_jsonl)
    logger.info(
        "memory ablation: %s | action samples averaged: %d | history fixed dims: %s | history smooth: %s | "
        "memory inference: %s (stream VAE %s)",
        server.memory_ablation, server.action_samples, server.history_fixed_dims or "none", server.history_smooth,
        server.memory_inference, server.stream_vae,
    )
    server.run()


if __name__ == "__main__":
    main()
