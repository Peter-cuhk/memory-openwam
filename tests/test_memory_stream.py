"""Memory-OpenWAM Phase 2: KV-cached memory prefix == parallel training prefix.

Tiny random Wan2.2-TI2V + ActionDiT through the real DualSystemMoTDriver, fp32:

1. ``prefix_once`` and ``streaming`` reproduce the parallel prefix's video and action predictions
   (gist gate open, proprio hidden from memory rows, both mask modes, with/without ``recent_kv``);
2. ``streaming`` stays exact across consecutive requests and repeated denoising steps, and falls
   back to an exact rebuild once the layout is truncated;
3. the streaming VAE encoder gives the latents of a whole-clip encode.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO, MUTUAL
from openwam.model.architectures.utils.memory_stream import MemoryKVStream
from openwam.model.architectures.utils.memory_tokens import MemoryConfig, MemoryTokens
from openwam.deploy.memory_buffer import memory_times_for
from tests.test_memory_layout import _tiny_backbone

B, Z, FW, H, W = 1, 4, 3, 4, 4  # window: F_cur + 2 noisy latents on a 2x2 token grid
S_ACT, A_DIM = 5, 7


def _setup(*, recent_kv: bool, mode: str):
    vb, dit = _tiny_backbone()
    torch.manual_seed(3)
    ab = ActionDiT(
        action_dim=A_DIM,
        dim=64,
        ffn_dim=96,
        num_heads=2,
        num_layers=2,
        video_dim=64,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
        text_dim=16,
    )
    cfg = MemoryConfig(enabled=True, gist_tokens=3, use_gist_gate=True, gate_init=0.0, recent_kv=recent_kv, max_latents=4)
    memory = MemoryTokens(cfg, dim=64, num_layers=2)
    with torch.no_grad():
        memory.gist_gate.copy_(torch.tensor([0.7, -1.3]))  # an open, layer-dependent gate
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False, attention_mask_mode=mode)
    for m in (dit, ab, memory):
        m.eval()
    return vb, ab, memory, driver


TEXT = torch.randn(B, 5, 16, generator=torch.Generator().manual_seed(99))  # one prompt per episode


def _inputs(seed: int):
    """Per-request inputs: new window, actions and proprio token; the episode's text is fixed."""
    g = torch.Generator().manual_seed(seed)
    window = torch.randn(B, Z, FW, H, W, generator=g)
    proprio_token = torch.randn(B, 1, 16, generator=g)  # memory rows must not read it
    context = torch.cat([TEXT, proprio_token], dim=1)
    context_mask = torch.ones(B, 6, dtype=torch.bool)
    context_mask[:, 3] = False  # a padded text position
    actions = torch.randn(B, S_ACT, A_DIM, generator=g)
    return dict(window=window, context=context, context_mask=context_mask, actions=actions)


@torch.no_grad()
def _forward(vb, ab, memory, driver, inp, *, v_t: float, a_t: float, **mem):
    window = inp["window"]
    vstate = vb.prepare(
        latents=window,
        timestep=torch.tensor([v_t]),
        context=inp["context"],
        context_mask=inp["context_mask"],
        fuse_vae_embedding_in_latents=True,
        first_frame_latents=window[:, :, 0:1],
        memory_tokens=memory,
        memory_ctx_hide_last_n=1,
        **mem,
    )
    vstate.extras["memory_mask_cfg"] = memory.cfg.mask_cfg()
    vstate.extras["memory_gist_gate"] = memory.gate
    vstate.extras["memory_reader_sees_memory"] = None
    astate = ab.prepare_state(
        inp["actions"], torch.tensor([a_t]), context=inp["context"], context_mask=inp["context_mask"]
    )
    vstate, astate = driver.run_joint_loop(vstate, astate)
    return vb.finalize(vstate), ab.extract_prediction(astate)


def _check(a, b):
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("recent_kv", [False, True])
@pytest.mark.parametrize("mode", [ACTION_SEES_VIDEO, MUTUAL])
def test_prefix_once_and_streaming_match_parallel(recent_kv, mode):
    vb, ab, memory, driver = _setup(recent_kv=recent_kv, mode=mode)
    n_lat = 7  # c = 0..6 with max_latents 4 -> the last three requests are truncated
    history = torch.randn(B, Z, n_lat, H, W, generator=torch.Generator().manual_seed(11))
    stream = MemoryKVStream("streaming")
    for c in range(n_lat):
        stream.append_latent(history[:, :, c : c + 1])
        times = memory_times_for(c, memory.cfg.max_latents)
        inp = _inputs(100 + c)
        mem = dict(memory_times=[times], memory_context_index=[c])
        once = MemoryKVStream("prefix_once")
        for t in range(c + 1):
            once.append_latent(history[:, :, t : t + 1])
        # two denoising steps of the same request: the second reuses the cached K/V
        for v_t, a_t in ((900.0, 800.0), (350.0, 120.0)):
            ref = _forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_latents=history[:, :, times], **mem)
            _check(_forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_stream=once, **mem), ref)
            _check(_forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_stream=stream, **mem), ref)
    # 4 streamed requests wrote one slot each; the truncated ones were rebuilt, not streamed
    assert stream.stats["slots_written"] == memory.cfg.max_latents
    assert stream.stats["prefix_rebuilds"] == n_lat - memory.cfg.max_latents
    assert stream.stats["requests"] == n_lat


def test_streaming_rejects_a_changed_prompt():
    vb, ab, memory, driver = _setup(recent_kv=False, mode=ACTION_SEES_VIDEO)
    stream = MemoryKVStream("streaming")
    stream.append_latent(torch.randn(B, Z, 1, H, W))
    inp = _inputs(0)
    _forward(vb, ab, memory, driver, inp, v_t=500.0, a_t=500.0, memory_stream=stream, memory_times=[[0]], memory_context_index=[0])
    stream.append_latent(torch.randn(B, Z, 1, H, W))
    inp = _inputs(1)
    inp["context"] = inp["context"].clone()
    inp["context"][:, 0] += 1.0  # a different prompt mid-episode
    with pytest.raises(ValueError, match="text context"):
        _forward(
            vb, ab, memory, driver, inp, v_t=500.0, a_t=500.0, memory_stream=stream, memory_times=[[0, 1]], memory_context_index=[1]
        )


def test_memory_stream_changes_the_prediction():
    """Guard against a vacuous pass: the cached memory must actually be read."""
    vb, ab, memory, driver = _setup(recent_kv=False, mode=ACTION_SEES_VIDEO)
    history = torch.randn(B, Z, 3, H, W, generator=torch.Generator().manual_seed(5))
    inp = _inputs(7)
    mem = dict(memory_times=[[0, 1, 2]], memory_context_index=[2])
    s1, s2 = MemoryKVStream("prefix_once"), MemoryKVStream("prefix_once")
    for t in range(3):
        s1.append_latent(history[:, :, t : t + 1])
        s2.append_latent(history[:, :, t : t + 1] * (1.0 if t == 0 else -1.0))
    out1 = _forward(vb, ab, memory, driver, inp, v_t=500.0, a_t=500.0, memory_stream=s1, **mem)
    out2 = _forward(vb, ab, memory, driver, inp, v_t=500.0, a_t=500.0, memory_stream=s2, **mem)
    # equivalence noise above is ~1e-7; a different history must move both outputs far beyond it
    assert (out1[0] - out2[0]).abs().max() > 1e-5
    assert (out1[1] - out2[1]).abs().max() > 1e-5


def test_streaming_vae_matches_whole_clip_encode():
    from PIL import Image

    from openwam.deploy.streaming_vae import StreamingVAEEncoder
    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.model.video_backbone.wan.models.vae import VideoVAE38_

    torch.manual_seed(0)
    model = VideoVAE38_(dim=16, z_dim=48, dec_dim=16).eval()
    scale = [torch.randn(48), torch.rand(48) + 0.5]
    vae = types.SimpleNamespace(model=model, scale=scale)
    vb = types.SimpleNamespace(vae=vae, video_encoder=None, dtype=torch.float32, device=torch.device("cpu"))
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)) for _ in range(9)]

    pixels = wan_encode.preprocess_video(frames, encoder=None, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        full = model.encode(pixels, scale)  # (1, 48, 3, H', W')

    enc = StreamingVAEEncoder(vb)
    lat = [enc.append(frames[:1]), enc.append(frames[1:5]), enc.append(frames[5:9])]
    torch.testing.assert_close(torch.cat(lat, dim=2), full, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        enc.append(frames[:3])
    enc.reset()
    torch.testing.assert_close(enc.append(frames[:1]), full[:, :, :1], atol=1e-5, rtol=1e-5)
