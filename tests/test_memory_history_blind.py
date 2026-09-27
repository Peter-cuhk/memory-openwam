"""Experiment B, ``memory.history_video_blind``: history slots keep their times and action history but see only the
anchor (first-frame) latent. CPU, fp32, tiny random Wan2.2-TI2V + ActionDiT through the real DualSystemMoTDriver.

1. the helper replaces exactly the real history slots (anchor and padding untouched, anchor copy detached);
2. parallel prefix (training forward / deploy ``recompute``): with the switch on, the history images no longer
   reach the output while the action history (and the anchor) still do; the switch == feeding anchor copies;
3. ``prefix_once`` / ``streaming`` (incl. the truncated-layout rebuild) reproduce the blind parallel prefix;
4. the switch is off by default, parses from a config dict and composes as a Hydra ``+`` override.
"""

from __future__ import annotations

import pytest
import torch

from openwam.deploy.memory_buffer import memory_times_for
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO, MUTUAL
from openwam.model.architectures.utils.memory_stream import MemoryKVStream
from openwam.model.architectures.utils.memory_tokens import (
    MemoryConfig,
    MemoryTokens,
    blind_history_latents,
    parse_memory_config,
)
from tests.test_memory_layout import _tiny_backbone
from tests.test_memory_stream import A_DIM, B, FW, H, W, Z, _check, _forward, _inputs  # noqa: F401

STEPS, ADIM = 16, 10  # action-history steps per slot / width (the MemoryConfig defaults)


def _setup(*, blind: bool, action_history: bool = False, mode: str = ACTION_SEES_VIDEO, recent_kv: bool = False):
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
    cfg = MemoryConfig(
        enabled=True,
        gist_tokens=3,
        use_gist_gate=True,
        gate_init=0.0,
        recent_kv=recent_kv,
        max_latents=4,
        action_history=action_history,
        history_video_blind=blind,
    )
    memory = MemoryTokens(cfg, dim=64, num_layers=2)
    with torch.no_grad():
        memory.gist_gate.copy_(torch.tensor([0.7, -1.3]))  # an open, layer-dependent gate
        if action_history:  # the last layer is zero-init (no-op at step 0): give it weights so the history matters
            g = torch.Generator().manual_seed(17)
            last = memory.action_history_mlp[-1]
            last.weight.copy_(torch.randn(last.weight.shape, generator=g) * 0.05)
            last.bias.copy_(torch.randn(last.bias.shape, generator=g) * 0.05)
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False, attention_mask_mode=mode)
    for m in (dit, ab, memory):
        m.eval()
    return vb, ab, memory, driver


def _anchor_copies(history: torch.Tensor) -> torch.Tensor:
    return history[:, :, :1].expand_as(history).clone()


def _max_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# 1. helper
# ----------------------------------------------------------------------------


def test_helper_replaces_history_keeps_anchor_padding_and_detaches():
    lat = torch.randn(2, 4, 4, 2, 2, requires_grad=True)
    times = [[0, 3, 5, 6], [0, 9]]  # sample 1: two padded slots
    out = blind_history_latents(lat, times)
    for s in range(1, 4):
        torch.testing.assert_close(out[0, :, s], lat[0, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(out[1, :, 1], lat[1, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(out[:, :, 0], lat[:, :, 0], atol=0, rtol=0)  # anchor untouched
    torch.testing.assert_close(out[1, :, 2:], lat[1, :, 2:], atol=0, rtol=0)  # padding untouched
    out.sum().backward()
    grad = lat.grad
    assert torch.all(grad[:, :, 0] == 1)  # anchor copies are detached: the anchor only gets its own gradient
    assert torch.all(grad[0, :, 1:] == 0) and torch.all(grad[1, :, 1] == 0)  # replaced slots: no gradient
    assert torch.all(grad[1, :, 2:] == 1)  # padding passes through
    with pytest.raises(ValueError):
        blind_history_latents(lat, [[1, 2], [0]])  # slot 0 must be the anchor
    with pytest.raises(ValueError):
        blind_history_latents(lat, [[0, 1]])  # batch mismatch


# ----------------------------------------------------------------------------
# 2. parallel prefix (training forward / deploy recompute), with action history
# ----------------------------------------------------------------------------


def _parallel(vb, ab, memory, driver, history, acts, *, c: int, seed: int = 7, v_t=500.0, a_t=300.0):
    times = memory_times_for(c, memory.cfg.max_latents)
    mem = dict(memory_times=[times], memory_context_index=[c], memory_latents=history[:, :, times], memory_actions=acts)
    return _forward(vb, ab, memory, driver, _inputs(seed), v_t=v_t, a_t=a_t, **mem)


@pytest.mark.parametrize("mode", [ACTION_SEES_VIDEO, MUTUAL])
def test_blind_parallel_ignores_history_images_but_reads_actions(mode):
    vb, ab, memory, driver = _setup(blind=True, action_history=True, mode=mode)
    g = torch.Generator().manual_seed(11)
    c = 3  # slots 0..3
    history = torch.randn(B, Z, c + 1, H, W, generator=g)
    other = history.clone()
    other[:, :, 1:] = torch.randn(B, Z, c, H, W, generator=g)  # different history images, same anchor
    acts = torch.randn(B, c + 1, STEPS, ADIM, generator=g)
    acts[:, 0] = 0.0  # anchor slot has no past actions
    ref = _parallel(vb, ab, memory, driver, history, acts, c=c)
    # history images are gone: bit-identical output
    for x, y in zip(ref, _parallel(vb, ab, memory, driver, other, acts, c=c)):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    # the action history still reaches both outputs
    acts2 = acts.clone()
    acts2[:, 1:] += torch.randn(B, c, STEPS, ADIM, generator=g)
    out_a = _parallel(vb, ab, memory, driver, history, acts2, c=c)
    assert _max_diff(ref, out_a) > 1e-5
    assert (ref[1] - out_a[1]).abs().max() > 1e-5  # the action prediction itself moves
    # the anchor image still reaches the output
    anchor2 = history.clone()
    anchor2[:, :, 0] = torch.randn(B, Z, H, W, generator=g)
    assert _max_diff(ref, _parallel(vb, ab, memory, driver, anchor2, acts, c=c)) > 1e-5


def test_blind_switch_equals_feeding_anchor_copies_and_off_reads_history():
    vb, ab, mem_on, driver = _setup(blind=True, action_history=True)
    _, _, mem_off, _ = _setup(blind=False, action_history=True)
    mem_off.load_state_dict(mem_on.state_dict())
    g = torch.Generator().manual_seed(23)
    c = 5  # c + 1 > max_latents: truncated layout [0, 3, 4, 5]
    history = torch.randn(B, Z, c + 1, H, W, generator=g)
    acts = torch.randn(B, len(memory_times_for(c, 4)), STEPS, ADIM, generator=g)
    on = _parallel(vb, ab, mem_on, driver, history, acts, c=c)
    manual = _parallel(vb, ab, mem_off, driver, _anchor_copies(history), acts, c=c)
    for x, y in zip(on, manual):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    # switch off (original behaviour): the history images do reach the output
    off = _parallel(vb, ab, mem_off, driver, history, acts, c=c)
    assert _max_diff(off, manual) > 1e-5


@torch.no_grad()
def _forward_batch(vb, ab, memory, driver, inp, *, v_t: float, a_t: float, **mem):
    """``tests.test_memory_stream._forward`` for a batch > 1 (per-sample timesteps)."""
    window = inp["window"]
    n = window.shape[0]
    vstate = vb.prepare(
        latents=window,
        timestep=torch.full((n,), v_t),
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
    astate = ab.prepare_state(inp["actions"], torch.full((n,), a_t), context=inp["context"], context_mask=inp["context_mask"])
    vstate, astate = driver.run_joint_loop(vstate, astate)
    return vb.finalize(vstate), ab.extract_prediction(astate)


def test_blind_padded_batch_matches_single_samples():
    """Training batches mix memory lengths: padded slots stay padding, each sample == its own forward."""
    vb, ab, memory, driver = _setup(blind=True, action_history=True)
    g = torch.Generator().manual_seed(31)
    hist = torch.randn(2, Z, 4, H, W, generator=g)
    acts = torch.randn(2, 4, STEPS, ADIM, generator=g)
    times = [[0, 1, 2, 3], [0, 1]]
    lat = hist.clone()
    lat[1, :, 2:] = 0.0  # zero padding, as _collect_memory_inputs builds it
    act = acts.clone()
    act[1, 2:] = 0.0
    inp0, inp1 = _inputs(3), _inputs(4)
    both = {k: torch.cat([inp0[k], inp1[k]], dim=0) for k in inp0}
    out = _forward_batch(vb, ab, memory, driver, both, v_t=400.0, a_t=200.0, memory_latents=lat, memory_times=times,
                         memory_context_index=[3, 1], memory_actions=act)
    for b, inp in enumerate((inp0, inp1)):
        n = len(times[b])
        single = _forward(vb, ab, memory, driver, inp, v_t=400.0, a_t=200.0, memory_latents=hist[b : b + 1, :, :n],
                          memory_times=[times[b]], memory_context_index=[times[b][-1]], memory_actions=acts[b : b + 1, :n])
        _check([o[b : b + 1] for o in out], single)


# ----------------------------------------------------------------------------
# 3. prefix_once / streaming (no action history: the KV modes do not support it)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("recent_kv", [False, True])
def test_blind_kv_modes_match_blind_parallel_and_ignore_history_images(recent_kv):
    vb, ab, memory, driver = _setup(blind=True, recent_kv=recent_kv)
    g = torch.Generator().manual_seed(41)
    n_lat = 7  # c = 0..6 with max_latents 4 -> the last three requests use the exact rebuild
    history = torch.randn(B, Z, n_lat, H, W, generator=g)
    other = history.clone()
    other[:, :, 1:] = torch.randn(B, Z, n_lat - 1, H, W, generator=g)
    stream, stream_other = MemoryKVStream("streaming"), MemoryKVStream("streaming")
    for c in range(n_lat):
        stream.append_latent(history[:, :, c : c + 1])
        stream_other.append_latent(other[:, :, c : c + 1])
        times = memory_times_for(c, memory.cfg.max_latents)
        inp = _inputs(200 + c)
        mem = dict(memory_times=[times], memory_context_index=[c])
        once = MemoryKVStream("prefix_once")
        for t in range(c + 1):
            once.append_latent(history[:, :, t : t + 1])
        for v_t, a_t in ((900.0, 800.0), (350.0, 120.0)):
            ref = _forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_latents=history[:, :, times], **mem)
            _check(_forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_stream=once, **mem), ref)
            _check(_forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_stream=stream, **mem), ref)
            _check(_forward(vb, ab, memory, driver, inp, v_t=v_t, a_t=a_t, memory_stream=stream_other, **mem), ref)
    assert stream.stats["slots_written"] == memory.cfg.max_latents
    assert stream.stats["prefix_rebuilds"] == n_lat - memory.cfg.max_latents


# ----------------------------------------------------------------------------
# 4. config
# ----------------------------------------------------------------------------


def test_switch_default_off_and_parses():
    assert MemoryConfig().history_video_blind is False
    assert parse_memory_config({"enabled": True}).history_video_blind is False
    assert parse_memory_config({"enabled": True, "history_video_blind": True}).history_video_blind is True


def test_switch_composes_as_hydra_plus_override():
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        on = compose(config_name="train_memmimic", overrides=["+model.architecture.memory.history_video_blind=true"])
        off = compose(config_name="train_memmimic")
    assert parse_memory_config(on.model.architecture.memory).history_video_blind is True
    assert parse_memory_config(off.model.architecture.memory).history_video_blind is False
