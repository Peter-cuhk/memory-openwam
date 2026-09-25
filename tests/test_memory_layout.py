"""Memory-OpenWAM: layout / mask / RoPE unit tests (CPU only).

1. The memory block of the role-aware mask matches CoCo MemoryWAM's
   ``MemoryTokenLayout.allowed`` element-wise (CoCo is only a test oracle).
2. Parallel teacher-forced memory forward == sequential per-latent KV-cache
   streaming forward (gist hidden states and window outputs), i.e. training
   layout and deployment write/read semantics agree.
3. Memory rows never see proprio in cross-attention; readers see the gist
   columns only through the gate selector.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from openwam.model.architectures.utils.memory_layout import (
    MemoryRole,
    bool_mask_to_additive,
    build_gist_gate_selector,
    build_memory_joint_mask,
    build_memory_rope_freqs,
    build_memory_token_layout,
    build_memory_video_mask,
)
from openwam.model.architectures.utils.memory_tokens import MemoryConfig, MemoryTokens

COCO_ROOT = Path("/mnt/cpfs/workspace/coco_pretrain_memory")


def _coco_layout_module():
    if not COCO_ROOT.is_dir():
        pytest.skip("CoCo memory_causal_wan checkout not available")
    # Load layout.py directly: the CoCo package __init__ pulls in diffsynth.
    import importlib.util

    path = COCO_ROOT / "memory_causal_wan" / "models" / "layout.py"
    if not path.is_file():
        pytest.skip("CoCo layout.py not available")
    spec = importlib.util.spec_from_file_location("coco_memory_layout_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CoCo layout import failed: {exc}")
    return module


# ----------------------------------------------------------------------------
# 1. Mask vs CoCo oracle
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("n_slots,tpf,gist", [(1, 4, 2), (3, 4, 2), (5, 6, 8)])
def test_memory_block_matches_coco_allowed(n_slots, tpf, gist):
    coco = _coco_layout_module()
    c = n_slots - 1
    grid_h, grid_w = 2, tpf // 2
    mine = build_memory_token_layout(
        memory_times=[list(range(n_slots))],
        context_index=[c],
        tokens_per_frame=tpf,
        grid_height=grid_h,
        grid_width=grid_w,
        gist_per_latent=gist,
        window_frames=3,
    )
    video = build_memory_video_mask(mine, recent_kv=True)[0]  # (S, S)
    n_mem = mine.n_memory

    # CoCo: clean frames F_0..F_{c+1}, gists G_0..G_{c+1}, one noisy latent at c+1.
    theirs = coco.build_memory_token_layout(
        latent_frames=n_slots + 1,
        spatial_tokens_per_latent=tpf,
        gist_tokens_per_latent=gist,
        noisy_group_size=1,
        context_indices=(c,),
    )
    dense = theirs.dense_mask()
    is_frame = theirs.role == int(coco.TokenRole.CLEAN_FRAME)
    is_gist = theirs.role == int(coco.TokenRole.GIST)
    is_noisy = theirs.role == int(coco.TokenRole.NOISY)
    keep = ((is_frame | is_gist) & (theirs.latent_index <= c)).nonzero().flatten()
    # CoCo orders [frames][gists]; ours [anchor, hist frames][gists] — identical order.
    assert torch.equal(video[:n_mem, :n_mem], dense[keep][:, keep])

    # Noisy rows over the memory columns: CoCo's N sees F_0, F_c, G_{<=c};
    # ours (recent_kv=True) sees anchor, hist F_c, G_{<=c}.
    coco_noisy_row = dense[is_noisy.nonzero().flatten()[0]][keep]
    my_noisy_row = video[mine.is_role(MemoryRole.NOISY).nonzero().flatten()[0], :n_mem]
    assert torch.equal(my_noisy_row, coco_noisy_row)


def test_memory_rows_are_causal_and_isolated_from_window_and_action():
    layout = build_memory_token_layout(
        memory_times=[[0, 1, 2, 3], [0, 1]],
        context_index=[3, 1],
        tokens_per_frame=4,
        grid_height=2,
        grid_width=2,
        gist_per_latent=2,
        window_frames=3,
    )
    s_action = 5
    mask = build_memory_joint_mask(layout, s_action, mode="mutual")[:, 0]  # (B, S, S)
    S_v = layout.sequence_length
    is_mem = layout.is_role(MemoryRole.ANCHOR) | layout.is_role(MemoryRole.HIST_FRAME) | layout.is_role(MemoryRole.GIST)
    is_cur = layout.is_role(MemoryRole.CUR_FRAME)
    is_noisy = layout.is_role(MemoryRole.NOISY)
    for b in range(2):
        m = mask[b]
        # memory rows: nothing from the window or the action stream
        assert not m[:S_v][is_mem][:, S_v:].any()
        assert not m[:S_v][is_mem][:, :S_v][:, is_cur | is_noisy].any()
        # memory rows: no future gists / frames
        tq = layout.time[b][is_mem][:, None]
        tk = layout.time[b][None, :S_v]
        sub = m[:S_v][is_mem][:, :S_v]
        assert not (sub & (tk > tq)).any()
        # padded slots are never keys, and padded rows only see themselves
        invalid = ~layout.valid[b]
        inv_rows = m[:S_v][invalid]
        assert torch.equal(inv_rows[:, :S_v], torch.eye(S_v, dtype=torch.bool)[invalid])
        assert not inv_rows[:, S_v:].any()
        real_rows = torch.ones(m.shape[0], dtype=torch.bool)
        real_rows[:S_v] = layout.valid[b]
        assert not m[real_rows][:, :S_v][:, invalid].any()
        # F_cur rows: self only (first_frame_causal), no action
        assert torch.equal(m[:S_v][is_cur][:, :S_v].any(0), is_cur)
        assert not m[:S_v][is_cur][:, S_v:].any()
        # noisy rows see anchor + valid gists <= c + F_cur + N + action (mutual)
        n_rows = m[:S_v][is_noisy]
        assert n_rows[:, S_v:].all()
        n_v = n_rows[:, :S_v]
        assert n_v[:, is_cur].all() and n_v[:, is_noisy].all()
        assert n_v[:, layout.is_role(MemoryRole.ANCHOR)].all()
        assert not n_v[:, layout.is_role(MemoryRole.HIST_FRAME)].any()
        # action rows: anchor + gists + F_cur + N (mutual) + action, never hist frames
        a_rows = m[S_v:]
        a_v = a_rows[:, :S_v]
        assert a_rows[:, S_v:].all() and a_v[:, is_cur].all() and a_v[:, is_noisy].all()
        assert not a_v[:, layout.is_role(MemoryRole.HIST_FRAME)].any()
        gist_valid = layout.is_role(MemoryRole.GIST) & layout.valid[b]
        assert a_v[:, gist_valid].all()
        assert not a_v[:, layout.is_role(MemoryRole.GIST) & ~layout.valid[b]].any()
    # action_sees_video: noisy rows must not read the action stream
    mask2 = build_memory_joint_mask(layout, s_action, mode="action_sees_video")[:, 0]
    assert not mask2[:, :S_v, S_v:].any()
    # memory dropout hides gists (keeps the anchor) from the readers
    mask3 = build_memory_joint_mask(layout, s_action, mode="mutual", reader_sees_memory=torch.tensor([False, True]))[:, 0]
    assert not mask3[0, S_v:, :S_v][:, layout.is_role(MemoryRole.GIST)].any()
    assert mask3[0, S_v:, :S_v][:, layout.is_role(MemoryRole.ANCHOR)].all()
    assert mask3[1, S_v:, :S_v][:, layout.is_role(MemoryRole.GIST) & layout.valid[1]].all()


def test_gist_gate_selector_and_additive_mask():
    layout = build_memory_token_layout(
        memory_times=[[0, 1, 2]],
        context_index=[2],
        tokens_per_frame=4,
        grid_height=2,
        grid_width=2,
        gist_per_latent=3,
        window_frames=2,
    )
    sel = build_gist_gate_selector(layout, 4)[0, 0]
    S_v = layout.sequence_length
    is_gist = layout.is_role(MemoryRole.GIST)
    is_noisy = layout.is_role(MemoryRole.NOISY)
    assert sel[S_v:][:, :S_v][:, is_gist].all()
    assert sel[:S_v][is_noisy][:, :S_v][:, is_gist].all()
    assert not sel[:S_v][layout.is_role(MemoryRole.CUR_FRAME)].any()
    assert not sel[:, :S_v][:, ~is_gist].any() and not sel[:, S_v:].any()
    mask = build_memory_joint_mask(layout, 4, mode="mutual")
    add = bool_mask_to_additive(mask, torch.float32)
    assert torch.isneginf(add[~mask]).all() and (add[mask] == 0).all()


# ----------------------------------------------------------------------------
# 2. Parallel == streaming
# ----------------------------------------------------------------------------


def _tiny_backbone():
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    torch.manual_seed(0)
    dit = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=96,
        out_dim=4,
        text_dim=16,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,  # head_dim 32 -> 3D RoPE split (12, 10, 10) sums to 32
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    holder = types.SimpleNamespace(dit=dit)
    vb = Wan22Ti2v(holder)
    vb.set_dtype_device(torch.float32, torch.device("cpu"))
    return vb, dit


def _block_stream(block, x, context, t_mod, freqs, cache_k, cache_v, self_mask=None):
    """CoCo ``_block_query_forward`` equivalent on OpenWAM's Wan block."""
    from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply

    mod = (block.modulation + t_mod).chunk(6, dim=2)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (v.squeeze(2) for v in mod)
    inp = modulate(block.norm1(x), shift_msa, scale_msa)
    sa = block.self_attn
    n = sa.num_heads
    q = rope_apply(sa.norm_q(sa.q(inp)), freqs, n)
    k = rope_apply(sa.norm_k(sa.k(inp)), freqs, n)
    v = sa.v(inp)
    K = k if cache_k is None else torch.cat([cache_k, k], dim=1)
    V = v if cache_v is None else torch.cat([cache_v, v], dim=1)

    def heads(t):
        return t.view(t.shape[0], t.shape[1], n, -1).transpose(1, 2)

    out = F.scaled_dot_product_attention(heads(q), heads(K), heads(V), attn_mask=self_mask)
    out = out.transpose(1, 2).reshape(q.shape)
    x = block.gate(x, gate_msa, sa.o(out))
    x = x + block.cross_attn(block.norm3(x), context)
    x = block.gate(x, gate_mlp, block.ffn(modulate(block.norm2(x), shift_mlp, scale_mlp)))
    return x, k, v


@pytest.mark.parametrize("n_slots", [1, 4])
def test_parallel_memory_forward_equals_streaming(n_slots):
    torch.manual_seed(1)
    vb, dit = _tiny_backbone()
    mem_cfg = MemoryConfig(enabled=True, gist_tokens=3, use_gist_gate=False)
    memory = MemoryTokens(mem_cfg, dim=64, num_layers=2)
    B, C, Fw, H, W = 1, 4, 3, 4, 4
    c = n_slots - 1
    window = torch.randn(B, C, Fw, H, W)
    history = torch.randn(B, C, n_slots, H, W)
    context = torch.randn(B, 5, 16)
    timestep = torch.tensor([437.0])
    common = dict(
        latents=window,
        timestep=timestep,
        context=context,
        fuse_vae_embedding_in_latents=True,
        first_frame_latents=window[:, :, 0:1],
    )

    # --- parallel (training layout) ---
    state = vb.prepare(
        **common,
        memory_latents=history,
        memory_times=[list(range(n_slots))],
        memory_context_index=[c],
        memory_tokens=memory,
    )
    layout = state.extras["memory_layout"]
    mask = build_memory_joint_mask(layout, 0, mode="mutual")
    state.extras["shared_attention_mask"] = mask
    for block_id in range(vb.num_layers):
        state = vb.run_block(block_id, state)
    hidden_par = state.hidden_states.clone()
    out_par = vb.finalize(state)
    assert out_par.shape == window.shape

    # --- streaming (deployment semantics) ---
    plain = vb.prepare(**common)  # vanilla window tokens / t_mod / head time embedding
    freqs_all = build_memory_rope_freqs(dit.freqs, layout, device=torch.device("cpu"))
    tpf = layout.tokens_per_frame
    n_layers = len(dit.blocks)
    anchor_kv = [None] * n_layers
    gist_k = [[] for _ in range(n_layers)]
    gist_v = [[] for _ in range(n_layers)]
    recent_kv = [None] * n_layers
    text_ctx = dit.text_embedding(context)
    zero_t = torch.zeros(B)
    from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d

    t_mod0 = dit.time_projection(dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, zero_t))).unflatten(
        1, (6, dit.dim)
    )
    mem_roles = layout.is_role(MemoryRole.ANCHOR) | layout.is_role(MemoryRole.HIST_FRAME) | layout.is_role(MemoryRole.GIST)
    gist_hidden_stream = {}
    for s in range(n_slots):
        idx = (mem_roles & (layout.slot == s)).nonzero().flatten()
        frame_tokens = dit.patchify(history[:, :, s : s + 1]).flatten(2).transpose(1, 2)  # (B, tpf, D)
        x = torch.cat([frame_tokens, memory.gist_tokens.expand(B, -1, -1)], dim=1)
        freqs = freqs_all[:, idx]
        t_mod = t_mod0[:, None].expand(B, x.shape[1], 6, dit.dim)
        for l, block in enumerate(dit.blocks):
            ks, vs = [], []
            if anchor_kv[l] is not None:
                ks.append(anchor_kv[l][0])
                vs.append(anchor_kv[l][1])
            if gist_k[l]:
                ks.extend(gist_k[l])
                vs.extend(gist_v[l])
            if recent_kv[l] is not None and s - 1 >= 1:
                ks.append(recent_kv[l][0])
                vs.append(recent_kv[l][1])
            ck = torch.cat(ks, dim=1) if ks else None
            cv = torch.cat(vs, dim=1) if vs else None
            x, k, v = _block_stream(block, x, text_ctx, t_mod, freqs, ck, cv)
            if s == 0:
                anchor_kv[l] = (k[:, :tpf], v[:, :tpf])
            else:
                recent_kv[l] = (k[:, :tpf], v[:, :tpf])
            gist_k[l].append(k[:, tpf:])
            gist_v[l].append(v[:, tpf:])
        gist_hidden_stream[s] = x[:, tpf:]

    # window read: keys = anchor + all gists (+ own); F_cur rows see only themselves
    x = plain.hidden_states
    freqs = freqs_all[:, layout.window_slice]
    t_mod = plain.time_mod
    n_win = x.shape[1]
    for l, block in enumerate(dit.blocks):
        ck = torch.cat([anchor_kv[l][0]] + gist_k[l], dim=1)
        cv = torch.cat([anchor_kv[l][1]] + gist_v[l], dim=1)
        n_cache = ck.shape[1]
        m = torch.ones(n_win, n_cache + n_win, dtype=torch.bool)
        m[:tpf, :] = False
        m[:tpf, n_cache : n_cache + tpf] = True
        x, _, _ = _block_stream(block, x, text_ctx, t_mod, freqs, ck, cv, self_mask=m[None, None])
    head_t = plain.extras["time_embed"]
    out_stream = dit.unpatchify(dit.head(x, head_t), (Fw, H // 2, W // 2))

    for s in range(n_slots):
        idx = (layout.is_role(MemoryRole.GIST) & (layout.slot == s)).nonzero().flatten()
        torch.testing.assert_close(hidden_par[:, idx], gist_hidden_stream[s], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_par, out_stream, atol=1e-4, rtol=1e-4)


def test_memory_disabled_prepare_is_unchanged():
    vb, dit = _tiny_backbone()
    window = torch.randn(1, 4, 3, 4, 4)
    state = vb.prepare(
        latents=window,
        timestep=torch.tensor([10.0]),
        context=torch.randn(1, 5, 16),
        fuse_vae_embedding_in_latents=True,
        first_frame_latents=window[:, :, 0:1],
    )
    assert "memory_layout" not in state.extras
    assert state.hidden_states.shape[1] == 3 * 4
    assert state.rope_freqs.dim() == 3


# ----------------------------------------------------------------------------
# 3. Cross-attention: memory rows never read the proprio token
# ----------------------------------------------------------------------------


def test_memory_rows_hide_proprio_in_cross_attention():
    vb, dit = _tiny_backbone()
    memory = MemoryTokens(MemoryConfig(enabled=True, gist_tokens=2, use_gist_gate=False), dim=64, num_layers=2)
    window = torch.randn(1, 4, 3, 4, 4)
    context = torch.randn(1, 6, 16)  # 5 text tokens + 1 proprio token
    context_mask = torch.ones(1, 6, dtype=torch.bool)
    state = vb.prepare(
        latents=window,
        timestep=torch.tensor([10.0]),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        first_frame_latents=window[:, :, 0:1],
        memory_latents=torch.randn(1, 4, 2, 4, 4),
        memory_times=[[0, 1]],
        memory_context_index=[1],
        memory_tokens=memory,
        memory_ctx_hide_last_n=1,
    )
    m = vb._memory_context_mask(state)
    layout = state.extras["memory_layout"]
    assert m.shape == (1, layout.sequence_length, 6)
    assert not m[0, : layout.n_memory, -1].any()
    assert m[0, : layout.n_memory, :-1].all()
    assert m[0, layout.n_memory :, :].all()
