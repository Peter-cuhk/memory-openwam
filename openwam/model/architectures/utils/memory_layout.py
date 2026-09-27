"""Memory-OpenWAM: token layout, role-aware joint attention mask, memory RoPE.

The video-side sequence of a memory-augmented forward is

    [ A | F_1 .. F_{T-1} | G_0 .. G_{T-1} | F_cur | N_1 .. N_{W-1} ]
      |<-------- memory prefix (n_memory tokens) -------->|<-- window -->|

where ``A`` is the episode anchor latent (absolute latent time 0), ``F_k`` are
historical chunk latents (queries whose only job is to write their gist
``G_k``), ``G_t`` are ``gist_per_latent`` learnable tokens per memory slot,
``F_cur`` is OpenWAM's clean conditioning frame (unchanged from the vanilla
window) and ``N`` are the noisy future frames. Memory slots carry absolute
latent times; the window sits at times ``c, c+1, ...``.

Visibility rules (rows = queries, cols = keys), mirroring CoCo MemoryWAM's
``MemoryTokenLayout.allowed`` for the memory block and OpenWAM's
``build_cross_modal_attention_mask`` for the ``{F_cur, N} <-> action`` block:

* memory-write rows ``A / F_k / G_k`` (time ``t``): anchor, ``F_j`` with
  ``j in {t, t-1}``, ``G_j`` with ``j <= t``. Never ``F_cur``, ``N`` or action.
* ``F_cur`` rows: themselves (OpenWAM ``first_frame_causal``); optionally the
  memory (``f_cur_reads_memory``).
* ``N`` rows: anchor, ``G_{<=c}``, ``F_cur``, ``N``; action per ``mode``.
* action rows: anchor, ``G_{<=c}``, ``F_cur``; ``N`` per ``mode``; action.
* ``recent_kv`` additionally exposes ``F_c`` (the last history chunk) to the
  readers, i.e. CoCo's "recent full frame" KV.

Batches may mix samples with different memory lengths / context indices: the
layout is padded to the longest sample, ``valid`` marks real slots, and the
mask / RoPE are built per sample (``(B, 1, S, S)`` mask, ``(B, S, 1, D/2)``
RoPE frequencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence

import torch
from torch import Tensor

from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    MUTUAL,
    VIDEO_SEES_ACTION,
    validate_attention_mask_mode,
)


class MemoryRole(IntEnum):
    ANCHOR = 0
    HIST_FRAME = 1
    GIST = 2
    CUR_FRAME = 3
    NOISY = 4


@dataclass(frozen=True)
class MemoryTokenLayout:
    """Per-token metadata of the memory-augmented video sequence.

    ``role`` / ``slot`` / ``hpos`` / ``wpos`` are batch-invariant ``(S,)``;
    ``time`` / ``valid`` are per sample ``(B, S)``. ``slot`` is the memory slot
    index (0 = anchor) for memory tokens and ``-1`` for window tokens.
    """

    role: Tensor
    slot: Tensor
    hpos: Tensor
    wpos: Tensor
    time: Tensor
    valid: Tensor
    n_memory: int
    n_slots: int
    tokens_per_frame: int
    gist_per_latent: int
    window_frames: int
    context_index: Tensor  # (B,)

    @property
    def sequence_length(self) -> int:
        return int(self.role.numel())

    @property
    def batch_size(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_window(self) -> int:
        return self.sequence_length - self.n_memory

    @property
    def window_slice(self) -> slice:
        return slice(self.n_memory, self.sequence_length)

    def is_role(self, role: MemoryRole) -> Tensor:
        return self.role == int(role)


def build_memory_token_layout(
    *,
    memory_times: Sequence[Sequence[int]],
    context_index: Sequence[int],
    tokens_per_frame: int,
    grid_height: int,
    grid_width: int,
    gist_per_latent: int,
    window_frames: int,
    device: torch.device | str = "cpu",
) -> MemoryTokenLayout:
    """Build the padded batch layout.

    Args:
        memory_times: per-sample absolute latent times of the memory slots
            (``memory_times[b][0]`` must be 0 = the anchor; strictly
            increasing; ``max <= context_index[b]``). Samples are padded to the
            longest list; padded slots are marked invalid.
        context_index: per-sample absolute latent time ``c`` of ``F_cur``.
        tokens_per_frame: ``grid_height * grid_width`` patch tokens per latent.
        window_frames: latent frames of the OpenWAM window (``F_cur`` + noisy).
    """
    if grid_height * grid_width != tokens_per_frame:
        raise ValueError("tokens_per_frame must equal grid_height * grid_width.")
    if gist_per_latent <= 0 or window_frames <= 0 or tokens_per_frame <= 0:
        raise ValueError("gist_per_latent, window_frames and tokens_per_frame must be positive.")
    batch = len(memory_times)
    if batch == 0 or len(context_index) != batch:
        raise ValueError("memory_times and context_index must have the same non-zero batch size.")
    n_slots = max(len(times) for times in memory_times)
    if n_slots == 0:
        raise ValueError("Every sample needs at least the anchor memory slot.")
    for b, times in enumerate(memory_times):
        times = [int(t) for t in times]
        if not times or times[0] != 0:
            raise ValueError(f"memory_times[{b}] must start with the anchor at time 0, got {times}.")
        if any(t1 <= t0 for t0, t1 in zip(times, times[1:])):
            raise ValueError(f"memory_times[{b}] must be strictly increasing, got {times}.")
        if times[-1] > int(context_index[b]):
            raise ValueError(
                f"memory_times[{b}] reaches time {times[-1]} beyond context_index {int(context_index[b])}."
            )

    tpf = tokens_per_frame
    n_frame_mem = n_slots * tpf
    n_gist = n_slots * gist_per_latent
    n_memory = n_frame_mem + n_gist
    n_window = window_frames * tpf
    seq = n_memory + n_window

    role = torch.empty(seq, dtype=torch.long)
    slot = torch.full((seq,), -1, dtype=torch.long)
    frame_of_token = torch.empty(seq, dtype=torch.long)  # frame index in window for window tokens
    # memory frames: slot 0 = anchor, slots 1.. = history chunks
    role[:tpf] = int(MemoryRole.ANCHOR)
    role[tpf:n_frame_mem] = int(MemoryRole.HIST_FRAME)
    slot[:n_frame_mem] = torch.arange(n_slots).repeat_interleave(tpf)
    role[n_frame_mem:n_memory] = int(MemoryRole.GIST)
    slot[n_frame_mem:n_memory] = torch.arange(n_slots).repeat_interleave(gist_per_latent)
    role[n_memory : n_memory + tpf] = int(MemoryRole.CUR_FRAME)
    role[n_memory + tpf :] = int(MemoryRole.NOISY)
    frame_of_token[n_memory:] = torch.arange(window_frames).repeat_interleave(tpf)

    grid_h = torch.arange(grid_height).repeat_interleave(grid_width)
    grid_w = torch.arange(grid_width).repeat(grid_height)
    hpos = torch.full((seq,), -1, dtype=torch.long)
    wpos = torch.full((seq,), -1, dtype=torch.long)
    hpos[:n_frame_mem] = grid_h.repeat(n_slots)
    wpos[:n_frame_mem] = grid_w.repeat(n_slots)
    hpos[n_memory:] = grid_h.repeat(window_frames)
    wpos[n_memory:] = grid_w.repeat(window_frames)

    time = torch.zeros(batch, seq, dtype=torch.long)
    valid = torch.ones(batch, seq, dtype=torch.bool)
    ctx = torch.tensor([int(c) for c in context_index], dtype=torch.long)
    for b, times in enumerate(memory_times):
        times = [int(t) for t in times]
        n_valid = len(times)
        # Padded slots get times after the window so they never alias a real
        # slot; they are masked out of every attention anyway.
        slot_time = torch.tensor(times + [ctx[b].item() + window_frames + i for i in range(n_slots - n_valid)])
        slot_valid = torch.tensor([True] * n_valid + [False] * (n_slots - n_valid))
        mem = slot[:n_memory]
        time[b, :n_memory] = slot_time[mem]
        valid[b, :n_memory] = slot_valid[mem]
        time[b, n_memory:] = ctx[b] + frame_of_token[n_memory:]

    dev = torch.device(device)
    return MemoryTokenLayout(
        role=role.to(dev),
        slot=slot.to(dev),
        hpos=hpos.to(dev),
        wpos=wpos.to(dev),
        time=time.to(dev),
        valid=valid.to(dev),
        n_memory=n_memory,
        n_slots=n_slots,
        tokens_per_frame=tpf,
        gist_per_latent=gist_per_latent,
        window_frames=window_frames,
        context_index=ctx.to(dev),
    )


def build_memory_video_mask(
    layout: MemoryTokenLayout,
    *,
    f_cur_reads_memory: bool = False,
    recent_kv: bool = False,
    reader_sees_memory: Optional[Tensor] = None,
) -> Tensor:
    """Video<->video visibility ``(B, S_v, S_v)`` bool (True = attend)."""
    role = layout.role
    time = layout.time  # (B, S)
    valid = layout.valid
    B, S = time.shape
    dev = role.device

    is_anchor = layout.is_role(MemoryRole.ANCHOR)
    is_hist = layout.is_role(MemoryRole.HIST_FRAME)
    is_gist = layout.is_role(MemoryRole.GIST)
    is_cur = layout.is_role(MemoryRole.CUR_FRAME)
    is_noisy = layout.is_role(MemoryRole.NOISY)
    is_memory = is_anchor | is_hist | is_gist

    tq = time[:, :, None]  # (B, S, 1)
    tk = time[:, None, :]  # (B, 1, S)
    c = layout.context_index[:, None, None]  # (B, 1, 1)
    kA = is_anchor[None, None, :]
    kH = is_hist[None, None, :]
    kG = is_gist[None, None, :]
    kC = is_cur[None, None, :]
    kN = is_noisy[None, None, :]
    qM = is_memory[None, :, None]
    qC = is_cur[None, :, None]
    qN = is_noisy[None, :, None]

    # Memory-write rows (CoCo rule): anchor | F_{t}, F_{t-1} | G_{<=t}.
    mem_rows = qM & (kA | (kH & ((tk == tq) | (tk == tq - 1))) | (kG & (tk <= tq)))

    # What readers may see of the memory: anchor + causal gists (+ F_c).
    reader_memory = kA | (kG & (tk <= c))
    if recent_kv:
        reader_memory = reader_memory | (kH & (tk == c))
    if reader_sees_memory is not None:
        drop = ~reader_sees_memory.to(device=dev, dtype=torch.bool)[:, None, None]
        # Memory dropout: hide gists / recent frame, keep the anchor.
        reader_memory = torch.where(drop, kA.expand(B, 1, S), reader_memory)

    cur_rows = qC & kC
    if f_cur_reads_memory:
        cur_rows = cur_rows | (qC & reader_memory)
    noisy_rows = qN & (reader_memory | kC | kN)

    mask = mem_rows | cur_rows | noisy_rows
    mask = mask & valid[:, None, :]
    # Padded query rows attend to themselves only (avoids NaN softmax rows).
    eye = torch.eye(S, dtype=torch.bool, device=dev)[None]
    mask = torch.where(valid[:, :, None], mask, eye.expand(B, S, S))
    return mask


def build_memory_joint_mask(
    layout: MemoryTokenLayout,
    s_action: int,
    *,
    mode: str,
    f_cur_reads_memory: bool = False,
    recent_kv: bool = False,
    reader_sees_memory: Optional[Tensor] = None,
) -> Tensor:
    """Joint ``[video, action]`` mask ``(B, 1, S, S)`` bool for the MoT driver."""
    validate_attention_mask_mode(mode)
    video = build_memory_video_mask(
        layout,
        f_cur_reads_memory=f_cur_reads_memory,
        recent_kv=recent_kv,
        reader_sees_memory=reader_sees_memory,
    )
    B, S_v, _ = video.shape
    dev = video.device
    S = S_v + s_action
    mask = torch.zeros(B, S, S, dtype=torch.bool, device=dev)
    mask[:, :S_v, :S_v] = video
    if s_action > 0:
        time = layout.time
        c = layout.context_index[:, None]
        is_anchor = layout.is_role(MemoryRole.ANCHOR)[None]
        is_hist = layout.is_role(MemoryRole.HIST_FRAME)[None]
        is_gist = layout.is_role(MemoryRole.GIST)[None]
        is_cur = layout.is_role(MemoryRole.CUR_FRAME)[None]
        is_noisy = layout.is_role(MemoryRole.NOISY)[None]
        a2v = is_anchor | (is_gist & (time <= c))  # (B, S_v)
        if recent_kv:
            a2v = a2v | (is_hist & (time == c))
        if reader_sees_memory is not None:
            drop = ~reader_sees_memory.to(device=dev, dtype=torch.bool)[:, None]
            a2v = torch.where(drop, is_anchor.expand(B, S_v), a2v)
        a2v = a2v | is_cur
        if mode in (ACTION_SEES_VIDEO, MUTUAL):
            a2v = a2v | is_noisy
        a2v = a2v & layout.valid
        mask[:, S_v:, :S_v] = a2v[:, None, :]
        mask[:, S_v:, S_v:] = True
        if mode in (MUTUAL, VIDEO_SEES_ACTION):
            # Only noisy video rows read the action stream (OpenWAM keeps the
            # clean first frame isolated; memory rows must stay causal).
            mask[:, :S_v, S_v:] = is_noisy[:, :, None].expand(B, S_v, s_action)
    return mask[:, None]


def build_gist_gate_selector(
    layout: MemoryTokenLayout,
    s_action: int,
    *,
    f_cur_reads_memory: bool = False,
) -> Tensor:
    """``(B, 1, S, S)`` bool: reader-row x gist-column cells that receive the
    per-layer additive gist gate bias."""
    is_gist = layout.is_role(MemoryRole.GIST)
    readers = layout.is_role(MemoryRole.NOISY)
    if f_cur_reads_memory:
        readers = readers | layout.is_role(MemoryRole.CUR_FRAME)
    B, S_v = layout.time.shape
    S = S_v + s_action
    rows = torch.zeros(S, dtype=torch.bool, device=is_gist.device)
    rows[:S_v] = readers
    rows[S_v:] = True
    cols = torch.zeros(S, dtype=torch.bool, device=is_gist.device)
    cols[:S_v] = is_gist
    sel = rows[:, None] & cols[None, :]
    sel = sel[None].expand(B, S, S) & torch.cat(
        [layout.valid, torch.ones(B, s_action, dtype=torch.bool, device=is_gist.device)], dim=1
    )[:, None, :]
    return sel[:, None]


def memory_reader_key_index(layout: MemoryTokenLayout, *, recent_kv: bool = False) -> Tensor:
    """Memory-prefix columns any reader row can see, in the order a streamed KV cache stores them:
    anchor frame tokens, then every gist (slot order), then ``F_c`` when ``recent_kv``.

    Readers never see the other history frames, so a deploy-time KV cache only needs these columns.
    """
    if layout.batch_size != 1:
        raise ValueError("memory_reader_key_index expects a single-sample layout.")
    idx = [
        layout.is_role(MemoryRole.ANCHOR).nonzero().flatten(),
        layout.is_role(MemoryRole.GIST).nonzero().flatten(),
    ]
    if recent_kv:
        c = int(layout.context_index[0])
        idx.append((layout.is_role(MemoryRole.HIST_FRAME) & (layout.time[0] == c)).nonzero().flatten())
    return torch.cat(idx)


def build_memory_reader_mask(
    layout: MemoryTokenLayout,
    s_action: int,
    key_index: Tensor,
    *,
    mode: str,
    f_cur_reads_memory: bool = False,
    recent_kv: bool = False,
    with_gate: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Reader-row slice of :func:`build_memory_joint_mask` for a KV-cached memory prefix.

    Rows are ``[window | action]`` (the memory rows are gone: their K/V are cached), columns are
    ``[key_index | window | action]``. Returns ``(mask, gate_selector)``; the mask is additive
    float32 when ``with_gate`` (as in the parallel path) and bool otherwise. Slicing the training
    mask keeps the visibility rules in one place.
    """
    full = build_memory_joint_mask(
        layout, s_action, mode=mode, f_cur_reads_memory=f_cur_reads_memory, recent_kv=recent_kv
    )
    S = full.shape[-1]
    rows = torch.arange(layout.n_memory, S, device=full.device)
    cols = torch.cat([key_index.to(full.device), rows])
    mask = full[:, :, rows][:, :, :, cols]
    selector = None
    if with_gate:
        sel = build_gist_gate_selector(layout, s_action, f_cur_reads_memory=f_cur_reads_memory)
        selector = sel[:, :, rows][:, :, :, cols]
        mask = bool_mask_to_additive(mask, torch.float32)
    return mask, selector


def bool_mask_to_additive(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """True -> 0, False -> -inf, as a float mask SDPA can add to the logits."""
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))


def _axis_freqs_at_position(dim: int, position: float, *, theta: float = 10000.0, device) -> Tensor:
    """Complex RoPE factors of one axis at an arbitrary (possibly negative) position."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device)[: dim // 2] / dim))
    angles = inv * position
    return torch.polar(torch.ones_like(angles), angles)


def build_memory_rope_freqs(
    dit_freqs: Sequence[Tensor],
    layout: MemoryTokenLayout,
    *,
    device: torch.device,
) -> Tensor:
    """Per-sample 3D RoPE factors ``(B, S, 1, head_dim/2)`` complex.

    Frame tokens use Wan's precomputed ``(time, h, w)`` tables at absolute
    latent time; gist tokens use the temporal factor of their slot time and a
    spatial marker at position ``-1`` on both spatial axes (CoCo convention).
    """
    f_table, h_table, w_table = (t.to(device) for t in dit_freqs)
    B, S = layout.time.shape
    max_time = int(layout.time.max())
    if max_time >= f_table.shape[0]:
        raise ValueError(f"Absolute latent time {max_time} exceeds the RoPE table ({f_table.shape[0]}).")
    temporal = f_table[layout.time.reshape(-1)].reshape(B, S, -1)
    is_gist = layout.is_role(MemoryRole.GIST)
    hpos = layout.hpos.clamp(min=0)
    wpos = layout.wpos.clamp(min=0)
    h_part = h_table[hpos][None].expand(B, S, -1).clone()
    w_part = w_table[wpos][None].expand(B, S, -1).clone()
    h_marker = _axis_freqs_at_position(h_table.shape[1] * 2, -1.0, device=device).to(h_table.dtype)
    w_marker = _axis_freqs_at_position(w_table.shape[1] * 2, -1.0, device=device).to(w_table.dtype)
    h_part[:, is_gist] = h_marker
    w_part[:, is_gist] = w_marker
    freqs = torch.cat([temporal, h_part, w_part], dim=-1)
    return freqs.reshape(B, S, 1, -1)


__all__ = [
    "MemoryRole",
    "MemoryTokenLayout",
    "build_memory_token_layout",
    "build_memory_video_mask",
    "build_memory_joint_mask",
    "build_gist_gate_selector",
    "memory_reader_key_index",
    "build_memory_reader_mask",
    "bool_mask_to_additive",
    "build_memory_rope_freqs",
]
