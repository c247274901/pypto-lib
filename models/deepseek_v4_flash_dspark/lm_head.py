# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 Flash DSpark LM head: fused matmul+push projection with DP-owned hidden and TP vocab shards."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from config import FLASH as M, FP32_NEG_INF, MOE_TOKENS


T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size
# Rows one owner projects: the TP group splits the step across its ranks, so a
# rank carries exactly the tokens it owns through MoE and the group window holds
# exactly one step. Taken from config rather than derived from TP, which the
# forward entries rewrite at import before this module loads.
MAX_LOGIT_ROWS = MOE_TOKENS

# parallelism
_TP_CHOICES = (1, 2, 4, 8, 16)
_DP_CHOICES = (1, 2, 4, 8, 16)
_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


# Vocabulary sharding and owner groups use an independent head TP.
TP_SIZE: int = _parse_int_argv("--lm-head-tp", _parse_int_argv("--tp", _TP_DEFAULT))
DP_SIZE: int = _parse_int_argv("--dp", 1)
# --dp describes the standalone head fixture; --ep is the composed model world.
WORLD_SIZE = _parse_int_argv("--ep", TP_SIZE * DP_SIZE)
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"LM-head TP must be one of {_TP_CHOICES}, got {TP_SIZE}")
if WORLD_SIZE <= 0 or WORLD_SIZE % TP_SIZE:
    raise ValueError(f"world size {WORLD_SIZE} must be a positive multiple of LM-head TP={TP_SIZE}")
if VOCAB % TP_SIZE:
    raise ValueError(f"vocab size {VOCAB} must be divisible by LM-head TP={TP_SIZE}")
VOCAB_PER_TP = VOCAB // TP_SIZE
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS
TEST_TOKENS = 2 * MAX_LOGIT_ROWS  # standalone fixture: hidden rows per card

# tiling
# MM_ROW_TILE x FUSED_VOCAB_TILE fp32 keeps the accumulator inside the 128KiB
# Acc space and divides MAX_LOGIT_ROWS: every row block is one owner's rows.
# One whole owner per block is the widest M the Acc space admits, and it halves
# the weight tile re-reads a narrower row tile forces.
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 256
MM_ROW_TILE = 128
HIDDEN_GATHER_TILE = 1024
HIDDEN_GATHER_ROW_TILE = min(GROUP_LOGIT_ROWS, 64)
PUSH_ROW_TILE = 16  # rows pushed per dispatch_push block (fewer, larger ring puts)
LOGITS_GATHER_ROW_TILE = min(MAX_LOGIT_ROWS, 8)
LOGITS_COMM_TILE = 2048
# Greedy sampling scans each row as a [GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH] grid.
# 8 rows keeps a reduction result at 32 B; alloc_tile rejects the 4 B a [1, 1] gives.
GREEDY_ROW_WIDTH = 808
GREEDY_BLOCK_ROWS = 8
GREEDY_MASK_VALID_BYTES = (GREEDY_ROW_WIDTH + 7) // 8
GREEDY_MASK_EVEN_BYTES = (GREEDY_MASK_VALID_BYTES + 1) // 2 * 2
SAMPLED_IDS_PAD = 8
FUSED_LM_HEAD_CORES = 24
# AIV lanes per AICore: the fused kernel splits each row block's accumulator
# across the two lanes, and each lane pushes a contiguous half of one owner's rows.
AIV_LANES = 2
GREEDY_SAMPLE_CORES = min(MAX_LOGIT_ROWS, FUSED_LM_HEAD_CORES * AIV_LANES)
LANE_ROWS = MM_ROW_TILE // AIV_LANES
VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE
VOCAB_FULL_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS
GREEDY_GRID_ROWS = VOCAB // GREEDY_ROW_WIDTH
GREEDY_BLOCK_SPAN = GREEDY_BLOCK_ROWS * GREEDY_ROW_WIDTH
GRAMMAR_LOOKUP_WIDTH = 816  # 808 logits plus the 32-byte int16 tile alignment.
GRAMMAR_SEGMENT_WORDS = 64  # 808 bits plus aligned padding per scan lane.
# 2^30: above every vocab id, clear of int32 overflow once a block base is added.
GREEDY_INDEX_SENTINEL = 1073741824
GREEDY_INDEX_SENTINEL_FP32 = float(GREEDY_INDEX_SENTINEL)
DONE_VALUE = 1
LM_HEAD_RING_HEAP = 1 << 30

assert MAX_LOGIT_ROWS % MM_ROW_TILE == 0, "each row block must be one owner's rows"
assert MAX_LOGIT_ROWS % PUSH_ROW_TILE == 0, "row block must cover whole rows"
assert MAX_LOGIT_ROWS % LOGITS_GATHER_ROW_TILE == 0, "logits row blocks must cover whole rows"
# TP1 notifies nobody, so its empty peer loop needs no lane pairing.
assert TP_SIZE == 1 or TP_SIZE % AIV_LANES == 0, "peer owners must pair across the AIV lanes"
assert GREEDY_BLOCK_ROWS * 4 % 32 == 0, "reduction result must clear the 32 B column floor"
assert GREEDY_ROW_WIDTH * 4 % 32 == 0, "block row must clear the 32 B row floor"
assert VOCAB < GREEDY_INDEX_SENTINEL, "sentinel must lose every row_min against a real id"


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    hidden_ready_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[
    pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    pl.Scalar[pl.TASK_ID],
]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    selected_hidden_packed = pl.create_tensor([D // FUSED_K_TILE, GROUP_LOGIT_ROWS, FUSED_K_TILE], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    # One block per PUSH_ROW_TILE rows: PUSH_ROW_TILE-row ring puts instead
    # of one [1, D] put per row. This cuts the per-dispatch put and notify
    # counts (512 -> 32 per card) to the MTP fixture's scale; the earlier
    # per-row form stalled lm_head_dispatch_gather on persistent re-dispatch
    # (SCHEDULER_TIMEOUT S1 with the gather task's core never exiting).
    with pl.spmd(
        MAX_LOGIT_ROWS // PUSH_ROW_TILE,
        name_hint="lm_head_dispatch_push",
        deps=[hidden_ready_tid],
    ) as _dispatch_push_tid:
        blk = pl.tile.get_block_idx()
        r0 = blk * PUSH_ROW_TILE
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        for i in pl.range(PUSH_ROW_TILE):
            row = r0 + i
            source_row_raw = pl.read(logit_row_indices, [row])
            # Clamp so the load address is always inside hidden_states even if a
            # caller hands over a stale index; the -1 guard below decides whether
            # the row is actually used.
            safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
            if TP_SIZE == 1:
                packed_zero_row = pl.full([1, D], dtype=pl.BF16, value=0.0)
                packed_zero_panels = pl.reshape(packed_zero_row, [D // FUSED_K_TILE, 1, FUSED_K_TILE])
                selected_hidden_packed[:, row : row + 1, :] = packed_zero_panels
                if source_row_raw >= 0:
                    packed_source_row = pl.cast(safe_raw, target_type=pl.INDEX)
                    packed_selected_row = hidden_states[packed_source_row : packed_source_row + 1, :]
                    packed_selected_panels = pl.reshape(packed_selected_row, [D // FUSED_K_TILE, 1, FUSED_K_TILE])
                    selected_hidden_packed[:, row : row + 1, :] = packed_selected_panels
            else:
                selected_hidden[row : row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
                if source_row_raw >= 0:
                    source_row = pl.cast(safe_raw, target_type=pl.INDEX)
                    selected_hidden[row : row + 1, :] = hidden_states[source_row : source_row + 1, :]

        if TP_SIZE > 1:
            # Publish this owner's selected rows to all head shards.
            for peer_tp in pl.range(TP_SIZE):
                pld.tensor.put(
                    dst=hidden_window,
                    peer=group_base + peer_tp,
                    src=selected_hidden,
                    dst_offsets=[tp_rank * MAX_LOGIT_ROWS + r0, 0],
                    src_offsets=[r0, 0],
                    shape=[PUSH_ROW_TILE, D],
                )

        # Notify folded into the push: one notify per block per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    if TP_SIZE > 1:
        # Join all published row slots before reusing the selection buffer.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[_dispatch_push_tid]
        ) as _dwait_tid:
            for owner_tp in pl.range(TP_SIZE):
                if owner_tp != tp_rank:
                    pld.system.wait(
                        signal=hidden_done,
                        offsets=[owner_tp, 0],
                        expected=pl.cast(done_epoch * (MAX_LOGIT_ROWS // PUSH_ROW_TILE), pl.INT32),
                        cmp=pld.WaitCmp.Ge,
                    )

        with pl.spmd(
            (GROUP_LOGIT_ROWS // HIDDEN_GATHER_ROW_TILE) * (D // HIDDEN_GATHER_TILE),
            name_hint="lm_head_dispatch_gather",
            deps=[_dwait_tid, _dispatch_push_tid],
        ):
            gblock = pl.tile.get_block_idx()
            gk0 = (gblock % (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_TILE
            gr0 = (gblock // (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_ROW_TILE
            selected_hidden[gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[
                gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE
            ]

    packed_hidden_grid = pl.reshape(selected_hidden_packed, [D // FUSED_K_TILE * GROUP_LOGIT_ROWS, FUSED_K_TILE])

    # Fused cube+comm kernel: matmul one [MM_ROW_TILE, FUSED_VOCAB_TILE] tile,
    # carry the accumulator across the C->V edge with pl.aiv_shard, and
    # remote_store each lane's row half to its owner's window while the cube
    # projects the next tile. No GM staging: every row block is one owner's rows,
    # so each lane's shard is already the exact rows it pushes.
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        deps=[_dispatch_push_tid],
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=2, local_slot_num=1)],
    ) as _push_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(lm_head_weight, pl.CachePolicy.BYPASS)
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_FULL_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            for mm_rb in pl.range(GROUP_LOGIT_ROWS // MM_ROW_TILE):
                mm_r0 = mm_rb * MM_ROW_TILE
                mm_acc = pl.create_tensor([MM_ROW_TILE, FUSED_VOCAB_TILE], dtype=pl.FP32)
                for mm_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                    mm_k0 = mm_kb * FUSED_K_TILE
                    if TP_SIZE == 1:
                        mm_packed_r0 = mm_kb * GROUP_LOGIT_ROWS + mm_r0
                        mm_hidden_tile = packed_hidden_grid[mm_packed_r0 : mm_packed_r0 + MM_ROW_TILE, :]
                    else:
                        mm_hidden_tile = selected_hidden[mm_r0 : mm_r0 + MM_ROW_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_weight_tile = lm_head_weight[mm_o0 : mm_o0 + FUSED_VOCAB_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_acc = pl.matmul_acc(mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True, init_cond=(mm_kb == 0))

                if TP_SIZE == 1:
                    logits[mm_r0 : mm_r0 + MM_ROW_TILE, mm_o0 : mm_o0 + FUSED_VOCAB_TILE] = mm_acc
                else:
                    # Push each lane's contiguous rows to their owner.
                    for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                        mm_shard = pl.aiv_shard(mm_acc)
                        owner_tp = mm_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                        owner_r0 = (mm_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)) * MM_ROW_TILE + aiv_id * LANE_ROWS
                        pld.tensor.remote_store(
                            mm_shard, logits_window, group_base + owner_tp,
                            [owner_r0, vocab_base + mm_o0],
                        )

        # Ragged tail: its own matmul on one block; the accumulator is already
        # VOCAB_TAIL wide. Hoisted out of the strided loop: a dynamic branch would
        # give the accumulator two static shapes across the C->V edge.
        if VOCAB_TAIL != 0:
            if lm_core == VOCAB_FULL_TILES % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_FULL_TILES * FUSED_VOCAB_TILE
                for tail_rb in pl.range(GROUP_LOGIT_ROWS // MM_ROW_TILE):
                    tail_r0 = tail_rb * MM_ROW_TILE
                    tail_acc = pl.create_tensor([MM_ROW_TILE, VOCAB_TAIL], dtype=pl.FP32)
                    for tail_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                        tail_k0 = tail_kb * FUSED_K_TILE
                        tail_hidden_tile = selected_hidden[
                            tail_r0 : tail_r0 + MM_ROW_TILE, tail_k0 : tail_k0 + FUSED_K_TILE
                        ]
                        tail_weight_tile = lm_head_weight[
                            mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL,
                            tail_k0 : tail_k0 + FUSED_K_TILE,
                        ]
                        tail_acc = pl.matmul_acc(tail_acc, tail_hidden_tile, tail_weight_tile, b_trans=True, init_cond=(tail_kb == 0))

                    if TP_SIZE == 1:
                        logits[tail_r0 : tail_r0 + MM_ROW_TILE, mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL] = tail_acc
                    else:
                        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            tail_shard = pl.aiv_shard(tail_acc)
                            owner_tp = tail_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                            owner_r0 = (tail_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)) * MM_ROW_TILE + aiv_id * LANE_ROWS
                            pld.tensor.remote_store(
                                tail_shard, logits_window, group_base + owner_tp,
                                [owner_r0, vocab_base + mm_tail_o0],
                            )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(TP_SIZE // AIV_LANES):
                owner_tp = notify_pair * AIV_LANES + aiv_id
                if owner_tp != tp_rank:
                    pld.system.notify(
                        target=logits_done,
                        peer=group_base + owner_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    if TP_SIZE > 1:
        # Wait for every peer vocabulary shard.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid]
        ) as _cwait_tid:
            for src_tp in pl.range(TP_SIZE):
                if src_tp != tp_rank:
                    pld.system.wait(
                        signal=logits_done,
                        offsets=[src_tp, 0],
                        expected=pl.cast(done_epoch * FUSED_LM_HEAD_CORES, pl.INT32),
                        cmp=pld.WaitCmp.Ge,
                    )

        # Assemble full-vocabulary logits from the group-local window.
        with pl.spmd(
            LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", deps=[_cwait_tid]
        ):
            gblk = pl.tile.get_block_idx()
            for src_tp in pl.range(TP_SIZE):
                src_vocab_base = src_tp * VOCAB_PER_TP
                for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                    o0 = ob * LOGITS_COMM_TILE
                    lo = src_vocab_base + o0
                    for gr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                        logits[gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE] = logits_window[
                            gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE
                        ]

                if LOGITS_COMM_TAIL != 0:
                    if gblk == LOGITS_TAIL_BLOCK:
                        tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                        tl = src_vocab_base + tail_o0
                        for tr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                            logits[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL] = logits_window[
                                tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL
                            ]

    # The logits read joins the final writer before clearing local counters.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_signal_clear",
        deps=[_push_tid],
    ) as _clear_tid:
        _logits_anchor = pl.read(logits, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits, _clear_tid


@pl.jit
def l2_lm_head(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_input_ready",
    ) as hidden_ready_tid:
        _input_anchor = pl.read(hidden_states, [0, 0])
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch, hidden_ready_tid,
    )
    return logits


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Select the first maximum allowed token id per active row; inactive rows read -1.

    The mask uses 16 bits per word for each greedy scan segment. The final
    padding word of segment zero is -1 for an unconstrained row and 0 for a
    constrained row; it never overlaps vocabulary bits. An all--1 mask takes
    the native argmax fast path without expanding the bitset.
    """
    # One pass per row into a [BLOCK_ROWS, ROW_WIDTH] accumulator carrying the
    # running maximum and the block that set it; a lane's column is its position.
    # The inactive-row mask rides the same block: a dead row skips the scan
    # instead of being overwritten by a second pass over sampled_ids.
    logits_grid = pl.reshape(logits, [MAX_LOGIT_ROWS * GREEDY_GRID_ROWS, GREEDY_ROW_WIDTH])
    grammar_grid = pl.reshape(grammar_mask, [MAX_LOGIT_ROWS * GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS])
    for sample_worker in pl.spmd(GREEDY_SAMPLE_CORES, name_hint="lm_head_greedy_sample"):
        for row in pl.range(sample_worker, MAX_LOGIT_ROWS, GREEDY_SAMPLE_CORES):
            if pl.read(logit_row_indices, [row]) < 0:
                sampled_ids[row : row + 1, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=-1)
            else:
                # The last padding word of segment zero is a row-mode header.
                # -1 selects the native argmax without reading or expanding mask bits.
                if pl.read(grammar_mask, [row, 0, GRAMMAR_SEGMENT_WORDS - 1]) == -1:
                    row_base = row * GREEDY_GRID_ROWS
                    max_tile = pl.tile.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], pl.FP32, FP32_NEG_INF)
                    base_tile = pl.tile.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], pl.INT32, 0)
                    for block in pl.range(GREEDY_GRID_ROWS // GREEDY_BLOCK_ROWS):
                        block_row = row_base + block * GREEDY_BLOCK_ROWS
                        scores_tile = pl.tile.load(logits_grid, [block_row, 0], [GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH])
                        # Strict greater-than retains the earliest block for each lane.
                        newer_tile = pl.tile.cmp(scores_tile, max_tile, cmp_type=4)
                        max_tile = pl.maximum(max_tile, scores_tile)
                        block_base_i32 = pl.cast(block * GREEDY_BLOCK_SPAN, pl.INT32)
                        newer_native_full = pl.set_validshape(newer_tile, GREEDY_BLOCK_ROWS, GREEDY_MASK_EVEN_BYTES)
                        for native_mask_row in pl.unroll(GREEDY_BLOCK_ROWS):
                            pl.write(newer_native_full, [native_mask_row, GREEDY_MASK_VALID_BYTES], pl.cast(0, pl.UINT8))
                        newer_native_i16 = pl.reinterpret_view(newer_native_full, pl.INT16)
                        retain_native = pl.tile.not_(newer_native_i16)
                        native_select_scratch = pl.tile.create([1, GREEDY_ROW_WIDTH], pl.INT32)
                        base_tile = pl.tile.sels(retain_native, base_tile, native_select_scratch, block_base_i32)

                    max_scratch = pl.tile.create([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], pl.FP32)
                    maxima_tile = pl.row_max(max_tile, max_scratch)
                    broadcast_zeros = pl.tile.full([GREEDY_BLOCK_ROWS, SAMPLED_IDS_PAD], pl.FP32, 0.0)
                    broadcast_tile = pl.row_expand_add(broadcast_zeros, maxima_tile)
                    best_value_tile = pl.read(pl.col_max(broadcast_tile), [0, 0])
                    ramp_tile = pl.tile.arange(0, [1, GREEDY_ROW_WIDTH], pl.INT32)
                    columns_tile = pl.col_expand(base_tile, ramp_tile)
                    indices_tile = pl.add(base_tile, columns_tile)
                    indices_f32 = pl.cast(indices_tile, pl.FP32)
                    hits_tile = pl.tile.cmps(max_tile, best_value_tile, cmp_type=0)
                    candidates_tile = pl.tile.select(hits_tile, indices_f32, GREEDY_INDEX_SENTINEL_FP32)
                    min_scratch = pl.tile.create([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], pl.FP32)
                    lane_min_f32 = pl.row_min(candidates_tile, min_scratch)
                    lane_indices_tile = pl.cast(lane_min_f32, pl.INT32)
                    best_index = pl.read(lane_indices_tile, [0, 0])
                    for lane in pl.range(1, GREEDY_BLOCK_ROWS):
                        lane_term = pl.cast(lane * GREEDY_ROW_WIDTH, pl.INT32)
                        lane_best = pl.read(lane_indices_tile, [lane, 0]) + lane_term
                        best_index = pl.min(best_index, lane_best)

                else:
                    row_base = row * GREEDY_GRID_ROWS
                    grammar_max = pl.tile.full([GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH], pl.FP32, FP32_NEG_INF)
                    grammar_max = pl.set_validshape(grammar_max, GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH)
                    grammar_block = pl.tile.full([GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH], pl.INT16, 0)
                    grammar_block = pl.set_validshape(grammar_block, GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH)
                    for block in pl.range(GREEDY_GRID_ROWS // GREEDY_BLOCK_ROWS):
                        block_row = row_base + block * GREEDY_BLOCK_ROWS
                        words_tile = pl.tile.load(grammar_grid, [block_row, 0], [GREEDY_BLOCK_ROWS, GRAMMAR_SEGMENT_WORDS])
                        grammar_scores = pl.tile.load(
                            logits_grid, [block_row, 0], [GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH],
                            valid_shape=[GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH],
                        )
                        penalized_scores = pl.add(grammar_scores, -1.0e30)
                        # Each 128-byte grammar row is the packed token predicate.
                        grammar_mask_scratch = pl.tile.create([1, 16], pl.UINT32)
                        masked_scores = pl.tile.sel(words_tile, grammar_scores, penalized_scores, grammar_mask_scratch)
                        newer = pl.tile.cmp(masked_scores, grammar_max, cmp_type=4)
                        grammar_max = pl.maximum(grammar_max, masked_scores)
                        block_i16 = pl.cast(pl.cast(block, pl.INT32), pl.INT16)
                        newer_grammar_full = pl.set_validshape(newer, GREEDY_BLOCK_ROWS, GREEDY_MASK_EVEN_BYTES)
                        for grammar_mask_row in pl.unroll(GREEDY_BLOCK_ROWS):
                            pl.write(newer_grammar_full, [grammar_mask_row, GREEDY_MASK_VALID_BYTES], pl.cast(0, pl.UINT8))
                        newer_grammar_i16 = pl.reinterpret_view(newer_grammar_full, pl.INT16)
                        retain_grammar = pl.tile.not_(newer_grammar_i16)
                        grammar_select_scratch = pl.tile.create([1, GRAMMAR_LOOKUP_WIDTH], pl.INT16)
                        grammar_block = pl.tile.sels(retain_grammar, grammar_block, grammar_select_scratch, block_i16)

                    grammar_max_scratch = pl.tile.create([GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH], pl.FP32)
                    lane_maxima = pl.row_max(grammar_max, grammar_max_scratch)
                    lane_zeros = pl.tile.full([GREEDY_BLOCK_ROWS, SAMPLED_IDS_PAD], pl.FP32, 0.0)
                    lane_broadcast = pl.row_expand_add(lane_zeros, lane_maxima)
                    best_value = pl.read(pl.col_max(lane_broadcast), [0, 0])
                    block_f32 = pl.cast(grammar_block, pl.FP32)
                    base_i32 = pl.mul(pl.cast(block_f32, pl.INT32), GREEDY_BLOCK_SPAN)
                    final_columns = pl.tile.arange(0, [1, GRAMMAR_LOOKUP_WIDTH], pl.INT32)
                    final_columns = pl.set_validshape(final_columns, 1, GREEDY_ROW_WIDTH)
                    final_shape = pl.tile.create([GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH], pl.INT32)
                    final_shape = pl.set_validshape(final_shape, GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH)
                    final_column_grid = pl.col_expand(final_shape, final_columns)
                    flat_index = pl.add(base_i32, final_column_grid)
                    flat_index_f32 = pl.cast(flat_index, pl.FP32)
                    hit = pl.tile.cmps(grammar_max, best_value, cmp_type=0)
                    candidates = pl.tile.select(hit, flat_index_f32, GREEDY_INDEX_SENTINEL_FP32)
                    grammar_min_scratch = pl.tile.create([GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH], pl.FP32)
                    grammar_lane_min_f32 = pl.row_min(candidates, grammar_min_scratch)
                    lane_indices = pl.cast(grammar_lane_min_f32, pl.INT32)
                    best_index = pl.read(lane_indices, [0, 0])
                    for lane in pl.range(1, GREEDY_BLOCK_ROWS):
                        lane_term = pl.cast(lane * GREEDY_ROW_WIDTH, pl.INT32)
                        lane_best = pl.read(lane_indices, [lane, 0]) + lane_term
                        best_index = pl.min(best_index, lane_best)

                sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
                sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
                pl.write(sampled_row, [0, 0], best_index)
                sampled_ids[row : row + 1, :] = sampled_row

    return sampled_ids


@pl.jit
def l2_lm_head_sample(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Projection plus greedy sampling: the decode-forward tail in one entry."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_input_ready",
    ) as hidden_ready_tid:
        _input_anchor = pl.read(hidden_states, [0, 0])
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch, hidden_ready_tid,
    )
    greedy_sample(logits, logit_row_indices, grammar_mask, sampled_ids)
    return logits, sampled_ids


@pl.jit.host
def l3_lm_head_sample(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_lm_head_sample(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r], grammar_mask[r], sampled_ids[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


def golden_lm_head_sample(tensors):
    import torch

    golden_lm_head(tensors)
    logits = tensors["logits"]
    indices = tensors["logit_row_indices"]
    sampled = torch.full(
        (WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), -1, dtype=torch.int32,
    )
    for rank in range(WORLD_SIZE):
        for row in range(MAX_LOGIT_ROWS):
            if int(indices[rank, row]) < 0:
                continue
            sampled[rank, row, :] = 0
            sampled[rank, row, 0] = expected_sampled_id(logits[rank, row], tensors["grammar_mask"][rank, row])
    tensors["sampled_ids"][:] = sampled


def expected_sampled_id(logits, grammar_mask):
    """Compute the masked argmax for a golden or device-output comparison."""
    import torch

    if int(grammar_mask[0, GRAMMAR_SEGMENT_WORDS - 1]) == -1:
        return int(torch.argmax(logits))
    token_ids = torch.arange(VOCAB, device=logits.device)
    segment = token_ids // GREEDY_ROW_WIDTH
    offset = token_ids % GREEDY_ROW_WIDTH
    words = grammar_mask[segment, offset // 16].to(torch.int32)
    allowed = (words & (1 << (offset % 16))) != 0
    return int(torch.argmax(logits + (~allowed).to(logits.dtype) * -1.0e30))


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
):
    # Windows are group-local: hidden_window holds one row slot per group member,
    # and every card receives only its own full-vocabulary logits.
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_lm_head(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


def golden_lm_head(tensors):
    import torch

    from utils import unpack_nz

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE; concatenating shards in index order
    # reproduces the global vocabulary order.
    weight = unpack_nz(tensors["lm_head_weight"]).float()
    full_weight = torch.cat([weight[tp] for tp in range(TP_SIZE)], dim=0)
    full_logits = []
    for owner_rank in range(WORLD_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight.t()))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def build_tensor_specs(num_tokens=TEST_TOKENS):
    import torch
    from golden import TensorSpec
    from utils import pack_nz

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(WORLD_SIZE, TEST_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        shards = (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        return pack_nz(torch.stack([shards[r % TP_SIZE] for r in range(WORLD_SIZE)], dim=0))

    def init_logit_row_indices():
        indices = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        return indices

    return [
        TensorSpec("hidden_states", [WORLD_SIZE, TEST_TOKENS, D], torch.bfloat16, init_value=init_hidden_states),
        # One vocab shard per DP rank: card r carries a copy of shard
        # r % TP_SIZE, matching how resident args are handed out per rank. Keep
        # each rank-local shard on its consuming card across dispatches.
        TensorSpec(
            "lm_head_weight", [WORLD_SIZE, VOCAB_PER_TP, D], torch.bfloat16,
            init_value=init_lm_head_weight, resident="stacked",
        ),
        TensorSpec("logits", [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], torch.float32),
        TensorSpec("logit_row_indices", [WORLD_SIZE, MAX_LOGIT_ROWS], torch.int32, init_value=init_logit_row_indices),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES), help="DP groups (world size = tp * dp)")
    parser.add_argument("--num-tokens", type=int, default=TEST_TOKENS, help="Active hidden rows each owner projects")
    parser.add_argument("--entry", type=str, default="projection", choices=("projection", "sample"))
    device_default = ",".join(str(i) for i in range(WORLD_SIZE))
    parser.add_argument("-d", "--device", type=str, default=device_default, help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = WORLD_SIZE
    assert len(device_ids) >= required_devices, f"need at least {required_devices} devices, got {device_ids}"
    assert args.tp == TP_SIZE and args.dp == DP_SIZE
    assert 1 <= args.num_tokens <= TEST_TOKENS

    fn = l3_lm_head
    specs = build_tensor_specs(args.num_tokens)
    golden_fn = golden_lm_head
    if args.entry == "sample":
        import torch

        from golden import TensorSpec

        fn = l3_lm_head_sample
        golden_fn = golden_lm_head_sample

        def init_grammar_mask():
            mask = torch.full(
                (WORLD_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS),
                -1, dtype=torch.int16,
            )
            mask[0, 0] = 0
            mask[0, 0, 0] = -32768  # Only token 15 is allowed in this row.
            return mask

        specs.append(
            TensorSpec(
                "grammar_mask", [WORLD_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], torch.int16,
                init_value=init_grammar_mask,
            ),
        )
        specs.append(
            TensorSpec(
                "sampled_ids", [WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32,
            ),
        )

    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:required_devices],
                num_sub_workers=0,
            ),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            ring_heap=LM_HEAD_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
