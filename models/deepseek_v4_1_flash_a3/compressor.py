# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compressor projections, ratio pooling and cache publication."""

from math import gcd

import pypto.language as pl
from .config import FLASH as M
from .gemm import VEC_M_TILE
from .operator_config import B_DYN, Q_START_DYN, STATE_BLOCKS_DYN, T_DYN
from .qkv_proj_rope import make_bf16_projection_staged, make_decode_bf16_projection_staged as decode_make_bf16_projection_staged, make_decode_qkv_w8a8_inline as decode_make_qkv_w8a8_inline, make_qkv_w8a8_inline
from .quant import make_decode_quantize as decode_make_quantize, make_quantize
from .rmsnorm import make_blocked_rmsnorm as make_norm, make_decode_blocked_rmsnorm as decode_make_norm
from .rope import make_adjacent_rope as make_rope, make_decode_adjacent_rope as decode_make_rope


STATE_WIDTH = 2 * M.head_dim


RING_ROWS_DYN = pl.dynamic("V41_COMPRESSOR_RING_ROWS_DYN")


HEAD_DIM = M.head_dim


EPS = M.rms_norm_eps


@pl.jit.inline
def compressor_pool(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    num_state_blocks: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
):
    """Assemble partners and pool eight rows before updating ring state."""
    ring_rows = pl.tensor.dim(state_cache, 1)
    state_rows = num_state_blocks * ring_rows
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(
        pl.max((num_tokens + 7) // 8, 1), name_hint="c2a_compressor_pool", deps=[pair_ready]
    ) as pool_tid:
        group = pl.tile.get_block_idx()
        t = group * 8
        if t < num_tokens:
            rows = pl.min(8, num_tokens - t)
            current_kv = pl.slice(kv_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_kv = pl.set_validshape(pl.fillpad(current_kv, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            current_score = pl.slice(score_proj, [8, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_score = pl.set_validshape(
                pl.fillpad(current_score, pad_value=pl.PadValue.zero), 8, HEAD_DIM
            )
            previous_kv = pl.full([8, HEAD_DIM], dtype=pl.FP32, value=0.0)
            previous_score = pl.full([8, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(8):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                if token > start:
                                    partner_kv = kv_proj[token - 1 : token, :]
                                    partner_score = score_proj[token - 1 : token, :]
                                else:
                                    slot = pl.cast(block * ring_rows + (position - 1) % ring_rows, pl.INDEX)
                                    partner_kv = flat[slot : slot + 1, :HEAD_DIM]
                                    partner_score = flat[slot : slot + 1, HEAD_DIM:]
                                previous_kv[lane : lane + 1, :] = partner_kv
                                previous_score[lane : lane + 1, :] = partner_score
            highest = pl.maximum(previous_score, current_score)
            previous_weight = pl.exp(pl.sub(previous_score, highest))
            current_weight = pl.exp(pl.sub(current_score, highest))
            weighted = pl.add(pl.mul(previous_kv, previous_weight), pl.mul(current_kv, current_weight))
            pooled = pl.div(weighted, pl.add(previous_weight, current_weight))
            # Round the pooled row to BF16 before normalizing.
            value = pl.cast(pl.cast(pooled, pl.BF16, mode="rint"), pl.FP32)
            square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / HEAD_DIM)
            inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
            gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, HEAD_DIM])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
            parity = pl.full([8, 8], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(8):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                pl.write(parity, [lane, 0], 1.0)
            parity = pl.row_sum(parity)
            published = pl.row_expand_mul(normalized, parity)
            latent[t : t + 8, :] = pl.set_validshape(pl.cast(published, pl.BF16, mode="rint"), rows, HEAD_DIM)
    return pool_tid


@pl.jit.inline
def compressor_state_write(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
):
    """Save only the final capacity rows of each chunk after all historical reads."""
    blocks = pl.tensor.dim(state_cache, 0)
    ring_rows = pl.tensor.dim(state_cache, 1)
    state_rows = blocks * ring_rows
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(pl.max(num_tokens, 1), name_hint="c2a_compressor_state", deps=[pair_ready]) as state_tid:
        t = pl.tile.get_block_idx()
        if t < num_tokens:
            request = pl.cast(pl.read(token_to_req_indices, [t]), pl.INDEX)
            if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                if block >= 0 and block < blocks:
                    start = pl.read(query_start_loc, [request])
                    end = pl.min(pl.read(query_start_loc, [request + 1]), num_tokens)
                    position = pl.read(position_ids, [t])
                    if t >= start and t < end and t >= end - ring_rows and position >= 0:
                        slot = pl.cast(block * ring_rows + position % ring_rows, pl.INDEX)
                        flat[slot : slot + 1, :HEAD_DIM] = kv_proj[t : t + 1, :]
                        flat[slot : slot + 1, HEAD_DIM:] = score_proj[t : t + 1, :]
    return state_tid


@pl.jit.inline
def compressor_ratio2(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
):
    """Pool completed pairs and persist request state using a shared prefill/decode ABI."""
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    pool_tid = compressor_pool(
        kv_proj,
        score_proj,
        state_cache,
        position_ids,
        norm_weight,
        latent,
        num_tokens,
        query_start_loc,
        token_to_req_indices,
        state_block_table,
        blocks,
        state_ready,
    )
    state_tid = compressor_state_write(
        kv_proj,
        score_proj,
        query_start_loc,
        position_ids,
        token_to_req_indices,
        state_block_table,
        state_cache,
        num_tokens,
        pool_tid,
    )
    return pool_tid, state_tid


def make_compressed_cache_publish(capacity, blocks, width):
    """Publish BF16 compressed KV rows into PA_BBND pages.

    ``slots`` contains the flattened page row (``page * 128 + offset``), as
    produced by the vLLM A2 scheduler.  A negative slot is padding and leaves
    the cache byte-for-byte untouched.
    """
    rows = blocks * 128

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [rows, width])
        with pl.spmd(active, name_hint="a2_c2_bf16_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                flat[slot : slot + 1, :] = value[token : token + 1, :]
        return cache

    return publish


def make_indexer_cache_publish(capacity, blocks, width):
    """Quantize index keys to INT8/FP16 and publish both cache tensors.

    Ascend's non-A5 indexer path uses one dynamic scale for each key row.  The
    A2 PyPTO representation stores that scale as ``[blocks,128,1,1]`` FP16,
    matching ``indexer_quant_scatter`` in vLLM-Ascend.  Quantization is the
    same symmetric per-token INT8 operation used by the A2 W8A8 projections.
    """
    rows = blocks * 128
    quantize = make_quantize(capacity, width)

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        row_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(value, payload, row_scales, active)
        cache_flat = pl.reshape(cache, [rows, width])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_int8_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        # A scalar FP16 row is only two bytes, while Ascend writes a full
        # cache line.  Serialize these writes so adjacent rows cannot race.
        with pl.spmd(1, name_hint="a2_c1_int8_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    scale = pl.cast(pl.read(row_scales, [token, 0]), pl.FP16)
                    pl.write(scale_flat, [slot, 0], scale)
        return cache, scales

    return publish


def make_index_cache_publish_observed(capacity, blocks, page_rows=64):
    """Quantize padded index rows and publish codes and FP16 scales with one writer."""
    width = 128
    quantize = make_quantize(capacity, width, rows_tile=16)

    @pl.jit
    def index_cache_publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT32],
        cache: pl.InOut[pl.Tensor[[blocks, page_rows, 1, width], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[blocks, page_rows, 1, 1], pl.FP16]],
        payload: pl.InOut[pl.Tensor[[capacity, width], pl.INT8]],
        row_scales: pl.InOut[pl.Tensor[[capacity, 8], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, row_scales, active)
        flat_cache = pl.reshape(cache, [blocks * page_rows, width])
        flat_scales = pl.reshape(scales, [blocks * page_rows, 1])
        scale_values = pl.reshape(row_scales, [1, capacity * 8])
        with pl.spmd(1, name_hint="index_fused_publication"):
            _writer = pl.tile.get_block_idx()
            staged_payload = payload[:, :]
            narrowed_scales = pl.cast(scale_values[:, :], pl.FP16, mode="rint")
            for token in pl.range(active):
                block = pl.read(slots, [token, 0])
                offset = pl.read(slots, [token, 1])
                if block >= 0 and offset >= 0:
                    row = pl.cast(block * page_rows + offset, pl.INDEX)
                    flat_cache[row : row + 1, :] = staged_payload[token : token + 1, :]
                    publication_scale = pl.read(narrowed_scales, [0, token * 8])
                    pl.write(flat_scales, [row, 0], publication_scale)
        return cache, scales, payload, row_scales

    return index_cache_publish


def make_decode_indexer_cache_publish(capacity, blocks, width):
    """Quantize index keys to INT8/FP16 and publish both cache tensors.

    Ascend's non-A5 indexer path uses one dynamic scale for each key row.  The
    A2 PyPTO representation stores that scale as ``[blocks,128,1,1]`` FP16,
    matching ``indexer_quant_scatter`` in vLLM-Ascend.  Quantization is the
    same symmetric per-token INT8 operation used by the A2 W8A8 projections.
    """
    rows = blocks * 128
    quantize = decode_make_quantize(capacity, width)

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        row_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(value, payload, row_scales, active)
        cache_flat = pl.reshape(cache, [rows, width])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_int8_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        # A scalar FP16 row is only two bytes, while Ascend writes a full
        # cache line.  Serialize these writes so adjacent rows cannot race.
        with pl.spmd(1, name_hint="a2_c1_int8_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    scale = pl.cast(pl.read(row_scales, [token, 0]), pl.FP16)
                    pl.write(scale_flat, [slot, 0], scale)
        return cache, scales

    return publish


COMPRESSOR_PROJECTION_HEAD_DIM = 512
INDEX_WIDTH = 128
PROJECTION_ROWS_DYN = pl.dynamic("V41_PROJECTION_ROWS_DYN")


def _make_fp32_projection_tile(width, output_width):
    """Accumulate one K partition of a BF16 projection into FP32 scratch."""
    m_tile, n_tile = 16, min(256, output_width)
    partition_width = gcd(1024, width)
    partitions = width // partition_width
    k_tile = gcd(256, partition_width)

    @pl.jit.inline
    def project_tile(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[PROJECTION_ROWS_DYN, output_width], pl.FP32],
        active: pl.Scalar[pl.INT32],
        block: pl.Scalar[pl.INT32],
    ):
        partition = block % partitions
        slab = block // partitions
        row = slab // (output_width // n_tile) * m_tile
        col = slab % (output_width // n_tile) * n_tile
        rows = pl.min(m_tile, active - row)
        acc = pl.create_tensor([m_tile, n_tile], dtype=pl.FP32)
        for step in pl.pipeline(0, partition_width, k_tile, stage=2):
            k0 = partition * partition_width + step
            a = pl.slice(source, [m_tile, k_tile], [row, k0], valid_shape=[rows, k_tile])
            b = pl.slice(weight, [k_tile, n_tile], [k0, col])
            acc = pl.matmul_acc(acc, a, b, init_cond=(step == 0))
        output_row = partition * pl.tensor.dim(source, 0) + row
        output[output_row : output_row + m_tile, col : col + n_tile] = pl.set_validshape(acc, rows, n_tile)
        return output

    return project_tile


def _make_fp32_projection_reduce(width, output_width):
    """Reduce FP32 projection partitions in order across each active output row."""
    partitions = width // gcd(1024, width)
    rows_tile = 8

    @pl.jit.inline
    def reduce_projection(
        partials: pl.Tensor[[PROJECTION_ROWS_DYN, output_width], pl.FP32],
        output: pl.Tensor[[T_DYN, output_width], pl.FP32],
        active: pl.Scalar[pl.INT32],
        block: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(output, 0)
        row = block * rows_tile
        rows = pl.min(rows_tile, active - row)
        total = pl.slice(partials, [rows_tile, output_width], [row, 0], valid_shape=[rows, output_width])
        for partition in pl.range(1, partitions):
            partial_row = partition * tokens + row
            part = pl.slice(partials, [rows_tile, output_width], [partial_row, 0], valid_shape=[rows, output_width])
            total = pl.add(total, part)
        output[row : row + rows_tile, :] = pl.set_validshape(total, rows, output_width)
        return output

    return reduce_projection


def make_fp32_projection(width, output_width):
    """Project BF16 operands with FP32 accumulation and output."""
    m_tile, n_tile = 16, min(256, output_width)
    partitions = width // gcd(1024, width)
    project_tile = _make_fp32_projection_tile(width, output_width)
    reduce_projection = _make_fp32_projection_reduce(width, output_width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        partial_rows = partitions * pl.tensor.dim(source, 0)
        partials = pl.create_tensor([partial_rows, output_width], dtype=pl.FP32)
        blocks = (active + m_tile - 1) // m_tile * (output_width // n_tile) * partitions
        with pl.spmd(blocks, name_hint="a2_fp32_projection_partial", allow_early_resolve=True) as partial_tid:
            block = pl.tile.get_block_idx()
            project_tile(source, weight, partials, active, block)
        with pl.spmd((active + 7) // 8, name_hint="a2_fp32_projection_reduce", deps=[partial_tid], allow_early_resolve=True) as projection_tid:
            block = pl.tile.get_block_idx()
            reduce_projection(partials, output, active, block)
        return projection_tid

    return project


def make_indexer_cache_publish_128(capacity, blocks):
    """Publish width-128 rows with the historical reciprocal rounding boundary."""
    rows = blocks * 128
    QUANT_M_TILE = 16

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.INT8)
        row_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        with pl.spmd((active + QUANT_M_TILE - 1) // QUANT_M_TILE, name_hint="a2_c1_index_quant"):
            block = pl.tile.get_block_idx()
            t0 = block * QUANT_M_TILE
            rows_valid = pl.min(QUANT_M_TILE, active - t0)
            source = pl.slice(
                value, [QUANT_M_TILE, INDEX_WIDTH], [t0, 0], valid_shape=[rows_valid, INDEX_WIDTH]
            )
            source = pl.set_validshape(
                pl.fillpad(source, pad_value=pl.PadValue.zero), QUANT_M_TILE, INDEX_WIDTH
            )
            value_fp32 = pl.cast(source, pl.FP32)
            maximum_row = pl.reshape(pl.row_max(pl.abs(value_fp32)), [1, QUANT_M_TILE])
            maximum_row = pl.maximum(maximum_row, 1e-30)
            scale = pl.reshape(pl.mul(maximum_row, 1.0 / 127.0), [QUANT_M_TILE, 1])
            # Match scalar Torch 127 / maximum: rounded unit reciprocal then 127.
            # Split the approximate reciprocal so BF16-maximum products retain
            # their residual on A2 without an FMA or A5-only precision flag.
            reciprocal = pl.div(pl.full([1, QUANT_M_TILE], dtype=pl.FP32, value=1.0), maximum_row)
            reciprocal_hi = pl.cast(pl.cast(reciprocal, pl.BF16, mode="rint"), pl.FP32)
            reciprocal_lo = pl.sub(reciprocal, reciprocal_hi)
            residual_hi = pl.sub(
                pl.full([1, QUANT_M_TILE], dtype=pl.FP32, value=1.0), pl.mul(reciprocal_hi, maximum_row)
            )
            residual = pl.sub(residual_hi, pl.mul(reciprocal_lo, maximum_row))
            refined_reciprocal = pl.add(reciprocal, pl.mul(residual, reciprocal))
            inverse = pl.reshape(pl.mul(refined_reciprocal, 127.0), [QUANT_M_TILE, 1])
            normalized = pl.row_expand_mul(value_fp32, inverse)
            rounded = pl.cast(pl.minimum(pl.maximum(normalized, -127.0), 127.0), pl.INT32, mode="rint")
            payload[t0 : t0 + QUANT_M_TILE, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), rows_valid, INDEX_WIDTH
            )
            row_scales[t0 : t0 + QUANT_M_TILE, :] = pl.row_expand_mul(
                pl.full([QUANT_M_TILE, 8], dtype=pl.FP32, value=1.0), scale
            )
        cache_flat = pl.reshape(cache, [rows, INDEX_WIDTH])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_index_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        # A scalar FP16 row is narrower than the device cache line.  Publish
        # scales from one worker so writes to adjacent slots cannot race and
        # silently restore a neighboring row's sentinel values.
        with pl.spmd(1, name_hint="a2_c1_index_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    pl.write(scale_flat, [slot, 0], pl.cast(pl.read(row_scales, [token, 0]), pl.FP16))
        return cache, scales

    return publish


def make_c2_index_publish_projected_inline(capacity, config, cache_blocks=2):
    """Pool projected FP32 rows, then normalize, rotate and publish caches."""
    pairs = config["qk_rope_head_dim"] // 2
    eps = config["rms_norm_eps"]
    project_index = pl.jit.inline(
        make_bf16_projection_staged(COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH)._func
    )
    norm_index = make_norm(capacity, INDEX_WIDTH, eps)
    rope_index = make_rope(capacity, 1, INDEX_WIDTH, config["qk_rope_head_dim"])
    rope_compressed = make_rope(capacity, 1, COMPRESSOR_PROJECTION_HEAD_DIM, config["qk_rope_head_dim"])
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, COMPRESSOR_PROJECTION_HEAD_DIM)
    publish_index = make_indexer_cache_publish_128(capacity, cache_blocks)

    @pl.jit.inline
    def publish(
        kv_proj: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        score_proj: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        compressor_norm: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        active: pl.Scalar[pl.INT32],
        projections_ready: pl.Scalar[pl.TASK_ID],
    ):
        norm_ones = pl.create_tensor([1, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16)
        for norm_row in pl.spmd(1, name_hint="a2_c2_unweighted_norm"):
            norm_ones[norm_row : norm_row + 1, :] = pl.full(
                [1, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16, value=1.0
            )
        norm_ones_flat = pl.reshape(norm_ones, [COMPRESSOR_PROJECTION_HEAD_DIM])
        unweighted_compressed = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16)
        compressor_ratio2(
            kv_proj,
            score_proj,
            query_start,
            positions,
            requests,
            state_block_table,
            state,
            norm_ones_flat,
            unweighted_compressed,
            active,
            projections_ready,
        )
        # Native RMSNorm narrows the normalized value before its BF16 gamma
        # multiplication. Keep that boundary while sharing the existing pool.
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_c2_norm_gamma"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            value = unweighted_compressed[t0 : t0 + VEC_M_TILE, :]
            gamma = pl.reshape(pl.cast(compressor_norm[:], pl.FP32), [1, COMPRESSOR_PROJECTION_HEAD_DIM])
            weighted = pl.col_expand_mul(pl.cast(value, pl.FP32), gamma)
            compressed[t0 : t0 + VEC_M_TILE, :] = pl.set_validshape(
                pl.cast(weighted, pl.BF16, mode="rint"),
                rows,
                COMPRESSOR_PROJECTION_HEAD_DIM,
            )
        compressed_rotated = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16)
        rope_compressed(compressed, source_cos, source_sin, compressed_rotated, active)
        publish_compressed(compressed_rotated, compressed_slots, compressed_cache, active)

        projected_index = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, compressed_slots, index_cache, index_scale, active)
        return state, compressed, compressed_cache, index_key, index_cache, index_scale

    return publish


def make_c2_index_publish_inline(capacity, config, cache_blocks=2):
    """Project BF16 inputs and publish C2 and index caches."""
    hidden = config["hidden_size"]
    pairs = config["qk_rope_head_dim"] // 2
    project_kv = make_fp32_projection(hidden, COMPRESSOR_PROJECTION_HEAD_DIM)
    publish_projected = make_c2_index_publish_projected_inline(capacity, config, cache_blocks)

    @pl.jit.inline
    def publish(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        compressor_wkv: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_norm: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        kv_proj = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.FP32)
        score_proj = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.FP32)
        project_kv(x, compressor_wkv, kv_proj, active)
        score_ready = project_kv(x, compressor_wgate, score_proj, active)
        publish_projected(
            kv_proj,
            score_proj,
            compressor_norm,
            index_wk,
            index_norm,
            source_cos,
            source_sin,
            positions,
            compressed_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            compressed,
            index_key,
            active,
            score_ready,
        )
        return state, compressed, compressed_cache, index_key, index_cache, index_scale

    return publish


def make_layer2_c1c2(capacity, config):
    """Build QKV, ratio-2 C2 pooling and C1 index-key publication."""
    hidden = config["hidden_size"]
    latent_width = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    query_width = heads * head_dim
    pair_width = config["qk_rope_head_dim"] // 2
    qkv = make_qkv_w8a8_inline(capacity, config)
    publish_c2 = make_c2_index_publish_inline(capacity, config)

    @pl.jit
    def layer2_c1c2(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent_width // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent_width], pl.FP32],
        q_norm_weight: pl.Tensor[[latent_width], pl.BF16],
        wq_b: pl.Tensor[[latent_width * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pair_width], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pair_width], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        positions32: pl.Tensor[[capacity], pl.INT32],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent_width], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        qkv_cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        compressor_wkv: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_norm: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.InOut[
            pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[2, 128, 1, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16]],
        index_cache: pl.InOut[pl.Tensor[[2, 128, 1, INDEX_WIDTH], pl.INT8]],
        index_scale: pl.InOut[pl.Tensor[[2, 128, 1, 1], pl.FP16]],
        compressed: pl.InOut[pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        qkv(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            positions,
            query_latent,
            query,
            window_kv,
            qkv_cache,
            cache_lens,
            active,
        )

        publish_c2(
            x,
            compressor_wkv,
            compressor_wgate,
            compressor_norm,
            index_wk,
            index_norm,
            rope_cos,
            rope_sin,
            positions32,
            compressed_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            compressed,
            index_key,
            active,
        )
        return (
            positions,
            query_latent,
            query,
            window_kv,
            qkv_cache,
            cache_lens,
            state,
            compressed,
            compressed_cache,
            index_key,
            index_cache,
            index_scale,
        )

    return layer2_c1c2


def make_c2_projection(capacity, hidden):
    """Expose both full-width FP32 compressor projections from BF16 checkpoint operands."""
    project_tile = _make_fp32_projection_tile(hidden, COMPRESSOR_PROJECTION_HEAD_DIM)
    reduce_projection = _make_fp32_projection_reduce(hidden, COMPRESSOR_PROJECTION_HEAD_DIM)
    partitions = hidden // gcd(1024, hidden)
    partial_rows = partitions * capacity

    @pl.jit
    def c2_projection(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wkv: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        wgate: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        kv: pl.InOut[pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32]],
        gate: pl.InOut[pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        kv_partials = pl.create_tensor([partial_rows, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.FP32)
        gate_partials = pl.create_tensor([partial_rows, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.FP32)
        blocks = (active + 15) // 16 * (COMPRESSOR_PROJECTION_HEAD_DIM // 256) * partitions
        with pl.spmd(2 * blocks, name_hint="c2_paired_projection_partial", allow_early_resolve=True) as partial_tid:
            block = pl.tile.get_block_idx()
            projection_block = block // 2
            if block % 2 == 0:
                project_tile(x, wkv, kv_partials, active, projection_block)
            else:
                project_tile(x, wgate, gate_partials, active, projection_block)
        with pl.spmd(2 * ((active + 7) // 8), name_hint="c2_paired_projection_reduce", deps=[partial_tid], allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            row_block = block // 2
            if block % 2 == 0:
                reduce_projection(kv_partials, kv, active, row_block)
            else:
                reduce_projection(gate_partials, gate, active, row_block)
        return kv, gate

    return c2_projection


def make_decode_c2_index_publish_projected_inline(capacity, config, cache_blocks=2):
    """Pool projected FP32 rows, then normalize, rotate and publish caches."""
    pairs = config["qk_rope_head_dim"] // 2
    eps = config["rms_norm_eps"]
    project_index = decode_make_bf16_projection_staged(COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH)
    norm_index = decode_make_norm(capacity, INDEX_WIDTH, eps)
    rope_index = decode_make_rope(capacity, 1, INDEX_WIDTH, config["qk_rope_head_dim"])
    rope_compressed = decode_make_rope(
        capacity, 1, COMPRESSOR_PROJECTION_HEAD_DIM, config["qk_rope_head_dim"]
    )
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, COMPRESSOR_PROJECTION_HEAD_DIM)
    publish_index = make_indexer_cache_publish_128(capacity, cache_blocks)

    @pl.jit.inline
    def publish(
        kv_proj: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        score_proj: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        compressor_norm: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        index_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        active: pl.Scalar[pl.INT32],
        projections_ready: pl.Scalar[pl.TASK_ID],
    ):
        norm_ones = pl.create_tensor([1, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16)
        for norm_row in pl.spmd(1, name_hint="a2_c2_unweighted_norm"):
            norm_ones[norm_row : norm_row + 1, :] = pl.full(
                [1, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16, value=1.0
            )
        norm_ones_flat = pl.reshape(norm_ones, [COMPRESSOR_PROJECTION_HEAD_DIM])
        unweighted_compressed = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16)
        compressor_ratio2(
            kv_proj,
            score_proj,
            query_start,
            positions,
            requests,
            state_block_table,
            state,
            norm_ones_flat,
            unweighted_compressed,
            active,
            projections_ready,
        )
        # Native RMSNorm narrows the normalized value before its BF16 gamma
        # multiplication. Keep that boundary while sharing the existing pool.
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_c2_norm_gamma"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            value = unweighted_compressed[t0 : t0 + VEC_M_TILE, :]
            gamma = pl.reshape(pl.cast(compressor_norm[:], pl.FP32), [1, COMPRESSOR_PROJECTION_HEAD_DIM])
            weighted = pl.col_expand_mul(pl.cast(value, pl.FP32), gamma)
            compressed[t0 : t0 + VEC_M_TILE, :] = pl.set_validshape(
                pl.cast(weighted, pl.BF16, mode="rint"),
                rows,
                COMPRESSOR_PROJECTION_HEAD_DIM,
            )
        compressed_rotated = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.BF16)
        rope_compressed(compressed, source_cos, source_sin, compressed_rotated, active)
        publish_compressed(compressed_rotated, compressed_slots, compressed_cache, active)

        projected_index = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, index_slots, index_cache, index_scale, active)
        return state, compressed, compressed_cache, index_key, index_cache, index_scale

    return publish


def make_decode_c2_index_publish_inline(capacity, config, cache_blocks=2):
    """Project BF16 inputs and publish C2 and index caches."""
    hidden = config["hidden_size"]
    pairs = config["qk_rope_head_dim"] // 2
    project_kv = make_fp32_projection(hidden, COMPRESSOR_PROJECTION_HEAD_DIM)
    publish_projected = make_decode_c2_index_publish_projected_inline(capacity, config, cache_blocks)

    @pl.jit.inline
    def publish(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        compressor_wkv: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_norm: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        index_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        kv_proj = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.FP32)
        score_proj = pl.create_tensor([capacity, COMPRESSOR_PROJECTION_HEAD_DIM], dtype=pl.FP32)
        project_kv(x, compressor_wkv, kv_proj, active)
        score_ready = project_kv(x, compressor_wgate, score_proj, active)
        publish_projected(
            kv_proj,
            score_proj,
            compressor_norm,
            index_wk,
            index_norm,
            source_cos,
            source_sin,
            positions,
            compressed_slots,
            index_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            compressed,
            index_key,
            active,
            score_ready,
        )
        return state, compressed, compressed_cache, index_key, index_cache, index_scale

    return publish


def make_decode_layer2_c1c2(capacity, config):
    """Build QKV, ratio-2 C2 pooling and C1 index-key publication."""
    hidden = config["hidden_size"]
    latent_width = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    query_width = heads * head_dim
    pair_width = config["qk_rope_head_dim"] // 2
    qkv = decode_make_qkv_w8a8_inline(capacity, config)
    publish_c2 = make_decode_c2_index_publish_inline(capacity, config)

    @pl.jit
    def layer2_c1c2(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent_width // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent_width], pl.FP32],
        q_norm_weight: pl.Tensor[[latent_width], pl.BF16],
        wq_b: pl.Tensor[[latent_width * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pair_width], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pair_width], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        positions32: pl.Tensor[[capacity], pl.INT32],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent_width], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        qkv_cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        compressor_wkv: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        compressor_norm: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[COMPRESSOR_PROJECTION_HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        index_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.InOut[
            pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * COMPRESSOR_PROJECTION_HEAD_DIM], pl.FP32]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[2, 128, 1, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16]],
        index_cache: pl.InOut[pl.Tensor[[2, 128, 1, INDEX_WIDTH], pl.INT8]],
        index_scale: pl.InOut[pl.Tensor[[2, 128, 1, 1], pl.FP16]],
        compressed: pl.InOut[pl.Tensor[[capacity, COMPRESSOR_PROJECTION_HEAD_DIM], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        qkv(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            positions,
            query_latent,
            query,
            window_kv,
            qkv_cache,
            cache_lens,
            active,
        )

        publish_c2(
            x,
            compressor_wkv,
            compressor_wgate,
            compressor_norm,
            index_wk,
            index_norm,
            rope_cos,
            rope_sin,
            positions32,
            compressed_slots,
            index_slots,
            query_start,
            requests,
            state_block_table,
            state,
            compressed_cache,
            index_cache,
            index_scale,
            compressed,
            index_key,
            active,
        )
        return (
            positions,
            query_latent,
            query,
            window_kv,
            qkv_cache,
            cache_lens,
            state,
            compressed,
            compressed_cache,
            index_key,
            index_cache,
            index_scale,
        )

    return layer2_c1c2


# tiling
T_TILE = 8


@pl.jit.inline
def pool_prefill(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    num_state_blocks: pl.Scalar[pl.INT32],
    pair_ready: pl.Scalar[pl.TASK_ID],
    pooled_output: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
):
    ring_rows = pl.tensor.dim(state_cache, 1)
    state_rows = num_state_blocks * ring_rows
    flat = pl.reshape(state_cache, [state_rows, STATE_WIDTH])
    with pl.spmd(
        pl.max((num_tokens + T_TILE - 1) // T_TILE, 1), name_hint="c2_pool", deps=[pair_ready],
        allow_early_resolve=True,
    ) as pool_tid:
        group = pl.tile.get_block_idx()
        t = group * T_TILE
        if t < num_tokens:
            rows = pl.min(T_TILE, num_tokens - t)
            current_kv = pl.slice(kv_proj, [T_TILE, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_kv = pl.fillpad(current_kv, pad_value=pl.PadValue.zero)
            current_kv = pl.set_validshape(current_kv, T_TILE, HEAD_DIM)
            current_score = pl.slice(score_proj, [T_TILE, HEAD_DIM], [t, 0], valid_shape=[rows, HEAD_DIM])
            current_score = pl.fillpad(current_score, pad_value=pl.PadValue.zero)
            current_score = pl.set_validshape(current_score, T_TILE, HEAD_DIM)
            previous_kv = pl.full([T_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            previous_score = pl.full([T_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            if t > 0:
                previous_kv = pl.slice(kv_proj, [T_TILE, HEAD_DIM], [t - 1, 0], valid_shape=[rows, HEAD_DIM])
                previous_score = pl.slice(score_proj, [T_TILE, HEAD_DIM], [t - 1, 0], valid_shape=[rows, HEAD_DIM])
            else:
                previous_rows = pl.min(T_TILE - 1, num_tokens)
                leading_kv = pl.slice(kv_proj, [T_TILE - 1, HEAD_DIM], [0, 0], valid_shape=[previous_rows, HEAD_DIM])
                leading_score = pl.slice(score_proj, [T_TILE - 1, HEAD_DIM], [0, 0], valid_shape=[previous_rows, HEAD_DIM])
                previous_kv[1:T_TILE, :] = leading_kv
                previous_score[1:T_TILE, :] = leading_score
            previous_kv = pl.set_validshape(previous_kv, T_TILE, HEAD_DIM)
            previous_score = pl.set_validshape(previous_score, T_TILE, HEAD_DIM)
            parity = pl.full([T_TILE, 8], dtype=pl.FP32, value=0.0)
            for lane in pl.range(T_TILE):
                token = t + lane
                if token < num_tokens:
                    request = pl.cast(pl.read(token_to_req_indices, [token]), pl.INDEX)
                    if request >= 0 and request < pl.tensor.dim(state_block_table, 0):
                        block = pl.cast(pl.read(state_block_table, [request, 0]), pl.INDEX)
                        if block >= 0 and block < num_state_blocks:
                            start = pl.read(query_start_loc, [request])
                            end = pl.read(query_start_loc, [request + 1])
                            position = pl.read(position_ids, [token])
                            if token >= start and token < end and position >= 0 and position % 2 == 1:
                                if token == start:
                                    slot_offset = block * ring_rows + (position - 1) % ring_rows
                                    slot = pl.cast(slot_offset, pl.INDEX)
                                    partner_kv = flat[slot : slot + 1, :HEAD_DIM]
                                    partner_score = flat[slot : slot + 1, HEAD_DIM:]
                                    previous_kv[lane : lane + 1, :] = partner_kv
                                    previous_score[lane : lane + 1, :] = partner_score
                                pl.write(parity, [lane, 0], 1.0)
            highest = pl.maximum(previous_score, current_score)
            previous_exp = pl.exp(pl.sub(previous_score, highest))
            current_exp = pl.exp(pl.sub(current_score, highest))
            denominator = pl.add(previous_exp, current_exp)
            previous_probability = pl.div(previous_exp, denominator)
            current_probability = pl.div(current_exp, denominator)
            previous_weighted = pl.mul(previous_kv, previous_probability)
            current_weighted = pl.mul(current_kv, current_probability)
            pooled = pl.add(previous_weighted, current_weighted)
            pooled_bf16 = pl.cast(pooled, pl.BF16, mode="rint")
            parity_value = pl.row_sum(parity)
            pooled_masked = pl.row_expand_mul(pl.cast(pooled_bf16, pl.FP32), parity_value)
            pooled_output[t : t + T_TILE, :] = pl.set_validshape(
                pl.cast(pooled_masked, pl.BF16), rows, HEAD_DIM
            )
            value = pl.cast(pooled_bf16, pl.FP32)
            square = pl.mul(value, value)
            square = pl.mul(pl.row_sum(square), 1.0 / HEAD_DIM)
            inverse = pl.rsqrt(pl.add(square, EPS), high_precision=True)
            normalized = pl.row_expand_mul(value, inverse)
            gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, HEAD_DIM])
            weighted = pl.col_expand_mul(normalized, gamma)
            parity = pl.row_sum(parity)
            published = pl.row_expand_mul(weighted, parity)
            published = pl.cast(published, pl.BF16, mode="rint")
            latent[t : t + T_TILE, :] = pl.set_validshape(published, rows, HEAD_DIM)
    return pool_tid


@pl.jit.inline
def compressor_ratio2_prefill(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, STATE_WIDTH], pl.FP32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    pooled_output = pl.create_tensor([pl.tensor.dim(kv_proj, 0), HEAD_DIM], dtype=pl.BF16)
    pool_tid = pool_prefill(
        kv_proj,
        score_proj,
        state_cache,
        position_ids,
        norm_weight,
        latent,
        num_tokens,
        query_start_loc,
        token_to_req_indices,
        state_block_table,
        blocks,
        state_ready,
        pooled_output,
    )
    state_tid = compressor_state_write(
        kv_proj,
        score_proj,
        query_start_loc,
        position_ids,
        token_to_req_indices,
        state_block_table,
        state_cache,
        num_tokens,
        pool_tid,
    )
    return pool_tid, state_tid


def make_c2_pool_observed(capacity, cache_blocks, requests):
    """Pool complete request pairs and publish the actual FP32 ring with BF16 observations."""

    @pl.jit
    def c2_pool_observed(
        kv: pl.Tensor[[capacity, HEAD_DIM], pl.FP32],
        gate: pl.Tensor[[capacity, HEAD_DIM], pl.FP32],
        starts: pl.Tensor[[requests + 1], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        blocks: pl.Tensor[[requests, 1], pl.INT32],
        gamma: pl.Tensor[[HEAD_DIM], pl.BF16],
        state: pl.InOut[pl.Tensor[[cache_blocks, 32, STATE_WIDTH], pl.FP32]],
        pooled: pl.InOut[pl.Tensor[[capacity, HEAD_DIM], pl.BF16]],
        normalized: pl.InOut[pl.Tensor[[capacity, HEAD_DIM], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        ready = pl.system.task_dummy(deps=[])
        pooled_tid = pool_prefill(
            kv,
            gate,
            state,
            positions,
            gamma,
            normalized,
            active,
            starts,
            request_ids,
            blocks,
            cache_blocks,
            ready,
            pooled,
        )
        compressor_state_write(kv, gate, starts, positions, request_ids, blocks, state, active, pooled_tid)
        return state, pooled, normalized

    return c2_pool_observed


WIDTH = 512


@pl.jit
def pool_projected(
    kv: pl.Tensor[[T_DYN, WIDTH], pl.FP32],
    scores: pl.Tensor[[T_DYN, WIDTH], pl.FP32],
    query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT32],
    requests: pl.Tensor[[T_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    norm: pl.Tensor[[WIDTH], pl.BF16],
    state: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * WIDTH], pl.FP32]],
    latent: pl.InOut[pl.Tensor[[T_DYN, WIDTH], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    kv.bind_dynamic(0, T_DYN)
    scores.bind_dynamic(0, T_DYN)
    positions.bind_dynamic(0, T_DYN)
    requests.bind_dynamic(0, T_DYN)
    latent.bind_dynamic(0, T_DYN)
    copied_kv = pl.create_tensor([pl.tensor.dim(kv, 0), WIDTH], dtype=pl.FP32)
    copied_scores = pl.create_tensor([pl.tensor.dim(kv, 0), WIDTH], dtype=pl.FP32)
    with pl.spmd(active, name_hint="a2_compressor_inputs") as ready:
        row = pl.tile.get_block_idx()
        copied_kv[row : row + 1, :WIDTH] = kv[row : row + 1, :WIDTH]
        copied_scores[row : row + 1, :WIDTH] = scores[row : row + 1, :WIDTH]
    compressor_ratio2(
        copied_kv,
        copied_scores,
        query_start,
        positions,
        requests,
        block_table,
        state,
        norm,
        latent,
        active,
        ready,
    )
    return state, latent


def make_decode_ratio1_compress_publish(capacity, config, cache_blocks=2, request_count=1):
    """Project each token once and publish its own-position BF16/INT8 rows."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    index_width = config["index_head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    if head_dim != 512 or index_width != 128:
        raise ValueError("Use the original full512 compressor and128-wide index cache")
    norm_compressed = decode_make_norm(capacity, head_dim, config["rms_norm_eps"])
    project_index = decode_make_bf16_projection_staged(head_dim, index_width)
    norm_index = decode_make_norm(capacity, index_width, config["rms_norm_eps"])
    rope_compressed = decode_make_rope(capacity, 1, head_dim, rope_dim)
    rope_index = decode_make_rope(capacity, 1, index_width, rope_dim)
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, head_dim)
    publish_index = make_indexer_cache_publish_128(capacity, cache_blocks)

    @pl.jit
    def ratio1_compress_publish(
        projected: pl.Tensor[[capacity, head_dim], pl.BF16],
        compressor_norm: pl.Tensor[[head_dim], pl.BF16],
        index_wk: pl.Tensor[[head_dim, index_width], pl.BF16],
        index_norm: pl.Tensor[[index_width], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        requests: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        compressed: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        long_cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, index_width], pl.BF16]],
        key_cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, index_width], pl.INT8]],
        key_scale: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16]],
        source_positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        cache_lengths: pl.InOut[pl.Tensor[[request_count, 2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        norm_compressed(projected, compressor_norm, compressed, active)
        rotated = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        rope_compressed(compressed, source_cos, source_sin, rotated, active)
        publish_compressed(rotated, compressed_slots, long_cache, active)
        projected_index = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, compressed_slots, key_cache, key_scale, active)
        with pl.spmd(1, name_hint="ratio1_lengths_and_sources"):
            block = pl.tile.get_block_idx()
            if block == 0:
                for token in pl.range(active):
                    position = pl.read(positions, [token])
                    request = pl.read(requests, [token])
                    length = pl.cast(position + 1, pl.INT32)
                    pl.write(source_positions, [token], position)
                    pl.write(cache_lengths, [request, 0], length)
                    pl.write(cache_lengths, [request, 1], length)
        return compressed, long_cache, index_key, key_cache, key_scale, source_positions, cache_lengths

    return ratio1_compress_publish


def make_ratio1_compress_publish(capacity, config, cache_blocks=2, request_count=1):
    """Project each token once and publish its own-position BF16/INT8 rows."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    index_width = config["index_head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    if head_dim != 512 or index_width != 128:
        raise ValueError("Use the original full512 compressor and128-wide index cache")
    norm_compressed = make_norm(capacity, head_dim, config["rms_norm_eps"])
    project_index = pl.jit.inline(make_bf16_projection_staged(head_dim, index_width)._func)
    norm_index = make_norm(capacity, index_width, config["rms_norm_eps"])
    rope_compressed = make_rope(capacity, 1, head_dim, rope_dim)
    rope_index = make_rope(capacity, 1, index_width, rope_dim)
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, head_dim)
    publish_index = make_indexer_cache_publish_128(capacity, cache_blocks)

    @pl.jit
    def ratio1_compress_publish(
        projected: pl.Tensor[[capacity, head_dim], pl.BF16],
        compressor_norm: pl.Tensor[[head_dim], pl.BF16],
        index_wk: pl.Tensor[[head_dim, index_width], pl.BF16],
        index_norm: pl.Tensor[[index_width], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        requests: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        compressed: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        long_cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, index_width], pl.BF16]],
        key_cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, index_width], pl.INT8]],
        key_scale: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16]],
        source_positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        cache_lengths: pl.InOut[pl.Tensor[[request_count, 2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        norm_compressed(projected, compressor_norm, compressed, active)
        rotated = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        rope_compressed(compressed, source_cos, source_sin, rotated, active)
        publish_compressed(rotated, compressed_slots, long_cache, active)
        projected_index = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, compressed_slots, key_cache, key_scale, active)
        with pl.spmd(1, name_hint="ratio1_lengths_and_sources"):
            block = pl.tile.get_block_idx()
            if block == 0:
                for token in pl.range(active):
                    position = pl.read(positions, [token])
                    request = pl.read(requests, [token])
                    length = pl.cast(position + 1, pl.INT32)
                    pl.write(source_positions, [token], position)
                    pl.write(cache_lengths, [request, 0], length)
                    pl.write(cache_lengths, [request, 1], length)
        return compressed, long_cache, index_key, key_cache, key_scale, source_positions, cache_lengths

    return ratio1_compress_publish
