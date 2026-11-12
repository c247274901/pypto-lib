# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sliding-window attention, cache publication, gathers and bounded validation."""

import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
import torch as decode_swa_cache_torch
import torch as decode_torch
import torch
from .operator_config import CONTRACT
from .qkv_proj_rope import qkv_shape


# tiling
HEAD_TILE = 16
WINDOW_TILE = 128
FUSED_HEAD_TILE = 64
FUSED_VALUE_TILE = 256

BLOCK_SIZE = 128


def make_swa_leaves(capacity, cache_blocks, config):
    """Build a BF16 PA_BBND SWA leaf for the full configured query width."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    window = config["sliding_window"]
    if window % WINDOW_TILE:
        raise ValueError("A2 SWA requires a window divisible by its 128-token softmax tile")
    width = heads * head_dim
    cache_rows = cache_blocks * 128
    score_scale = head_dim**-0.5

    @pl.jit.inline
    def publish(
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for token in pl.spmd(active, name_hint="a2_swa_publish"):
            page = pl.read(slots, [token, 0])
            offset = pl.read(slots, [token, 1])
            if page >= 0:
                address = page * 128 + offset
                row = pl.cast(address, pl.INDEX)
                flat[row : row + 1, :] = kv[token : token + 1, :]
        return cache

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(active * window, name_hint="a2_swa_gather"):
            token = block // window
            column = block % window
            index = pl.read(indices, [token, column])
            if index >= 0:
                row = pl.cast(index, pl.INDEX)
                selected[block : block + 1, :] = flat[row : row + 1, :]
            else:
                selected[block : block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        return selected

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, width], pl.BF16],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        for block in pl.spmd(active * (heads // HEAD_TILE), name_hint="a2_swa_attention"):
            token = block // (heads // HEAD_TILE)
            head = block % (heads // HEAD_TILE) * HEAD_TILE
            query_row = token * heads + head
            # Include the zero-value sink before narrowing exp(scores-max) to
            # BF16, matching the native softmax's rounding boundary.
            maximum = pl.reshape(sink[head : head + HEAD_TILE], [1, HEAD_TILE])
            denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            numerator = pl.full([HEAD_TILE, head_dim], dtype=pl.FP32, value=0.0)
            sparse_maximum = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=-1e30)
            if window == 640:
                for sparse_part in pl.range(4):
                    sparse_column = 128 + sparse_part * WINDOW_TILE
                    sparse_row = token * window + sparse_column
                    sparse_query = qflat[query_row : query_row + HEAD_TILE, :]
                    sparse_kv = selected[sparse_row : sparse_row + WINDOW_TILE, :]
                    sparse_dot = pl.matmul(sparse_query, sparse_kv, b_trans=True)
                    sparse_scores = pl.mul(sparse_dot, score_scale)
                    sparse_ids = indices[token : token + 1, sparse_column : sparse_column + WINDOW_TILE]
                    sparse_ids_fp32 = pl.cast(sparse_ids, pl.FP32)
                    sparse_shifted = pl.add(sparse_ids_fp32, 1.0)
                    sparse_positive = pl.maximum(sparse_shifted, 0.0)
                    sparse_valid = pl.minimum(sparse_positive, 1.0)
                    sparse_invalid = pl.sub(sparse_valid, 1.0)
                    sparse_bias = pl.mul(sparse_invalid, 1e30)
                    sparse_scores = pl.col_expand_add(sparse_scores, sparse_bias)
                    sparse_row_max = pl.row_max(sparse_scores)
                    sparse_local_max = pl.reshape(sparse_row_max, [1, HEAD_TILE])
                    sparse_maximum = pl.maximum(sparse_maximum, sparse_local_max)
            for part in pl.range(window // WINDOW_TILE):
                cache_row = token * window + part * WINDOW_TILE
                q = qflat[query_row : query_row + HEAD_TILE, :]
                kv = selected[cache_row : cache_row + WINDOW_TILE, :]
                dot = pl.matmul(q, kv, b_trans=True)
                scores = pl.mul(dot, score_scale)
                index_tile = indices[token : token + 1, part * WINDOW_TILE : part * WINDOW_TILE + WINDOW_TILE]
                index_float = pl.cast(index_tile, pl.FP32)
                shifted_index = pl.add(index_float, 1.0)
                positive_index = pl.maximum(shifted_index, 0.0)
                valid = pl.minimum(positive_index, 1.0)
                invalid = pl.sub(valid, 1.0)
                bias = pl.mul(invalid, 1e30)
                scores = pl.col_expand_add(scores, bias)
                row_max = pl.row_max(scores)
                local_max = pl.reshape(row_max, [1, HEAD_TILE])
                if window == 640:
                    if part > 0:
                        local_max = pl.maximum(local_max, sparse_maximum)
                next_max = pl.maximum(maximum, local_max)
                max_difference = pl.sub(maximum, next_max)
                correction = pl.exp(max_difference)
                max_column = pl.reshape(next_max, [HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_column)
                exponential = pl.exp(shifted)
                probabilities = pl.col_expand_mul(exponential, valid)
                row_sum = pl.row_sum(probabilities)
                probability_sum = pl.reshape(row_sum, [1, HEAD_TILE])
                old_denominator = pl.mul(denominator, correction)
                denominator = pl.add(old_denominator, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="round")
                weighted = pl.matmul(weights, kv)
                correction_column = pl.reshape(correction, [HEAD_TILE, 1])
                corrected = pl.row_expand_mul(numerator, correction_column)
                numerator = pl.add(corrected, weighted)
                maximum = next_max
            denominator_column = pl.reshape(denominator, [HEAD_TILE, 1])
            result = pl.row_expand_div(numerator, denominator_column)
            oflat[query_row : query_row + HEAD_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        return output

    return publish, gather, attend


def _make_swa_read_single_window(capacity, cache_blocks, config):
    """Gather contiguous cache runs into L1 and reuse them for QK and PV."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    width = heads * head_dim
    cache_rows = cache_blocks * WINDOW_TILE
    score_scale = head_dim ** -0.5

    @pl.jit
    def swa_read_single_window(
        query: pl.Tensor[[capacity, width], pl.BF16],
        indices: pl.Tensor[[capacity, WINDOW_TILE], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[cache_blocks, WINDOW_TILE, 1, head_dim], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(active * (heads // FUSED_HEAD_TILE), name_hint="prefill_swa_read_fused"):
            token = block // (heads // FUSED_HEAD_TILE)
            head = block % (heads // FUSED_HEAD_TILE) * FUSED_HEAD_TILE
            query_row = token * heads + head
            q = qflat[query_row:query_row + FUSED_HEAD_TILE, :]
            kv = pl.create_l1([WINDOW_TILE, head_dim], pl.BF16)
            run_start = pl.cast(0, pl.INDEX)
            run_source = pl.read(indices, [token, 0])
            previous = run_source
            for row in pl.range(1, WINDOW_TILE + 1):
                if row < WINDOW_TILE:
                    current = pl.read(indices, [token, row])
                else:
                    current = pl.cast(-2, pl.INT32)
                run_break = (previous >= 0 and current != previous + 1) or (previous < 0 and current >= 0)
                if row == WINDOW_TILE or run_break:
                    source = pl.cast(pl.max(run_source, 0), pl.INDEX)
                    run_rows = row - run_start
                    kv = pl.gather_row(kv, flat, [run_start, 0], [source, 0], [WINDOW_TILE, head_dim], valid_shape=[run_rows, head_dim])
                    run_start = row
                    run_source = current
                previous = current
            dot = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
            for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                dot_part = pl.aiv_shard(dot)
                lane_head = head + lane * (FUSED_HEAD_TILE // 2)
                scores = pl.mul(dot_part, score_scale)
                ids = pl.cast(indices[token:token + 1, :], pl.FP32)
                valid = pl.minimum(pl.maximum(pl.add(ids, 1.0), 0.0), 1.0)
                bias = pl.mul(pl.sub(valid, 1.0), 1e30)
                scores = pl.col_expand_add(scores, bias)
                local_max = pl.reshape(pl.row_max(scores), [1, FUSED_HEAD_TILE // 2])
                sink_row = pl.reshape(sink[lane_head:lane_head + FUSED_HEAD_TILE // 2], [1, FUSED_HEAD_TILE // 2])
                maximum = pl.maximum(local_max, sink_row)
                sink_exp = pl.exp(pl.sub(sink_row, maximum))
                max_column = pl.reshape(maximum, [FUSED_HEAD_TILE // 2, 1])
                exponential = pl.exp(pl.row_expand_sub(scores, max_column))
                probabilities = pl.col_expand_mul(exponential, valid)
                probability_sum = pl.reshape(pl.row_sum(probabilities), [1, FUSED_HEAD_TILE // 2])
                denominator = pl.add(sink_exp, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="round")
                full_weights = pl.aic_gather(weights)
                denominator_column = pl.reshape(denominator, [FUSED_HEAD_TILE // 2, 1])
            for value_column in pl.range(0, head_dim, FUSED_VALUE_TILE):
                values = kv[:, value_column:value_column + FUSED_VALUE_TILE]
                weighted = pl.matmul(full_weights, values, out_dtype=pl.FP32)
                for lane_out in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    weighted_part = pl.aiv_shard(weighted)
                    result = pl.row_expand_div(weighted_part, denominator_column)
                    output_row = query_row + lane_out * (FUSED_HEAD_TILE // 2)
                    narrowed = pl.cast(result, pl.BF16, mode="rint")
                    oflat[output_row:output_row + FUSED_HEAD_TILE // 2, value_column:value_column + FUSED_VALUE_TILE] = narrowed
        return output

    return swa_read_single_window


def make_swa_read(capacity, cache_blocks, config):
    """Read a published cache with an independent CP query row count."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    width = heads * head_dim
    window = config["sliding_window"]
    if window == WINDOW_TILE and heads % FUSED_HEAD_TILE == 0 and head_dim % FUSED_VALUE_TILE == 0:
        return _make_swa_read_single_window(capacity, cache_blocks, config)
    _publish, gather, attend = make_swa_leaves(capacity, cache_blocks, config)

    @pl.jit
    def swa_read(
        query: pl.Tensor[[capacity, width], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        selected = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, selected, active)
        attend(query, selected, indices, sink, output, active)
        return output

    return swa_read


def make_swa_inline(capacity, cache_blocks, config):
    """Publish and read a BF16 paged SWA cache."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    width = heads * head_dim
    window = config["sliding_window"]
    publish, _gather, _attend = make_swa_leaves(capacity, cache_blocks, config)
    read = pl.jit.inline(make_swa_read(capacity, cache_blocks, config)._func)

    @pl.jit.inline
    def swa_inline(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(kv, slots, cache, active)
        read(query, indices, sink, cache, output, active)
        return cache, output

    return swa_inline


def make_swa(capacity, cache_blocks, config):
    """Build the standalone JIT wrapper for shared inline SWA compute."""
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    window = config["sliding_window"]
    compute = make_swa_inline(capacity, cache_blocks, config)

    @pl.jit
    def swa(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(query, kv, slots, indices, sink, cache, output, active)
        return cache, output

    return swa


def swa_output_compare(active, all_masked):
    """Compare active values while requiring exact masked rows and inactive tails."""
    from golden.validation import ratio_allclose

    numeric = ratio_allclose(rtol=0.05, atol=0.05, max_error_ratio=0.0)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        if active:
            passed, detail = numeric(actual[:active], expected[:active], **context)
            if not passed:
                return False, "active: " + detail
        for label, region in (("untouched", slice(active, None)), ("masked", all_masked)):
            if actual[region].numel():
                passed, detail = exact(actual[region], expected[region], **context)
                if not passed:
                    return False, label + ": " + detail
        return True, ""

    return compare


def make_swa_attention(capacity: int, heads: int, head_dim: int, window: int):
    """Build the sink-aware A2/A3 attention reduction kernel."""

    if capacity <= 0 or heads <= 0 or head_dim <= 0 or window <= 0:
        raise ValueError("attention dimensions must be positive")
    if heads % 16:
        raise ValueError("A2/A3 attention uses a 16-head vector tile")
    publish = make_swa_cache_publish(capacity, 2, head_dim)
    gather = make_swa_cache_gather(capacity, window, 2, head_dim)
    head_tile = 16
    scale = head_dim**-0.5

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(window_kv, slots, cache, active)
        selected_flat = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, selected_flat, active)
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        kflat = selected_flat
        for block in pl.spmd((active + 0) * (heads // head_tile), name_hint="dsv41_a2a3_swa_attention"):
            token = block // (heads // head_tile)
            head = (block % (heads // head_tile)) * head_tile
            q = qflat[token * heads + head : token * heads + head + head_tile, :]
            k = kflat[token * window : token * window + window, :]
            scores = pl.mul(pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32), scale)
            idx = pl.cast(indices[token : token + 1, :], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            scores = pl.col_expand_add(scores, pl.mul(pl.sub(valid, 1.0), 1.0e30))
            maximum = pl.reshape(pl.row_max(scores), [1, head_tile])
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(maximum, [head_tile, 1]))), valid
            )
            denominator = pl.row_sum(probabilities)
            sink_score = pl.reshape(sink[head : head + head_tile], [1, head_tile])
            final_max = pl.maximum(maximum, sink_score)
            correction = pl.exp(pl.sub(maximum, final_max))
            denominator = pl.add(
                pl.mul(pl.reshape(denominator, [1, head_tile]), correction),
                pl.exp(pl.sub(sink_score, final_max)),
            )
            weighted = pl.matmul(pl.cast(probabilities, pl.BF16, mode="rint"), k, out_dtype=pl.FP32)
            result = pl.row_expand_mul(weighted, pl.reshape(pl.div(correction, denominator), [head_tile, 1]))
            oflat[token * heads + head : token * heads + head + head_tile, :] = pl.cast(
                result, pl.BF16, mode="rint"
            )
        return output, cache, selected_flat

    return attend


def make_swa_entry(capacity: int, config: Mapping[str, object]):
    shape = qkv_shape(config)
    window = int(config["sliding_window"])
    attend = make_swa_attention(capacity, shape.num_attention_heads, shape.head_dim, window)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, shape.query_width], pl.BF16],
        window_kv: pl.Tensor[[capacity, shape.head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[shape.num_attention_heads], pl.FP32],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, shape.head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, shape.query_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        attend(query, window_kv, slots, indices, sink, cache, output, active)
        return cache, output

    return kernel


def make_swa_cache_publish(capacity: int, cache_blocks: int, head_dim: int):
    cache_rows = cache_blocks * BLOCK_SIZE

    @pl.jit.inline
    def publish(
        values: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for token in pl.spmd(capacity, name_hint="dsv41_a2a3_swa_publish"):
            if token < active:
                page = pl.read(slots, [token, 0])
                offset = pl.read(slots, [token, 1])
                if page >= 0:
                    physical = pl.cast(page * 128 + offset, pl.INDEX)
                    flat[physical : physical + 1, :] = values[token : token + 1, :]
        return cache

    return publish


def make_swa_cache_gather(capacity: int, window: int, cache_blocks: int, head_dim: int):
    cache_rows = cache_blocks * BLOCK_SIZE

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        output: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(capacity * window, name_hint="dsv41_a2a3_swa_gather"):
            token = block // window
            column = block % window
            if token < active:
                index = pl.read(indices, [token, column])
                if index >= 0:
                    physical = pl.cast(index, pl.INDEX)
                    output[block : block + 1, :] = flat[physical : physical + 1, :]
                else:
                    output[block : block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        return output

    return gather


def make_decode_swa_attention(
    capacity: int,
    heads: int,
    head_dim: int,
    window: int,
    *,
    decode_state: bool = False,
):
    """Build the sink-aware A2/A3 attention reduction kernel.

    The default keeps the prefill/layer ABI. Decode requests opt into the
    request-state controls explicitly so existing callers retain their
    public signature.
    """

    if capacity <= 0 or heads <= 0 or head_dim <= 0 or window <= 0:
        raise ValueError("attention dimensions must be positive")
    if heads % 16:
        raise ValueError("A2/A3 attention uses a 16-head vector tile")
    publish = make_decode_swa_cache_publish(capacity, 2, head_dim, decode_state=True)
    gather = make_decode_swa_cache_gather(capacity, window, 2, head_dim, decode_state=True)
    head_tile = 16
    scale = head_dim**-0.5

    @pl.jit.inline
    def decode_attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        cache_source_ids: pl.Tensor[[capacity], pl.INT32],
        finished: pl.Tensor[[capacity], pl.INT8],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(window_kv, slots, positions, request_ids, cache_source_ids, finished, cache, active)
        selected_flat = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, finished, selected_flat, active)
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        kflat = selected_flat
        for block in pl.spmd((active + 0) * (heads // head_tile), name_hint="dsv41_a2a3_swa_attention"):
            token = block // (heads // head_tile)
            head = (block % (heads // head_tile)) * head_tile
            q = qflat[token * heads + head : token * heads + head + head_tile, :]
            k = kflat[token * window : token * window + window, :]
            scores = pl.mul(pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32), scale)
            idx = pl.cast(indices[token : token + 1, :], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            scores = pl.col_expand_add(scores, pl.mul(pl.sub(valid, 1.0), 1.0e30))
            maximum = pl.reshape(pl.row_max(scores), [1, head_tile])
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(maximum, [head_tile, 1]))), valid
            )
            denominator = pl.row_sum(probabilities)
            sink_score = pl.reshape(sink[head : head + head_tile], [1, head_tile])
            final_max = pl.maximum(maximum, sink_score)
            correction = pl.exp(pl.sub(maximum, final_max))
            denominator = pl.add(
                pl.mul(pl.reshape(denominator, [1, head_tile]), correction),
                pl.exp(pl.sub(sink_score, final_max)),
            )
            weighted = pl.matmul(pl.cast(probabilities, pl.BF16, mode="rint"), k, out_dtype=pl.FP32)
            result = pl.row_expand_mul(weighted, pl.reshape(pl.div(correction, denominator), [head_tile, 1]))
            oflat[token * heads + head : token * heads + head + head_tile, :] = pl.cast(
                result, pl.BF16, mode="rint"
            )
        return output, cache, selected_flat

    if decode_state:
        return decode_attend

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        positions = pl.full([capacity], dtype=pl.INT64, value=0)
        request_ids = pl.full([capacity], dtype=pl.INT32, value=0)
        cache_source_ids = pl.full([capacity], dtype=pl.INT32, value=0)
        finished = pl.full([capacity], dtype=pl.INT8, value=0)
        return decode_attend(
            query,
            window_kv,
            slots,
            indices,
            sink,
            positions,
            request_ids,
            cache_source_ids,
            finished,
            cache,
            output,
            active,
        )

    return attend


def make_decode_swa_cache_publish(
    capacity: int,
    cache_blocks: int,
    head_dim: int,
    *,
    decode_state: bool = False,
):
    cache_rows = cache_blocks * BLOCK_SIZE

    if decode_state:

        @pl.jit.inline
        def publish(
            values: pl.Tensor[[capacity, head_dim], pl.BF16],
            slots: pl.Tensor[[capacity, 2], pl.INT64],
            positions: pl.Tensor[[capacity], pl.INT64],
            request_ids: pl.Tensor[[capacity], pl.INT32],
            cache_source_ids: pl.Tensor[[capacity], pl.INT32],
            finished: pl.Tensor[[capacity], pl.INT8],
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for token in pl.spmd(capacity, name_hint="dsv41_a2a3_swa_publish"):
                position = pl.read(positions, [token])
                request_id = pl.read(request_ids, [token])
                source_id = pl.read(cache_source_ids, [token])
                done = pl.read(finished, [token])
                if token < active and position >= 0 and request_id >= 0 and source_id >= 0 and done == 0:
                    page = pl.read(slots, [token, 0])
                    offset = pl.read(slots, [token, 1])
                    if page >= 0:
                        physical = pl.cast(page * 128 + offset, pl.INDEX)
                        flat[physical : physical + 1, :] = values[token : token + 1, :]
            return cache

    else:

        @pl.jit.inline
        def publish(
            values: pl.Tensor[[capacity, head_dim], pl.BF16],
            slots: pl.Tensor[[capacity, 2], pl.INT64],
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for token in pl.spmd(capacity, name_hint="dsv41_a2a3_swa_publish"):
                if token < active:
                    page = pl.read(slots, [token, 0])
                    offset = pl.read(slots, [token, 1])
                    if page >= 0:
                        physical = pl.cast(page * 128 + offset, pl.INDEX)
                        flat[physical : physical + 1, :] = values[token : token + 1, :]
            return cache

    return publish


def make_decode_swa_cache_gather(
    capacity: int,
    window: int,
    cache_blocks: int,
    head_dim: int,
    *,
    decode_state: bool = False,
):
    cache_rows = cache_blocks * BLOCK_SIZE

    if decode_state:

        @pl.jit.inline
        def gather(
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            indices: pl.Tensor[[capacity, window], pl.INT32],
            finished: pl.Tensor[[capacity], pl.INT8],
            output: pl.Tensor[[capacity * window, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for block in pl.spmd(capacity * window, name_hint="dsv41_a2a3_swa_gather"):
                token = block // window
                column = block % window
                if token < active and pl.read(finished, [token]) == 0:
                    index = pl.read(indices, [token, column])
                    if index >= 0:
                        physical = pl.cast(index, pl.INDEX)
                        output[block : block + 1, :] = flat[physical : physical + 1, :]
                    else:
                        output[block : block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
            return output

    else:

        @pl.jit.inline
        def gather(
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            indices: pl.Tensor[[capacity, window], pl.INT32],
            output: pl.Tensor[[capacity * window, head_dim], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for block in pl.spmd(capacity * window, name_hint="dsv41_a2a3_swa_gather"):
                token = block // window
                column = block % window
                if token < active:
                    index = pl.read(indices, [token, column])
                    if index >= 0:
                        physical = pl.cast(index, pl.INDEX)
                        output[block : block + 1, :] = flat[physical : physical + 1, :]
                    else:
                        output[block : block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
            return output

    return gather


def make_decode_swa_inline(capacity, cache_blocks, config):
    """Build a BF16 PA_BBND SWA leaf for the full configured query width."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    window = config["sliding_window"]
    if window % WINDOW_TILE:
        raise ValueError("A2 SWA requires a window divisible by its 128-token softmax tile")
    width = heads * head_dim
    cache_rows = cache_blocks * 128
    score_scale = head_dim**-0.5

    @pl.jit.inline
    def publish(
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for token in pl.spmd(active, name_hint="a2_swa_publish"):
            page = pl.read(slots, [token, 0])
            offset = pl.read(slots, [token, 1])
            if page >= 0:
                address = page * 128 + offset
                row = pl.cast(address, pl.INDEX)
                flat[row : row + 1, :] = kv[token : token + 1, :]
        return cache

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(active * window, name_hint="a2_swa_gather"):
            token = block // window
            column = block % window
            index = pl.read(indices, [token, column])
            if index >= 0:
                row = pl.cast(index, pl.INDEX)
                selected[block : block + 1, :] = flat[row : row + 1, :]
            else:
                selected[block : block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        return selected

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, width], pl.BF16],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        for block in pl.spmd(active * (heads // HEAD_TILE), name_hint="a2_swa_attention"):
            token = block // (heads // HEAD_TILE)
            head = block % (heads // HEAD_TILE) * HEAD_TILE
            query_row = token * heads + head
            # Include the zero-value sink before narrowing exp(scores-max) to
            # BF16, matching the native softmax's rounding boundary.
            maximum = pl.reshape(sink[head : head + HEAD_TILE], [1, HEAD_TILE])
            denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            numerator = pl.full([HEAD_TILE, head_dim], dtype=pl.FP32, value=0.0)
            for part in pl.range(window // WINDOW_TILE):
                cache_row = token * window + part * WINDOW_TILE
                q = qflat[query_row : query_row + HEAD_TILE, :]
                kv = selected[cache_row : cache_row + WINDOW_TILE, :]
                dot = pl.matmul(q, kv, b_trans=True)
                scores = pl.mul(dot, score_scale)
                index_tile = indices[token : token + 1, part * WINDOW_TILE : part * WINDOW_TILE + WINDOW_TILE]
                index_float = pl.cast(index_tile, pl.FP32)
                shifted_index = pl.add(index_float, 1.0)
                positive_index = pl.maximum(shifted_index, 0.0)
                valid = pl.minimum(positive_index, 1.0)
                invalid = pl.sub(valid, 1.0)
                bias = pl.mul(invalid, 1e30)
                scores = pl.col_expand_add(scores, bias)
                row_max = pl.row_max(scores)
                local_max = pl.reshape(row_max, [1, HEAD_TILE])
                next_max = pl.maximum(maximum, local_max)
                max_difference = pl.sub(maximum, next_max)
                correction = pl.exp(max_difference)
                max_column = pl.reshape(next_max, [HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_column)
                exponential = pl.exp(shifted)
                probabilities = pl.col_expand_mul(exponential, valid)
                row_sum = pl.row_sum(probabilities)
                probability_sum = pl.reshape(row_sum, [1, HEAD_TILE])
                old_denominator = pl.mul(denominator, correction)
                denominator = pl.add(old_denominator, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="rint")
                weighted = pl.matmul(weights, kv)
                correction_column = pl.reshape(correction, [HEAD_TILE, 1])
                corrected = pl.row_expand_mul(numerator, correction_column)
                numerator = pl.add(corrected, weighted)
                maximum = next_max
            one = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=1.0)
            normalized = pl.div(one, denominator, high_precision=True)
            normalized_column = pl.reshape(normalized, [HEAD_TILE, 1])
            result = pl.row_expand_mul(numerator, normalized_column)
            oflat[query_row : query_row + HEAD_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        return output

    @pl.jit.inline
    def swa_inline(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(kv, slots, cache, active)
        selected = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, selected, active)
        attend(query, selected, indices, sink, output, active)
        return cache, output

    return swa_inline


def make_decode_swa(capacity, cache_blocks, config):
    """Build the standalone JIT wrapper for shared inline SWA compute."""
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    window = config["sliding_window"]
    compute = make_decode_swa_inline(capacity, cache_blocks, config)

    @pl.jit
    def swa(
        query: pl.Tensor[[capacity, width], pl.BF16],
        kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.InOut[pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(query, kv, slots, indices, sink, cache, output, active)
        return cache, output

    return swa


def make_decode_swa_entry(capacity: int, config: Mapping[str, object]):
    # Keep the product entrypoint independent from the Torch-only reference
    # helpers in qkv_proj_rope.py. This shape is a model boundary.
    query_width = int(config["num_attention_heads"]) * int(config["head_dim"])
    head_dim = int(config["head_dim"])
    heads = int(config["num_attention_heads"])
    window = int(config["sliding_window"])
    attend = make_decode_swa_attention(capacity, heads, head_dim, window, decode_state=True)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, query_width], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        cache_source_ids: pl.Tensor[[capacity], pl.INT32],
        finished: pl.Tensor[[capacity], pl.INT8],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        attend(
            query,
            window_kv,
            slots,
            indices,
            sink,
            positions,
            request_ids,
            cache_source_ids,
            finished,
            cache,
            output,
            active,
        )
        return cache, output

    return kernel


def golden_swa(tensors, *, config):
    """Publish legal slots and evaluate sink softmax with BF16 probability rounding."""
    import torch

    active = int(tensors["active"])
    heads, dim, window = (config[name] for name in ("num_attention_heads", "head_dim", "sliding_window"))
    if not 0 <= active <= tensors["query"].shape[0]:
        raise ValueError("SWA active rows must fit the actual capacity")
    if not active:
        return
    flat = tensors["cache"].view(-1, dim)
    for token in range(active):
        page, offset = (int(value) for value in tensors["slots"][token])
        if page < 0:
            continue
        if page >= tensors["cache"].shape[0] or not 0 <= offset < 128:
            raise ValueError("Positive SWA publication slots must remain in the allocated cache")
        flat[page * 128 + offset] = tensors["kv"][token]
    indices = tensors["indices"][:active].long()
    if (indices >= len(flat)).any():
        raise ValueError("SWA selected cache rows must remain in the allocated cache")
    for token in range(active):
        valid = indices[token] >= 0
        selected = torch.zeros((window, dim), dtype=torch.bfloat16)
        selected[valid] = flat[indices[token, valid]]
        query = tensors["query"][token].reshape(heads, dim).float()
        scores = (query @ selected.float().T) * (dim**-0.5)
        scores = scores + (valid.float() - 1.0)[None] * 1e30
        maximum = tensors["sink"].float().clone()
        denominator = torch.ones(heads, dtype=torch.float32)
        numerator = torch.zeros((heads, dim), dtype=torch.float32)
        sparse_maximum = torch.full((heads,), -1e30, dtype=torch.float32)
        if window == 640:
            for part in range(4):
                start = 128 + part * WINDOW_TILE
                sparse_maximum = torch.maximum(
                    sparse_maximum, scores[:, start : start + WINDOW_TILE].amax(dim=1)
                )
        for part in range(window // WINDOW_TILE):
            start = part * WINDOW_TILE
            score = scores[:, start : start + WINDOW_TILE]
            local_max = score.amax(dim=1)
            if window == 640 and part > 0:
                local_max = torch.maximum(local_max, sparse_maximum)
            next_max = torch.maximum(maximum, local_max)
            correction = torch.exp(maximum - next_max)
            probability = (
                torch.exp(score - next_max[:, None]) * valid[start : start + WINDOW_TILE].float()[None]
            )
            denominator = denominator * correction + probability.sum(dim=1)
            rounded = probability.to(torch.bfloat16).float()
            weighted = rounded @ selected[start : start + WINDOW_TILE].float()
            numerator = numerator * correction[:, None] + weighted
            maximum = next_max
        tensors["output"][token] = (numerator / denominator[:, None]).reshape(-1).to(torch.bfloat16)


def swa_attention_reference(
    query: torch.Tensor,
    selected: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    head_dim: int,
    *,
    valid_tokens: int | None = None,
) -> torch.Tensor:
    """Compute sink-aware attention over gathered BF16 SWA rows."""

    if query.ndim != 2 or query.shape[-1] % head_dim:
        raise ValueError("query must be [T, H*D] with a divisible head dimension")
    tokens = query.shape[0]
    heads = query.shape[-1] // head_dim
    if selected.ndim == 2:
        if selected.shape[0] != tokens * indices.shape[1]:
            raise ValueError("flat selected rows do not match indices")
        selected = selected.reshape(tokens, indices.shape[1], head_dim)
    if selected.shape[:2] != indices.shape or selected.shape[-1] != head_dim:
        raise ValueError("selected and indices shape mismatch")
    if sink.shape != (heads,):
        raise ValueError(f"sink must have shape {(heads,)}, got {tuple(sink.shape)}")
    active = tokens if valid_tokens is None else int(valid_tokens)
    output = torch.zeros_like(query)
    if active == 0:
        return output
    q = query[:active].float().reshape(active, heads, head_dim)
    k = selected[:active].float()
    score = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
    valid = indices[:active].ge(0).unsqueeze(1)
    score = score.masked_fill(~valid, float("-inf"))
    sink_score = sink.float().reshape(1, heads, 1).expand(active, -1, -1)
    all_score = torch.cat((score, sink_score), dim=-1)
    probability = torch.softmax(all_score, dim=-1)
    result = torch.matmul(probability[..., :-1], k)
    output[:active] = result.reshape(active, heads * head_dim).to(torch.bfloat16)
    return output


def write_swa_cache_reference(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Write BF16 rows into a ``[blocks, 128, 1, head_dim]`` cache."""

    if cache.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise TypeError("A2/A3 SWA cache and values must be BF16")
    result = cache.clone()
    flat = result.view(-1, result.shape[-1])
    for row, slot in zip(values, slot_mapping.reshape(-1, 2).tolist()):
        page, offset = slot
        if page >= 0:
            flat[page * CONTRACT.swa_block_size + offset].copy_(row)
    return result


def gather_swa_cache_reference(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather physical cache rows; negative indices produce zero rows."""

    flat = cache.reshape(-1, cache.shape[-1])
    result = torch.zeros(*indices.shape, cache.shape[-1], dtype=cache.dtype, device=cache.device)
    valid = indices >= 0
    if valid.any():
        result[valid] = flat[indices[valid].long()]
    return result


def decode_swa_attention_reference(
    query: decode_torch.Tensor,
    selected: decode_torch.Tensor,
    indices: decode_torch.Tensor,
    sink: decode_torch.Tensor,
    head_dim: int,
    *,
    valid_tokens: int | None = None,
) -> decode_torch.Tensor:
    """Compute sink-aware attention over gathered BF16 SWA rows."""

    import torch

    if query.ndim != 2 or query.shape[-1] % head_dim:
        raise ValueError("query must be [T, H*D] with a divisible head dimension")
    tokens = query.shape[0]
    heads = query.shape[-1] // head_dim
    if selected.ndim == 2:
        if selected.shape[0] != tokens * indices.shape[1]:
            raise ValueError("flat selected rows do not match indices")
        selected = selected.reshape(tokens, indices.shape[1], head_dim)
    if selected.shape[:2] != indices.shape or selected.shape[-1] != head_dim:
        raise ValueError("selected and indices shape mismatch")
    if sink.shape != (heads,):
        raise ValueError(f"sink must have shape {(heads,)}, got {tuple(sink.shape)}")
    active = tokens if valid_tokens is None else int(valid_tokens)
    output = torch.zeros_like(query)
    if active == 0:
        return output
    q = query[:active].float().reshape(active, heads, head_dim)
    k = selected[:active].float()
    score = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
    valid = indices[:active].ge(0).unsqueeze(1)
    score = score.masked_fill(~valid, float("-inf"))
    sink_score = sink.float().reshape(1, heads, 1).expand(active, -1, -1)
    all_score = torch.cat((score, sink_score), dim=-1)
    probability = torch.softmax(all_score, dim=-1)
    result = torch.matmul(probability[..., :-1], k)
    output[:active] = result.reshape(active, heads * head_dim).to(torch.bfloat16)
    return output


def decode_write_swa_cache_reference(
    cache: decode_swa_cache_torch.Tensor,
    values: decode_swa_cache_torch.Tensor,
    slot_mapping: decode_swa_cache_torch.Tensor,
) -> decode_swa_cache_torch.Tensor:
    """Write BF16 rows into a ``[blocks, 128, 1, head_dim]`` cache."""

    import torch

    if cache.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise TypeError("A2/A3 SWA cache and values must be BF16")
    result = cache.clone()
    flat = result.view(-1, result.shape[-1])
    for row, slot in zip(values, slot_mapping.reshape(-1, 2).tolist()):
        page, offset = slot
        if page >= 0:
            flat[page * CONTRACT.swa_block_size + offset].copy_(row)
    return result


def decode_gather_swa_cache_reference(
    cache: decode_swa_cache_torch.Tensor, indices: decode_swa_cache_torch.Tensor
) -> decode_swa_cache_torch.Tensor:
    """Gather physical cache rows; negative indices produce zero rows."""

    import torch

    flat = cache.reshape(-1, cache.shape[-1])
    result = torch.zeros(*indices.shape, cache.shape[-1], dtype=cache.dtype, device=cache.device)
    valid = indices >= 0
    if valid.any():
        result[valid] = flat[indices[valid].long()]
    return result


def swa_golden_cases(*, build=True):
    """Cover 128/640 windows, page edges, skipped slots, masks and inactive rows."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec
    from golden.validation import ratio_allclose

    capacity, blocks, heads, dim = 16, 6, 16, 64
    generator = torch.Generator().manual_seed(517)
    query = (torch.randn((capacity, heads * dim), generator=generator) * 0.2).to(torch.bfloat16)
    kv = (torch.randn((capacity, dim), generator=generator) * 0.3).to(torch.bfloat16)
    cache = (torch.randn((blocks, 128, 1, dim), generator=generator) * 0.3).to(torch.bfloat16)
    sink = torch.linspace(-2.0, 2.0, heads)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)
    for window in (128, 640):
        config = dict(num_attention_heads=heads, head_dim=dim, sliding_window=window)
        for active in (0, 1, 7, capacity):
            slots = torch.tensor(
                [[token // 4, token * 37 % 128] for token in range(capacity)], dtype=torch.int64
            )
            slots[0] = torch.tensor([0, 127])
            slots[1] = torch.tensor([1, 0])
            slots[3] = torch.tensor([-1, 999])
            slots[active:] = torch.tensor([blocks + 1, 999])
            indices = (torch.arange(capacity * window).reshape(capacity, window) * 7 % (blocks * 128)).int()
            indices[:, ::5] = -1
            indices[0] = -1
            if active > 1:
                indices[1, 0] = 128
            indices[active:] = blocks * 128 + 999
            all_masked = list(range(min(active, 1)))
            specs = [
                TensorSpec("query", [capacity, heads * dim], torch.bfloat16, init_value=query),
                TensorSpec("kv", [capacity, dim], torch.bfloat16, init_value=kv),
                TensorSpec("slots", [capacity, 2], torch.int64, init_value=slots),
                TensorSpec("indices", [capacity, window], torch.int32, init_value=indices),
                TensorSpec("sink", [heads], torch.float32, init_value=sink),
                TensorSpec("cache", [blocks, 128, 1, dim], torch.bfloat16, init_value=cache),
                TensorSpec("output", [capacity, heads * dim], torch.bfloat16, init_value=-43.0),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ]
            case = dict(
                name=f"swa-window{window}-active{active}",
                specs=specs,
                golden_fn=partial(golden_swa, config=config),
                compare_fn=dict(cache=exact, output=swa_output_compare(active, all_masked)),
            )
            if build:
                case["fn"] = make_swa(capacity, blocks, config)
            yield case


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Run bounded SWA goldens; CPU mode does not compile or execute a kernel."""
    import torch
    from golden import run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    for case in swa_golden_cases(build=not cpu_only):
        name = case.pop("name")
        if cpu_only:
            values = {
                spec.name: spec.create_tensor().clone()
                if hasattr(spec, "create_tensor")
                else spec.to_python()
                for spec in case["specs"]
            }
            before = {key: value.clone() for key, value in values.items() if isinstance(value, torch.Tensor)}
            case["golden_fn"](values)
            active = int(values["active"])
            for key in before:
                if key not in ("cache", "output") and not torch.equal(values[key], before[key]):
                    raise RuntimeError("SWA golden mutated input " + key)
            expected_cache = before["cache"].clone()
            for token in range(active):
                page, offset = (int(value) for value in values["slots"][token])
                if page >= 0:
                    expected_cache[page, offset, 0] = values["kv"][token]
            if not torch.equal(values["cache"], expected_cache):
                raise RuntimeError("SWA cache publication changed an untouched byte")
            if not torch.equal(values["output"][active:], before["output"][active:]):
                raise RuntimeError("SWA golden changed an inactive output row")
            if active and values["output"][0].count_nonzero():
                raise RuntimeError("All-masked SWA row must retain the zero-value sink result")
            if not torch.isfinite(values["output"]).all():
                raise RuntimeError("SWA golden produced nonfinite output")
            if active == 1:
                for invalid_name in ("slot", "selection"):
                    invalid = {
                        key: value.clone() if isinstance(value, torch.Tensor) else value
                        for key, value in values.items()
                    }
                    if invalid_name == "slot":
                        invalid["slots"][0] = torch.tensor([values["cache"].shape[0], 0])
                    else:
                        invalid["indices"][0, 0] = values["cache"].shape[0] * 128
                    try:
                        case["golden_fn"](invalid)
                    except ValueError:
                        pass
                    else:
                        raise RuntimeError("SWA golden accepted an out-of-allocation " + invalid_name)
            records.append(dict(name=name, cpu_golden="passed", device="not_run"))
            continue
        result = run(
            **case,
            config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
            compile_only=compile_only,
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
        records.append(dict(name=name, compiled=True, device="not_run" if compile_only else "passed"))
    return records


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(device=args.device, sim=args.sim, compile_only=args.compile_only, cpu_only=args.cpu_only)
        )
    )
