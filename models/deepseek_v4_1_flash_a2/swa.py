# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 paged SWA cache publication, window gather and sink attention."""

import pypto.language as pl


# tiling
HEAD_TILE = 64
SINGLE_HEAD_TILE = 32
SINGLE_VALUE_TILE = 512
WINDOW_TILE = 128
GATHER_TILE = 64
VALUE_TILE = 256
SCAN_TILE = 4
ONLINE_HEAD_TILE = 16


def make_swa_inline(capacity, cache_blocks, config):
    """Build a BF16 PA_BBND SWA leaf for the full configured query width."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    window = config["sliding_window"]
    if window % WINDOW_TILE:
        raise ValueError("A2 SWA requires a window divisible by its 128-token softmax tile")
    width = heads * head_dim
    cache_rows = cache_blocks * 128
    score_scale = head_dim ** -0.5
    use_single_window = window == WINDOW_TILE and heads % HEAD_TILE == 0 and head_dim % VALUE_TILE == 0

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
                flat[row:row + 1, :] = kv[token:token + 1, :]
        return cache

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        selected: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(active * (window // GATHER_TILE), name_hint="a2_swa_gather"):
            token = block // (window // GATHER_TILE)
            column = block % (window // GATHER_TILE) * GATHER_TILE
            index = pl.load(indices, [token, column], [1, GATHER_TILE])
            safe_index = pl.maximum(index, 0)
            gathered = pl.tile.mgather(flat, safe_index, coalesce="row")
            shifted = pl.add(index, 1)
            nonnegative = pl.maximum(shifted, 0)
            valid = pl.minimum(nonnegative, 1)
            valid_column = pl.reshape(valid, [GATHER_TILE, 1])
            bits = pl.tile.full([GATHER_TILE, head_dim // 32], dtype=pl.INT32, value=-1)
            mask_words = pl.row_expand_mul(bits, valid_column)
            mask_bytes = pl.reinterpret_view(mask_words, pl.UINT8)
            scratch = pl.tile.create([1, head_dim], dtype=pl.UINT16)
            gathered_bits = pl.reinterpret_view(gathered, pl.UINT16)
            masked_bits = pl.tile.sels(mask_bytes, gathered_bits, scratch, 0)
            rows = pl.reinterpret_view(masked_bits, pl.BF16)
            selected_row = token * window + column
            pl.store(rows, [selected_row, 0], selected)
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
        for block in pl.spmd(active * (heads // ONLINE_HEAD_TILE), name_hint="a2_swa_attention", optimizations=[pl.cross_core_slot(slot_num=1)]):
            token = block // (heads // ONLINE_HEAD_TILE)
            head = block % (heads // ONLINE_HEAD_TILE) * ONLINE_HEAD_TILE
            query_row = token * heads + head
            # Include the zero-value sink before narrowing exp(scores-max) to
            # BF16, matching the native softmax's rounding boundary.
            maximum = pl.reshape(sink[head:head + ONLINE_HEAD_TILE], [1, ONLINE_HEAD_TILE])
            denominator = pl.full([1, ONLINE_HEAD_TILE], dtype=pl.FP32, value=1.0)
            numerator = pl.full([ONLINE_HEAD_TILE, head_dim], dtype=pl.FP32, value=0.0)
            for part in pl.range(window // WINDOW_TILE):
                cache_row = token * window + part * WINDOW_TILE
                q = qflat[query_row:query_row + ONLINE_HEAD_TILE, :]
                kv = selected[cache_row:cache_row + WINDOW_TILE, :]
                dot = pl.matmul(q, kv, b_trans=True)
                scores = pl.mul(dot, score_scale)
                index_tile = indices[token:token + 1, part * WINDOW_TILE:part * WINDOW_TILE + WINDOW_TILE]
                index_float = pl.cast(index_tile, pl.FP32)
                shifted_index = pl.add(index_float, 1.0)
                positive_index = pl.maximum(shifted_index, 0.0)
                valid = pl.minimum(positive_index, 1.0)
                invalid = pl.sub(valid, 1.0)
                bias = pl.mul(invalid, 1e30)
                scores = pl.col_expand_add(scores, bias)
                row_max = pl.row_max(scores)
                local_max = pl.reshape(row_max, [1, ONLINE_HEAD_TILE])
                next_max = pl.maximum(maximum, local_max)
                max_difference = pl.sub(maximum, next_max)
                correction = pl.exp(max_difference)
                max_column = pl.reshape(next_max, [ONLINE_HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_column)
                exponential = pl.exp(shifted)
                probabilities = pl.col_expand_mul(exponential, valid)
                row_sum = pl.row_sum(probabilities)
                probability_sum = pl.reshape(row_sum, [1, ONLINE_HEAD_TILE])
                old_denominator = pl.mul(denominator, correction)
                denominator = pl.add(old_denominator, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="rint")
                weighted = pl.matmul(weights, kv)
                correction_column = pl.reshape(correction, [ONLINE_HEAD_TILE, 1])
                corrected = pl.row_expand_mul(numerator, correction_column)
                numerator = pl.add(corrected, weighted)
                maximum = next_max
            one = pl.full([1, ONLINE_HEAD_TILE], dtype=pl.FP32, value=1.0)
            normalized = pl.div(one, denominator, high_precision=True)
            normalized_column = pl.reshape(normalized, [ONLINE_HEAD_TILE, 1])
            result = pl.row_expand_mul(numerator, normalized_column)
            oflat[query_row:query_row + ONLINE_HEAD_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        return output

    def make_attend_single(head_tile, value_tile):
        single_head_tile = min(head_tile, heads)
        single_value_tile = min(value_tile, head_dim)

        @pl.jit.inline
        def attend_single(
            query: pl.Tensor[[capacity, width], pl.BF16],
            incoming: pl.Tensor[[capacity, head_dim], pl.BF16],
            slots: pl.Tensor[[capacity, 2], pl.INT64],
            cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
            indices: pl.Tensor[[capacity, window], pl.INT32],
            sink: pl.Tensor[[heads], pl.FP32],
            output: pl.Tensor[[capacity, width], pl.BF16],
            active: pl.Scalar[pl.INT32],
            local_publish: pl.Scalar[pl.INT32],
        ):
            qflat = pl.reshape(query, [capacity * heads, head_dim])
            oflat = pl.reshape(output, [capacity * heads, head_dim])
            flat = pl.reshape(cache, [cache_rows, head_dim])
            for block in pl.spmd(active * (heads // single_head_tile), name_hint="a2_swa_attention", optimizations=[pl.cross_core_slot(slot_num=2)]):
                token = block // (heads // single_head_tile)
                head = block % (heads // single_head_tile) * single_head_tile
                query_row = token * heads + head
                publish_page = pl.cast(-1, pl.INT64)
                publish_row = pl.cast(0, pl.INDEX)
                if local_publish != 0:
                    publish_page = pl.read(slots, [token, 0])
                    publish_offset = pl.read(slots, [token, 1])
                    publish_row = pl.cast(publish_page * 128 + publish_offset, pl.INDEX)
                q = qflat[query_row:query_row + single_head_tile, :]
                segment_count = pl.cast(0, pl.INDEX)
                first_start = pl.cast(0, pl.INDEX)
                first_source = pl.cast(0, pl.INDEX)
                first_rows = pl.cast(0, pl.INDEX)
                second_start = pl.cast(0, pl.INDEX)
                second_source = pl.cast(0, pl.INDEX)
                second_rows = pl.cast(0, pl.INDEX)
                last_slot = pl.cast(-2, pl.INDEX)
                if local_publish == 2:
                    for probe_token in pl.range(active):
                        probe_page = pl.read(slots, [probe_token, 0])
                        probe_offset = pl.read(slots, [probe_token, 1])
                        probe_address = pl.cast(probe_page * 128 + probe_offset, pl.INDEX)
                        if probe_page >= 0:
                            if probe_address != last_slot + 1:
                                segment_count = segment_count + 1
                                if segment_count == 1:
                                    first_start = probe_token
                                    first_source = probe_address
                                if segment_count == 2:
                                    second_start = probe_token
                                    second_source = probe_address
                            if segment_count == 1:
                                first_rows = first_rows + 1
                            if segment_count == 2:
                                second_rows = second_rows + 1
                            last_slot = probe_address
                        else:
                            last_slot = pl.cast(-2, pl.INDEX)
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
                        if local_publish == 1 and publish_page >= 0 and run_source >= 0:
                            if publish_row >= source and publish_row < source + run_rows:
                                patch_row = run_start + publish_row - source
                                kv = pl.gather_row(kv, incoming, [patch_row, 0], [0, 0], [1, head_dim])
                        if local_publish == 2 and run_source >= 0:
                            if segment_count <= 2:
                                for fast_id in pl.unroll(2):
                                    if fast_id == 0:
                                        fast_base = first_source
                                        fast_start = first_start
                                        fast_rows = first_rows
                                    else:
                                        fast_base = second_source
                                        fast_start = second_start
                                        fast_rows = second_rows
                                    copy_begin = pl.max(source, fast_base)
                                    copy_end = pl.min(source + run_rows, fast_base + fast_rows)
                                    if copy_end > copy_begin:
                                        copy_target = run_start + copy_begin - source
                                        copy_source = fast_start + copy_begin - fast_base
                                        copy_rows = copy_end - copy_begin
                                        kv = pl.gather_row(kv, incoming, [copy_target, 0], [copy_source, 0], [WINDOW_TILE, head_dim], valid_shape=[copy_rows, head_dim])
                            else:
                                patch_start = pl.cast(0, pl.INDEX)
                                patch_source = pl.cast(0, pl.INDEX)
                                previous_address = pl.cast(-2, pl.INDEX)
                                patch_open = pl.cast(False, pl.BOOL)
                                for published in pl.range(active + 1):
                                    next_address = pl.cast(-2, pl.INDEX)
                                    matching = pl.cast(False, pl.BOOL)
                                    if published < active:
                                        next_page = pl.read(slots, [published, 0])
                                        next_offset = pl.read(slots, [published, 1])
                                        next_address = pl.cast(next_page * 128 + next_offset, pl.INDEX)
                                        matching = next_page >= 0 and next_address >= source and next_address < source + run_rows
                                    if patch_open and (not matching or next_address != previous_address + 1):
                                        patch_rows = published - patch_start
                                        target_row = run_start + patch_source - source
                                        kv = pl.gather_row(kv, incoming, [target_row, 0], [patch_start, 0], [WINDOW_TILE, head_dim], valid_shape=[patch_rows, head_dim])
                                    if matching and (not patch_open or next_address != previous_address + 1):
                                        patch_start = published
                                        patch_source = next_address
                                    patch_open = matching
                                    previous_address = next_address
                        run_start = row
                        run_source = current
                    previous = current
                dot = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    if local_publish != 0 and lane == 0 and head == 0 and publish_page >= 0:
                        flat[publish_row:publish_row + 1, :] = incoming[token:token + 1, :]
                    dot_part = pl.aiv_shard(dot)
                    lane_head = head + lane * (single_head_tile // 2)
                    scores = pl.mul(dot_part, score_scale)
                    index_tile = indices[token:token + 1, :WINDOW_TILE]
                    index_float = pl.cast(index_tile, pl.FP32)
                    shifted_index = pl.add(index_float, 1.0)
                    positive_index = pl.maximum(shifted_index, 0.0)
                    valid = pl.minimum(positive_index, 1.0)
                    invalid = pl.sub(valid, 1.0)
                    bias = pl.mul(invalid, 1e30)
                    scores = pl.col_expand_add(scores, bias)
                    row_max = pl.row_max(scores)
                    local_max = pl.reshape(row_max, [1, single_head_tile // 2])
                    sink_row = pl.reshape(sink[lane_head:lane_head + single_head_tile // 2], [1, single_head_tile // 2])
                    maximum = pl.maximum(local_max, sink_row)
                    sink_difference = pl.sub(sink_row, maximum)
                    sink_exp = pl.exp(sink_difference)
                    max_column = pl.reshape(maximum, [single_head_tile // 2, 1])
                    shifted = pl.row_expand_sub(scores, max_column)
                    exponential = pl.exp(shifted)
                    probabilities = pl.col_expand_mul(exponential, valid)
                    row_sum = pl.row_sum(probabilities)
                    probability_sum = pl.reshape(row_sum, [1, single_head_tile // 2])
                    denominator = pl.add(sink_exp, probability_sum)
                    weights = pl.cast(probabilities, pl.BF16, mode="rint")
                    full_weights = pl.aic_gather(weights)
                    one = pl.full([1, single_head_tile // 2], dtype=pl.FP32, value=1.0)
                    normalized = pl.div(one, denominator, high_precision=True)
                    normalized_column = pl.reshape(normalized, [single_head_tile // 2, 1])
                for value_column in pl.range(0, head_dim, single_value_tile):
                    values = kv[:, value_column:value_column + single_value_tile]
                    weighted = pl.matmul(full_weights, values, out_dtype=pl.FP32)
                    for lane_out in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        weighted_part = pl.aiv_shard(weighted)
                        result = pl.row_expand_mul(weighted_part, normalized_column)
                        output_row = query_row + lane_out * (single_head_tile // 2)
                        narrowed = pl.cast(result, pl.BF16, mode="rint")
                        oflat[output_row:output_row + single_head_tile // 2, value_column:value_column + single_value_tile] = narrowed
            return output

        return attend_single

    attend_single_token = make_attend_single(SINGLE_HEAD_TILE, SINGLE_VALUE_TILE)
    attend_batch = make_attend_single(HEAD_TILE, VALUE_TILE)

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
        if use_single_window:
            if active == 1:
                attend_single_token(query, kv, slots, cache, indices, sink, output, active, 1)
            else:
                attend_batch(query, kv, slots, cache, indices, sink, output, active, 2)
        else:
            publish(kv, slots, cache, active)
            selected = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
            gather(cache, indices, selected, active)
            attend(query, selected, indices, sink, output, active)
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
