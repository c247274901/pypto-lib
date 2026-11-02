# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 HC coefficients, delayed stream collapse and BF16 residual expansion."""

import pypto.language as pl


# tiling
D_TILE = 640
PIPELINE_TILE = 4
MERGE_D_TILE = 5120
POST_STREAM_TILE = 2
TOKEN_TILE = 8
CHANNEL_TILE = 8
MIX_TILE = 8


def make_hc_pre_inline(capacity, config):
    """Build four-stream HC coefficient generation and incoming-premix collapse."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    flat_width = hidden * hc
    channels = (hc + 2) * hc
    iterations = config["hc_sinkhorn_iters"]
    eps = config["hc_eps"]
    norm_eps = config["rms_norm_eps"]

    @pl.jit.inline
    def pre(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, flat_width], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        next_pre: pl.Tensor[[capacity, hc], pl.FP32],
        post: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        collapsed: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(x, [capacity, flat_width])
        projected = pl.create_tensor([capacity, channels], dtype=pl.FP32)
        for block in pl.spmd(active * (channels // CHANNEL_TILE), name_hint="a2_hc_projection"):
            token = block // (channels // CHANNEL_TILE)
            channel = block % (channels // CHANNEL_TILE) * CHANNEL_TILE
            dot_acc = pl.full([CHANNEL_TILE, D_TILE], dtype=pl.FP32, value=0.0)
            square_acc = pl.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.pipeline(0, flat_width, D_TILE, stage=PIPELINE_TILE):
                source = pl.cast(flat[token:token + 1, inner:inner + D_TILE], pl.FP32)
                square = pl.mul(source, source)
                square_acc = pl.add(square_acc, square)
                weight = function[channel:channel + CHANNEL_TILE, inner:inner + D_TILE]
                product = pl.col_expand_mul(weight, source)
                dot_acc = pl.add(dot_acc, product)
            dot_sum = pl.reshape(pl.row_sum(dot_acc), [1, CHANNEL_TILE])
            square_rows = pl.reshape(square_acc, [8, D_TILE // 8])
            square_partials = pl.row_sum(square_rows)
            square_total = pl.cast(0.0, pl.FP32)
            for lane in pl.unroll(8):
                square_total = square_total + pl.read(square_partials, [lane, 0])
            mean = square_total * (1.0 / flat_width)
            ones = pl.full([1, CHANNEL_TILE], dtype=pl.FP32, value=1.0)
            argument = pl.mul(ones, mean + norm_eps)
            inverse_tile = pl.rsqrt(argument, high_precision=False)
            normalized = pl.mul(dot_sum, inverse_tile)
            projected[token:token + 1, channel:channel + CHANNEL_TILE] = normalized

        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        base_row = pl.reshape(base, [1, channels])
        for token in pl.spmd(active, name_hint="a2_hc_sinkhorn"):
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])
            packed_logits = pl.tile.full([2, MIX_TILE], dtype=pl.FP32, value=0.0)
            for column in pl.unroll(MIX_TILE):
                logical = column % hc
                pre_value = pl.read(projected, [token, logical]) * scale0 + pl.read(base, [logical])
                post_value = pl.read(projected, [token, hc + logical]) * scale1 + pl.read(base, [hc + logical])
                pl.tile.write(packed_logits, [0, column], pre_value)
                pl.tile.write(packed_logits, [1, column], post_value)
            packed_neg = pl.neg(packed_logits)
            packed_exp = pl.exp(packed_neg)
            packed_denominator = pl.add(packed_exp, 1.0)
            packed_sigmoid = pl.recip(packed_denominator)
            pre_sigmoid = pl.slice(packed_sigmoid, [1, MIX_TILE], [0, 0])
            post_sigmoid = pl.slice(packed_sigmoid, [1, MIX_TILE], [1, 0])
            pre_result = pl.add(pre_sigmoid, eps)
            post_result = pl.mul(post_sigmoid, 2.0)
            pre_valid = pl.set_validshape(pre_result, 1, hc)
            post_valid = pl.set_validshape(post_result, 1, hc)
            pl.store(pre_valid, [token, 0], next_pre)
            pl.store(post_valid, [token, 0], post)
            mix_projected = pl.load(projected, [token, 2 * hc], [1, hc * hc])
            mix_base = pl.load(base_row, [0, 2 * hc], [1, hc * hc])
            mix_scaled = pl.mul(mix_projected, scale2)
            mix_logits = pl.add(mix_scaled, mix_base)
            flat_indices = pl.tile.arange(0, [1, MIX_TILE * MIX_TILE], dtype=pl.INT16)
            row_indices = pl.and_(pl.shr(flat_indices, 3), hc - 1)
            column_indices = pl.and_(flat_indices, hc - 1)
            row_offsets = pl.mul(row_indices, hc)
            mix_indices16 = pl.add(row_offsets, column_indices)
            mix_indices_fp16 = pl.cast(mix_indices16, pl.FP16)
            mix_indices_fp32 = pl.cast(mix_indices_fp16, pl.FP32)
            mix_indices = pl.cast(mix_indices_fp32, pl.INT32, mode="rint")
            gather_tmp = pl.create_tile([1, MIX_TILE * MIX_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            logits_flat = pl.tile.gather(mix_logits, mix_indices, gather_tmp)
            logits = pl.reshape(logits_flat, [MIX_TILE, MIX_TILE])
            row_tmp = pl.create_tile([MIX_TILE, MIX_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_max = pl.row_max(logits, row_tmp)
            shifted = pl.row_expand_sub(logits, row_max)
            exponential = pl.exp(shifted)
            row_sum = pl.row_sum(exponential, row_tmp)
            row_sum = pl.mul(row_sum, hc / MIX_TILE)
            probabilities = pl.row_expand_div(exponential, row_sum)
            probabilities = pl.add(probabilities, eps)
            column_sum = pl.col_sum(probabilities)
            column_sum = pl.mul(column_sum, hc / MIX_TILE)
            column_sum = pl.add(column_sum, eps)
            probabilities = pl.col_expand_div(probabilities, column_sum)
            for _ in pl.range(iterations - 1):
                row_sum = pl.row_sum(probabilities, row_tmp)
                row_sum = pl.mul(row_sum, hc / MIX_TILE)
                row_sum = pl.add(row_sum, eps)
                probabilities = pl.row_expand_div(probabilities, row_sum)
                column_sum = pl.col_sum(probabilities)
                column_sum = pl.mul(column_sum, hc / MIX_TILE)
                column_sum = pl.add(column_sum, eps)
                probabilities = pl.col_expand_div(probabilities, column_sum)
            for row in pl.unroll(hc):
                comb_value = pl.slice(probabilities, [1, MIX_TILE], [row, 0], valid_shape=[1, hc])
                pl.store(comb_value, [token, row * hc], comb_flat)

        for block in pl.spmd(active * (hidden // MERGE_D_TILE), name_hint="a2_hc_collapse"):
            token = block // (hidden // MERGE_D_TILE)
            offset = block % (hidden // MERGE_D_TILE) * MERGE_D_TILE
            collapsed_tile = pl.full([1, MERGE_D_TILE], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(hc):
                source_offset = stream * hidden + offset
                source_bf16 = flat[token:token + 1, source_offset:source_offset + MERGE_D_TILE]
                source_fp32 = pl.cast(source_bf16, pl.FP32)
                coefficient = pl.read(incoming, [token, stream])
                term = pl.mul(source_fp32, coefficient)
                collapsed_tile = pl.add(collapsed_tile, term)
            collapsed[token:token + 1, offset:offset + MERGE_D_TILE] = pl.cast(collapsed_tile, pl.BF16, mode="rint")
        return next_pre, post, comb, collapsed

    return pre


def make_hc_post_inline(capacity, config):
    """Build BF16 residual expansion with FP32 combination arithmetic."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]

    @pl.jit.inline
    def post(
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        residual: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        post_mix: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        output: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        residual_rows = pl.reshape(residual, [capacity * hc, hidden])
        output_rows = pl.reshape(output, [capacity * hc, hidden])
        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for block in pl.spmd(active * (hc // POST_STREAM_TILE) * (hidden // MERGE_D_TILE), name_hint="a2_hc_post"):
            token = block // ((hc // POST_STREAM_TILE) * (hidden // MERGE_D_TILE))
            stream_start = block // (hidden // MERGE_D_TILE) % (hc // POST_STREAM_TILE) * POST_STREAM_TILE
            offset = block % (hidden // MERGE_D_TILE) * MERGE_D_TILE
            value = pl.cast(sublayer[token:token + 1, offset:offset + MERGE_D_TILE], pl.FP32)
            source_rows = residual_rows[token * hc:token * hc + hc, offset:offset + MERGE_D_TILE]
            source_values = pl.cast(source_rows, pl.FP32)
            for stream_lane in pl.unroll(POST_STREAM_TILE):
                stream = stream_start + stream_lane
                coefficient = pl.read(post_mix, [token, stream])
                merged = pl.mul(value, coefficient)
                for source_stream in pl.unroll(hc):
                    source = source_values[source_stream:source_stream + 1, :]
                    coefficient = pl.read(comb_flat, [token, source_stream * hc + stream])
                    term = pl.mul(source, coefficient)
                    merged = pl.add(merged, term)
                destination = token * hc + stream
                merged_bf16 = pl.cast(merged, pl.BF16, mode="rint")
                output_rows[destination:destination + 1, offset:offset + MERGE_D_TILE] = merged_bf16
        return output

    return post

def make_hc(capacity, config):
    """Build the standalone HC operator diagnostic entry."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    pre = make_hc_pre_inline(capacity, config)
    post_fn = make_hc_post_inline(capacity, config)

    @pl.jit
    def hc_entry(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        merged: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        pre(x, function, scale, base, incoming, next_pre, post, comb, collapsed, active)
        post_fn(sublayer, x, post, comb, merged, active)
        return next_pre, post, comb, collapsed, merged

    return hc_entry
