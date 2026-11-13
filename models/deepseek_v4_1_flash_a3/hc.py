# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hyper-connection kernels, Torch goldens and validation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl


# tiling
D_TILE = 256
TOKEN_TILE = 8
CHANNEL_TILE = 8
MIX_TILE = 8
PROJ_T_TILE = 16
PROJ_N_TILE = 16
PROJ_K_TILE = 64
MERGE_D_TILE = 5120
POST_STREAM_TILE = 2
PREFILL_PROJ_K_TILE = 1024
PREFILL_PROJ_REDUCE_TILE = 4096
PREFILL_RMS_D_TILE = 2304
PREFILL_WEIGHT_D_TILE = 20480


def make_hc_collapse_inline(capacity, config):
    """Collapse four BF16 residual streams with FP32 premix weights."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    merge_tile = MERGE_D_TILE if hidden % MERGE_D_TILE == 0 else D_TILE

    @pl.jit.inline
    def collapse(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(x, [capacity, hc * hidden])
        for block in pl.spmd(active * (hidden // merge_tile), name_hint="a2_hc_collapse"):
            token = block // (hidden // merge_tile)
            offset = block % (hidden // merge_tile) * merge_tile
            collapsed_tile = pl.full([1, merge_tile], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(hc):
                source_offset = stream * hidden + offset
                source_bf16 = flat[token : token + 1, source_offset : source_offset + merge_tile]
                source_fp32 = pl.cast(source_bf16, pl.FP32)
                coefficient = pl.read(incoming, [token, stream])
                term = pl.mul(source_fp32, coefficient)
                collapsed_tile = pl.add(collapsed_tile, term)
            result = pl.cast(collapsed_tile, pl.BF16, mode="rint")
            collapsed[token : token + 1, offset : offset + merge_tile] = result
        return collapsed

    return collapse


def make_hc_pre_inline(capacity, config):
    """Build four-stream HC coefficient generation and incoming-premix collapse."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    flat_width = hidden * hc
    channels = (hc + 2) * hc
    iterations = config["hc_sinkhorn_iters"]
    eps = config["hc_eps"]
    norm_eps = config["rms_norm_eps"]
    projection_parts = (flat_width + PREFILL_PROJ_REDUCE_TILE - 1) // PREFILL_PROJ_REDUCE_TILE
    rms_width = flat_width // TOKEN_TILE
    rms_tile = min(rms_width, PREFILL_RMS_D_TILE)
    collapse = make_hc_collapse_inline(capacity, config)

    @pl.jit.inline
    def pre(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, flat_width], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        rounded_function = pl.create_tensor([channels, flat_width], dtype=pl.FP32)
        for channel_pair in pl.spmd(pl.min(active, 1) * ((channels + 1) // 2), name_hint="a2_hc_weight_round", allow_early_resolve=True):
            for weight_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                channel_row = channel_pair * 2 + weight_lane
                if channel_row < channels:
                    for weight_offset in pl.range(0, flat_width, PREFILL_WEIGHT_D_TILE):
                        weight_valid = pl.min(PREFILL_WEIGHT_D_TILE, flat_width - weight_offset)
                        weight_values = pl.slice(function, [1, PREFILL_WEIGHT_D_TILE], [channel_row, weight_offset], valid_shape=[1, weight_valid])
                        weight_int = pl.reinterpret_view(weight_values, pl.INT32)
                        weight_units = pl.shrs(pl.add(weight_int, 2048), 12)
                        weight_rounded = pl.shls(weight_units, 12)
                        rounded_values = pl.reinterpret_view(weight_rounded, pl.FP32)
                        rounded_function[channel_row : channel_row + 1, weight_offset : weight_offset + PREFILL_WEIGHT_D_TILE] = rounded_values

        rms_rows = pl.reshape(x, [capacity * TOKEN_TILE, rms_width])
        converted = pl.create_tensor([capacity, flat_width], dtype=pl.FP32)
        converted_rows = pl.reshape(converted, [capacity * TOKEN_TILE, rms_width])
        inverse = pl.create_tensor([capacity, TOKEN_TILE], dtype=pl.FP32, manual_dep=True)
        for token in pl.spmd(active, name_hint="a2_hc_rms", allow_early_resolve=True):
            square_sum = pl.tile.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, rms_width, rms_tile):
                valid_d = pl.min(rms_tile, rms_width - inner)
                values = pl.load(rms_rows, [token * TOKEN_TILE, inner], [TOKEN_TILE, rms_tile],
                                  valid_shape=[TOKEN_TILE, valid_d])
                values = pl.fillpad(values, pad_value=pl.PadValue.zero)
                values_fp32 = pl.cast(values, pl.FP32)
                converted_tile = pl.set_validshape(values_fp32, TOKEN_TILE, valid_d)
                pl.store(converted_tile, [token * TOKEN_TILE, inner], converted_rows)
                square: pl.Tile[[TOKEN_TILE, rms_tile], pl.FP32, pl.MemRef("hc_rms_square"), pl.Mem.Vec] = pl.mul(values_fp32, values_fp32)
                rms_tmp = pl.create_tile([TOKEN_TILE, rms_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                rms_partial = pl.reshape(pl.row_sum(square, rms_tmp), [1, TOKEN_TILE])
                square_sum = pl.add(square_sum, rms_partial)
            partial_column = pl.reshape(square_sum, [TOKEN_TILE, 1])
            ones = pl.tile.full([TOKEN_TILE, TOKEN_TILE], dtype=pl.FP32, value=1.0)
            partial_rows = pl.row_expand_mul(ones, partial_column)
            total = pl.col_sum(partial_rows)
            mean = pl.mul(total, 1.0 / flat_width)
            argument = pl.add(mean, norm_eps)
            inverse_tmp = pl.create_tile([1, TOKEN_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            inverse_tile = pl.tile.rsqrt(argument, inverse_tmp)
            pl.store(inverse_tile, [token, 0], inverse)

        accumulated = pl.create_tensor([projection_parts * capacity, channels], dtype=pl.FP32)
        channel_blocks = (channels + PROJ_N_TILE - 1) // PROJ_N_TILE
        for block in pl.spmd(
            (active + PROJ_T_TILE - 1) // PROJ_T_TILE * channel_blocks * projection_parts,
            name_hint="a2_hc_projection_cube",
            allow_early_resolve=True,
        ):
            token = block // (channel_blocks * projection_parts) * PROJ_T_TILE
            channel = block % channel_blocks * PROJ_N_TILE
            partition = block // channel_blocks % projection_parts
            partition_start = partition * PREFILL_PROJ_REDUCE_TILE
            partition_stop = pl.min(partition_start + PREFILL_PROJ_REDUCE_TILE, flat_width)
            rows = pl.min(PROJ_T_TILE, active - token)
            columns = pl.min(PROJ_N_TILE, channels - channel)
            acc = pl.create_tensor([PROJ_T_TILE, PROJ_N_TILE], dtype=pl.FP32)
            for inner in pl.range(partition_start, partition_stop, PREFILL_PROJ_K_TILE):
                valid_k = pl.min(PREFILL_PROJ_K_TILE, flat_width - inner)
                source = pl.slice(
                    converted, [PROJ_T_TILE, PREFILL_PROJ_K_TILE], [token, inner], valid_shape=[rows, valid_k]
                )
                weight = pl.slice(
                    rounded_function, [PROJ_N_TILE, PREFILL_PROJ_K_TILE], [channel, inner], valid_shape=[columns, valid_k]
                )
                acc = pl.matmul_acc(acc, source, weight, b_trans=True, init_cond=(inner == partition_start))
            projection_tile = pl.set_validshape(acc, rows, columns)
            output_row = partition * capacity + token
            accumulated[output_row : output_row + PROJ_T_TILE, channel : channel + PROJ_N_TILE] = projection_tile

        comb_rows = pl.reshape(comb, [capacity * hc, hc])
        base_row = pl.reshape(base, [1, channels])
        for token in pl.spmd(active, name_hint="a2_hc_sinkhorn", allow_early_resolve=True):
            inverse_scalar = pl.read(inverse, [token, 0])
            projection_sum = pl.load(accumulated, [token, 0], [1, channels])
            for part in pl.unroll(1, projection_parts):
                partial_row = part * capacity + token
                partial = pl.load(accumulated, [partial_row, 0], [1, channels])
                projection_sum = pl.add(projection_sum, partial)
            projected = pl.mul(projection_sum, inverse_scalar)
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])
            base_values = pl.load(base_row, [0, 0], [1, channels])
            pre_logit_values = pl.add(pl.mul(projected, scale0), base_values)
            post_logit_values = pl.add(pl.mul(projected, scale1), base_values)
            mix_logit_values = pl.add(pl.mul(projected, scale2), base_values)
            pre_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            post_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            for column in pl.unroll(MIX_TILE):
                logical = column % hc
                pre_value = pl.tile.read(pre_logit_values, [0, logical])
                post_value = pl.tile.read(post_logit_values, [0, hc + logical])
                pl.tile.write(pre_logits, [0, column], pre_value)
                pl.tile.write(post_logits, [0, column], post_value)
            pre_neg = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_neg)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre_sigmoid = pl.recip(pre_denominator)
            pre_result = pl.add(pre_sigmoid, eps)
            post_neg = pl.neg(post_logits)
            post_exp = pl.exp(post_neg)
            post_denominator = pl.add(post_exp, 1.0)
            post_sigmoid = pl.recip(post_denominator)
            post_result = pl.mul(post_sigmoid, 2.0)
            pre_valid = pl.set_validshape(pre_result, 1, hc)
            post_valid = pl.set_validshape(post_result, 1, hc)
            pl.store(pre_valid, [token, 0], next_pre)
            pl.store(post_valid, [token, 0], post)
            logits = pl.tile.full([MIX_TILE, MIX_TILE], dtype=pl.FP32, value=0.0)
            for row in pl.unroll(hc):
                for column in pl.unroll(MIX_TILE):
                    logical = 2 * hc + (row % hc) * hc + column % hc
                    value = pl.tile.read(mix_logit_values, [0, logical])
                    pl.tile.write(logits, [row, column], value)
            logits = pl.set_validshape(logits, hc, MIX_TILE)
            row_tmp = pl.create_tile([MIX_TILE, MIX_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_max = pl.row_max(logits, row_tmp)
            shifted = pl.row_expand_sub(logits, row_max)
            exponential = pl.exp(shifted)
            row_sum = pl.row_sum(exponential, row_tmp)
            row_sum = pl.mul(row_sum, hc / MIX_TILE)
            probabilities = pl.row_expand_div(exponential, row_sum)
            probabilities = pl.add(probabilities, eps)
            column_sum = pl.col_sum(probabilities)
            column_sum = pl.add(column_sum, eps)
            probabilities = pl.col_expand_div(probabilities, column_sum)
            for _ in pl.range(iterations - 1):
                row_sum = pl.row_sum(probabilities, row_tmp)
                row_sum = pl.mul(row_sum, hc / MIX_TILE)
                row_sum = pl.add(row_sum, eps)
                probabilities = pl.row_expand_div(probabilities, row_sum)
                column_sum = pl.col_sum(probabilities)
                column_sum = pl.add(column_sum, eps)
                probabilities = pl.col_expand_div(probabilities, column_sum)
            comb_values = pl.set_validshape(probabilities, hc, hc)
            pl.store(comb_values, [token * hc, 0], comb_rows)

        collapse(x, incoming, collapsed, active)
        return next_pre, post, comb, collapsed

    return pre


def make_hc_post_inline(capacity, config):
    """Build BF16 residual expansion with FP32 combination arithmetic."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    merge_tile = MERGE_D_TILE if hidden % MERGE_D_TILE == 0 else D_TILE
    stream_tile = POST_STREAM_TILE if hc % POST_STREAM_TILE == 0 else 1

    @pl.jit.inline
    def post(
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        residual: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        post_mix: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        residual_rows = pl.reshape(residual, [capacity * hc, hidden])
        output_rows = pl.reshape(output, [capacity * hc, hidden])
        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for block in pl.spmd(active * (hc // stream_tile) * (hidden // merge_tile), name_hint="a2_hc_post"):
            token = block // ((hc // stream_tile) * (hidden // merge_tile))
            stream_start = block // (hidden // merge_tile) % (hc // stream_tile) * stream_tile
            offset = block % (hidden // merge_tile) * merge_tile
            value = pl.cast(sublayer[token : token + 1, offset : offset + merge_tile], pl.FP32)
            source_rows = residual_rows[token * hc : token * hc + hc, offset : offset + merge_tile]
            source_values = pl.cast(source_rows, pl.FP32)
            for stream_lane in pl.unroll(stream_tile):
                stream = stream_start + stream_lane
                coefficient = pl.read(post_mix, [token, stream])
                merged = pl.mul(value, coefficient)
                for source_stream in pl.unroll(hc):
                    source = source_values[source_stream : source_stream + 1, :]
                    coefficient = pl.read(comb_flat, [token, source_stream * hc + stream])
                    term = pl.mul(source, coefficient)
                    merged = pl.add(merged, term)
                destination = token * hc + stream
                merged_bf16 = pl.cast(merged, pl.BF16, mode="rint")
                output_rows[destination : destination + 1, offset : offset + merge_tile] = merged_bf16
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


def make_decode_hc_pre_inline(capacity, config):
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
        collapsed_fp32: pl.Tensor[[capacity, hidden], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(x, [capacity, flat_width])
        inverse = pl.create_tensor([capacity, TOKEN_TILE], dtype=pl.FP32)
        for block in pl.spmd((active + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="a2_hc_rms"):
            token = block * TOKEN_TILE
            rows = pl.min(TOKEN_TILE, active - token)
            square_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, flat_width, D_TILE):
                values = pl.slice(flat, [TOKEN_TILE, D_TILE], [token, inner], valid_shape=[rows, D_TILE])
                values = pl.fillpad(values, pad_value=pl.PadValue.zero)
                values_fp32 = pl.cast(values, pl.FP32)
                square = pl.mul(values_fp32, values_fp32)
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square), [1, TOKEN_TILE]))
            mean = pl.mul(square_sum, 1.0 / flat_width)
            argument = pl.add(mean, norm_eps)
            inverse_column = pl.reshape(pl.rsqrt(argument, high_precision=True), [TOKEN_TILE, 1])
            ones = pl.full([TOKEN_TILE, TOKEN_TILE], dtype=pl.FP32, value=1.0)
            inverse_tile = pl.row_expand_mul(ones, inverse_column)
            inverse[token : token + TOKEN_TILE, :] = pl.set_validshape(inverse_tile, rows, TOKEN_TILE)

        projected = pl.create_tensor([capacity, channels], dtype=pl.FP32)
        for block in pl.spmd(active * (channels // CHANNEL_TILE), name_hint="a2_hc_projection"):
            token = block // (channels // CHANNEL_TILE)
            channel = block % (channels // CHANNEL_TILE) * CHANNEL_TILE
            dot_sum = pl.full([1, CHANNEL_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, flat_width, D_TILE):
                source = pl.cast(flat[token : token + 1, inner : inner + D_TILE], pl.FP32)
                weight = function[channel : channel + CHANNEL_TILE, inner : inner + D_TILE]
                product = pl.col_expand_mul(weight, source)
                dot_sum = pl.add(dot_sum, pl.reshape(pl.row_sum(product), [1, CHANNEL_TILE]))
            inverse_scalar = pl.read(inverse, [token, 0])
            normalized = pl.mul(dot_sum, inverse_scalar)
            projected[token : token + 1, channel : channel + CHANNEL_TILE] = normalized

        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for token in pl.spmd(active, name_hint="a2_hc_sinkhorn"):
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])
            pre_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            post_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            for column in pl.unroll(MIX_TILE):
                logical = column % hc
                pre_value = pl.read(projected, [token, logical]) * scale0 + pl.read(base, [logical])
                post_value = pl.read(projected, [token, hc + logical]) * scale1 + pl.read(
                    base, [hc + logical]
                )
                pl.tile.write(pre_logits, [0, column], pre_value)
                pl.tile.write(post_logits, [0, column], post_value)
            pre_neg = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_neg)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre_sigmoid = pl.recip(pre_denominator)
            pre_result = pl.add(pre_sigmoid, eps)
            post_neg = pl.neg(post_logits)
            post_exp = pl.exp(post_neg)
            post_denominator = pl.add(post_exp, 1.0)
            post_sigmoid = pl.recip(post_denominator)
            post_result = pl.mul(post_sigmoid, 2.0)
            pre_valid = pl.set_validshape(pre_result, 1, hc)
            post_valid = pl.set_validshape(post_result, 1, hc)
            pl.store(pre_valid, [token, 0], next_pre)
            pl.store(post_valid, [token, 0], post)
            logits = pl.tile.full([MIX_TILE, MIX_TILE], dtype=pl.FP32, value=0.0)
            for row in pl.unroll(MIX_TILE):
                for column in pl.unroll(MIX_TILE):
                    logical = 2 * hc + (row % hc) * hc + column % hc
                    value = pl.read(projected, [token, logical]) * scale2 + pl.read(base, [logical])
                    pl.tile.write(logits, [row, column], value)
            logits_valid = pl.set_validshape(logits, hc, hc)
            logits_padded = pl.fillpad(logits_valid, pad_value=pl.PadValue.min)
            row_tmp = pl.create_tile([MIX_TILE, MIX_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_max = pl.row_max(logits_padded, row_tmp)
            shifted = pl.row_expand_sub(logits_padded, row_max)
            exponential = pl.exp(shifted)
            row_sum = pl.row_sum(exponential, row_tmp)
            probabilities_softmax = pl.add(pl.row_expand_div(exponential, row_sum), eps)
            probabilities_valid = pl.set_validshape(probabilities_softmax, hc, hc)
            probabilities_padded = pl.fillpad(probabilities_valid, pad_value=pl.PadValue.zero)
            column_sum = pl.col_sum(probabilities_padded)
            column_sum = pl.add(column_sum, eps)
            probabilities = pl.col_expand_div(probabilities_padded, column_sum)
            for _ in pl.range(iterations - 1):
                iter_valid = pl.set_validshape(probabilities, hc, hc)
                iter_padded = pl.fillpad(iter_valid, pad_value=pl.PadValue.zero)
                row_sum = pl.row_sum(iter_padded, row_tmp)
                row_sum = pl.add(row_sum, eps)
                row_normalized = pl.row_expand_div(iter_padded, row_sum)
                row_normalized_valid = pl.set_validshape(row_normalized, hc, hc)
                row_normalized_padded = pl.fillpad(row_normalized_valid, pad_value=pl.PadValue.zero)
                column_sum = pl.col_sum(row_normalized_padded)
                column_sum = pl.add(column_sum, eps)
                probabilities = pl.col_expand_div(row_normalized_padded, column_sum)
            for row in pl.unroll(hc):
                comb_value = pl.slice(probabilities, [1, MIX_TILE], [row, 0], valid_shape=[1, hc])
                pl.store(comb_value, [token, row * hc], comb_flat)

        for block in pl.spmd(active * (hidden // D_TILE), name_hint="a2_hc_collapse"):
            token = block // (hidden // D_TILE)
            offset = block % (hidden // D_TILE) * D_TILE
            collapsed_tile = pl.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(hc):
                source_offset = stream * hidden + offset
                source_bf16 = flat[token : token + 1, source_offset : source_offset + D_TILE]
                source_fp32 = pl.cast(source_bf16, pl.FP32)
                coefficient = pl.read(incoming, [token, stream])
                term = pl.mul(source_fp32, coefficient)
                collapsed_tile = pl.add(collapsed_tile, term)
            collapsed_fp32[token : token + 1, offset : offset + D_TILE] = collapsed_tile
            collapsed[token : token + 1, offset : offset + D_TILE] = pl.cast(
                collapsed_tile, pl.BF16, mode="rint"
            )
        return next_pre, post, comb, collapsed

    return pre


def make_decode_hc_post_inline(capacity, config):
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
        flat = pl.reshape(residual, [capacity, hc * hidden])
        output_flat = pl.reshape(output, [capacity, hc * hidden])
        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for block in pl.spmd(active * hc * (hidden // D_TILE), name_hint="a2_hc_post"):
            token = block // (hc * (hidden // D_TILE))
            stream = block // (hidden // D_TILE) % hc
            offset = block % (hidden // D_TILE) * D_TILE
            value = pl.cast(sublayer[token : token + 1, offset : offset + D_TILE], pl.FP32)
            coefficient = pl.read(post_mix, [token, stream])
            merged = pl.mul(value, coefficient)
            for source_stream in pl.unroll(hc):
                source_offset = source_stream * hidden + offset
                source = flat[token : token + 1, source_offset : source_offset + D_TILE]
                source = pl.cast(source, pl.FP32)
                coefficient = pl.read(comb_flat, [token, source_stream * hc + stream])
                term = pl.mul(source, coefficient)
                merged = pl.add(merged, term)
            destination = stream * hidden + offset
            merged_bf16 = pl.cast(merged, pl.BF16, mode="rint")
            output_flat[token : token + 1, destination : destination + D_TILE] = merged_bf16
        return output

    return post


def make_decode_hc(capacity, config):
    """Build the standalone HC operator diagnostic entry."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    pre = make_decode_hc_pre_inline(capacity, config)
    post_fn = make_decode_hc_post_inline(capacity, config)

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
        collapsed_fp32 = pl.create_tensor([capacity, hidden], dtype=pl.FP32)
        pre(
            x,
            function,
            scale,
            base,
            incoming,
            next_pre,
            post,
            comb,
            collapsed,
            collapsed_fp32,
            active,
        )
        post_fn(sublayer, x, post, comb, merged, active)
        return next_pre, post, comb, collapsed, merged

    return hc_entry


def golden_hc(tensors, *, config):
    """Evaluate rounded projection weights, tiled FP32 sums and ordered HC mixing."""
    import torch

    active = int(tensors["active"])
    if not 0 <= active <= tensors["x"].shape[0]:
        raise ValueError("HC active rows must fit the actual capacity")
    if not active:
        return
    x = tensors["x"][:active].float()
    hc, hidden = config["hc_mult"], config["hidden_size"]
    flat = x.reshape(active, hc * hidden)
    bits = tensors["function"].contiguous().view(torch.int32).to(torch.int64)
    rounded_bits = (((bits + 2048) >> 12) << 12).to(torch.int32)
    weight = rounded_bits.contiguous().view(torch.float32)
    squares = torch.zeros(active, dtype=torch.float32)
    for column in range(0, flat.shape[1], D_TILE):
        chunk = flat[:, column : column + D_TILE]
        squares = squares + (chunk * chunk).sum(dim=1)
    inverse = torch.rsqrt(squares * (1.0 / flat.shape[1]) + config["rms_norm_eps"])
    channels = (hc + 2) * hc
    projected = torch.zeros((active, channels), dtype=torch.float32)
    for column in range(0, flat.shape[1], PROJ_K_TILE):
        product = flat[:, column : column + PROJ_K_TILE] @ weight[:, column : column + PROJ_K_TILE].T
        projected = projected + product
    projected = projected * inverse[:, None]
    scale, base = tensors["scale"], tensors["base"]
    pre_logits = projected[:, :hc] * scale[0] + base[:hc]
    post_logits = projected[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    pre = torch.reciprocal(torch.exp(-pre_logits) + 1.0) + config["hc_eps"]
    post = torch.reciprocal(torch.exp(-post_logits) + 1.0) * 2.0
    logits = projected[:, 2 * hc :].reshape(active, hc, hc) * scale[2] + base[2 * hc :].reshape(hc, hc)
    repeats = MIX_TILE // hc
    expanded = logits.repeat(1, repeats, repeats)
    shifted = expanded - expanded.amax(dim=-1, keepdim=True)
    exponential = torch.exp(shifted)
    probability = exponential / (exponential.sum(dim=-1, keepdim=True) * (hc / MIX_TILE))
    probability = probability + config["hc_eps"]
    denominator = probability.sum(dim=-2, keepdim=True) * (hc / MIX_TILE) + config["hc_eps"]
    probability = probability / denominator
    for _ in range(config["hc_sinkhorn_iters"] - 1):
        denominator = probability.sum(dim=-1, keepdim=True) * (hc / MIX_TILE) + config["hc_eps"]
        probability = probability / denominator
        denominator = probability.sum(dim=-2, keepdim=True) * (hc / MIX_TILE) + config["hc_eps"]
        probability = probability / denominator
    comb = probability[:, :hc, :hc]
    collapsed = torch.zeros((active, hidden), dtype=torch.float32)
    for stream in range(hc):
        collapsed = collapsed + x[:, stream] * tensors["incoming"][:active, stream, None]
    merged = torch.empty_like(x)
    for stream in range(hc):
        value = tensors["sublayer"][:active].float() * post[:, stream, None]
        for source in range(hc):
            value = value + x[:, source] * comb[:, source, stream, None]
        merged[:, stream] = value
    for name, value in (
        ("next_pre", pre),
        ("post", post),
        ("comb", comb),
        ("collapsed", collapsed),
        ("merged", merged),
    ):
        tensors[name][:active] = value.to(tensors[name].dtype)


def hc_prefix_compare(active):
    """Use the shared floating comparator and exact untouched InOut tails."""
    from golden.validation import ratio_allclose

    numeric = ratio_allclose(rtol=0.05, atol=0.05, max_error_ratio=0.0)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        for label, region, comparator in (
            ("active", slice(None, active), numeric),
            ("untouched", slice(active, None), exact),
        ):
            if actual[region].numel():
                passed, detail = comparator(actual[region], expected[region], **context)
                if not passed:
                    return False, label + ": " + detail
        return True, ""

    return compare


def hc_golden_cases(*, build=True):
    """Cover empty, singleton, seven-row tail and full-capacity four-stream HC."""
    from functools import partial
    import torch
    from golden import ScalarSpec, TensorSpec

    capacity, hidden, streams = 16, 256, 4
    channels = (streams + 2) * streams
    config = dict(hidden_size=hidden, hc_mult=streams, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
    generator = torch.Generator().manual_seed(417)
    x = torch.randn((capacity, streams, hidden), generator=generator).to(torch.bfloat16)
    function = torch.randn((channels, streams * hidden), generator=generator) * 0.03 + 2.0**-20
    incoming = torch.rand((capacity, streams), generator=generator)
    sublayer = torch.randn((capacity, hidden), generator=generator).to(torch.bfloat16)
    for active in (0, 1, 7, capacity):
        specs = [
            TensorSpec("x", [capacity, streams, hidden], torch.bfloat16, init_value=x),
            TensorSpec("function", [channels, streams * hidden], torch.float32, init_value=function),
            TensorSpec("scale", [3], torch.float32, init_value=torch.tensor([0.3, -0.2, 0.1])),
            TensorSpec("base", [channels], torch.float32, init_value=torch.linspace(-0.2, 0.2, channels)),
            TensorSpec("incoming", [capacity, streams], torch.float32, init_value=incoming),
            TensorSpec("sublayer", [capacity, hidden], torch.bfloat16, init_value=sublayer),
            TensorSpec("next_pre", [capacity, streams], torch.float32, init_value=-23.0),
            TensorSpec("post", [capacity, streams], torch.float32, init_value=-29.0),
            TensorSpec("comb", [capacity, streams, streams], torch.float32, init_value=-31.0),
            TensorSpec("collapsed", [capacity, hidden], torch.bfloat16, init_value=-37.0),
            TensorSpec("merged", [capacity, streams, hidden], torch.bfloat16, init_value=-41.0),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        outputs = ("next_pre", "post", "comb", "collapsed", "merged")
        case = dict(
            name=f"hc-four-stream-active{active}",
            specs=specs,
            golden_fn=partial(golden_hc, config=config),
            compare_fn={name: hc_prefix_compare(active) for name in outputs},
        )
        if build:
            case["fn"] = make_hc(capacity, config)
        yield case


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Run bounded HC goldens; CPU mode checks only the reference implementation."""
    import torch
    from golden import run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    outputs = ("next_pre", "post", "comb", "collapsed", "merged")
    for case in hc_golden_cases(build=not cpu_only):
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
                if key not in outputs and not torch.equal(values[key], before[key]):
                    raise RuntimeError("HC golden mutated input " + key)
            for key in outputs:
                if not torch.isfinite(values[key]).all() or not torch.equal(
                    values[key][active:], before[key][active:]
                ):
                    raise RuntimeError("HC golden changed an inactive tail or produced a nonfinite result")
            if active:
                sums = values["comb"][:active].sum(dim=-2)
                torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0.0, atol=1e-5)
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
