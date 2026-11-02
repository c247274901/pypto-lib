# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 blocked INT8 projections, FP32 scaling, RMSNorm and adjacent-pair RoPE."""

import pypto.language as pl


# tiling
M_TILE = 16
VEC_M_TILE = 8
K_TILE = 256
N_TILE = 64


def make_cast_tile(rows, width, input_dtype, output_dtype):
    """Choose identity or a rounding cast before the DSL parses either body."""
    if input_dtype == output_dtype:
        @pl.jit.inline
        def identity(value: pl.Tensor[[rows, width], input_dtype]):
            return value
        return identity

    @pl.jit.inline
    def cast_tile(value: pl.Tensor[[rows, width], input_dtype]):
        return pl.cast(value, output_dtype, mode="rint")

    return cast_tile


def make_quantize(capacity, width, input_dtype=pl.BF16, row_tile=VEC_M_TILE, whole_row=False, early_resolve=False):
    """Specialize symmetric per-token INT8 quantization."""
    if whole_row:
        @pl.jit.inline
        def quantize_whole_row(
            x: pl.Tensor[[capacity, width], input_dtype],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            for token in pl.spmd(capacity, name_hint="a2_token_quantize_whole_row", allow_early_resolve=early_resolve):
                rows = pl.max(0, pl.min(1, active - token))
                source = pl.slice(x, [1, width], [token, 0], valid_shape=[rows, width])
                padded = pl.fillpad(source, pad_value=pl.PadValue.zero)
                complete = pl.set_validshape(padded, 1, width)
                value = pl.cast(complete, pl.FP32)
                segments = pl.reshape(pl.abs(value), [8, width // 8])
                partial = pl.row_max(segments)
                ones = pl.full([8, 8], dtype=pl.FP32, value=1.0)
                repeated = pl.row_expand_mul(ones, partial)
                maximum_row = pl.maximum(pl.col_max(repeated), 1e-30)
                scale_row = pl.mul(maximum_row, 1.0 / 127.0)
                scales[token:token + 1, :] = scale_row
                numerator = pl.full([1, 8], dtype=pl.FP32, value=127.0)
                inverse_row = pl.div(numerator, maximum_row)
                inverse = pl.read(inverse_row, [0, 0])
                normalized = pl.mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[token:token + 1, :] = pl.cast(exact_half, pl.INT8, mode="trunc")
            return payload, scales

        return quantize_whole_row

    widen = make_cast_tile(row_tile, K_TILE, input_dtype, pl.FP32)

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], input_dtype],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + row_tile - 1) // row_tile, name_hint="a2_token_quantize"):
            t0 = block * row_tile
            rows = pl.min(row_tile, active - t0)
            maximum_row = pl.full([1, row_tile], dtype=pl.FP32, value=1e-30)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [row_tile, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), row_tile, K_TILE)
                value = widen(source)
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, row_tile])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [row_tile, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, row_tile], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [row_tile, 1])
            scale_rows = pl.full([row_tile, 8], dtype=pl.FP32, value=1.0)
            scales[t0:t0 + row_tile, :] = pl.row_expand_mul(scale_rows, scale)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [row_tile, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), row_tile, K_TILE)
                value = widen(source)
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[t0:t0 + row_tile, k0:k0 + K_TILE] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_pack_weight(width, output_width, n_tile=N_TILE, k_splits=1):
    """Reorder canonical [K,N] INT8 bytes to [N/n_tile,K,n_tile]."""
    packed_rows = width * (output_width // n_tile)

    @pl.jit.inline
    def pack_weight(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, n_tile], pl.INT8],
    ):
        for block in pl.spmd(output_width // n_tile * k_splits, name_hint="a2_pack_weight"):
            nb = block // k_splits
            split = block % k_splits
            n0 = nb * n_tile
            for k0 in pl.pipeline(split * (width // k_splits), (split + 1) * (width // k_splits), K_TILE, stage=2):
                value = weight[k0:k0 + K_TILE, n0:n0 + n_tile]
                packed_k0 = nb * width + k0
                packed[packed_k0:packed_k0 + K_TILE, :] = value
        return packed

    return pack_weight


def make_dot(
    capacity, width, output_width, padded_payload=False, column_group=1, column_reuse=1,
    row_tile=M_TILE, full_k_reuse=False, early_resolve=False, column_stages=1,
):
    """Accumulate complete blocked INT8 products in INT32."""
    packed_rows = width * (output_width // N_TILE)

    if full_k_reuse:
        @pl.jit.inline
        def dot_full_k(
            payload: pl.Tensor[[capacity, width], pl.INT8],
            packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            active: pl.Scalar[pl.INT32],
        ):
            row_blocks = (active + row_tile - 1) // row_tile
            column_blocks = (output_width // N_TILE + column_group - 1) // column_group
            for block in pl.spmd(row_blocks * column_blocks, name_hint="a2_int8_dot_full_k", allow_early_resolve=early_resolve):
                t0 = (block // column_blocks) * row_tile
                first_nb = (block % column_blocks) * column_group
                last_nb = pl.min(first_nb + column_group, output_width // N_TILE)
                rows = pl.min(row_tile, active - t0)
                if padded_payload:
                    activation = payload[t0:t0 + row_tile, :]
                else:
                    source = pl.slice(payload, [row_tile, width], [t0, 0], valid_shape=[rows, width])
                    activation = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), row_tile, width)
                for nb in pl.pipeline(first_nb, last_nb, stage=column_stages):
                    n0 = nb * N_TILE
                    packed_k0 = nb * width
                    weight = packed[packed_k0:packed_k0 + width, :]
                    product = pl.matmul(activation, weight, out_dtype=pl.INT32)
                    accumulator[t0:t0 + row_tile, n0:n0 + N_TILE] = product
            return accumulator

        return dot_full_k

    @pl.jit.inline
    def dot(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + row_tile - 1) // row_tile
        for block in pl.spmd(row_blocks * (output_width // (N_TILE * column_group)), name_hint="a2_int8_dot"):
            t0 = (block // (output_width // (N_TILE * column_group))) * row_tile
            first_nb = (block % (output_width // (N_TILE * column_group))) * column_group
            rows = pl.min(row_tile, active - t0)
            for nb in pl.range(first_nb, first_nb + column_group, column_reuse):
                n0 = nb * N_TILE
                acc = pl.create_tensor([row_tile, N_TILE], dtype=pl.INT32)
                acc_next = pl.create_tensor([row_tile, N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                    if padded_payload:
                        a = payload[t0:t0 + row_tile, k0:k0 + K_TILE]
                    else:
                        source = pl.slice(payload, [row_tile, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                        a = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), row_tile, K_TILE)
                    packed_k0 = nb * width + k0
                    b = packed[packed_k0:packed_k0 + K_TILE, :]
                    acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
                    if column_reuse == 2:
                        next_k0 = packed_k0 + width
                        b_next = packed[next_k0:next_k0 + K_TILE, :]
                        acc_next = pl.matmul_acc(acc_next, a, b_next, init_cond=(k0 == 0))
                accumulator[t0:t0 + row_tile, n0:n0 + N_TILE] = acc
                if column_reuse == 2:
                    accumulator[t0:t0 + row_tile, n0 + N_TILE:n0 + 2 * N_TILE] = acc_next
        return accumulator

    return dot


def make_dequantize(capacity, output_width, combine_scales=False, output_tile=N_TILE):
    """Apply the selected native FP32 scale boundary to INT32 products.

    The native per-token optimized kernel combines channel/token scales before
    multiplying the accumulator. Other native kernels scale channel-first.
    Callers select their source boundary explicitly; QKV keeps its existing ABI.
    """
    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (output_width // output_tile), name_hint="a2_dequantize"):
            t0 = (block // (output_width // output_tile)) * VEC_M_TILE
            n0 = (block % (output_width // output_tile)) * output_tile
            rows = pl.min(VEC_M_TILE, active - t0)
            integer = accumulator[t0:t0 + VEC_M_TILE, n0:n0 + output_tile]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0:n0 + output_tile], [1, output_tile])
            scale = pl.row_min(token_scale[t0:t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, output_tile], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.col_expand_mul(value, channels)
                scaled = pl.row_expand_mul(scaled_channels, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0:t0 + VEC_M_TILE, n0:n0 + output_tile] = pl.set_validshape(narrowed, rows, output_tile)
        return output

    return dequantize


def make_norm(capacity, width, eps):
    """Specialize weighted FP32 RMSNorm with a BF16 output."""

    @pl.jit.inline
    def norm(
        x: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_rms_norm"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            source = pl.slice(x, [VEC_M_TILE, width], [t0, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
            value = pl.cast(source, pl.FP32)
            square = pl.mul(value, value)
            mean = pl.mul(pl.row_sum(square), 1.0 / width)
            inverse = pl.rsqrt(pl.add(mean, eps), high_precision=True)
            normalized = pl.row_expand_mul(value, inverse)
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            weighted = pl.col_expand_mul(normalized, gamma)
            narrowed = pl.cast(weighted, pl.BF16, mode="rint")
            output[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(narrowed, rows, width)
        return output

    return norm


def make_dequantize_norm_quantize(capacity, width, eps, early_resolve=False):
    """Fuse channel-first dequantization, RMSNorm and padded INT8 quantization."""

    @pl.jit.inline
    def dequantize_norm_quantize(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channel_scale: pl.Tensor[[width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(capacity // VEC_M_TILE, name_hint="a2_dequantize_norm_quantize", allow_early_resolve=early_resolve):
            t0 = block * VEC_M_TILE
            rows = pl.max(0, pl.min(VEC_M_TILE, active - t0))
            integer = pl.slice(accumulator, [VEC_M_TILE, width], [t0, 0], valid_shape=[rows, width])
            padded = pl.fillpad(integer, pad_value=pl.PadValue.zero)
            integer_rows = pl.set_validshape(padded, VEC_M_TILE, width)
            value = pl.cast(integer_rows, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[:], [1, width])
            scale = pl.row_min(token_scale[t0:t0 + VEC_M_TILE, :])
            scaled_channels = pl.col_expand_mul(value, channels)
            scaled = pl.row_expand_mul(scaled_channels, scale)
            dequantized = pl.cast(scaled, pl.BF16, mode="rint")
            norm_value = pl.cast(dequantized, pl.FP32)
            square = pl.mul(norm_value, norm_value)
            mean = pl.mul(pl.row_sum(square), 1.0 / width)
            inverse_norm = pl.rsqrt(pl.add(mean, eps), high_precision=True)
            normalized = pl.row_expand_mul(norm_value, inverse_norm)
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            weighted = pl.col_expand_mul(normalized, gamma)
            narrowed = pl.cast(weighted, pl.BF16, mode="rint")
            quant_value = pl.cast(narrowed, pl.FP32)
            row_maximum = pl.row_max(pl.abs(quant_value))
            maximum_row = pl.reshape(pl.maximum(row_maximum, 1e-30), [1, VEC_M_TILE])
            maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
            quant_scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
            scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
            scales[t0:t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, quant_scale)
            quant_normalized = pl.row_expand_mul(quant_value, inverse)
            bounded = pl.minimum(pl.maximum(quant_normalized, -127.0), 127.0)
            rounded = pl.cast(bounded, pl.INT32, mode="rint")
            exact_half = pl.cast(rounded, pl.FP16, mode="rint")
            payload[t0:t0 + VEC_M_TILE, :] = pl.cast(exact_half, pl.INT8, mode="trunc")
            output[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(narrowed, rows, width)
        return output, payload, scales

    return dequantize_norm_quantize


def make_dequantize_norm_rope(capacity, width, rope_dim, eps):
    """Fuse KV dequantization, RMSNorm and adjacent-pair RoPE."""
    nope_dim = width - rope_dim
    pair_width = rope_dim // 2
    pair_shift = pair_width.bit_length() - 1

    @pl.jit.inline
    def dequantize_norm_rope(
        accumulator: pl.Tensor[[capacity, width], pl.INT32],
        channel_scale: pl.Tensor[[width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        weight: pl.Tensor[[width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_kv_epilogue"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            source = pl.slice(accumulator, [VEC_M_TILE, width], [t0, 0], valid_shape=[rows, width])
            padded = pl.fillpad(source, pad_value=pl.PadValue.zero)
            integer = pl.set_validshape(padded, VEC_M_TILE, width)
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[:], [1, width])
            scale = pl.row_min(token_scale[t0:t0 + VEC_M_TILE, :])
            scaled_channels = pl.col_expand_mul(value, channels)
            scaled = pl.row_expand_mul(scaled_channels, scale)
            dequantized = pl.cast(scaled, pl.BF16, mode="rint")
            norm_value = pl.cast(dequantized, pl.FP32)
            square = pl.mul(norm_value, norm_value)
            mean = pl.mul(pl.row_sum(square), 1.0 / width)
            inverse = pl.rsqrt(pl.add(mean, eps), high_precision=True)
            normalized = pl.row_expand_mul(norm_value, inverse)
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            weighted = pl.col_expand_mul(normalized, gamma)
            narrowed = pl.cast(weighted, pl.BF16, mode="rint")
            prefix = narrowed[:, :nope_dim]
            tail = pl.cast(narrowed[:, nope_dim:width], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = pl.slice(cosine, [VEC_M_TILE, pair_width], [t0, 0], valid_shape=[rows, pair_width])
            c = pl.set_validshape(pl.fillpad(c, pad_value=pl.PadValue.zero), VEC_M_TILE, pair_width)
            s = pl.slice(sine, [VEC_M_TILE, pair_width], [t0, 0], valid_shape=[rows, pair_width])
            s = pl.set_validshape(pl.fillpad(s, pad_value=pl.PadValue.zero), VEC_M_TILE, pair_width)
            even_c = pl.mul(even, c)
            odd_s = pl.mul(odd, s)
            odd_c = pl.mul(odd, c)
            even_s = pl.mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            joined = pl.concat(real, imag)
            ramp = pl.arange(0, [1, VEC_M_TILE * rope_dim], dtype=pl.INT16)
            row_base = pl.and_(ramp, -rope_dim)
            pair_index = pl.shr(pl.and_(ramp, rope_dim - 1), 1)
            half_index = pl.shl(pl.and_(ramp, 1), pair_shift)
            index_bits = pl.or_(row_base, pl.or_(pair_index, half_index))
            index_half = pl.cast(index_bits, pl.FP16, mode="rint")
            index_float = pl.cast(index_half, pl.FP32, mode="rint")
            index_integer = pl.cast(index_float, pl.INT32, mode="rint")
            indices = pl.reshape(index_integer, [VEC_M_TILE, rope_dim])
            rotated = pl.gather(joined, indices)
            rotated_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
            complete = pl.concat(prefix, rotated_bf16)
            output[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(complete, rows, width)
        return output

    return dequantize_norm_rope


def make_rope(capacity, heads, head_dim, rope_dim, row_tile=1, vector_interleave=False):
    """Rotate the final channels of each head in adjacent pairs."""
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2
    pair_shift = pair_width.bit_length() - 1
    assert not vector_interleave or (rope_dim == 1 << (pair_shift + 1) and row_tile * rope_dim <= 2048)

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, output_width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + row_tile - 1) // row_tile) * heads, name_hint="a2_adjacent_rope"):
            t = (block // heads) * row_tile
            base = (block % heads) * head_dim
            rows = pl.min(row_tile, active - t)
            prefix = pl.slice(x, [row_tile, nope_dim], [t, base], valid_shape=[rows, nope_dim])
            output[t:t + row_tile, base:base + nope_dim] = prefix
            source = pl.slice(x, [row_tile, rope_dim], [t, base + nope_dim], valid_shape=[rows, rope_dim])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), row_tile, rope_dim)
            tail = pl.cast(source, pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = pl.slice(cosine, [row_tile, pair_width], [t, 0], valid_shape=[rows, pair_width])
            c = pl.set_validshape(pl.fillpad(c, pad_value=pl.PadValue.zero), row_tile, pair_width)
            s = pl.slice(sine, [row_tile, pair_width], [t, 0], valid_shape=[rows, pair_width])
            s = pl.set_validshape(pl.fillpad(s, pad_value=pl.PadValue.zero), row_tile, pair_width)
            even_c = pl.mul(even, c)
            odd_s = pl.mul(odd, s)
            odd_c = pl.mul(odd, c)
            even_s = pl.mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            if vector_interleave:
                joined = pl.concat(real, imag)
                ramp = pl.arange(0, [1, row_tile * rope_dim], dtype=pl.INT16)
                row_base = pl.and_(ramp, -rope_dim)
                pair_index = pl.shr(pl.and_(ramp, rope_dim - 1), 1)
                half_index = pl.shl(pl.and_(ramp, 1), pair_shift)
                index_half = pl.cast(pl.or_(row_base, pl.or_(pair_index, half_index)), pl.FP16, mode="rint")
                index_float = pl.cast(index_half, pl.FP32, mode="rint")
                indices = pl.reshape(pl.cast(index_float, pl.INT32, mode="rint"), [row_tile, rope_dim])
                rotated = pl.gather(joined, indices)
            else:
                zero_rotated = pl.full([row_tile, rope_dim], dtype=pl.FP32, value=0.0)
                even_rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=zero_rotated)
                rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=even_rotated)
            narrowed = pl.cast(rotated, pl.BF16, mode="rint")
            output[t:t + row_tile, base + nope_dim:base + head_dim] = pl.set_validshape(narrowed, rows, rope_dim)
        return output

    return rope


def make_dequantize_rope(capacity, heads, head_dim, rope_dim, row_tile=VEC_M_TILE, head_tile=1, scalar_channels=False):
    """Fuse query dequantization and RoPE while retaining the BF16 boundary."""
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2
    pair_shift = pair_width.bit_length() - 1
    assert rope_dim == 1 << (pair_shift + 1) and row_tile * rope_dim <= 2048

    if head_tile > 1:
        head_width = head_tile * head_dim

        @pl.jit.inline
        def dequantize_rope_heads(
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            channel_scale: pl.Tensor[[output_width], pl.FP32],
            token_scale: pl.Tensor[[capacity, 8], pl.FP32],
            cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
            sine: pl.Tensor[[capacity, pair_width], pl.FP32],
            output: pl.Tensor[[capacity, output_width], pl.BF16],
            active: pl.Scalar[pl.INT32],
        ):
            for block in pl.spmd(active * (heads // (2 * head_tile)), name_hint="a2_dequantize_rope_heads"):
                t0 = block // (heads // (2 * head_tile))
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    base = ((block % (heads // (2 * head_tile))) * 2 + aiv_id) * head_width
                    integer = accumulator[t0:t0 + 1, base:base + head_width]
                    value = pl.cast(integer, pl.FP32, mode="rint")
                    channels = pl.reshape(channel_scale[base:base + head_width], [1, head_width])
                    scale = pl.read(token_scale, [t0, 0])
                    if scalar_channels:
                        channel_rows = pl.mul(channels, 1.0)
                    else:
                        ones = pl.full([1, head_width], dtype=pl.FP32, value=1.0)
                        channel_rows = pl.col_expand_mul(ones, channels)
                    combined = pl.mul(channel_rows, scale)
                    scaled = pl.mul(value, combined)
                    narrowed = pl.cast(scaled, pl.BF16, mode="rint")
                    head_values = pl.reshape(narrowed, [head_tile, head_dim])
                    prefix = head_values[:, :nope_dim]
                    source = head_values[:, nope_dim:head_dim]
                    tail = pl.cast(source, pl.FP32)
                    even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                    odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                    c = cosine[t0:t0 + 1, :]
                    s = sine[t0:t0 + 1, :]
                    even_c = pl.col_expand_mul(even, c)
                    odd_s = pl.col_expand_mul(odd, s)
                    odd_c = pl.col_expand_mul(odd, c)
                    even_s = pl.col_expand_mul(even, s)
                    real = pl.sub(even_c, odd_s)
                    imag = pl.add(odd_c, even_s)
                    joined = pl.concat(real, imag)
                    ramp = pl.arange(0, [1, head_tile * rope_dim], dtype=pl.INT16)
                    row_base = pl.and_(ramp, -rope_dim)
                    pair_index = pl.shr(pl.and_(ramp, rope_dim - 1), 1)
                    half_index = pl.shl(pl.and_(ramp, 1), pair_shift)
                    index_bits = pl.or_(row_base, pl.or_(pair_index, half_index))
                    index_half = pl.cast(index_bits, pl.FP16, mode="rint")
                    index_float = pl.cast(index_half, pl.FP32, mode="rint")
                    index_integer = pl.cast(index_float, pl.INT32, mode="rint")
                    indices = pl.reshape(index_integer, [head_tile, rope_dim])
                    rotated = pl.gather(joined, indices)
                    rotated_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
                    complete = pl.concat(prefix, rotated_bf16)
                    flattened = pl.reshape(complete, [1, head_width])
                    output[t0:t0 + 1, base:base + head_width] = flattened
            return output

        return dequantize_rope_heads

    @pl.jit.inline
    def dequantize_rope(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + row_tile - 1) // row_tile) * heads, name_hint="a2_dequantize_rope"):
            t0 = (block // heads) * row_tile
            base = (block % heads) * head_dim
            rows = pl.min(row_tile, active - t0)
            integer = accumulator[t0:t0 + row_tile, base:base + head_dim]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[base:base + head_dim], [1, head_dim])
            scale = pl.row_min(token_scale[t0:t0 + row_tile, :])
            ones = pl.full([row_tile, head_dim], dtype=pl.FP32, value=1.0)
            channel_rows = pl.col_expand_mul(ones, channels)
            combined = pl.row_expand_mul(channel_rows, scale)
            scaled = pl.mul(value, combined)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            prefix = pl.slice(narrowed, [row_tile, nope_dim], [0, 0], valid_shape=[rows, nope_dim])
            output[t0:t0 + row_tile, base:base + nope_dim] = prefix
            source = pl.slice(narrowed, [row_tile, rope_dim], [0, nope_dim])
            tail = pl.cast(source, pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = pl.slice(cosine, [row_tile, pair_width], [t0, 0], valid_shape=[rows, pair_width])
            c = pl.set_validshape(pl.fillpad(c, pad_value=pl.PadValue.zero), row_tile, pair_width)
            s = pl.slice(sine, [row_tile, pair_width], [t0, 0], valid_shape=[rows, pair_width])
            s = pl.set_validshape(pl.fillpad(s, pad_value=pl.PadValue.zero), row_tile, pair_width)
            even_c = pl.mul(even, c)
            odd_s = pl.mul(odd, s)
            odd_c = pl.mul(odd, c)
            even_s = pl.mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            joined = pl.concat(real, imag)
            ramp = pl.arange(0, [1, row_tile * rope_dim], dtype=pl.INT16)
            row_base = pl.and_(ramp, -rope_dim)
            pair_index = pl.shr(pl.and_(ramp, rope_dim - 1), 1)
            half_index = pl.shl(pl.and_(ramp, 1), pair_shift)
            index_half = pl.cast(pl.or_(row_base, pl.or_(pair_index, half_index)), pl.FP16, mode="rint")
            index_float = pl.cast(index_half, pl.FP32, mode="rint")
            indices = pl.reshape(pl.cast(index_float, pl.INT32, mode="rint"), [row_tile, rope_dim])
            rotated = pl.gather(joined, indices)
            rotated_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
            output[t0:t0 + row_tile, base + nope_dim:base + head_dim] = pl.set_validshape(rotated_bf16, rows, rope_dim)
        return output

    return dequantize_rope


def make_shared_quantize(capacity, width):
    """Quantize shared activations in eight-row wide fragments."""
    m_tile = 8
    k_tile = 2560 if width % 2560 == 0 else 768

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[1, capacity], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + m_tile - 1) // m_tile, name_hint="a2_shared_quantize"):
            t0 = block * m_tile
            rows = pl.min(m_tile, active - t0)
            maximum_row = pl.full([1, m_tile], dtype=pl.FP32, value=1e-30)
            for k0 in pl.range(0, width, k_tile):
                source = pl.slice(x, [m_tile, k_tile], [t0, k0], valid_shape=[rows, k_tile], pad_value=0)
                source = pl.set_validshape(source, m_tile, k_tile)
                value = pl.cast(source, pl.FP32)
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, m_tile])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [m_tile, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, m_tile], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [m_tile, 1])
            scales[:, t0:t0 + m_tile] = pl.reshape(scale, [1, m_tile])
            for k0 in pl.range(0, width, k_tile):
                source = pl.slice(x, [m_tile, k_tile], [t0, k0], valid_shape=[rows, k_tile], pad_value=0)
                source = pl.set_validshape(source, m_tile, k_tile)
                value = pl.cast(source, pl.FP32)
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[t0:t0 + m_tile, k0:k0 + k_tile] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_shared_quantize_from_maxima(capacity, width, activation_tile):
    """Quantize BF16 activations using replicated per-tile partial maxima."""
    quant_tile = width
    quant_m_tile = 8
    maxima_width = width // activation_tile * 8

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], pl.BF16],
        partial_maxima: pl.Tensor[[capacity, maxima_width], pl.FP32],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[1, capacity], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + quant_m_tile - 1) // quant_m_tile, name_hint="a2_shared_quantize_maxima"):
            t0 = block * quant_m_tile
            rows = pl.max(0, pl.min(quant_m_tile, active - t0))
            maxima = pl.slice(partial_maxima, [quant_m_tile, maxima_width], [t0, 0], valid_shape=[rows, maxima_width], pad_value=0)
            maxima = pl.set_validshape(maxima, quant_m_tile, maxima_width)
            maximum_row = pl.reshape(pl.row_max(maxima), [1, quant_m_tile])
            maximum_row = pl.maximum(maximum_row, 1e-30)
            maximum = pl.reshape(maximum_row, [quant_m_tile, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, quant_m_tile], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [quant_m_tile, 1])
            scales[:, t0:t0 + quant_m_tile] = pl.reshape(scale, [1, quant_m_tile])
            for k0 in pl.range(0, width, quant_tile):
                source = pl.slice(x, [quant_m_tile, quant_tile], [t0, k0], valid_shape=[rows, quant_tile], pad_value=0)
                value = pl.cast(pl.set_validshape(source, quant_m_tile, quant_tile), pl.FP32)
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[t0:t0 + quant_m_tile, k0:k0 + quant_tile] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_shared_pack_weight(width, output_width):
    """Reorder shared weights to [N/256,K,256]."""
    n_tile = 256
    k_tile = 256
    group_width = ((width // k_tile + 1) // 2) * k_tile
    packed_rows = width * (output_width // n_tile)

    @pl.jit.inline
    def pack_weight(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, n_tile], pl.INT8],
    ):
        for task in pl.spmd((output_width // n_tile) * 2, name_hint="a2_shared_pack_weight"):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            nb = task // 2
            n0 = nb * n_tile
            k_begin = task % 2 * group_width
            k_end = pl.min(k_begin + group_width, width)
            for k0 in pl.pipeline(k_begin, k_end, k_tile, stage=2):
                block = weight[k0:k0 + k_tile, n0:n0 + n_tile]
                packed_k0 = nb * width + k0
                packed[packed_k0:packed_k0 + k_tile, :] = block
        return packed

    return pack_weight


def make_shared_dot(capacity, width, output_width):
    """Accumulate shared INT8 products with 256-column fragments."""
    m_tile = min(32, capacity)
    n_tile = 256
    k_tile = 256
    packed_rows = width * (output_width // n_tile)

    @pl.jit.inline
    def dot(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, n_tile], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + m_tile - 1) // m_tile
        for block in pl.spmd(row_blocks * (output_width // n_tile), name_hint="a2_shared_int8_dot"):
            t0 = block // (output_width // n_tile) * m_tile
            nb = block % (output_width // n_tile)
            n0 = nb * n_tile
            rows = pl.min(m_tile, active - t0)
            a0 = pl.slice(payload, [m_tile, k_tile], [t0, 0], valid_shape=[rows, k_tile], pad_value=0)
            packed_start = nb * width
            b0 = packed[packed_start:packed_start + k_tile, :]
            acc = pl.matmul(a0, b0, out_dtype=pl.INT32)
            for k0 in pl.pipeline(k_tile, width, k_tile, stage=2):
                a = pl.slice(payload, [m_tile, k_tile], [t0, k0], valid_shape=[rows, k_tile], pad_value=0)
                packed_k0 = nb * width + k0
                b = packed[packed_k0:packed_k0 + k_tile, :]
                acc = pl.matmul_acc(acc, a, b)
            accumulator[t0:t0 + m_tile, n0:n0 + n_tile] = acc
        return accumulator

    return dot


def make_shared_down(capacity, width, output_width):
    """Accumulate the shared down projection and scale eight-row fragments."""
    dot = make_shared_dot(capacity, width, output_width)
    m_tile = 8
    n_tile = 1280
    packed_rows = width * (output_width // 256)

    @pl.jit.inline
    def down(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, 256], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[1, capacity], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        accumulator = pl.create_tensor([capacity, output_width], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        row_blocks = (active + m_tile - 1) // m_tile
        for block in pl.spmd(row_blocks * (output_width // n_tile), name_hint="a2_shared_down_scale"):
            t0 = block // (output_width // n_tile) * m_tile
            n0 = block % (output_width // n_tile) * n_tile
            rows = pl.min(m_tile, active - t0)
            integer = accumulator[t0:t0 + m_tile, n0:n0 + n_tile]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0:n0 + n_tile], [1, n_tile])
            scaled_channels = pl.col_expand_mul(value, channels)
            scale = pl.reshape(token_scale[:, t0:t0 + m_tile], [m_tile, 1])
            scaled = pl.row_expand_mul(scaled_channels, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0:t0 + m_tile, n0:n0 + n_tile] = pl.set_validshape(narrowed, rows, n_tile)
        return output

    return down
