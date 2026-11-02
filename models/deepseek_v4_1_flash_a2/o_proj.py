# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inverse partial RoPE, grouped BF16 Wo-A and dynamic W8A8 Wo-B."""

import pypto.language as pl

from .ops import M_TILE, K_TILE, N_TILE, make_rope, make_quantize, make_pack_weight, make_dot, make_dequantize


def make_o_proj_inline(capacity, config):
    """Build the complete grouped attention output projection."""
    n_tile = 256
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    width = heads * head_dim
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    group_width = groups * rank
    packed_rows = group_width * (hidden // n_tile)
    nope_dim = head_dim - rope_dim
    rope_head_tile = 8 if heads % 8 == 0 else 1
    row_tile = min(capacity, 2 * M_TILE)
    a_n_tile = 512 if rank % 512 == 0 else n_tile
    a_k_tile = 128 if group_input % 128 == 0 else K_TILE
    b_k_tile = 512 if group_width % 512 == 0 else K_TILE
    quant_row_tile = 16
    quant_tile = 1024 if group_width % 1024 == 0 else K_TILE
    partial_width = group_width // a_n_tile
    pack = make_pack_weight(group_width, hidden, n_tile=n_tile, k_splits=2)

    @pl.jit.inline
    def quantize_partials(
        values: pl.Tensor[[capacity, group_width], pl.BF16],
        maxima: pl.Tensor[[partial_width, capacity], pl.FP32],
        quantized: pl.Tensor[[capacity, group_width], pl.INT8],
        scales: pl.Tensor[[1, capacity], pl.FP32],
        padded_active: pl.Scalar[pl.INT32],
    ):
        quant_tasks = (padded_active + quant_row_tile - 1) // quant_row_tile * (group_width // quant_tile)
        for quant_block in pl.spmd(quant_tasks, name_hint="a2_partial_quantize"):
            t0 = quant_block // (group_width // quant_tile) * quant_row_tile
            q_inner = quant_block % (group_width // quant_tile) * quant_tile
            maximum_row = pl.maximum(pl.col_max(maxima[:, t0:t0 + quant_row_tile]), 1e-30)
            numerator = pl.full([1, quant_row_tile], dtype=pl.FP32, value=127.0)
            inverse = pl.reshape(pl.div(numerator, maximum_row), [quant_row_tile, 1])
            if q_inner == 0:
                scales[:, t0:t0 + quant_row_tile] = pl.mul(maximum_row, 1.0 / 127.0)
            source = values[t0:t0 + quant_row_tile, q_inner:q_inner + quant_tile]
            value = pl.cast(source, pl.FP32)
            normalized = pl.row_expand_mul(value, inverse)
            bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
            rounded = pl.cast(bounded, pl.INT32, mode="rint")
            exact_half = pl.cast(rounded, pl.FP16, mode="rint")
            quantized[t0:t0 + quant_row_tile, q_inner:q_inner + quant_tile] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return quantized, scales

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[group_width, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        attention_heads = pl.reshape(attention, [capacity * heads, head_dim])
        rotated_heads = pl.reshape(rotated, [capacity * heads, head_dim])
        for block in pl.spmd(active * (heads // rope_head_tile), name_hint="a2_inverse_rope"):
            token = block // (heads // rope_head_tile)
            head = block % (heads // rope_head_tile) * rope_head_tile
            row = token * heads + head
            full_head = pl.load(attention_heads, [row, 0], [rope_head_tile, head_dim], target_memory=pl.MemorySpace.Vec)
            source = pl.tile.slice(full_head, [rope_head_tile, nope_dim], [0, 0])
            rope_source = pl.tile.slice(full_head, [rope_head_tile, rope_dim], [0, nope_dim])
            tail = pl.cast(rope_source, pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(rope_cos, [token, 0], [1, pairs], target_memory=pl.MemorySpace.Vec)
            sine_source = pl.load(rope_sin, [token, 0], [1, pairs], target_memory=pl.MemorySpace.Vec)
            sine = pl.mul(sine_source, -1.0)
            even_c = pl.col_expand_mul(even, cosine)
            odd_s = pl.col_expand_mul(odd, sine)
            odd_c = pl.col_expand_mul(odd, cosine)
            even_s = pl.col_expand_mul(even, sine)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            real_field = pl.tile.full([rope_head_tile, rope_dim], dtype=pl.FP32, value=0.0)
            imag_field = pl.tile.full([rope_head_tile, rope_dim], dtype=pl.FP32, value=0.0)
            real_field = pl.tile.scatter_mask(real_field, real, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_field = pl.tile.scatter_mask(imag_field, imag, mask_pattern=pl.tile.MaskPattern.P1010)
            interleaved = pl.add(real_field, imag_field)
            narrowed_rope = pl.cast(interleaved, pl.BF16, mode="rint")
            full_output = pl.concat(source, narrowed_rope)
            pl.store(full_output, [row, 0], rotated_heads)

        grouped = pl.create_tensor([capacity, group_width], dtype=pl.BF16)
        partial_maxima = pl.create_tensor([partial_width, capacity], dtype=pl.FP32)
        a_tasks = (active + row_tile - 1) // row_tile * (group_width // a_n_tile)
        for block in pl.spmd(a_tasks, name_hint="a2_wo_a", optimizations=[pl.cross_core_slot(slot_num=1)]):
            pl.set_cache_policy(wo_a, pl.CachePolicy.BYPASS)
            token = block // (group_width // a_n_tile) * row_tile
            column = block % (group_width // a_n_tile) * a_n_tile
            group = column // rank
            group_column = column % rank
            rows = pl.min(row_tile, active - token)
            group_offset = group * group_input
            first_x = pl.slice(rotated, [row_tile, a_k_tile], [token, group_offset], valid_shape=[rows, a_k_tile])
            first_w = wo_a[group:group + 1, :a_k_tile, group_column:group_column + a_n_tile]
            first_w = pl.reshape(first_w, [a_k_tile, a_n_tile])
            accumulated = pl.matmul(first_x, first_w, out_dtype=pl.FP32)
            for part in pl.pipeline(1, group_input // a_k_tile, stage=2):
                inner = part * a_k_tile
                offset = group * group_input + inner
                x = pl.slice(rotated, [row_tile, a_k_tile], [token, offset], valid_shape=[rows, a_k_tile])
                w = wo_a[group:group + 1, inner:inner + a_k_tile, group_column:group_column + a_n_tile]
                w = pl.reshape(w, [a_k_tile, a_n_tile])
                accumulated = pl.matmul_acc(accumulated, x, w)
            narrowed = pl.cast(accumulated, pl.BF16, mode="rint")
            narrowed = pl.fillpad(narrowed, pad_value=pl.PadValue.zero)
            narrowed = pl.set_validshape(narrowed, row_tile, a_n_tile)
            grouped[token:token + row_tile, column:column + a_n_tile] = narrowed
            partial_left = pl.cast(narrowed[:, :a_n_tile // 2], pl.FP32)
            maximum_left = pl.row_max(pl.abs(partial_left))
            partial_right = pl.cast(narrowed[:, a_n_tile // 2:a_n_tile], pl.FP32)
            maximum_right = pl.row_max(pl.abs(partial_right))
            partial_maximum = pl.reshape(pl.maximum(maximum_left, maximum_right), [1, row_tile])
            partial_column = column // a_n_tile
            partial_maxima[partial_column:partial_column + 1, token:token + row_tile] = partial_maximum

        payload = pl.create_tensor([capacity, group_width], dtype=pl.INT8)
        token_scale = pl.create_tensor([1, capacity], dtype=pl.FP32)
        quantize_partials(grouped, partial_maxima, payload, token_scale, (active + row_tile - 1) // row_tile * row_tile)
        packed = pl.create_tensor([packed_rows, n_tile], dtype=pl.INT8)
        pack(wo_b, packed)
        for block in pl.spmd((active + row_tile - 1) // row_tile * (hidden // n_tile), name_hint="a2_wo_b"):
            token = block // (hidden // n_tile) * row_tile
            nb = block % (hidden // n_tile)
            column = nb * n_tile
            rows = pl.min(row_tile, active - token)
            integer = pl.create_tensor([row_tile, n_tile], dtype=pl.INT32)
            for b_inner in pl.pipeline(0, group_width, b_k_tile, stage=2):
                b_payload = payload[token:token + row_tile, b_inner:b_inner + b_k_tile]
                packed_inner = nb * group_width + b_inner
                b_weight = packed[packed_inner:packed_inner + b_k_tile, :]
                integer = pl.matmul_acc(integer, b_payload, b_weight, init_cond=(b_inner == 0))
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(wo_b_scale[column:column + n_tile], [1, n_tile])
            scale = pl.reshape(token_scale[:, token:token + row_tile], [row_tile, 1])
            ones = pl.full([row_tile, n_tile], dtype=pl.FP32, value=1.0)
            channel_rows = pl.col_expand_mul(ones, channels)
            combined = pl.row_expand_mul(channel_rows, scale)
            scaled = pl.mul(value, combined)
            b_output = pl.cast(scaled, pl.BF16, mode="rint")
            output[token:token + row_tile, column:column + n_tile] = pl.set_validshape(b_output, rows, n_tile)
        return output

    return project


def make_o_proj_w8a8(capacity, config):
    """Build inverse RoPE and all grouped W8A8 Wo-A/Wo-B projections.

    Wo-A input quantization is independent for each native group. The package
    stores [group_input, groups*rank] with per-output-channel FP32 scales.
    """
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    pairs = rope_dim // 2
    width = heads * head_dim
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    group_width = groups * rank
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize_a = make_quantize(capacity, group_input)
    pack_a = make_pack_weight(group_input, rank)
    dot_a = make_dot(capacity, group_input, rank)
    dequantize_a = make_dequantize(capacity, rank)
    quantize_b = make_quantize(capacity, group_width)
    pack_b = make_pack_weight(group_width, hidden)
    dot_b = make_dot(capacity, group_width, hidden)
    dequantize_b = make_dequantize(capacity, hidden, combine_scales=True)
    packed_a_rows = group_input * (rank // N_TILE)
    packed_b_rows = group_width * (hidden // N_TILE)

    @pl.jit
    def project(
        attention: pl.Tensor[[capacity, width], pl.FP32],
        wo_a: pl.Tensor[[group_input, group_width], pl.INT8],
        wo_a_scale: pl.Tensor[[group_width], pl.FP32],
        wo_b: pl.Tensor[[group_width, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        rotated: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        grouped: pl.InOut[pl.Tensor[[capacity, group_width], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        narrowed = pl.create_tensor([capacity, width], dtype=pl.BF16)
        for block in pl.spmd(active * (width // N_TILE), name_hint="a2_attention_bf16_boundary"):
            token = block // (width // N_TILE)
            col = block % (width // N_TILE) * N_TILE
            source = attention[token:token + 1, col:col + N_TILE]
            narrowed[token:token + 1, col:col + N_TILE] = pl.cast(source, pl.BF16, mode="rint")
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for token in pl.spmd(active, name_hint="a2_w8a8_inverse_sine"):
            inverse_sine[token:token + 1, :] = pl.mul(rope_sin[token:token + 1, :], -1.0)
        rotate(narrowed, rope_cos, inverse_sine, rotated, active)

        for group in pl.range(groups):
            group_source = pl.create_tensor([capacity, group_input], dtype=pl.BF16)
            for block in pl.spmd(active * (group_input // N_TILE), name_hint="a2_wo_a_group_input"):
                token = block // (group_input // N_TILE)
                column = block % (group_input // N_TILE) * N_TILE
                offset = group * group_input + column
                group_source[token:token + 1, column:column + N_TILE] = rotated[token:token + 1, offset:offset + N_TILE]
            payload_a = pl.create_tensor([capacity, group_input], dtype=pl.INT8)
            scale_a = pl.create_tensor([capacity, 8], dtype=pl.FP32)
            quantize_a(group_source, payload_a, scale_a, active)
            group_weight = pl.create_tensor([group_input, rank], dtype=pl.INT8)
            for block in pl.spmd((group_input // K_TILE) * (rank // N_TILE), name_hint="a2_wo_a_group_weight"):
                inner = block // (rank // N_TILE) * K_TILE
                column = block % (rank // N_TILE) * N_TILE
                offset = group * rank + column
                group_weight[inner:inner + K_TILE, column:column + N_TILE] = wo_a[inner:inner + K_TILE, offset:offset + N_TILE]
            packed_a = pl.create_tensor([packed_a_rows, N_TILE], dtype=pl.INT8)
            pack_a(group_weight, packed_a)
            accumulator_a = pl.create_tensor([capacity, rank], dtype=pl.INT32)
            dot_a(payload_a, packed_a, accumulator_a, active)
            channel_scale = wo_a_scale[group * rank:(group + 1) * rank]
            group_output = pl.create_tensor([capacity, rank], dtype=pl.BF16)
            dequantize_a(accumulator_a, channel_scale, scale_a, group_output, active)
            for block in pl.spmd(active * (rank // N_TILE), name_hint="a2_wo_a_group_output"):
                token = block // (rank // N_TILE)
                column = block % (rank // N_TILE) * N_TILE
                offset = group * rank + column
                grouped[token:token + 1, offset:offset + N_TILE] = group_output[token:token + 1, column:column + N_TILE]

        payload_b = pl.create_tensor([capacity, group_width], dtype=pl.INT8)
        scale_b = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize_b(grouped, payload_b, scale_b, active)
        packed_b = pl.create_tensor([packed_b_rows, N_TILE], dtype=pl.INT8)
        pack_b(wo_b, packed_b)
        accumulator_b = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_b(payload_b, packed_b, accumulator_b, active)
        dequantize_b(accumulator_b, wo_b_scale, scale_b, output, active)
        return rotated, grouped, output

    return project
