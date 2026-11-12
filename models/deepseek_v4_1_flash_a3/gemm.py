# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Blocked GEMM, BF16 projections and Eco-Tech W8A8 linear compositions."""

import math
import pypto.language as pl
import torch
from .quant import make_dequantize, make_quantize
from .utils import pack_blocked_weight

M_TILE = 16
N_TILE = 64
K_TILE = 256
VEC_M_TILE = 8
FULL_K_TILE = 5120

NAMESPACE = "pypto_dsv41_bf16_linear_v1"
SCHEMA = "linear(Tensor source, Tensor weight_nk) -> Tensor"


def make_pack_weight(width: int, output_width: int):
    """Pack canonical ``[K, N]`` INT8 weights into 64-column blocks."""

    if output_width % N_TILE or width % K_TILE:
        raise ValueError("A2/A3 weight packing requires K divisible by 256 and N by 64")
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def pack(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
    ):
        for block in pl.spmd(output_width // N_TILE, name_hint="dsv41_a2a3_pack_weight"):
            n0 = block * N_TILE
            for k0 in pl.range(0, width, K_TILE):
                packed[block * width + k0 : block * width + k0 + K_TILE, :] = weight[
                    k0 : k0 + K_TILE, n0 : n0 + N_TILE
                ]
        return packed

    return pack


def make_int8_gemm(capacity: int, width: int, output_width: int):
    """Accumulate INT8 activations and packed INT8 weights into INT32."""

    if capacity % M_TILE or width % K_TILE:
        raise ValueError("capacity must be divisible by 16 and K by 256")
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def gemm(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = output_width // N_TILE
        for block in pl.spmd(((active + 15) // M_TILE) * blocks, name_hint="dsv41_a2a3_int8_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                lhs = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                rhs = packed[(block % blocks) * width + k0 : (block % blocks) * width + k0 + K_TILE, :]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            accumulator[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(acc, rows, N_TILE)
        return accumulator

    return gemm


def make_fp32_dequantize(capacity: int, output_width: int):
    """Apply token/channel FP32 scales and narrow the result to BF16."""

    if output_width % N_TILE:
        raise ValueError("output width must be divisible by 64")

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        token_scale: pl.Tensor[[capacity, 1], pl.FP32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = output_width // N_TILE
        for block in pl.spmd(((active + 7) // 8) * blocks, name_hint="dsv41_a2a3_dequant"):
            t0 = (block // blocks) * 8
            n0 = (block % blocks) * N_TILE
            rows = pl.min(8, active - t0)
            value = pl.cast(accumulator[t0 : t0 + 8, n0 : n0 + N_TILE], pl.FP32)
            channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
            value = pl.col_expand_mul(value, channels)
            # Preserve the 2-D [rows, 1] view so the backend emits the legal
            # row-scale load used by the A2/A3 W8A8 kernels.
            scale = token_scale[t0 : t0 + 8, 0:1]
            value = pl.row_expand_mul(value, scale)
            output[t0 : t0 + 8, n0 : n0 + N_TILE] = pl.set_validshape(
                pl.cast(value, pl.BF16, mode="rint"), rows, N_TILE
            )
        return output

    return dequantize


def make_int8_linear_direct(capacity: int, width: int, output_width: int):
    """Build a W8A8 linear that consumes canonical ``[K, N]`` weights.

    The packed-weight helper is useful for the leaf ABI, but materializing a
    full packed matrix is too large for the A2/A3 Mat buffer at V4.1 widths.
    This variant loads one ``[K_TILE, N_TILE]`` canonical tile per GEMM
    reduction and keeps only the accumulator tile on core.
    """

    if capacity % M_TILE or width % K_TILE or output_width % N_TILE:
        raise ValueError("direct INT8 GEMM requires aligned capacity, K and N")

    from .quant import make_dynamic_int8_quantize

    quantize = make_dynamic_int8_quantize(capacity, width)
    blocks = output_width // N_TILE

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(value, payload, token_scale, active)
        for block in pl.spmd(
            ((active + M_TILE - 1) // M_TILE) * blocks, name_hint="dsv41_a2a3_direct_int8_linear"
        ):
            t0 = (block // blocks) * M_TILE
            n0 = (block % blocks) * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                lhs = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                rhs = weight[k0 : k0 + K_TILE, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
            value_fp32 = pl.col_expand_mul(value_fp32, channels)
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + M_TILE, :])
            output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, N_TILE
            )
        return output

    return linear


def make_bf16_linear(capacity: int, width: int, output_width: int):
    """BF16-input/FP32-accumulate linear for compressor and Engram paths."""

    if capacity % 16 or width % 256 or output_width % 32:
        raise ValueError("BF16 linear dimensions are not A2/A3 aligned")
    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_bf16_linear"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0 : k0 + 256, n0 : n0 + n_tile]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            output[t0 : t0 + 16, n0 : n0 + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        return output

    return linear


def make_fp32_linear(capacity: int, width: int, output_width: int):
    """BF16-input/FP32-weight linear used by the ratio-2 compressor."""

    if capacity % 16 or width % 256 or output_width % 32:
        raise ValueError("FP32 linear dimensions are not A2/A3 aligned")
    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_fp32_linear"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0 : k0 + 256, n0 : n0 + n_tile]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            output[t0 : t0 + 16, n0 : n0 + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        return output

    return linear


def make_blocked_weight_pack(width, output_width):
    """Reorder canonical [K,N] INT8 bytes to [N/64,K,64]."""
    packed_rows = width * (output_width // N_TILE)

    @pl.jit.inline
    def pack_weight(
        weight: pl.Tensor[[width, output_width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
    ):
        for nb in pl.spmd(output_width // N_TILE, name_hint="a2_pack_weight"):
            n0 = nb * N_TILE
            for k0 in pl.range(0, width, K_TILE):
                block = weight[k0 : k0 + K_TILE, n0 : n0 + N_TILE]
                packed_k0 = nb * width + k0
                packed[packed_k0 : packed_k0 + K_TILE, :] = block
        return packed

    return pack_weight


def make_dot(capacity, width, output_width, *, full_k_reuse=False, column_group=1, column_stages=1):
    """Accumulate complete blocked INT8 products in INT32."""
    packed_rows = width * (output_width // N_TILE)

    if full_k_reuse and width <= FULL_K_TILE:
        row_tile = min(capacity, M_TILE)

        @pl.jit.inline
        def dot_full_k(
            payload: pl.Tensor[[capacity, width], pl.INT8],
            packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            active: pl.Scalar[pl.INT32],
        ):
            row_blocks = (active + row_tile - 1) // row_tile
            column_blocks = (output_width // N_TILE + column_group - 1) // column_group
            for block in pl.spmd(row_blocks * column_blocks, name_hint="prefill_int8_dot_full_k"):
                t0 = block // column_blocks * row_tile
                first_nb = block % column_blocks * column_group
                last_nb = pl.min(first_nb + column_group, output_width // N_TILE)
                rows = pl.min(row_tile, active - t0)
                source = pl.slice(payload, [row_tile, width], [t0, 0], valid_shape=[rows, width])
                activation = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), row_tile, width)
                for nb in pl.pipeline(first_nb, last_nb, stage=column_stages):
                    n0 = nb * N_TILE
                    packed_k0 = nb * width
                    weight = packed[packed_k0 : packed_k0 + width, :]
                    product = pl.matmul(activation, weight, out_dtype=pl.INT32)
                    accumulator[t0 : t0 + row_tile, n0 : n0 + N_TILE] = product
            return accumulator

        return dot_full_k

    @pl.jit.inline
    def dot(
        payload: pl.Tensor[[capacity, width], pl.INT8],
        packed: pl.Tensor[[packed_rows, N_TILE], pl.INT8],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + M_TILE - 1) // M_TILE
        for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_int8_dot"):
            t0 = (block // (output_width // N_TILE)) * M_TILE
            nb = block % (output_width // N_TILE)
            n0 = nb * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                a = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                packed_k0 = nb * width + k0
                b = packed[packed_k0 : packed_k0 + K_TILE, :]
                acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
            accumulator[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = acc
        return accumulator

    return dot


def make_decode_int8_linear_direct(capacity: int, width: int, output_width: int):
    """Build a W8A8 linear that consumes canonical ``[K, N]`` weights.

    The packed-weight helper is useful for the leaf ABI, but materializing a
    full packed matrix is too large for the A2/A3 Mat buffer at V4.1 widths.
    This variant loads one ``[K_TILE, N_TILE]`` canonical tile per GEMM
    reduction and keeps only the accumulator tile on core.
    """

    if capacity % M_TILE or width % K_TILE or output_width % N_TILE:
        raise ValueError("direct INT8 GEMM requires aligned capacity, K and N")

    from .quant import make_decode_dynamic_int8_quantize as make_dynamic_int8_quantize

    quantize = make_dynamic_int8_quantize(capacity, width)
    blocks = output_width // N_TILE

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(value, payload, token_scale, active)
        for block in pl.spmd(
            ((active + M_TILE - 1) // M_TILE) * blocks, name_hint="dsv41_a2a3_direct_int8_linear"
        ):
            t0 = (block // blocks) * M_TILE
            n0 = (block % blocks) * N_TILE
            rows = pl.min(M_TILE, active - t0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, width, K_TILE, stage=2):
                lhs = pl.slice(payload, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
                rhs = weight[k0 : k0 + K_TILE, n0 : n0 + N_TILE]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
            value_fp32 = pl.col_expand_mul(value_fp32, channels)
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + M_TILE, :])
            output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, N_TILE
            )
        return output

    return linear


def prepare_ecotech_dynamic_weight(weight_nk, source_scale, source_offset):
    """Pack CPU checkpoint weights once and retain effective FP32 channel scales."""
    values = (weight_nk, source_scale, source_offset)
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("Checkpoint preprocessing requires CPU source tensors")
    if weight_nk.ndim != 2 or weight_nk.dtype != torch.int8:
        raise ValueError("The source weight must be a complete NK INT8 tensor")
    output_width, input_width = weight_nk.shape
    if input_width % 256 or output_width % 64:
        raise ValueError("The A3 blocked projection requires K/256 and N/64 alignment")
    for value in (source_scale, source_offset):
        if value.shape != (output_width, 1) or value.dtype != torch.float32:
            raise ValueError("Source per-channel scale and offset require FP32 [N,1]")
        if any(not math.isfinite(item) for item in value.flatten().tolist()):
            raise ValueError("Source scale and offset must be finite")
    if any(item != 0 for item in source_offset.flatten().tolist()):
        raise ValueError("The symmetric native INT8 path requires zero checkpoint offsets")
    native_scale = source_scale.to(torch.bfloat16)
    if any(not math.isfinite(item) or item <= 0 for item in native_scale.flatten().tolist()):
        raise ValueError("Effective BF16 channel scales must remain finite and positive")
    weight_kn = weight_nk.T.contiguous()
    channel_scale = native_scale.to(torch.float32).flatten()
    return pack_blocked_weight(weight_kn), channel_scale


def make_ecotech_dynamic_linear(capacity, input_width, output_width, combine_scales=False):
    """Consume prepacked INT8 weights with unchanged quantization and scaling."""
    if capacity <= 0 or capacity % 16 or input_width % 256 or output_width % 64:
        raise ValueError("Use a positive capacity/16, K/256 and N/64")
    quantize = make_quantize(capacity, input_width, whole_row=True)
    qb_columns = input_width == 1280 and output_width == 32768
    dot = make_dot(
        capacity, input_width, output_width, full_k_reuse=True,
        column_group=26 if qb_columns else 1, column_stages=2 if qb_columns else 1,
    )
    dequantize = make_dequantize(capacity, output_width, combine_scales=combine_scales)

    @pl.jit
    def linear(
        x: pl.Tensor[[capacity, input_width], pl.BF16],
        weight: pl.Tensor[[input_width * (output_width // 64), 64], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, input_width], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(x, payload, token_scale, active)
        accumulator = pl.create_tensor([capacity, output_width], dtype=pl.INT32)
        dot(payload, weight, accumulator, active)
        dequantize(accumulator, channel_scale, token_scale, output, active)
        return output

    return linear


def padded_dimensions(input_width, output_width):
    if type(input_width) is not int or type(output_width) is not int or input_width <= 0 or output_width <= 0:
        raise ValueError("Require positive logical dimensions")
    if input_width % 32 or output_width % 32:
        raise ValueError("This tail path requires logical K/32 and N/32")
    return ((input_width + 255) // 256) * 256, ((output_width + 63) // 64) * 64


def prepare_ecotech_tail_weight(weight_nk, source_scale, source_offset):
    values = (weight_nk, source_scale, source_offset)
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("Checkpoint preprocessing requires CPU operands")
    if weight_nk.ndim != 2 or weight_nk.dtype != torch.int8:
        raise ValueError("Require source NK INT8 bytes")
    output_width, input_width = weight_nk.shape
    padded_k, padded_n = padded_dimensions(input_width, output_width)
    for value in (source_scale, source_offset):
        if value.shape != (output_width, 1) or value.dtype != torch.float32:
            raise ValueError("Require original F32 [N,1] scale/offset")
    padded_weight = torch.zeros((padded_n, padded_k), dtype=torch.int8)
    padded_weight[:output_width, :input_width].copy_(weight_nk)
    padded_scale = torch.ones((padded_n, 1), dtype=torch.float32)
    padded_scale[:output_width].copy_(source_scale)
    padded_offset = torch.zeros((padded_n, 1), dtype=torch.float32)
    padded_offset[:output_width].copy_(source_offset)
    return prepare_ecotech_dynamic_weight(padded_weight, padded_scale, padded_offset)


def _make_pad_input(capacity, input_width, padded_k, input_dtype):
    @pl.jit.inline
    def pad_input(
        x: pl.Tensor[[capacity, input_width], input_dtype],
        padded: pl.Tensor[[capacity, padded_k], input_dtype],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + 7) // 8, name_hint="ecotech_tail_pad_input"):
            t0 = block * 8
            rows = pl.min(8, active - t0)
            for k0 in pl.range(0, padded_k, 256):
                columns = pl.min(256, input_width - k0)
                source = pl.slice(x, [8, 256], [t0, k0], valid_shape=[rows, columns])
                filled = pl.fillpad(source, pad_value=pl.PadValue.zero)
                padded[t0 : t0 + 8, k0 : k0 + 256] = pl.set_validshape(filled, 8, 256)
        return padded

    return pad_input


def _make_crop_output(capacity, output_width, padded_n):
    @pl.jit.inline
    def crop_output(
        padded: pl.Tensor[[capacity, padded_n], pl.BF16],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(
            ((active + 7) // 8) * (output_width // 32), name_hint="ecotech_tail_crop_output"
        ):
            t0 = (block // (output_width // 32)) * 8
            n0 = (block % (output_width // 32)) * 32
            rows = pl.min(8, active - t0)
            value = padded[t0 : t0 + 8, n0 : n0 + 32]
            output[t0 : t0 + 8, n0 : n0 + 32] = pl.set_validshape(value, rows, 32)
        return output

    return crop_output


def make_ecotech_tail_linear(capacity, input_width, output_width):
    if capacity <= 0 or capacity % 16:
        raise ValueError("Require a positive capacity/16")
    padded_k, padded_n = padded_dimensions(input_width, output_width)
    standalone = make_ecotech_dynamic_linear(capacity, padded_k, padded_n)
    aligned = pl.jit.inline(standalone._func)

    pad_input = _make_pad_input(capacity, input_width, padded_k, pl.BF16)
    crop_output = _make_crop_output(capacity, output_width, padded_n)

    @pl.jit
    def linear(
        x: pl.Tensor[[capacity, input_width], pl.BF16],
        weight: pl.Tensor[[padded_k * (padded_n // 64), 64], pl.INT8],
        channel_scale: pl.Tensor[[padded_n], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        padded_x = pl.create_tensor([capacity, padded_k], dtype=pl.BF16)
        pad_input(x, padded_x, active)
        padded_output = pl.create_tensor([capacity, padded_n], dtype=pl.BF16)
        aligned(padded_x, weight, channel_scale, padded_output, active)
        crop_output(padded_output, output, active)
        return output

    return linear


def prepare_ecotech_token_scales(source_scale):
    """Broadcast exact CPU F32 per-token scale bits to the eight-lane ABI."""
    if source_scale.device.type != "cpu" or source_scale.dtype != torch.float32 or source_scale.ndim != 1:
        raise ValueError("Require original CPU F32 per-token vector")
    if not bool(torch.isfinite(source_scale).all()) or not bool((source_scale > 0).all()):
        raise ValueError("Per-token scales must be finite and positive")
    return source_scale[:, None].expand(-1, 8).contiguous()


def make_ecotech_int8_tail_linear(capacity, input_width, output_width, *, combine_scales=False):
    if capacity <= 0 or capacity % 16:
        raise ValueError("Require a positive capacity/16")
    padded_k, padded_n = padded_dimensions(input_width, output_width)
    dot = make_dot(capacity, padded_k, padded_n)
    dequantize = make_dequantize(capacity, padded_n, combine_scales=combine_scales)
    pad_input = _make_pad_input(capacity, input_width, padded_k, pl.INT8)
    crop_output = _make_crop_output(capacity, output_width, padded_n)

    @pl.jit
    def linear(
        payload: pl.Tensor[[capacity, input_width], pl.INT8],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        weight: pl.Tensor[[padded_k * (padded_n // 64), 64], pl.INT8],
        channel_scale: pl.Tensor[[padded_n], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        padded_payload = pl.create_tensor([capacity, padded_k], dtype=pl.INT8)
        pad_input(payload, padded_payload, active)
        accumulator = pl.create_tensor([capacity, padded_n], dtype=pl.INT32)
        dot(padded_payload, weight, accumulator, active)
        padded_output = pl.create_tensor([capacity, padded_n], dtype=pl.BF16)
        dequantize(accumulator, channel_scale, token_scale, padded_output, active)
        crop_output(padded_output, output, active)
        return output

    return linear


def register(torch):
    from torch.utils._python_dispatch import TorchDispatchMode

    trace = []

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            return [x for v in value for x in tensors(v)]
        if isinstance(value, dict):
            return [x for v in value.values() for x in tensors(v)]
        return []

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            values = tensors((args, kwargs or {}))
            if any(v.device.type != "npu" or v.dtype != torch.bfloat16 for v in values):
                raise ValueError("BF16 linear dispatch reached a different device/dtype")
            result = func(*args, **(kwargs or {}))
            if any(v.device.type != "npu" or v.dtype != torch.bfloat16 for v in tensors(result)):
                raise ValueError("BF16 linear dispatch returned a different device/dtype")
            trace.append(
                dict(
                    op=str(func),
                    schema=str(func._schema),
                    inputs=[
                        dict(
                            shape=list(v.shape),
                            stride=list(v.stride()),
                            dtype=str(v.dtype),
                            device=str(v.device),
                        )
                        for v in values
                    ],
                )
            )
            return result

    def linear(source, weight_nk):
        if (
            source.device.type != "npu"
            or weight_nk.device.type != "npu"
            or source.dtype != torch.bfloat16
            or weight_nk.dtype != torch.bfloat16
            or source.ndim != 2
            or weight_nk.ndim != 2
            or source.shape[1] != weight_nk.shape[1]
        ):
            raise ValueError("Generic NPU BF16 M,K / N,K operands required")
        with Observe():
            return torch.ops.aten.linear.default(source, weight_nk, None)

    definition = torch.library.Library(NAMESPACE, "DEF")
    definition.define(SCHEMA)
    implementation = torch.library.Library(NAMESPACE, "IMPL", "PrivateUse1")
    implementation.impl("linear", linear)
    return getattr(torch.ops, NAMESPACE).linear.default, (definition, implementation), trace


def make_decode_bf16_linear(capacity: int, width: int, output_width: int):
    """BF16-input/FP32-accumulate linear for compressor and Engram paths."""

    if capacity % 16 or width % 256 or output_width % 32:
        raise ValueError("BF16 linear dimensions are not A2/A3 aligned")
    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_bf16_linear"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = pl.slice(weight, [256, n_tile], [k0, n0])
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            output[t0 : t0 + 16, n0 : n0 + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        return output

    return linear
