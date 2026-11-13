# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared W8A8 goldens and phase-specific device quantization."""

import pypto.language as pl
import torch
from .operator_config import CONTRACT

K_TILE = 256
N_TILE = 64
VEC_M_TILE = 8
WHOLE_ROW_TILE = 5120

INT8_AMAX_EPS = CONTRACT.int8_amax_epsilon


DECODE_INT8_AMAX_EPS = CONTRACT.int8_amax_epsilon


def make_dynamic_int8_quantize(capacity: int, width: int):
    """Build a PyPTO kernel for BF16 to INT8 per-token quantization.

    The scale uses the vLLM/A2/A3 contract directly: one FP32 dequantization
    scale per token, stored as ``[T, 1]``.
    """

    if capacity % 8 or width % K_TILE:
        raise ValueError("capacity must be divisible by 8 and width by 256")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 1], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(capacity // 8, name_hint="dsv41_a2a3_dynamic_quant"):
            t0 = block * 8
            maximum = pl.full([1, 8], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(value, [8, K_TILE], [t0, k0])
                maximum = pl.maximum(
                    maximum, pl.reshape(pl.row_max(pl.abs(pl.cast(source, pl.FP32))), [1, 8])
                )
            inverse = pl.reshape(pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), maximum), [8, 1])
            scale = pl.reshape(pl.div(maximum, 127.0), [8, 1])
            scales[t0 : t0 + 8, 0:1] = scale
            for k0 in pl.range(0, width, K_TILE):
                source_fp32 = pl.cast(pl.slice(value, [8, K_TILE], [t0, k0]), pl.FP32)
                normalized = pl.row_expand_mul(source_fp32, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                payload[t0 : t0 + 8, k0 : k0 + K_TILE] = pl.cast(rounded, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_decode_dynamic_int8_quantize(capacity: int, width: int, *, cann_dispatch: bool = False):
    """Build a PyPTO kernel for BF16 to INT8 per-token quantization.

    The scale uses the vLLM/A2/A3 contract directly: one FP32 dequantization
    scale per token, stored as ``[T, 1]``.  CANN's fused dispatch path
    uses FP16 truncation before its rounded INT8 store; other paths retain
    the existing INT32 staging sequence.
    """

    if capacity % 8 or width % K_TILE:
        raise ValueError("capacity must be divisible by 8 and width by 256")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 1], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(capacity // 8, name_hint="dsv41_a2a3_dynamic_quant"):
            t0 = block * 8
            maximum = pl.full([1, 8], dtype=pl.FP32, value=DECODE_INT8_AMAX_EPS)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(value, [8, K_TILE], [t0, k0])
                maximum = pl.maximum(
                    maximum, pl.reshape(pl.row_max(pl.abs(pl.cast(source, pl.FP32))), [1, 8])
                )
            inverse = pl.reshape(pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), maximum), [8, 1])
            scale = pl.reshape(pl.div(maximum, 127.0), [8, 1])
            scales[t0 : t0 + 8, 0:1] = scale
            for k0 in pl.range(0, width, K_TILE):
                source_fp32 = pl.cast(pl.slice(value, [8, K_TILE], [t0, k0]), pl.FP32)
                normalized = pl.row_expand_mul(source_fp32, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                if cann_dispatch:
                    payload[t0 : t0 + 8, k0 : k0 + K_TILE] = pl.cast(
                        pl.cast(bounded, target_type=pl.FP16, mode="trunc"),
                        target_type=pl.INT8,
                        mode="round",
                    )
                else:
                    rounded = pl.cast(bounded, pl.INT32, mode="rint")
                    payload[t0 : t0 + 8, k0 : k0 + K_TILE] = pl.cast(rounded, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_cast_tile(rows, width, input_dtype, output_dtype):
    """Select a typed identity or cross-dtype rounding conversion."""
    if input_dtype == output_dtype:

        @pl.jit.inline
        def identity(value: pl.Tensor[[rows, width], input_dtype]):
            return value

        return identity

    @pl.jit.inline
    def cast_tile(value: pl.Tensor[[rows, width], input_dtype]):
        return pl.cast(value, output_dtype, mode="rint")

    return cast_tile


def make_decode_cast_tile(rows, width, input_dtype, output_dtype):
    """Choose identity or a rounding cast before the DSL parses either body."""
    # PyPTO needs a concrete Tensor annotation while parsing an inline body.
    # A closure over ``input_dtype`` is a Python value, so the generic
    # annotation used here previously lost the tensor metadata during a full
    # layer compile.  Keep the small set of dtypes used by the model explicit.
    if input_dtype == pl.BF16 and output_dtype == pl.BF16:

        @pl.jit.inline
        def identity_bf16(value: pl.Tensor[[rows, width], pl.BF16]):
            return value

        return identity_bf16
    if input_dtype == pl.BF16 and output_dtype == pl.FP32:

        @pl.jit.inline
        def cast_bf16_fp32(value: pl.Tensor[[rows, width], pl.BF16]):
            return pl.cast(value, pl.FP32, mode="rint")

        return cast_bf16_fp32
    if input_dtype == pl.FP32 and output_dtype == pl.FP32:

        @pl.jit.inline
        def identity_fp32(value: pl.Tensor[[rows, width], pl.FP32]):
            return value

        return identity_fp32
    if input_dtype == pl.FP32 and output_dtype == pl.BF16:

        @pl.jit.inline
        def cast_fp32_bf16(value: pl.Tensor[[rows, width], pl.FP32]):
            return pl.cast(value, pl.BF16, mode="rint")

        return cast_fp32_bf16
    raise ValueError(f"unsupported cast tile dtypes: {input_dtype} -> {output_dtype}")


def make_quantize_row(width):
    """Quantize one FP32 row with eight-segment maximum reduction."""
    @pl.jit.inline
    def quantize_row(value: pl.Tensor[[1, width], pl.FP32]):
        segments = pl.reshape(pl.abs(value), [8, width // 8])
        partial = pl.row_max(segments)
        ones = pl.full([8, 8], dtype=pl.FP32, value=1.0)
        repeated = pl.row_expand_mul(ones, partial)
        maximum = pl.col_max(repeated)
        scale = pl.mul(maximum, 1.0 / 127.0)
        safe_maximum = pl.maximum(maximum, 1e-30)
        numerator = pl.full([1, 8], dtype=pl.FP32, value=127.0)
        inverse_row = pl.div(numerator, safe_maximum)
        inverse = pl.read(inverse_row, [0, 0])
        normalized = pl.mul(value, inverse)
        lower_bounded = pl.maximum(normalized, -127.0)
        bounded = pl.minimum(lower_bounded, 127.0)
        rounded = pl.cast(bounded, pl.INT32, mode="rint")
        exact_half = pl.cast(rounded, pl.FP16, mode="rint")
        codes = pl.cast(exact_half, pl.INT8, mode="trunc")
        return codes, scale

    return quantize_row


def make_quantize(capacity, width, input_dtype=pl.BF16, *, rows_tile=VEC_M_TILE, whole_row=False):
    """Specialize symmetric per-token INT8 quantization."""
    if whole_row and width <= WHOLE_ROW_TILE:
        if width % 8:
            raise ValueError("Whole-row quantization requires eight equal reduction segments")
        widen_row = make_cast_tile(1, width, input_dtype, pl.FP32)
        quantize_row = make_quantize_row(width)

        @pl.jit.inline
        def quantize_whole_row(
            x: pl.Tensor[[capacity, width], input_dtype],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            for token in pl.spmd(active, name_hint="prefill_token_quantize_whole_row"):
                source = pl.slice(x, [1, width], [token, 0], valid_shape=[1, width])
                value = pl.create_tensor([1, width], dtype=pl.FP32)
                value[:, :] = widen_row(source)
                codes, scale = quantize_row(value)
                scales[token : token + 1, :] = scale
                payload[token : token + 1, :] = codes
            return payload, scales

        return quantize_whole_row

    width_tile = min(K_TILE, width)
    widen = make_cast_tile(rows_tile, width_tile, input_dtype, pl.FP32)

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], input_dtype],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + rows_tile - 1) // rows_tile, name_hint="a2_token_quantize"):
            t0 = block * rows_tile
            rows = pl.min(rows_tile, active - t0)
            maximum_row = pl.full([1, rows_tile], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, width, width_tile):
                source = pl.slice(x, [rows_tile, width_tile], [t0, k0], valid_shape=[rows, width_tile])
                widened = widen(source)
                value = pl.set_validshape(
                    pl.fillpad(widened, pad_value=pl.PadValue.zero), rows_tile, width_tile
                )
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, rows_tile])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [rows_tile, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, rows_tile], dtype=pl.FP32, value=127.0)
            safe_maximum_row = pl.maximum(maximum_row, 1e-30)
            inverse_row = pl.div(numerator, safe_maximum_row)
            inverse = pl.reshape(inverse_row, [rows_tile, 1])
            scale_rows = pl.full([rows_tile, 8], dtype=pl.FP32, value=1.0)
            expanded_scale = pl.row_expand_mul(scale_rows, scale)
            scales[t0 : t0 + rows_tile, :] = pl.set_validshape(expanded_scale, rows, 8)
            for k0 in pl.range(0, width, width_tile):
                source = pl.slice(x, [rows_tile, width_tile], [t0, k0], valid_shape=[rows, width_tile])
                widened = widen(source)
                value = pl.set_validshape(
                    pl.fillpad(widened, pad_value=pl.PadValue.zero), rows_tile, width_tile
                )
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                codes = pl.cast(exact_half, pl.INT8, mode="trunc")
                payload[t0 : t0 + rows_tile, k0 : k0 + width_tile] = pl.set_validshape(
                    codes, rows, width_tile
                )
        return payload, scales

    return quantize


def make_decode_quantize(capacity, width, input_dtype=pl.BF16):
    """Specialize symmetric per-token INT8 quantization."""
    if input_dtype == pl.FP32:

        @pl.jit.inline
        def quantize(
            x: pl.Tensor[[capacity, width], pl.FP32],
            payload: pl.Tensor[[capacity, width], pl.INT8],
            scales: pl.Tensor[[capacity, 8], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_token_quantize"):
                t0 = block * VEC_M_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                maximum_row = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
                for k0 in pl.range(0, width, K_TILE):
                    source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                    source = pl.set_validshape(
                        pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE
                    )
                    chunk_maximum = pl.reshape(pl.row_max(pl.abs(source)), [1, VEC_M_TILE])
                    maximum_row = pl.maximum(maximum_row, chunk_maximum)
                maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
                scale = pl.mul(maximum, 1.0 / 127.0)
                numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
                inverse_row = pl.div(numerator, maximum_row)
                inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
                scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
                scales[t0 : t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, scale)
                for k0 in pl.range(0, width, K_TILE):
                    source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                    source = pl.set_validshape(
                        pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE
                    )
                    normalized = pl.row_expand_mul(source, inverse)
                    bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                    rounded = pl.cast(bounded, pl.INT32, mode="rint")
                    exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                    payload[t0 : t0 + VEC_M_TILE, k0 : k0 + K_TILE] = pl.cast(
                        exact_half, pl.INT8, mode="trunc"
                    )
            return payload, scales

        return quantize

    @pl.jit.inline
    def quantize(
        x: pl.Tensor[[capacity, width], input_dtype],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_token_quantize"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            maximum_row = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                value = pl.cast(source, pl.FP32, mode="rint")
                chunk_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, VEC_M_TILE])
                maximum_row = pl.maximum(maximum_row, chunk_maximum)
            maximum = pl.reshape(maximum_row, [VEC_M_TILE, 1])
            scale = pl.mul(maximum, 1.0 / 127.0)
            numerator = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
            inverse_row = pl.div(numerator, maximum_row)
            inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
            scale_rows = pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0)
            scales[t0 : t0 + VEC_M_TILE, :] = pl.row_expand_mul(scale_rows, scale)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(x, [VEC_M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, K_TILE)
                value = pl.cast(source, pl.FP32, mode="rint")
                normalized = pl.row_expand_mul(value, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                exact_half = pl.cast(rounded, pl.FP16, mode="rint")
                payload[t0 : t0 + VEC_M_TILE, k0 : k0 + K_TILE] = pl.cast(exact_half, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


def make_dequantize(capacity, output_width, combine_scales=False, channel_dtype=pl.FP32):
    """Apply channel-first or combined scales with the selected channel rounding."""
    column_tile = 1280 if output_width == 5120 else 1024
    while column_tile > N_TILE and output_width % column_tile:
        column_tile //= 2
    narrow_channel = make_cast_tile(VEC_M_TILE, column_tile, pl.FP32, channel_dtype)
    widen_channel = make_cast_tile(VEC_M_TILE, column_tile, channel_dtype, pl.FP32)

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (output_width // column_tile), name_hint="a2_dequantize"):
            t0 = (block // (output_width // column_tile)) * VEC_M_TILE
            n0 = (block % (output_width // column_tile)) * column_tile
            rows = pl.min(VEC_M_TILE, active - t0)
            integer = accumulator[t0 : t0 + VEC_M_TILE, n0 : n0 + column_tile]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0 : n0 + column_tile], [1, column_tile])
            scale = pl.row_min(token_scale[t0 : t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, column_tile], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.create_tensor([VEC_M_TILE, column_tile], dtype=pl.FP32)
                scaled_channels[:, :] = pl.col_expand_mul(value, channels)
                rounded_channels = pl.create_tensor([VEC_M_TILE, column_tile], dtype=channel_dtype)
                rounded_channels[:, :] = narrow_channel(scaled_channels)
                channel_fp32 = widen_channel(rounded_channels)
                scaled = pl.row_expand_mul(channel_fp32, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0 : t0 + VEC_M_TILE, n0 : n0 + column_tile] = pl.set_validshape(narrowed, rows, column_tile)
        return output

    return dequantize


def make_decode_dequantize(capacity, output_width, combine_scales=False, output_dtype=pl.BF16):
    """Apply the selected native FP32 scale boundary to INT32 products.

    The native per-token optimized kernel combines channel/token scales before
    multiplying the accumulator. Other native kernels scale channel-first.
    Callers select their source boundary explicitly; QKV keeps its existing ABI.
    """
    if output_dtype == pl.FP32:

        @pl.jit.inline
        def dequantize(
            accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
            channel_scale: pl.Tensor[[output_width], pl.FP32],
            token_scale: pl.Tensor[[capacity, 8], pl.FP32],
            output: pl.Tensor[[capacity, output_width], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
            for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_dequantize_fp32"):
                t0 = (block // (output_width // N_TILE)) * VEC_M_TILE
                n0 = (block % (output_width // N_TILE)) * N_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                integer = accumulator[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE]
                value = pl.cast(integer, pl.FP32, mode="rint")
                channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
                scale = pl.row_min(token_scale[t0 : t0 + VEC_M_TILE, :])
                if combine_scales:
                    ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                    channel_rows = pl.col_expand_mul(ones, channels)
                    combined = pl.row_expand_mul(channel_rows, scale)
                    scaled = pl.mul(value, combined)
                else:
                    scaled_channels = pl.col_expand_mul(value, channels)
                    scaled = pl.row_expand_mul(scaled_channels, scale)
                output[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(scaled, rows, N_TILE)
            return output

        return dequantize

    @pl.jit.inline
    def dequantize(
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        token_scale: pl.Tensor[[capacity, 8], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        row_blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
        for block in pl.spmd(row_blocks * (output_width // N_TILE), name_hint="a2_dequantize"):
            t0 = (block // (output_width // N_TILE)) * VEC_M_TILE
            n0 = (block % (output_width // N_TILE)) * N_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            integer = accumulator[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE]
            value = pl.cast(integer, pl.FP32, mode="rint")
            channels = pl.reshape(channel_scale[n0 : n0 + N_TILE], [1, N_TILE])
            scale = pl.row_min(token_scale[t0 : t0 + VEC_M_TILE, :])
            if combine_scales:
                ones = pl.full([VEC_M_TILE, N_TILE], dtype=pl.FP32, value=1.0)
                channel_rows = pl.col_expand_mul(ones, channels)
                combined = pl.row_expand_mul(channel_rows, scale)
                scaled = pl.mul(value, combined)
            else:
                scaled_channels = pl.col_expand_mul(value, channels)
                scaled = pl.row_expand_mul(scaled_channels, scale)
            narrowed = pl.cast(scaled, pl.BF16, mode="rint")
            output[t0 : t0 + VEC_M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(narrowed, rows, N_TILE)
        return output

    return dequantize


def _round_symmetric_int8(value: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(value).clamp(-CONTRACT.int8_scale_max, CONTRACT.int8_scale_max)
    return rounded.to(torch.int32).to(torch.float16).to(torch.int8)


def quantize_per_token_int8(
    value: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row to INT8 and return an FP32 dequant scale.

    This mirrors ``torch_npu.npu_dynamic_quant`` used by
    ``AscendW8A8DynamicLinearMethod``.  The returned scale has a trailing
    singleton dimension and is suitable for broadcasting over the row.
    """

    source = value.float()
    if smooth_scale is not None:
        source = source * smooth_scale.float()
    amax = source.abs().amax(dim=-1, keepdim=True).clamp_min(CONTRACT.int8_amax_epsilon)
    dequant_scale = amax / CONTRACT.int8_scale_max
    payload = _round_symmetric_int8(source / dequant_scale)
    return payload, dequant_scale


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an output-major ``[N, K]`` weight with FP32 channel scales."""

    source = weight.float()
    amax = source.abs().amax(dim=-1, keepdim=True).clamp_min(CONTRACT.int8_amax_epsilon)
    dequant_scale = amax / CONTRACT.int8_scale_max
    payload = _round_symmetric_int8(source / dequant_scale)
    return payload, dequant_scale


def int8_matmul(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``[..., K] @ [N, K].T`` with an INT32 accumulator."""

    if value.shape[-1] != weight.shape[-1]:
        raise ValueError(f"K mismatch: activation={value.shape[-1]}, weight={weight.shape[-1]}")
    if value.shape[-1] % CONTRACT.int8_k_alignment:
        raise ValueError(
            f"A2/A3 INT8 GEMM requires K aligned to {CONTRACT.int8_k_alignment}, got {value.shape[-1]}"
        )
    return torch.matmul(value.to(torch.int32), weight.to(torch.int32).transpose(-2, -1))


def dequantize_int32(
    accumulator: torch.Tensor,
    token_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Apply FP32 token/channel scales after INT32 accumulation."""

    result = accumulator.float() * token_scale.float()
    result = result * channel_scale.float().reshape(*([1] * (result.ndim - 1)), -1)
    return result.to(out_dtype)


def w8a8_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference for dynamic W8A8 linear on A2/A3."""

    payload, token_scale = quantize_per_token_int8(value, smooth_scale)
    accumulator = int8_matmul(payload, weight)
    result = dequantize_int32(accumulator, token_scale, weight_scale, out_dtype)
    if bias is not None:
        result = (result.float() + bias.float()).to(out_dtype)
    return result
