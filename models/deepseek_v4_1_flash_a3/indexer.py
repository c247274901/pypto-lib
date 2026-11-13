# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Indexer query, candidate pooling, causal selection and cache operations."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
import torch
from .gemm import make_dot
from .operator_config import CONTRACT
from .qkv_proj_rope import make_decode_bf16_projection_staged as make_bf16_projection_staged
from .quant import make_decode_dequantize as make_dequantize, make_decode_dynamic_int8_quantize as decode_make_dynamic_int8_quantize, make_decode_quantize as make_quantize, make_dynamic_int8_quantize, w8a8_linear
from .rope import decode_rope_reference, make_decode_adjacent_rope as decode_indexer_projection_make_rope, make_decode_rope as decode_make_rope, make_rope, rope_reference


# ``npu_dynamic_quant`` is nearest rounding for ordinary values, but an exact
# half-way value is rounded toward zero on the A2/A3 kernel.  Keep the small
# bias in FP32 so the published FP16 scale remains unchanged.
_ROUND_TIE_EPS = 2.0**-20


# ``npu_dynamic_quant`` is nearest rounding for ordinary values, but an exact
# half-way value is rounded toward zero on the A2/A3 kernel.  Keep the small
# bias in FP32 so the published FP16 scale remains unchanged.
DECODE__ROUND_TIE_EPS = 2.0**-20


INDEXER_TOPK = 512
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


INDEX_HEAD_DIM = 128
INDEX_HEADS = 32
INDEX_ROPE_DIM = 64
INDEX_SCALE_MIN = 2.0**-24


DECODE_INDEX_SCALE_MIN = 2.0**-24


# model config
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 2048
SCORE_WIDTH = 2048
POOL_CAPACITY = SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


DECODE_SCORE_WIDTH = 512
DECODE_POOL_CAPACITY = DECODE_SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


INDEX_TOPK = 512
KEY_CAPACITY = 512
KEY_TILE = 32
HEAD_TILE = 16
INDEXER_SELECT_KEY_CAPACITY = 2048
SORT_TILE = 2048
INDEX_WIDTH = 128
WINDOW_SIZE = 128

# Tiling.
QUANT_ROWS_TILE = 16
WEIGHT_ROWS_TILE = 8


def quantize_index_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., 128]`` index vectors using FP16 dequant scales."""

    if value.shape[-1] != CONTRACT.index_head_dim:
        raise ValueError(f"index cache width must be {CONTRACT.index_head_dim}")
    source = value.float()
    amax = source.abs().amax(dim=-1, keepdim=True)
    # vLLM-Ascend computes the INT8 payload with the unrounded FP32 scale and
    # only converts the published row scale to FP16.  Reusing the rounded
    # FP16 scale for the payload changes exact half-way cases (for example
    # -63.5) and causes a one-code mismatch with ``npu_dynamic_quant``.
    raw_scale = torch.maximum(
        amax / CONTRACT.int8_scale_max,
        torch.full_like(amax, CONTRACT.int8_amax_epsilon),
    )
    scale = torch.maximum(
        raw_scale.to(torch.float16), torch.full_like(raw_scale, 2.0**-24, dtype=torch.float16)
    )
    normalized = source / raw_scale
    epsilon = torch.maximum(normalized.abs(), torch.ones_like(normalized)) * _ROUND_TIE_EPS
    # The A2/A3 kernel's observed tie behavior is asymmetric: positive
    # half-way values round down while negative values retain the normal rint
    # rule.  This only affects exact half-way cases.
    rounded = torch.where(
        normalized >= 0,
        torch.round(normalized - epsilon),
        torch.round(normalized),
    )
    payload = rounded.clamp(-127, 127).to(torch.int8)
    return payload, scale.squeeze(-1)


def dequantize_index_cache(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return payload.float() * scale.float().unsqueeze(-1)


def make_index_cache_quantize(capacity: int, heads: int, head_dim: int = 128):
    if head_dim != CONTRACT.index_head_dim:
        raise ValueError("A2/A3 index cache uses 128-wide heads")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        scales: pl.Tensor[[capacity, heads], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [capacity * heads, head_dim])
        output = pl.reshape(payload, [capacity * heads, head_dim])
        rows = (active * heads + 15) // 16
        with pl.spmd(rows, name_hint="dsv41_a2a3_index_cache_quant"):
            block = pl.tile.get_block_idx()
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.maximum(pl.div(amax, 127.0), 1.0e-30)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0**-24)
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            # Match the A2/A3 dynamic-quant tie rule: nearest for ordinary
            # values, half-way values toward zero.  Splitting signs keeps the
            # conversion in truncation mode, which is legal for INT32 output.
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), _ROUND_TIE_EPS)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            rounded = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row : row + 16, :] = pl.set_validshape(rounded, valid_rows, head_dim)
            head = row % heads
            token = row // heads
            scale_row = pl.reshape(pl.cast(scale, pl.FP16), [1, 16])
            scales[token : token + 1, head : head + 16] = scale_row
        return payload, scales

    return quantize


def make_index_quantize(capacity: int, heads: int, head_dim: int = 128):
    quantize = make_index_cache_quantize(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def decode_quantize_index_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., 128]`` index vectors using FP16 dequant scales."""

    if value.shape[-1] != CONTRACT.index_head_dim:
        raise ValueError(f"index cache width must be {CONTRACT.index_head_dim}")
    source = value.float()
    amax = source.abs().amax(dim=-1, keepdim=True)
    # vLLM-Ascend computes the INT8 payload with the unrounded FP32 scale and
    # only converts the published row scale to FP16.  Reusing the rounded
    # FP16 scale for the payload changes exact half-way cases (for example
    # -63.5) and causes a one-code mismatch with ``npu_dynamic_quant``.
    raw_scale = torch.maximum(
        amax / CONTRACT.int8_scale_max,
        torch.full_like(amax, CONTRACT.int8_amax_epsilon),
    )
    scale = torch.maximum(
        raw_scale.to(torch.float16), torch.full_like(raw_scale, 2.0**-24, dtype=torch.float16)
    )
    normalized = source / raw_scale
    epsilon = torch.maximum(normalized.abs(), torch.ones_like(normalized)) * DECODE__ROUND_TIE_EPS
    # The A2/A3 kernel's observed tie behavior is asymmetric: positive
    # half-way values round down while negative values retain the normal rint
    # rule.  This only affects exact half-way cases.
    rounded = torch.where(
        normalized >= 0,
        torch.round(normalized - epsilon),
        torch.round(normalized),
    )
    payload = rounded.clamp(-127, 127).to(torch.int8)
    return payload, scale.squeeze(-1)


def make_decode_index_cache_quantize(capacity: int, heads: int, head_dim: int = 128):
    if head_dim != CONTRACT.index_head_dim:
        raise ValueError("A2/A3 index cache uses 128-wide heads")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        scales: pl.Tensor[[capacity, heads], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [capacity * heads, head_dim])
        output = pl.reshape(payload, [capacity * heads, head_dim])
        rows = (active * heads + 15) // 16
        with pl.spmd(rows, name_hint="dsv41_a2a3_index_cache_quant"):
            block = pl.tile.get_block_idx()
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.maximum(pl.div(amax, 127.0), 1.0e-30)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0**-24)
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            # Match the A2/A3 dynamic-quant tie rule: nearest for ordinary
            # values, half-way values toward zero.  Splitting signs keeps the
            # conversion in truncation mode, which is legal for INT32 output.
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), DECODE__ROUND_TIE_EPS)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            rounded = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row : row + 16, :] = pl.set_validshape(rounded, valid_rows, head_dim)
            head = row % heads
            token = row // heads
            scale_row = pl.reshape(pl.cast(scale, pl.FP16), [1, 16])
            scales[token : token + 1, head : head + 16] = scale_row
        return payload, scales

    return quantize


def make_indexer_prepare(
    capacity: int,
    topk: int = INDEXER_TOPK,
    *,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket A2/A3 index preparation kernel."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    if topk <= 0 or topk % 32 or topk > SORT_LEAF:
        raise ValueError("topk must be a positive multiple of 32 and <= 512")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit
    def kernel(
        raw_selected: pl.Tensor[[capacity, topk], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="dsv41_a2a3_prepare_indexer_indices"):
            token = pl.tile.get_block_idx()
            position = pl.max(pl.read(positions, [token]), 0)
            visible = pl.min((position + 1) // compress_ratio, topk)
            # sort32 is descending.  Negated logical IDs therefore produce
            # ascending chronological IDs, while the very negative sentinel
            # remains at the tail.
            scores = pl.tile.full([1, topk], dtype=pl.FP32, value=NEGATIVE_INF)
            ids = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
            for lane in pl.range(topk):
                index = pl.read(raw_selected, [token, lane])
                if index >= 0:
                    if index < visible:
                        pl.tile.write(scores, [0, lane], -pl.cast(index, pl.FP32))
                        pl.tile.write(ids, [0, lane], index)
            pairs = pl.tile.sort32(scores, pl.reinterpret_view(ids, pl.UINT32))
            if topk >= 128:
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
            ordered = pl.tile.slice(pairs, [1, 2 * topk], [0, 0])
            output = pl.tile.gather_mask(
                ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            pl.store(output, [token, 0], indices)
        return indices

    return kernel


def _make_indexer_query_quantize_inline(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build the PyPTO per-head INT8 query quantizer used by QLI."""

    if capacity % 16 or heads <= 0 or head_dim != INDEX_HEAD_DIM:
        raise ValueError("indexer quantization requires capacity divisible by 16 and head_dim=128")
    total_rows = capacity * heads

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [total_rows, head_dim])
        output = pl.reshape(payload, [total_rows, head_dim])
        scale_output = pl.reshape(scales, [total_rows, 1])
        blocks = (active * heads + 15) // 16
        for block in pl.spmd(blocks, name_hint="dsv41_a2a3_indexer_query_quant"):
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.div(amax, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            rounded_scale = pl.maximum(rounded_scale, INDEX_SCALE_MIN)
            normalized = pl.row_expand_div(value_fp32, rounded_scale)
            rounded = pl.cast(
                pl.minimum(pl.maximum(normalized, -127.0), 127.0),
                pl.INT32,
                mode="rint",
            )
            output[row : row + 16, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), valid_rows, head_dim
            )
            scale_output[row : row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(rounded_scale, pl.FP16, mode="rint"), [16, 1]),
                valid_rows,
                1,
            )
        return payload, scales

    return quantize


def make_indexer_query_quantize(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build a standalone PyPTO indexer quantization entry point."""

    quantize = _make_indexer_query_quantize_inline(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def make_l3_indexer_query(
    capacity: int,
    q_lora_rank: int = 1280,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
):
    """Project ``q_lora`` and publish the QLI INT8 query boundary.

    ``weight`` is canonical A2/A3 ``[K, N]`` INT8 and ``weight_scale`` is
    FP32 per output channel.  ``query`` remains BF16 for diagnostics while
    ``payload``/``scales`` are the exact QLI ABI.
    """

    if heads * head_dim != 4096 or rope_dim != INDEX_ROPE_DIM:
        raise ValueError("V4.1 A2/A3 indexer uses 32 heads x 128 with a 64-wide RoPE tail")
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize = _make_indexer_query_quantize_inline(capacity, heads, head_dim)
    quantize_input = make_dynamic_int8_quantize(capacity, q_lora_rank)
    pairs = rope_dim // 2
    width = heads * head_dim
    blocks = width // 64

    @pl.jit.inline
    def project(
        value: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, q_lora_rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize_input(value, payload, token_scale, active)
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l3_indexer_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * 64
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, 64], dtype=pl.INT32)
            for k0 in pl.pipeline(0, q_lora_rank, 256, stage=2):
                lhs = pl.slice(payload, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0 : k0 + 256, n0 : n0 + 64]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            value_fp32 = pl.col_expand_mul(value_fp32, pl.reshape(weight_scale[n0 : n0 + 64], [1, 64]))
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + 16, :])
            output[t0 : t0 + 16, n0 : n0 + 64] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, 64
            )
        return output

    @pl.jit
    def kernel(
        q_lora: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        projected = pl.create_tensor([capacity, width], dtype=pl.BF16)
        project(q_lora, weight, weight_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, width])
        rotate(projected, cosine, sine, query_flat, active)
        quantize(query, payload, scales, active)
        return query, payload, scales

    return kernel


def _decode_make_indexer_query_quantize_inline(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build the PyPTO per-head INT8 query quantizer used by QLI."""

    if capacity % 16 or heads <= 0 or head_dim != INDEX_HEAD_DIM:
        raise ValueError("indexer quantization requires capacity divisible by 16 and head_dim=128")
    total_rows = capacity * heads

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [total_rows, head_dim])
        output = pl.reshape(payload, [total_rows, head_dim])
        scale_output = pl.reshape(scales, [total_rows, 1])
        blocks = (active * heads + 15) // 16
        for block in pl.spmd(blocks, name_hint="dsv41_a2a3_indexer_query_quant"):
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.div(amax, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            rounded_scale = pl.maximum(rounded_scale, DECODE_INDEX_SCALE_MIN)
            normalized = pl.row_expand_div(value_fp32, rounded_scale)
            rounded = pl.cast(
                pl.minimum(pl.maximum(normalized, -127.0), 127.0),
                pl.INT32,
                mode="rint",
            )
            output[row : row + 16, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), valid_rows, head_dim
            )
            scale_output[row : row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(rounded_scale, pl.FP16, mode="rint"), [16, 1]),
                valid_rows,
                1,
            )
        return payload, scales

    return quantize


def make_decode_indexer_query_quantize(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build a standalone PyPTO indexer quantization entry point."""

    quantize = _decode_make_indexer_query_quantize_inline(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def make_decode_l3_indexer_query(
    capacity: int,
    q_lora_rank: int = 1280,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
):
    """Project ``q_lora`` and publish the QLI INT8 query boundary.

    ``weight`` is canonical A2/A3 ``[K, N]`` INT8 and ``weight_scale`` is
    FP32 per output channel.  ``query`` remains BF16 for diagnostics while
    ``payload``/``scales`` are the exact QLI ABI.
    """

    if heads * head_dim != 4096 or rope_dim != INDEX_ROPE_DIM:
        raise ValueError("V4.1 A2/A3 indexer uses 32 heads x 128 with a 64-wide RoPE tail")
    rotate = decode_make_rope(capacity, heads, head_dim, rope_dim)
    quantize = _decode_make_indexer_query_quantize_inline(capacity, heads, head_dim)
    quantize_input = decode_make_dynamic_int8_quantize(capacity, q_lora_rank)
    pairs = rope_dim // 2
    width = heads * head_dim
    blocks = width // 64

    @pl.jit.inline
    def project(
        value: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, q_lora_rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize_input(value, payload, token_scale, active)
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l3_indexer_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * 64
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, 64], dtype=pl.INT32)
            for k0 in pl.pipeline(0, q_lora_rank, 256, stage=2):
                lhs = pl.slice(payload, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0 : k0 + 256, n0 : n0 + 64]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            value_fp32 = pl.col_expand_mul(value_fp32, pl.reshape(weight_scale[n0 : n0 + 64], [1, 64]))
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0 : t0 + 16, :])
            output[t0 : t0 + 16, n0 : n0 + 64] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, 64
            )
        return output

    @pl.jit
    def kernel(
        q_lora: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        projected = pl.create_tensor([capacity, width], dtype=pl.BF16)
        project(q_lora, weight, weight_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, width])
        rotate(projected, cosine, sine, query_flat, active)
        quantize(query, payload, scales, active)
        return query, payload, scales

    return kernel


def make_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by Prefill."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 2048-row score bucket")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("A2/A3 candidate blocks contain eight compressed rows")

    @pl.jit
    def kernel(
        scores: pl.Tensor[[capacity, score_width], pl.FP32],
        compressed_lengths: pl.Tensor[[capacity], pl.INT32],
        pool_block_ids: pl.Out[pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32]],
        pool_block_counts: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_candidate_pool"):
            token = pl.tile.get_block_idx()
            if token < active:
                raw = pl.load(scores, [token, 0], [1, score_width])
                grouped = pl.reshape(raw, [POOL_CAPACITY, CANDIDATE_BLOCK_SIZE])
                reduce_tmp = pl.tile.create([POOL_CAPACITY, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
                maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, POOL_CAPACITY])
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    POOL_CAPACITY,
                )
                lanes = pl.tile.arange(0, [1, POOL_CAPACITY], dtype=pl.INT32)
                # Build masks and IDs with vector operations. Scalar tile writes
                # can race the following sort on A3.
                last_block = pl.cast(visible_blocks - 1, pl.INT32)
                invalid = pl.minimum(pl.maximum(pl.sub(lanes, last_block), 0), 1)
                one = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=1)
                valid = pl.sub(one, invalid)
                maxima = pl.add(
                    pl.mul(maxima, pl.cast(valid, pl.FP32)), pl.mul(pl.cast(invalid, pl.FP32), -1.0e30)
                )
                delta = pl.sub(lanes, last_block)
                distance = pl.maximum(delta, pl.mul(delta, -1))
                pinned = pl.sub(one, pl.minimum(distance, 1))
                maxima = pl.add(
                    pl.mul(maxima, pl.cast(pl.sub(one, pinned), pl.FP32)),
                    pl.mul(pl.cast(pinned, pl.FP32), 1.0e30),
                )
                logical_ids = pl.sub(lanes, pl.mul(pl.add(lanes, 1), invalid))
                sort_scores = pl.tile.full([1, POOL_CAPACITY], dtype=pl.FP32, value=-1.0e30)
                sort_scores = pl.tile.assemble(sort_scores, maxima, [0, 0])
                sort_ids = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                sort_ids = pl.tile.assemble(sort_ids, logical_ids, [0, 0])
                sort_tmp = pl.tile.create([1, 2 * POOL_CAPACITY], dtype=pl.FP32)
                unsigned_ids = pl.reinterpret_view(sort_ids, pl.UINT32)
                pairs = pl.tile.sort32(sort_scores, unsigned_ids, tmp=sort_tmp)
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
                leading_pairs = pl.tile.slice(pairs, [1, 2 * POOL_CAPACITY], [0, 0])
                sorted_ids = pl.tile.gather_mask(
                    leading_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                )
                # Native A3 preserves score order, with the newest block first.
                pl.store(sorted_ids, [token, 0], pool_block_ids)
            else:
                empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], pool_block_ids)
        # Scalar INT32 stores from one worker per token can share a cache line
        # on A2/A3.  Publish the compact count vector from a single worker.
        with pl.spmd(1, name_hint="dsv41_a2a3_candidate_counts"):
            worker = pl.tile.get_block_idx()
            if worker == 0:
                for token in pl.range(capacity):
                    length = pl.max(pl.read(compressed_lengths, [token]), 0)
                    visible_blocks = pl.min(
                        (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                        POOL_CAPACITY,
                    )
                    count = pl.min(visible_blocks, POOL_CAPACITY)
                    if token >= active:
                        count = 0
                    pl.write(pool_block_counts, [token], pl.cast(count, pl.INT32))
        return pool_block_ids, pool_block_counts

    return kernel


def make_candidate_pool_publish(capacity: int, shared_rows: int = 8192):
    """Publish compact candidate blocks into active persistent-state rows."""

    @pl.jit
    def candidate_pool_publish(
        blocks: pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32],
        counts: pl.Tensor[[capacity], pl.INT32],
        shared_blocks: pl.InOut[pl.Tensor[[shared_rows, 1, CANDIDATE_TOPK_BLOCKS], pl.INT32]],
        shared_counts: pl.InOut[pl.Tensor[[shared_rows, 1], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(shared_blocks, [shared_rows, CANDIDATE_TOPK_BLOCKS])
        for section in pl.spmd(
            capacity * (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY), name_hint="prefill_candidate_blocks_publish"
        ):
            token = section // (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY)
            column = section % (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY) * POOL_CAPACITY
            if token < active:
                if column == 0:
                    ids = pl.load(blocks, [token, 0], [1, POOL_CAPACITY])
                    pl.store(ids, [token, 0], flat)
                else:
                    empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                    pl.store(empty, [token, column], flat)
        # The pinned A3 native QLI returns candidates but leaves this persistent
        # length buffer unchanged on its nonempty source path.
        return shared_blocks, shared_counts

    return candidate_pool_publish


def make_decode_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = DECODE_SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by the decoder stages."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != DECODE_SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 512-row score bucket")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("A2/A3 candidate blocks contain eight compressed rows")

    @pl.jit
    def kernel(
        scores: pl.Tensor[[capacity, score_width], pl.FP32],
        compressed_lengths: pl.Tensor[[capacity], pl.INT32],
        pool_block_ids: pl.Out[pl.Tensor[[capacity, DECODE_POOL_CAPACITY], pl.INT32]],
        pool_block_counts: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_candidate_pool"):
            token = pl.tile.get_block_idx()
            if token < active:
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    DECODE_POOL_CAPACITY,
                )
                # Candidate publication retains every visible causal block.  A
                # score sort is unnecessary here because the pool capacity is
                # the full visible prefix; sorting the mixed FP32/UINT32 pair
                # on A2/A3 also reinterprets the score sentinel as a block ID.
                ordered_ids = pl.tile.full([1, DECODE_POOL_CAPACITY], dtype=pl.INT32, value=-1)
                for lane in pl.range(DECODE_POOL_CAPACITY):
                    if lane < visible_blocks:
                        pl.tile.write(ordered_ids, [0, lane], pl.cast(lane, pl.INT32))
                pl.store(ordered_ids, [token, 0], pool_block_ids)
            else:
                empty = pl.tile.full([1, DECODE_POOL_CAPACITY], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], pool_block_ids)
        # Scalar INT32 stores from one worker per token can share a cache line
        # on A2/A3.  Publish the compact count vector from a single worker.
        with pl.spmd(1, name_hint="dsv41_a2a3_candidate_counts"):
            worker = pl.tile.get_block_idx()
            if worker == 0:
                for token in pl.range(capacity):
                    length = pl.max(pl.read(compressed_lengths, [token]), 0)
                    visible_blocks = pl.min(
                        (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                        DECODE_POOL_CAPACITY,
                    )
                    count = pl.min(visible_blocks, DECODE_POOL_CAPACITY)
                    if token >= active:
                        count = 0
                    pl.write(pool_block_counts, [token], pl.cast(count, pl.INT32))
        return pool_block_ids, pool_block_counts

    return kernel


def make_indexer_select(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket PyPTO QLI score/Top-K kernel."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    # The native A2/A3 QLI handoff allocates one 512-entry candidate row.
    # Smaller visible prefixes are represented by score padding and are
    # filtered by ``prepare_indexer_indices`` in the next boundary.  The
    # fixed signature is deliberate: larger or variable rows are not stable
    # on the current A2/A3 sort32/mrgsort backend.
    if key_capacity != KEY_CAPACITY:
        raise ValueError(f"A2/A3 QLI uses fixed key_capacity={KEY_CAPACITY}")
    if topk != INDEX_TOPK:
        raise ValueError(f"A2/A3 QLI uses fixed topk={INDEX_TOPK}")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit.incore
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ) -> None:
        token = pl.tile.get_block_idx()
        # The A2/A3 QLI cache ABI is one fixed 512-entry candidate row.  Sort
        # the complete row here; ``prepare_indexer_indices`` applies the
        # request-specific visible-prefix filter before cache gather.  Keeping
        # this boundary fixed avoids dynamic tile tail writes on A2/A3 while
        # preserving the final selected set.
        visible = key_capacity
        # QLI writes a score tile for every visible key.  Invalid rows never
        # enter the sort, and score padding stays at the hardware minimum.
        # Materialize that floor explicitly: dynamic ``valid_shape`` on a
        # sort32 input is not reliable on the A2/A3 backend for partial rows.
        pl.store(
            pl.tile.full([1, key_capacity], dtype=pl.FP32, value=NEGATIVE_INF),
            [token, 0],
            scores,
        )
        for k0 in pl.range(0, visible, KEY_TILE):
            valid = pl.min(KEY_TILE, visible - k0)
            key_tile = key[k0 : k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0 : k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token : token + 1, h0 : h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                dots = pl.row_expand_mul(dots, q_scale_tile)
                dots = pl.col_expand_mul(dots, scale_tile)
                dots = pl.maximum(dots, 0.0)
                dots = pl.row_expand_mul(dots, weight_tile)
                total = pl.add(total, pl.reshape(pl.col_sum(dots), [1, KEY_TILE]))
            scores[token : token + 1, k0 : k0 + KEY_TILE] = pl.fillpad(
                pl.set_validshape(total, 1, valid), pad_value=pl.PadValue.min
            )

        # Match QLI's score-order result with a bounded merge sort.  The
        # payload carries absolute key positions, so no host-side offset is
        # needed for the fixed flattened cache used by the A2/A3 path.
        # The A2/A3 ABI uses one 512-entry QLI candidate row.  Always execute
        # the fixed sort once and represent a short visible prefix with the
        # explicit score floor above; a dynamic ``pl.range(leaf_count)`` leaves
        # the pair arena uninitialized for partial prefixes on A2/A3.
        leaf0 = 0
        row_raw = pl.load(scores, [token, leaf0], [1, SORT_LEAF])
        row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
        row = pl.maximum(row, pl.tile.full([1, SORT_LEAF], dtype=pl.FP32, value=NEGATIVE_INF))
        ramp = pl.tile.arange(0, [1, SORT_LEAF], dtype=pl.INT32)
        leaf_indices = pl.tile.add(ramp, pl.cast(leaf0, pl.INT32))
        sort_tmp = pl.tile.create([1, 2 * SORT_LEAF], dtype=pl.FP32)
        leaf_pairs = pl.tile.sort32(row, pl.reinterpret_view(leaf_indices, pl.UINT32), tmp=sort_tmp)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
        leaf_pairs = pl.tile.slice(leaf_pairs, [1, 2 * topk], [0, 0])
        pl.store(leaf_pairs, [token, 0], running_pairs)

        # Materialize the complete padding row in one UB tile.  Explicitly
        # initialize every lane because the backend may preserve stale pair
        # lanes when only a dynamic prefix is written.
        selected = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
        if visible > 0:
            pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
            picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected = picked
        pl.store(selected, [token, 0], indices)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        with pl.spmd(active, name_hint="dsv41_a2a3_qli_select"):
            select_one(
                query,
                query_scale,
                key,
                key_scale,
                weights,
                positions,
                running_pairs,
                scores,
                indices,
            )
        return scores, indices

    return kernel


def make_indexer_select_restricted(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 1,
    candidate_block_size: int = 8,
    candidate_block_capacity: int = 64,
):
    """Select C1A QLI rows inside the source candidate block pool."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    if key_capacity != KEY_CAPACITY or topk != INDEX_TOPK:
        raise ValueError("A2/A3 restricted QLI uses fixed 512 rows and topk")
    if compress_ratio != 1 or candidate_block_size != 8:
        raise ValueError("C1A reindex uses ratio-1 eight-row candidate blocks")
    if candidate_block_capacity <= 0:
        raise ValueError("candidate block capacity must be positive")

    @pl.jit.incore
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        candidate_blocks: pl.Tensor[[capacity, candidate_block_capacity], pl.INT32],
        candidate_counts: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ) -> None:
        token = pl.tile.get_block_idx()
        pl.store(
            pl.tile.full([1, key_capacity], dtype=pl.FP32, value=NEGATIVE_INF),
            [token, 0],
            scores,
        )
        for k0 in pl.range(0, key_capacity, KEY_TILE):
            key_tile = key[k0 : k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0 : k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token : token + 1, h0 : h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0 : h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                dots = pl.row_expand_mul(dots, q_scale_tile)
                dots = pl.col_expand_mul(dots, scale_tile)
                dots = pl.maximum(dots, 0.0)
                dots = pl.row_expand_mul(dots, weight_tile)
                total = pl.add(total, pl.reshape(pl.col_sum(dots), [1, KEY_TILE]))
            visible = pl.min(pl.read(positions, [token]) + 1, key_capacity)
            for lane in pl.range(KEY_TILE):
                key_index = k0 + lane
                block_id = key_index // candidate_block_size
                allowed = False
                for candidate in pl.range(candidate_block_capacity):
                    if candidate < pl.read(candidate_counts, [token]):
                        if pl.read(candidate_blocks, [token, candidate]) == block_id:
                            allowed = True
                if key_index >= visible or not allowed:
                    pl.write(total, [0, lane], NEGATIVE_INF)
            scores[token : token + 1, k0 : k0 + KEY_TILE] = total

        row = pl.load(scores, [token, 0], [1, SORT_LEAF])
        row = pl.tile.fillpad(row, pad_value=pl.PadValue.min)
        ramp = pl.tile.arange(0, [1, SORT_LEAF], dtype=pl.INT32)
        leaf_pairs = pl.tile.sort32(
            row,
            pl.reinterpret_view(ramp, pl.UINT32),
            tmp=pl.tile.create([1, 2 * SORT_LEAF], dtype=pl.FP32),
        )
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
        leaf_pairs = pl.tile.slice(leaf_pairs, [1, 2 * topk], [0, 0])
        pl.store(leaf_pairs, [token, 0], running_pairs)
        pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
        picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        for lane in pl.range(topk):
            score = pl.read(running_pairs, [token, 2 * lane])
            if score <= NEGATIVE_INF / 2:
                pl.tile.write(picked, [0, lane], pl.cast(-1, pl.INT32))
        pl.store(picked, [token, 0], indices)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        candidate_blocks: pl.Tensor[[capacity, candidate_block_capacity], pl.INT32],
        candidate_counts: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        with pl.spmd(active, name_hint="dsv41_a2a3_qli_select_restricted"):
            select_one(
                query,
                query_scale,
                key,
                key_scale,
                weights,
                positions,
                candidate_blocks,
                candidate_counts,
                running_pairs,
                scores,
                indices,
            )
        return scores, indices

    return kernel


def make_prefill_indexer_select(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket PyPTO QLI score/Top-K kernel."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    if key_capacity not in (512, 1024, 2048):
        raise ValueError("A2/A3 QLI key_capacity must be 512, 1024 or 2048")
    if topk != INDEX_TOPK:
        raise ValueError(f"A2/A3 QLI uses fixed topk={INDEX_TOPK}")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    KEY_ROWS_TILE = 512
    SORT_ROWS_TILE = 512 if key_capacity == 512 else SORT_TILE

    @pl.jit.inline
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ):
        token = pl.tile.get_block_idx()
        position = pl.read(positions, [token])
        visible = pl.min(key_capacity, pl.max(0, (position + 1) // compress_ratio))
        q_tile = pl.reshape(query[token : token + 1, :, :], [heads, head_dim])
        q_scale_row = pl.cast(query_scale[token : token + 1, :], pl.FP32)
        weight_row = pl.cast(weights[token : token + 1, :], pl.FP32)
        combined = pl.mul(q_scale_row, weight_row)
        combined_half = pl.cast(combined, pl.FP16, mode="rint")
        weight_column = pl.reshape(pl.cast(combined_half, pl.FP32), [heads, 1])
        for k0 in pl.range(0, key_capacity, KEY_ROWS_TILE):
            key_tile = key[k0 : k0 + KEY_ROWS_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0 : k0 + KEY_ROWS_TILE], pl.FP32), [1, KEY_ROWS_TILE])
            dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
            dots_fp32 = pl.cast(dots, pl.FP32, mode="none")
            dots_scaled = pl.mul(pl.maximum(dots_fp32, 0.0), 0.0009765625)
            dots_half = pl.cast(dots_scaled, pl.FP16, mode="rint")
            weighted = pl.row_expand_mul(pl.cast(dots_half, pl.FP32), weight_column)
            reduced_row = pl.col_sum(weighted)
            total = pl.mul(reduced_row, scale_tile)
            scores[token : token + 1, k0 : k0 + KEY_ROWS_TILE] = total

        if visible <= 32:
            short_scores = pl.load(scores, [token, 0], [1, 32])
            short_absolute = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
            short_distance = pl.mul(pl.sub(short_absolute, pl.cast(visible, pl.INT32)), -1)
            short_valid = pl.minimum(pl.maximum(short_distance, 0), 1)
            short_invalid = pl.sub(pl.tile.full([1, 32], dtype=pl.INT32, value=1), short_valid)
            short_values = pl.mul(short_scores, pl.cast(short_valid, pl.FP32))
            short_padding = pl.mul(pl.cast(short_invalid, pl.FP32), NEGATIVE_INF)
            short_row = pl.add(short_values, short_padding)
            short_ids = pl.sub(short_absolute, pl.mul(pl.add(short_absolute, 1), short_invalid))
            short_tmp = pl.tile.create([1, 64], dtype=pl.FP32)
            short_pairs = pl.tile.sort32(short_row, pl.reinterpret_view(short_ids, pl.UINT32), tmp=short_tmp)
            short_selected = pl.tile.gather_mask(short_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(short_selected, [token, 0], indices)
            short_tail = pl.tile.full([1, topk - 32], dtype=pl.INT32, value=-1)
            pl.store(short_tail, [token, 32], indices)
        else:
            # Use the smallest complete four-way merge tree for this key bucket.
            actual_scores = pl.load(scores, [token, 0], [1, key_capacity])
            row_raw = pl.tile.full([1, SORT_ROWS_TILE], dtype=pl.FP32, value=NEGATIVE_INF)
            row_raw = pl.tile.assemble(row_raw, actual_scores, [0, 0])
            absolute = pl.tile.arange(0, [1, SORT_ROWS_TILE], dtype=pl.INT32)
            distance = pl.mul(pl.sub(absolute, pl.cast(visible, pl.INT32)), -1)
            valid = pl.minimum(pl.maximum(distance, 0), 1)
            invalid = pl.sub(pl.tile.full([1, SORT_ROWS_TILE], dtype=pl.INT32, value=1), valid)
            valid_scores = pl.mul(row_raw, pl.cast(valid, pl.FP32))
            invalid_scores = pl.mul(pl.cast(invalid, pl.FP32), NEGATIVE_INF)
            row = pl.add(valid_scores, invalid_scores)
            logical_ids = pl.sub(absolute, pl.mul(pl.add(absolute, 1), invalid))
            sort_tmp = pl.tile.create([1, 2 * SORT_ROWS_TILE], dtype=pl.FP32)
            pairs = pl.tile.sort32(row, pl.reinterpret_view(logical_ids, pl.UINT32), tmp=sort_tmp)
            pairs = pl.tile.mrgsort(pairs, block_len=64)
            pairs = pl.tile.mrgsort(pairs, block_len=256)
            if SORT_ROWS_TILE > 512:
                pairs = pl.tile.mrgsort(pairs, block_len=1024)
            leading = pl.tile.slice(pairs, [1, 2 * topk], [0, 0])
            selected = pl.tile.gather_mask(leading, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(selected, [token, 0], indices)
        return scores, indices

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        for _worker in pl.spmd(active, name_hint="dsv41_a2a3_qli_select", optimizations=[pl.cross_core_slot(slot_num=1)]):
            select_one(
                query,
                query_scale,
                key,
                key_scale,
                weights,
                positions,
                scores,
                indices,
            )
        return scores, indices

    return kernel


def indexer_active_compare(active, *, exact=False):
    """Compare the defined prefix; pure Out inactive rows have no preservation contract."""
    from golden.validation import ratio_allclose

    tolerance = 0.0 if exact else 0.05
    comparator = ratio_allclose(rtol=tolerance, atol=tolerance, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        if not active:
            return True, "No active pure-Out rows; inactive rows are undefined"
        return comparator(actual[:active], expected[:active], **context)

    return compare


def make_decode_index_quantize(capacity: int, heads: int, head_dim: int = 128):
    quantize = make_decode_index_cache_quantize(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def validate_prefill_select(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Check causal selection at supported key capacities and active-row boundaries."""
    from golden import run
    from itertools import chain

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    cases = (indexer_select_golden_cases(build=not cpu_only, key_capacity=keys) for keys in (512, 1024, 2048))
    for case in chain.from_iterable(cases):
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
                if key not in ("scores", "indices") and not torch.equal(values[key], before[key]):
                    raise RuntimeError("Indexer golden mutated input " + key)
            if active:
                scores = values["scores"][:active]
                ordered = torch.sort(scores, dim=1).values
                if not (torch.diff(ordered, dim=1) > 0).all():
                    raise RuntimeError("The exact-index fixture must contain no tied scores")
                ratio = int(name.split("ratio")[1].split("-")[0])
                expected = torch.full((active, INDEX_TOPK), -1, dtype=torch.int32)
                for token in range(active):
                    visible = min(len(values["key"]), max(0, (int(values["positions"][token]) + 1) // ratio))
                    count = min(visible, INDEX_TOPK)
                    expected[token, :count] = torch.arange(visible - 1, visible - count - 1, -1).int()
                if not torch.equal(values["indices"][:active], expected):
                    raise RuntimeError("Causal Top-512 selection changed its exact score ordering")
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Indexer golden produced nonfinite scores")
            records.append(
                dict(name=name, cpu_golden="passed", device="not_run", inactive_Out_rows="undefined")
            )
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
        records.append(
            dict(
                name=name,
                compiled=True,
                device="not_run" if compile_only else "passed",
                inactive_Out_rows="undefined",
            )
        )
    return records


def make_indexer_projection(capacity, config):
    """Project and rotate index queries and emit scaled FP16 head weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    pairs = rope_width // 2
    index_width = INDEX_HEADS * INDEX_WIDTH
    weights_scale = INDEX_WIDTH**-0.5 * INDEX_HEADS**-0.5
    quantize = make_quantize(capacity, latent)
    dot = make_dot(capacity, latent, index_width)
    dequantize = make_dequantize(capacity, index_width)
    rotate = decode_indexer_projection_make_rope(capacity, INDEX_HEADS, INDEX_WIDTH, rope_width)
    project_weights = make_bf16_projection_staged(hidden, INDEX_HEADS)

    @pl.jit
    def indexer_projection(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        index_wq_b: pl.Tensor[[latent * (index_width // 64), 64], pl.INT8],
        index_wq_b_scale: pl.Tensor[[index_width], pl.FP32],
        weights_proj: pl.Tensor[[hidden, INDEX_HEADS], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16]],
        weights: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        latent_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        latent_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(query_latent, latent_i8, latent_scale, active)
        accumulator = pl.create_tensor([capacity, index_width], dtype=pl.INT32)
        dot(latent_i8, index_wq_b, accumulator, active)
        projected = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        dequantize(accumulator, index_wq_b_scale, latent_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, index_width])
        rotate(projected, cosine, sine, query_flat, active)

        projected_weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
        project_weights(x, weights_proj, projected_weights, active)
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint="a2_index_weights"):
            block = pl.tile.get_block_idx()
            row = block * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            value = pl.slice(
                projected_weights, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS]
            )
            scaled = pl.mul(pl.cast(value, pl.FP32), weights_scale)
            narrowed = pl.cast(scaled, pl.FP16, mode="rint")
            weights[row : row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(narrowed, rows, INDEX_HEADS)
        return query, weights

    return indexer_projection


def make_indexer_query_quantize_fp16_scale(capacity):
    """Quantize each index head using its rounded FP16 scale."""

    @pl.jit
    def indexer_query_quantize(
        query: pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(query, [capacity * INDEX_HEADS, INDEX_WIDTH])
        output = pl.reshape(payload, [capacity * INDEX_HEADS, INDEX_WIDTH])
        blocks = (active * INDEX_HEADS + QUANT_ROWS_TILE - 1) // QUANT_ROWS_TILE
        with pl.spmd(blocks, name_hint="a2_index_query_quant"):
            block = pl.tile.get_block_idx()
            row = block * QUANT_ROWS_TILE
            rows = pl.min(QUANT_ROWS_TILE, active * INDEX_HEADS - row)
            q = pl.slice(source, [QUANT_ROWS_TILE, INDEX_WIDTH], [row, 0], valid_shape=[rows, INDEX_WIDTH])
            q = pl.set_validshape(pl.fillpad(q, pad_value=pl.PadValue.zero), QUANT_ROWS_TILE, INDEX_WIDTH)
            value = pl.cast(q, pl.FP32)
            maximum = pl.row_max(pl.abs(value))
            raw_scale = pl.div(maximum, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0**-24)
            normalized = pl.row_expand_div(value, scale)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            quantized = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row : row + QUANT_ROWS_TILE, :] = pl.set_validshape(quantized, rows, INDEX_WIDTH)
            scale_row = pl.reshape(pl.cast(scale, pl.FP16, mode="rint"), [1, QUANT_ROWS_TILE])
            token = row // INDEX_HEADS
            head = row % INDEX_HEADS
            scales[token : token + 1, head : head + QUANT_ROWS_TILE] = scale_row
        return payload, scales

    return indexer_query_quantize


def make_empty_selection(capacity):
    """Publish the source empty-cache selection sentinel."""

    @pl.jit
    def empty_selection(
        raw_selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_empty_sparse_selection"):
            token = pl.tile.get_block_idx()
            raw_selected[token : token + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        return raw_selected

    return empty_selection


def make_sparse_controls(
    capacity, requests, long_pages, window_pages, long_page_rows=64, *, compress_ratio=2
):
    """Filter and sort QLI logical IDs, then map both caches to physical rows."""
    if long_page_rows not in (64, 128):
        raise ValueError("A2 compressed page rows must match the actual 64/128-row cache ABI")
    if compress_ratio not in (1, 2):
        raise ValueError("Sparse controls require the physical source compression ratio 1 or 2")

    @pl.jit
    def sparse_controls(
        raw_selected: pl.Tensor[[capacity, INDEX_TOPK], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        long_block_table: pl.Tensor[[requests, long_pages], pl.INT32],
        window_block_table: pl.Tensor[[requests, window_pages], pl.INT32],
        selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        selected_lengths: pl.InOut[pl.Tensor[[capacity, 1], pl.INT32]],
        compressed_indices: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        window_indices: pl.InOut[pl.Tensor[[capacity, WINDOW_SIZE], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_prepare_sparse_indices") as controls_tid:
            token = pl.tile.get_block_idx()
            position = pl.cast(pl.read(positions, [token]), pl.INT32)
            visible = (position + 1) // compress_ratio
            request = pl.read(request_ids, [token])
            sort_scores = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=-1e30)
            sort_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                index = pl.read(raw_selected, [token, lane])
                if index >= 0:
                    if index < visible:
                        pl.tile.write(sort_scores, [0, lane], -pl.cast(index, pl.FP32))
                        pl.tile.write(sort_ids, [0, lane], index)
            sorted_pairs = pl.tile.sort32(sort_scores, pl.reinterpret_view(sort_ids, pl.UINT32))
            sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=256)
            logical_ids = pl.tile.gather_mask(
                sorted_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            pl.store(logical_ids, [token, 0], selected)
            physical_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                logical = pl.tile.read(logical_ids, [0, lane])
                if logical >= 0:
                    page = logical // long_page_rows
                    offset = logical % long_page_rows
                    block = pl.read(long_block_table, [request, page])
                    physical = pl.cast(block * long_page_rows + offset, pl.INT32)
                    pl.tile.write(physical_ids, [0, lane], physical)
            pl.store(physical_ids, [token, 0], compressed_indices)

            left = pl.max(0, position + 1 - WINDOW_SIZE)
            window_count = position + 1 - left
            window_ids = pl.tile.full([1, WINDOW_SIZE], dtype=pl.INT32, value=-1)
            for lane in pl.range(WINDOW_SIZE):
                if lane < window_count:
                    window_logical = left + lane
                    window_page = window_logical // WINDOW_SIZE
                    window_offset = window_logical % WINDOW_SIZE
                    window_block = pl.read(window_block_table, [request, window_page])
                    window_physical = pl.cast(window_block * WINDOW_SIZE + window_offset, pl.INT32)
                    pl.tile.write(window_ids, [0, lane], window_physical)
            pl.store(window_ids, [token, 0], window_indices)
        with pl.spmd(1, name_hint="a2_sparse_lengths_publish", deps=[controls_tid]):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                count = 0
                for lane in pl.range(INDEX_TOPK):
                    if pl.read(selected, [token, lane]) >= 0:
                        count = count + 1
                pl.write(selected_lengths, [token, 0], pl.cast(count, pl.INT32))
        return selected, selected_lengths, compressed_indices, window_indices

    return sparse_controls


def prepare_indexer_indices_reference(
    selected: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """CPU reference for vLLM's visible-prefix filter and chronological sort."""

    if selected.ndim != 2 or selected.dtype != torch.int32:
        raise ValueError("selected must be a 2-D INT32 tensor")
    if positions.shape != (selected.shape[0],) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a 1-D INT32/INT64 tensor")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    tokens, topk = selected.shape
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=topk)
    valid = selected.ge(0) & selected.lt(visible[:, None])
    sentinel = torch.iinfo(torch.int32).max
    sortable = torch.where(valid, selected, torch.full_like(selected, sentinel))
    ordered = sortable.sort(dim=-1, stable=True).values
    output = torch.where(ordered.eq(sentinel), torch.full_like(ordered, -1), ordered)
    lengths = valid.sum(dim=-1, keepdim=True, dtype=torch.int32)
    return {"indices": output, "lengths": lengths}


def quantize_indexer_query_reference(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for vLLM's A2/A3 ``quantize_indexer_query``.

    The scale is rounded to FP16 before it is used to form the INT8 payload;
    this differs from the index-cache writer, whose payload uses the raw FP32
    scale and publishes an FP16 copy.
    """

    if query.ndim != 3 or query.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("indexer query must have shape [tokens, heads, 128]")
    source = query.detach().float()
    scale = (source.abs().amax(dim=-1) / 127.0).to(torch.float16)
    scale = torch.maximum(scale, torch.full_like(scale, INDEX_SCALE_MIN, dtype=torch.float16))
    payload = torch.round(source / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload, scale


def indexer_query_reference(
    q_lora: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
) -> dict[str, torch.Tensor]:
    """CPU reference for the complete L3 indexer query boundary."""

    # The PyPTO ABI stores canonical [K, N], while the Torch helper consumes
    # output-major [N, K] weights.
    projected = w8a8_linear(q_lora, weight.transpose(0, 1).contiguous(), weight_scale)
    query = rope_reference(projected, cosine, sine, heads, head_dim, rope_dim)
    query = query.reshape(query.shape[0], heads, head_dim)
    payload, scales = quantize_indexer_query_reference(query)
    return {"query": query, "payload": payload, "scales": scales}


def decode_quantize_indexer_query_reference(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for vLLM's A2/A3 ``quantize_indexer_query``.

    The scale is rounded to FP16 before it is used to form the INT8 payload;
    this differs from the index-cache writer, whose payload uses the raw FP32
    scale and publishes an FP16 copy.
    """

    if query.ndim != 3 or query.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("indexer query must have shape [tokens, heads, 128]")
    source = query.detach().float()
    scale = (source.abs().amax(dim=-1) / 127.0).to(torch.float16)
    scale = torch.maximum(scale, torch.full_like(scale, DECODE_INDEX_SCALE_MIN, dtype=torch.float16))
    payload = torch.round(source / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload, scale


def decode_indexer_query_reference(
    q_lora: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
) -> dict[str, torch.Tensor]:
    """CPU reference for the complete L3 indexer query boundary."""

    # The PyPTO ABI stores canonical [K, N], while the Torch helper consumes
    # output-major [N, K] weights.
    projected = w8a8_linear(q_lora, weight.transpose(0, 1).contiguous(), weight_scale)
    query = decode_rope_reference(projected, cosine, sine, heads, head_dim, rope_dim)
    query = query.reshape(query.shape[0], heads, head_dim)
    payload, scales = decode_quantize_indexer_query_reference(query)
    return {"query": query, "payload": payload, "scales": scales}


def golden_candidate_block_pool(args):
    """Order visible logical blocks by maximum score, with the newest block first."""
    import torch

    args["pool_block_ids"].fill_(-1)
    args["pool_block_counts"].zero_()
    maxima = args["scores"].reshape(-1, POOL_CAPACITY, CANDIDATE_BLOCK_SIZE).amax(dim=-1)
    for token in range(int(args["active"])):
        count = min(POOL_CAPACITY, max(0, (int(args["compressed_lengths"][token]) + 7) // 8))
        if count:
            ordered = torch.argsort(maxima[token, : count - 1], descending=True)
            ids = torch.cat((torch.tensor([count - 1]), ordered)).int()
            args["pool_block_ids"][token, :count] = ids
            args["pool_block_counts"][token] = count


def golden_candidate_pool_publish(args):
    """Replace active candidate rows and preserve persistent count bytes and inactive rows."""
    active = int(args["active"])
    args["shared_blocks"][:active].fill_(-1)
    args["shared_blocks"][:active, 0, :POOL_CAPACITY] = args["blocks"][:active]


def indexer_select_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """CPU oracle for the A2/A3 QLI score and score-order Top-K contract.

    ``query`` is ``[T, H, 128]`` INT8, ``key`` is ``[K, 128]`` INT8, and the
    two scale tensors are FP16.  ``positions`` is the raw token position used
    by vLLM to derive the visible compressed-key prefix.
    """

    if query.ndim != 3 or tuple(query.shape[1:]) != (INDEX_HEADS, INDEX_HEAD_DIM):
        raise ValueError("query must have shape [T, 32, 128]")
    if key.ndim != 2 or key.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("key must have shape [K, 128]")
    tokens, heads, _ = query.shape
    key_rows = key.shape[0]
    if query_scale.shape != (tokens, heads) or weights.shape != (tokens, heads):
        raise ValueError("query_scale and weights must have shape [T, 32]")
    if key_scale.numel() != key_rows or positions.shape != (tokens,):
        raise ValueError("key_scale must be [K] and positions must be [T]")
    if not 1 <= topk <= key_rows:
        raise ValueError("topk must be in [1, key.shape[0]]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, key_rows))
    scores = torch.relu(dots)
    scores.mul_(weights.float().unsqueeze(-1))
    scores = scores.sum(dim=1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key_rows)
    for token in range(tokens):
        count = min(int(visible[token]), topk)
        if count:
            # QLI returns score order.  ``sorted=False`` matches the native
            # operator's contract; ties are not expected for random fixtures.
            picked = torch.topk(scores[token, : int(visible[token])], count, sorted=False).indices
            indices[token, :count] = picked.to(torch.int32)
    return {"scores": scores, "indices": indices}


def prefill_indexer_select_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """Ideal visibility-aware oracle without the executable kernel's FP16 boundaries.

    ``query`` is ``[T, H, 128]`` INT8, ``key`` is ``[K, 128]`` INT8, and the
    two scale tensors are FP16.  ``positions`` is the raw token position used
    by vLLM to derive the visible compressed-key prefix.
    """

    if query.ndim != 3 or tuple(query.shape[1:]) != (INDEX_HEADS, INDEX_HEAD_DIM):
        raise ValueError("query must have shape [T, 32, 128]")
    if key.ndim != 2 or key.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("key must have shape [K, 128]")
    tokens, heads, _ = query.shape
    key_rows = key.shape[0]
    if query_scale.shape != (tokens, heads) or weights.shape != (tokens, heads):
        raise ValueError("query_scale and weights must have shape [T, 32]")
    if key_scale.numel() != key_rows or positions.shape != (tokens,):
        raise ValueError("key_scale must be [K] and positions must be [T]")
    if not 1 <= topk <= key_rows:
        raise ValueError("topk must be in [1, key.shape[0]]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, key_rows))
    scores = torch.relu(dots)
    scores.mul_(weights.float().unsqueeze(-1))
    scores = scores.sum(dim=1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key_rows)
    for token in range(tokens):
        count = min(int(visible[token]), topk)
        if count:
            # QLI returns score order.  ``sorted=False`` matches the native
            # operator's contract; ties are not expected for random fixtures.
            picked = torch.topk(scores[token, : int(visible[token])], count, sorted=False).indices
            indices[token, :count] = picked.to(torch.int32)
    return {"scores": scores, "indices": indices}


def golden_indexer_select(tensors, *, compress_ratio=2):
    """Score all keys and select the causal Top-512 with both FP16 boundaries."""
    active = int(tensors["active"])
    if not 0 <= active <= tensors["query"].shape[0]:
        raise ValueError("Indexer active rows must fit the actual capacity")
    if not active:
        return
    key = tensors["key"].to(torch.int32)
    for token in range(active):
        dots = tensors["query"][token].to(torch.int32) @ key.T
        positive = dots.float().clamp_min(0.0) * 0.0009765625
        narrowed_dots = positive.to(torch.float16).float()
        combined = tensors["query_scale"][token].float() * tensors["weights"][token].float()
        narrowed_weight = combined.to(torch.float16).float()
        reduced = narrowed_weight[None] @ narrowed_dots
        scores = reduced[0] * tensors["key_scale"].float()
        tensors["scores"][token] = scores
        visible = min(len(key), max(0, (int(tensors["positions"][token]) + 1) // compress_ratio))
        count = min(INDEX_TOPK, visible)
        tensors["indices"][token].fill_(-1)
        tensors["indices"][token, :count] = torch.argsort(scores[:visible], descending=True)[:count].int()


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Check empty rows, partial blocks, 512/1024 boundaries and the 2048-row limit."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    capacity, shared_rows = 16, 32
    lengths = torch.tensor([0, 1, 7, 8, 9, 511, 512, 513, 1023, 1024, 1025, 2040, 2041, 2047, 2048, -1]).int()
    block_scores = (torch.arange(POOL_CAPACITY) * 73) % POOL_CAPACITY
    scores = (block_scores[:, None] + torch.arange(8)[None, :] / 16).flatten().repeat(capacity, 1)
    records = []
    for active in (0, 1, 7, capacity):
        pool_specs = [
            TensorSpec("scores", [capacity, SCORE_WIDTH], torch.float32, init_value=scores),
            TensorSpec("compressed_lengths", [capacity], torch.int32, init_value=lengths),
            TensorSpec("pool_block_ids", [capacity, POOL_CAPACITY], torch.int32, init_value=-53),
            TensorSpec("pool_block_counts", [capacity], torch.int32, init_value=-59),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        pool = {
            spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
            for spec in pool_specs
        }
        golden_candidate_block_pool(pool)
        for token in range(capacity):
            count = int(pool["pool_block_counts"][token])
            expected = min(POOL_CAPACITY, max(0, (int(lengths[token]) + 7) // 8)) if token < active else 0
            ids = pool["pool_block_ids"][token]
            if (
                count != expected
                or set(ids[:count].tolist()) != set(range(count))
                or (ids[count:] != -1).any()
            ):
                raise RuntimeError("Candidate pool lost a visible block or published an invalid block")
            if count and int(ids[0]) != count - 1:
                raise RuntimeError("Candidate pool did not pin the newest block")
        publish_specs = [
            TensorSpec("blocks", [capacity, POOL_CAPACITY], torch.int32, init_value=pool["pool_block_ids"]),
            TensorSpec("counts", [capacity], torch.int32, init_value=pool["pool_block_counts"]),
            TensorSpec("shared_blocks", [shared_rows, 1, CANDIDATE_TOPK_BLOCKS], torch.int32, init_value=-67),
            TensorSpec("shared_counts", [shared_rows, 1], torch.int32, init_value=-71),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        cases = [
            ("pool", pool_specs, golden_candidate_block_pool, make_candidate_block_pool),
            ("publish", publish_specs, golden_candidate_pool_publish, make_candidate_pool_publish),
        ]
        for name, specs, golden_fn, builder in cases:
            if cpu_only:
                values = {
                    spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
                    for spec in specs
                }
                golden_fn(values)
                if name == "publish":
                    if (
                        (values["shared_blocks"][active:] != -67).any()
                        or (values["shared_counts"] != -71).any()
                        or (values["shared_blocks"][:active, :, POOL_CAPACITY:] != -1).any()
                    ):
                        raise RuntimeError(
                            "Candidate publication changed readonly bytes or retained stale tail IDs"
                        )
            else:
                fn = builder(capacity) if name == "pool" else builder(capacity, shared_rows)
                result = run(
                    fn=fn,
                    specs=specs,
                    golden_fn=golden_fn,
                    config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
                    compile_only=compile_only,
                    rtol=0,
                    atol=0,
                )
                if not result.passed:
                    raise RuntimeError(f"candidate-{name}-active{active}: {result.error}")
            records.append(
                dict(
                    case=f"{name}-active{active}",
                    cpu_golden="passed" if cpu_only else "not_run",
                    kernel="not_run" if cpu_only or compile_only else "passed",
                    compiled=not cpu_only,
                )
            )
    return records


def indexer_select_golden_cases(*, build=True, key_capacity=INDEXER_SELECT_KEY_CAPACITY):
    """Cover short and long histories, causal boundaries and negative-dot heads."""
    from functools import partial
    from golden import ScalarSpec, TensorSpec

    capacity = 16
    rows = torch.arange(INDEXER_SELECT_KEY_CAPACITY, dtype=torch.int32)
    key = torch.zeros((INDEXER_SELECT_KEY_CAPACITY, INDEX_HEAD_DIM), dtype=torch.int8)
    key[:, 0] = (rows // 16).to(torch.int8)
    key[:, 1] = (rows % 16).to(torch.int8)
    query = torch.ones((capacity, INDEX_HEADS, INDEX_HEAD_DIM), dtype=torch.int8)
    query[:, :, 0] = 16
    query[:, ::3] *= -1
    heads = torch.arange(INDEX_HEADS, dtype=torch.float32)
    query_scale = ((heads % 7 + 8.0) / 17.0).repeat(capacity, 1).to(torch.float16)
    weights = ((heads % 5 + 3.0) / 11.0).repeat(capacity, 1).to(torch.float16)
    boundaries = [-1, 0, 1, 31, 127, 511, 512, 513, 1023, 1024, 1025, 2046, 2047, 2048, 4095, 7]
    positions = torch.tensor(boundaries, dtype=torch.int32)
    key = key[:key_capacity]
    for ratio in (1, 2):
        for active in (0, 1, 7, capacity):
            specs = [
                TensorSpec("query", [capacity, INDEX_HEADS, INDEX_HEAD_DIM], torch.int8, init_value=query),
                TensorSpec("query_scale", [capacity, INDEX_HEADS], torch.float16, init_value=query_scale),
                TensorSpec("key", [key_capacity, INDEX_HEAD_DIM], torch.int8, init_value=key),
                TensorSpec("key_scale", [key_capacity], torch.float16, init_value=0.75),
                TensorSpec("weights", [capacity, INDEX_HEADS], torch.float16, init_value=weights),
                TensorSpec("positions", [capacity], torch.int32, init_value=positions),
                TensorSpec("scores", [capacity, key_capacity], torch.float32, init_value=-53.0),
                TensorSpec("indices", [capacity, INDEX_TOPK], torch.int32, init_value=-59),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ]
            case = dict(
                name=f"indexer-keys{key_capacity}-top512-ratio{ratio}-active{active}",
                specs=specs,
                golden_fn=partial(golden_indexer_select, compress_ratio=ratio),
                compare_fn=dict(
                    scores=indexer_active_compare(active), indices=indexer_active_compare(active, exact=True)
                ),
            )
            if build:
                case["fn"] = make_prefill_indexer_select(capacity, key_capacity, compress_ratio=ratio)
            yield case


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
    result = validate(
        device=args.device, sim=args.sim, compile_only=args.compile_only, cpu_only=args.cpu_only
    )
    print(json.dumps(result))
    indexer_select_result = validate_prefill_select(
        device=args.device,
        sim=args.sim,
        compile_only=args.compile_only,
        cpu_only=args.cpu_only,
    )
    print(json.dumps(indexer_select_result))
