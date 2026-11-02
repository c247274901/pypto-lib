# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 W8A8 QKV entry point for the independently frozen V4.1 module contract."""

import pypto.language as pl

from .ops import (
    N_TILE, make_quantize, make_pack_weight, make_dot,
    make_dequantize_rope, make_dequantize_norm_quantize, make_dequantize_norm_rope,
)


def _make_weight_preparation(width, output_width, prepacked):
    """Select canonical packing or a resident [N/64,K,64] weight view."""
    packed_rows = width * (output_width // N_TILE)
    if prepacked:
        @pl.jit.inline
        def resident(weight: pl.Tensor[[packed_rows, N_TILE], pl.INT8]):
            return weight

        return resident

    pack = make_pack_weight(width, output_width)

    @pl.jit.inline
    def prepare(weight: pl.Tensor[[width, output_width], pl.INT8]):
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(weight, packed)
        return packed

    return prepare


def make_qkv_w8a8_inline(capacity, config, prepacked=False):
    """Build shared inline QKV compute with canonical or resident weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    query_width = heads * head_dim
    pair_width = rope_dim // 2
    eps = config["rms_norm_eps"]
    qa_packed_rows = hidden * (latent // N_TILE)
    qb_packed_rows = latent * (query_width // N_TILE)
    kv_packed_rows = hidden * (head_dim // N_TILE)
    qa_rows, qa_cols = (qa_packed_rows, N_TILE) if prepacked else (hidden, latent)
    qb_rows, qb_cols = (qb_packed_rows, N_TILE) if prepacked else (latent, query_width)
    kv_rows, kv_cols = (kv_packed_rows, N_TILE) if prepacked else (hidden, head_dim)

    projection_rows = min(capacity, 32)
    quant_x = make_quantize(capacity, hidden, whole_row=True, early_resolve=True)
    prepare_qa = _make_weight_preparation(hidden, latent, prepacked)
    prepare_qb = _make_weight_preparation(latent, query_width, prepacked)
    prepare_kv = _make_weight_preparation(hidden, head_dim, prepacked)
    dot_qa = make_dot(capacity, hidden, latent, padded_payload=True, row_tile=projection_rows, full_k_reuse=True, early_resolve=True)
    dot_qb = make_dot(
        capacity, latent, query_width, padded_payload=True, column_group=26,
        row_tile=projection_rows, full_k_reuse=True, early_resolve=True, column_stages=2,
    )
    dot_kv = make_dot(capacity, hidden, head_dim, padded_payload=True, row_tile=projection_rows, full_k_reuse=True, early_resolve=True)
    latent_epilogue = make_dequantize_norm_quantize(capacity, latent, eps, early_resolve=True)
    dequant_rope_q = make_dequantize_rope(capacity, heads, head_dim, rope_dim, row_tile=1, head_tile=32, scalar_channels=True)
    kv_epilogue = make_dequantize_norm_rope(capacity, head_dim, rope_dim, eps)

    @pl.jit.inline
    def qkv_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[qa_rows, qa_cols], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[qb_rows, qb_cols], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[kv_rows, kv_cols], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pair_width], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pair_width], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        query: pl.Tensor[[capacity, query_width], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        cache: pl.Tensor[[2, 128, head_dim], pl.BF16],
        cache_lens: pl.Tensor[[2], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        x_i8 = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_x(x, x_i8, x_scale, active)

        kv_packed = prepare_kv(wkv)
        kv_int = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, kv_packed, kv_int, active)
        kv_epilogue(kv_int, wkv_scale, x_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, active)

        qa_packed = prepare_qa(wq_a)
        qa_int = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, qa_packed, qa_int, active)
        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        latent_epilogue(qa_int, wq_a_scale, x_scale, q_norm_weight, query_latent, q_i8, q_scale, active)
        qb_packed = prepare_qb(wq_b)
        qb_int = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, qb_packed, qb_int, active)
        dequant_rope_q(qb_int, wq_b_scale, q_scale, rope_cos, rope_sin, query, active)

        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8_inline


def make_qkv_w8a8(capacity, config, prepacked=False):
    """Build the standalone JIT wrapper for shared inline QKV compute."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    width = config["num_attention_heads"] * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    qa_rows, qa_cols = (hidden * (latent // N_TILE), N_TILE) if prepacked else (hidden, latent)
    qb_rows, qb_cols = (latent * (width // N_TILE), N_TILE) if prepacked else (latent, width)
    kv_rows, kv_cols = (hidden * (head_dim // N_TILE), N_TILE) if prepacked else (hidden, head_dim)
    compute = make_qkv_w8a8_inline(capacity, config, prepacked=prepacked)

    @pl.jit
    def qkv_w8a8(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[qa_rows, qa_cols], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[qb_rows, qb_cols], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[kv_rows, kv_cols], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, rope_cos, rope_sin,
            positions, query_latent, query, window_kv, cache, cache_lens, active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8
