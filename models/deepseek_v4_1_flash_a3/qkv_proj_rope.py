# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""QKV projection, grouped projections and their rotary-embedding compositions."""

import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import pypto.language as pl
import torch
from .config import FLASH as M
from .gemm import make_dot, make_fp32_dequantize, make_int8_gemm
from .operator_config import T_DYN
from .quant import int8_matmul, make_decode_dequantize as decode_make_dequantize, make_decode_dynamic_int8_quantize as decode_make_dynamic_int8_quantize, make_decode_quantize as decode_make_quantize, make_dequantize, make_dynamic_int8_quantize, make_quantize, quantize_per_token_int8
from .rmsnorm import decode_rms_norm_reference, make_blocked_rmsnorm as make_norm, make_decode_blocked_rmsnorm as decode_make_norm, make_decode_rmsnorm as decode_make_rmsnorm, make_rmsnorm, rms_norm_reference
from .rope import decode_rope_reference, make_adjacent_rope as qkv_w8a8_make_rope, make_decode_adjacent_rope as decode_make_rope, make_decode_rope as decode_qkv_make_rope, make_rope as qkv_make_rope, rope_reference
from .runtime_config import validate_tp
from .utils import unpack_blocked_weight


# model config
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim


# tiling
M_TILE = 16
N_TILE = 128
K_TILE = 64
BF16_N_TILE = 256
BF16_K_TILE = 256


DECODE_HEAD_DIM = M.head_dim


DECODE_ROPE_DIM = M.qk_rope_head_dim


DECODE_K_TILE = 256


@dataclass(frozen=True)
class QKVShape:
    """Static dimensions required by the layer-0 QKV leaf."""

    hidden_size: int
    q_lora_rank: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    rms_norm_eps: float = 1.0e-20

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def rope_pairs(self) -> int:
        return self.qk_rope_head_dim // 2


def _make_rope_block(heads, inverse, head_dim, rope_dim):
    sign = -1.0 if inverse else 1.0
    nope_dim = head_dim - rope_dim

    @pl.jit.inline
    def rotate_block(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        block: pl.Scalar[pl.INT32],
    ):
        t = block // heads
        h = block % heads
        base = h * head_dim
        output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
        tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
        even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
        c = cos[t : t + 1, :]
        s = pl.mul(sin[t : t + 1, :], sign)
        re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
        im = pl.add(pl.mul(even, s), pl.mul(odd, c))
        rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
        rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
        rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
        output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rotate_block


def make_rope(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with direct or capped strided scheduling."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return output

    return rotate


def make_bf16_projection_staged(width, output_width, *, capacity=None):
    """Accumulate BF16 Cube slabs and narrow through a fused Vector epilogue.

    The BF16 operand passes through Vector before Cube starts its result transfer
    (hw-native-sys/pypto#2829). Results pass through Vector before GM stores.
    """
    n_tile = gcd(BF16_N_TILE, output_width)
    k_tile = gcd(BF16_K_TILE, width)
    n_owner_tile = gcd(32, output_width)
    n_owners = output_width // n_owner_tile
    n_pairs = max(1, n_owners // 2)
    shift_enabled = int(4 <= n_owners <= 2 * (width // k_tile) + 1)
    shift_step = (width // k_tile) // n_pairs * shift_enabled
    extent = T_DYN if capacity is None else capacity

    @pl.jit
    def project(
        source: pl.Tensor[[extent, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.InOut[pl.Tensor[[extent, output_width], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cube"):
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            owner = col // n_owner_tile
            parity = owner % 2 * shift_enabled
            phase = shift_step * ((owner // 2) % n_pairs)
            acc = pl.create_tile([M_TILE, n_tile], dtype=pl.FP32)
            first = (parity * (width // k_tile - 1) + phase) % (width // k_tile) * k_tile
            initial_a = pl.load(source, [row, first], [M_TILE, k_tile], valid_shape=[rows, k_tile], target_memory=pl.Mem.Vec)
            initial_b = pl.load(weight, [first, col], [k_tile, n_tile], target_memory=pl.Mem.Mat)
            acc = pl.matmul_acc(acc, initial_a, initial_b, init_cond=True)
            for step in pl.pipeline(1, width // k_tile, stage=2):
                directed = step + parity * (width // k_tile - 1 - 2 * step)
                physical = (directed + phase) % (width // k_tile)
                k = physical * k_tile
                a = pl.load(source, [row, k], [M_TILE, k_tile], valid_shape=[rows, k_tile], target_memory=pl.Mem.Mat)
                b = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.Mem.Mat)
                acc = pl.matmul_acc(acc, a, b)
            transferred = pl.mul(acc, 1.0)
            transferred = pl.set_validshape(transferred, rows, n_tile)
            narrowed = pl.cast(transferred, pl.BF16, mode="rint")
            pl.store(narrowed, [row, col], output)
        return output

    return project


def make_prefill_grouped_output(tp=1):
    """Specialize grouped Wo-A for an explicit TP1/2/4 shard."""
    validate_tp(tp)
    k_tile = 512
    n_tile = 96
    LOCAL_H = M.num_attention_heads // tp

    LOCAL_O_GROUPS = M.o_groups // tp

    LOCAL_O_WIDTH = M.o_groups // tp * M.o_lora_rank

    HEAD_DIM = M.head_dim

    O_LORA = M.o_lora_rank

    O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
    n_blocks = (O_LORA + n_tile - 1) // n_tile
    row_blocks = LOCAL_O_GROUPS * n_blocks

    @pl.jit.inline
    def prefill_grouped_output(
        x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        """Stage Wo-A values with Vector startup and fused BF16 Vector stores."""
        flat_weight = pl.reshape(weight, [LOCAL_O_WIDTH, O_GROUP_IN])
        blocks = (num_tokens + M_TILE - 1) // M_TILE * row_blocks
        with pl.spmd(blocks, name_hint="prefill_grouped_o_a_cube", allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            t0 = block // row_blocks * M_TILE
            owner = block % row_blocks
            group = owner // n_blocks
            local_n = owner % n_blocks * n_tile
            n0 = group * O_LORA + local_n
            rows = pl.min(M_TILE, num_tokens - t0)
            columns = pl.min(n_tile, O_LORA - local_n)
            acc = pl.create_tile([M_TILE, n_tile], dtype=pl.FP32)
            initial_a = pl.load(x, [t0, group * O_GROUP_IN], [M_TILE, k_tile], valid_shape=[rows, k_tile], target_memory=pl.Mem.Vec)
            initial_w = pl.load(flat_weight, [n0, 0], [n_tile, k_tile], valid_shape=[columns, k_tile], target_memory=pl.Mem.Mat)
            initial_w = pl.tile.transpose_view(initial_w)
            acc = pl.matmul_acc(acc, initial_a, initial_w, init_cond=True)
            for kb in pl.pipeline(1, O_GROUP_IN // k_tile, stage=4):
                k0 = kb * k_tile
                x_offset = group * O_GROUP_IN + k0
                a = pl.load(x, [t0, x_offset], [M_TILE, k_tile], valid_shape=[rows, k_tile], target_memory=pl.Mem.Mat)
                w = pl.load(flat_weight, [n0, k0], [n_tile, k_tile], valid_shape=[columns, k_tile], target_memory=pl.Mem.Mat)
                w = pl.tile.transpose_view(w)
                acc = pl.matmul_acc(acc, a, w)
            value = pl.set_validshape(pl.mul(acc, 1.0), rows, columns)
            narrowed = pl.cast(value, pl.BF16, mode="rint")
            pl.store(narrowed, [t0, n0], output)
        return output

    return prefill_grouped_output


def make_qkv_w8a8_inline(capacity, config):
    """Build shared inline QKV compute with prepacked 64-column INT8 weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    query_width = heads * head_dim
    pair_width = rope_dim // 2
    eps = config["rms_norm_eps"]

    quant_x = make_quantize(capacity, hidden)
    quant_q = make_quantize(capacity, latent)
    dot_qa = make_dot(capacity, hidden, latent)
    dot_qb = make_dot(capacity, latent, query_width)
    dot_kv = make_dot(capacity, hidden, head_dim)
    dequant_qa = make_dequantize(capacity, latent)
    dequant_qb = make_dequantize(capacity, query_width, combine_scales=True)
    dequant_kv = make_dequantize(capacity, head_dim)
    norm_q = make_norm(capacity, latent, eps)
    norm_kv = make_norm(capacity, head_dim, eps)
    rope_q = qkv_w8a8_make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = qkv_w8a8_make_rope(capacity, 1, head_dim, rope_dim)

    @pl.jit.inline
    def qkv_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
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

        qa_int = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, wq_a, qa_int, active)
        qa_bf16 = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        dequant_qa(qa_int, wq_a_scale, x_scale, qa_bf16, active)
        norm_q(qa_bf16, q_norm_weight, query_latent, active)

        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_int = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, wq_b, qb_int, active)
        qb_bf16 = pl.create_tensor([capacity, query_width], dtype=pl.BF16)
        dequant_qb(qb_int, wq_b_scale, q_scale, qb_bf16, active)
        rope_q(qb_bf16, rope_cos, rope_sin, query, active)

        kv_int = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, wkv, kv_int, active)
        kv_bf16 = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        dequant_kv(kv_int, wkv_scale, x_scale, kv_bf16, active)
        kv_normalized = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        norm_kv(kv_bf16, kv_norm_weight, kv_normalized, active)
        rope_kv(kv_normalized, rope_cos, rope_sin, window_kv, active)
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8_inline


def make_qkv_w8a8(capacity, config):
    """Build the standalone JIT wrapper for shared inline QKV compute."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    width = config["num_attention_heads"] * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    compute = make_qkv_w8a8_inline(capacity, config)

    @pl.jit
    def qkv_w8a8(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
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
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            positions,
            query_latent,
            query,
            window_kv,
            cache,
            cache_lens,
            active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8


def make_decode_rope(
    heads,
    inverse=False,
    *,
    head_dim=DECODE_HEAD_DIM,
    rope_dim=DECODE_ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with direct or capped strided scheduling."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return output

    return rotate


def make_decode_bf16_projection_staged(width, output_width):
    """Stage FP32 projection values before a separate BF16 narrowing task.

    The BF16 operand passes through Vector before Cube starts its result transfer
    (hw-native-sys/pypto#2829). Store FP32 values through Vector as well: direct
    Acc-to-GM stores were unstable in the concurrent A5 prefill workload.
    Accumulation and RNE rounding are unchanged.
    """
    n_tile = min(N_TILE, output_width)
    k_tile = min(DECODE_K_TILE, width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(source, 0)
        accumulated = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cube") as cube_tid:
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
            for k in pl.range(0, width, k_tile):
                a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
                # Preserve BF16 values while establishing Vector-to-Cube startup.
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                b = pl.slice(weight, [k_tile, n_tile], [k, col])
                acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
            accumulated[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(
                pl.mul(acc, 1.0), rows, n_tile
            )
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cast", deps=[cube_tid]):
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            value = pl.slice(accumulated, [M_TILE, n_tile], [row, col], valid_shape=[rows, n_tile])
            output[row : row + M_TILE, col : col + n_tile] = pl.cast(value, pl.BF16, mode="rint")
        return output

    return project


def make_decode_qkv_w8a8_inline(capacity, config):
    """Build shared inline QKV compute with prepacked 64-column INT8 weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    query_width = heads * head_dim
    pair_width = rope_dim // 2
    eps = config["rms_norm_eps"]

    quant_x = decode_make_quantize(capacity, hidden)
    quant_q = decode_make_quantize(capacity, latent)
    dot_qa = make_dot(capacity, hidden, latent)
    dot_qb = make_dot(capacity, latent, query_width)
    dot_kv = make_dot(capacity, hidden, head_dim)
    dequant_qa = decode_make_dequantize(capacity, latent)
    dequant_qb = decode_make_dequantize(capacity, query_width, combine_scales=True)
    dequant_kv = decode_make_dequantize(capacity, head_dim)
    norm_q = decode_make_norm(capacity, latent, eps)
    norm_kv = decode_make_norm(capacity, head_dim, eps)
    rope_q = decode_make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = decode_make_rope(capacity, 1, head_dim, rope_dim)

    @pl.jit.inline
    def qkv_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
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

        qa_int = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, wq_a, qa_int, active)
        qa_bf16 = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        dequant_qa(qa_int, wq_a_scale, x_scale, qa_bf16, active)
        norm_q(qa_bf16, q_norm_weight, query_latent, active)

        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_int = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, wq_b, qb_int, active)
        qb_bf16 = pl.create_tensor([capacity, query_width], dtype=pl.BF16)
        dequant_qb(qb_int, wq_b_scale, q_scale, qb_bf16, active)
        rope_q(qb_bf16, rope_cos, rope_sin, query, active)

        kv_int = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, wkv, kv_int, active)
        kv_bf16 = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        dequant_kv(kv_int, wkv_scale, x_scale, kv_bf16, active)
        kv_normalized = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        norm_kv(kv_bf16, kv_norm_weight, kv_normalized, active)
        rope_kv(kv_normalized, rope_cos, rope_sin, window_kv, active)
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8_inline


def make_decode_qkv_w8a8(capacity, config):
    """Build the standalone JIT wrapper for shared inline QKV compute."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    width = config["num_attention_heads"] * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    compute = make_decode_qkv_w8a8_inline(capacity, config)

    @pl.jit
    def qkv_w8a8(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
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
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            positions,
            query_latent,
            query,
            window_kv,
            cache,
            cache_lens,
            active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8


def qkv_shape(config: Mapping[str, Any] | QKVShape) -> QKVShape:
    """Normalize a model config or an already materialized shape object."""

    if isinstance(config, QKVShape):
        return config
    return QKVShape(
        hidden_size=int(config["hidden_size"]),
        q_lora_rank=int(config["q_lora_rank"]),
        num_attention_heads=int(config["num_attention_heads"]),
        head_dim=int(config["head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        rms_norm_eps=float(config.get("rms_norm_eps", 1.0e-20)),
    )


def make_prefill_qkv_with_state_guards(capacity: int, config: Mapping[str, Any] | QKVShape):
    """Build the standalone layer-0 QKV W8A8 PyPTO entry point.

    ``cache`` and ``cache_lens`` are opaque state guards here.  The QKV leaf
    does not write SWA state; cache publication is intentionally performed by
    :mod:`swa` so its slot mapping can be validated independently.
    """

    shape = qkv_shape(config)
    if capacity <= 0 or capacity % 16:
        raise ValueError("QKV capacity must be a positive multiple of 16 for the INT8 GEMM tile")
    if shape.qk_rope_head_dim % 2 or shape.qk_rope_head_dim > shape.head_dim:
        raise ValueError("qk_rope_head_dim must be even and no larger than head_dim")

    hidden = shape.hidden_size
    latent = shape.q_lora_rank
    heads = shape.num_attention_heads
    head_dim = shape.head_dim
    query_width = shape.query_width
    rope_pairs = shape.rope_pairs
    rope_dim = shape.qk_rope_head_dim

    quant_x = make_dynamic_int8_quantize(capacity, hidden)
    quant_q = make_dynamic_int8_quantize(capacity, latent)
    dot_qa = make_int8_gemm(capacity, hidden, latent)
    dot_qb = make_int8_gemm(capacity, latent, query_width)
    dot_kv = make_int8_gemm(capacity, hidden, head_dim)
    dequant_qa = make_fp32_dequantize(capacity, latent)
    dequant_qb = make_fp32_dequantize(capacity, query_width)
    dequant_kv = make_fp32_dequantize(capacity, head_dim)
    norm_q = make_rmsnorm(capacity, latent, shape.rms_norm_eps)
    norm_kv = make_rmsnorm(capacity, head_dim, shape.rms_norm_eps)
    rope_q = qkv_make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = qkv_make_rope(capacity, 1, head_dim, rope_dim)

    @pl.jit.inline
    def qkv_w8a8_inline(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        sine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wq_a_output: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        wq_b_output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        wkv_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wkv_normed_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        x_i8 = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_i8, x_scale, active)

        qa_i32 = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, wq_a, qa_i32, active)
        dequant_qa(qa_i32, x_scale, wq_a_scale, wq_a_output, active)
        norm_q(wq_a_output, q_norm_weight, query_latent, active)

        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_i32 = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, wq_b, qb_i32, active)
        dequant_qb(qb_i32, q_scale, wq_b_scale, wq_b_output, active)
        rope_q(wq_b_output, cosine, sine, query, active)

        kv_i32 = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, wkv, kv_i32, active)
        dequant_kv(kv_i32, x_scale, wkv_scale, wkv_output, active)
        norm_kv(wkv_output, kv_norm_weight, wkv_normed_output, active)
        rope_kv(wkv_normed_output, cosine, sine, window_kv, active)
        return (
            positions,
            query_latent,
            query,
            window_kv,
            wq_a_output,
            wq_b_output,
            wkv_output,
            wkv_normed_output,
            cache,
            cache_lens,
        )

    @pl.jit
    def qkv_w8a8(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        sine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wq_a_output: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        wq_b_output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        wkv_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wkv_normed_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        return qkv_w8a8_inline(
            value,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            cosine,
            sine,
            positions,
            query_latent,
            query,
            window_kv,
            wq_a_output,
            wq_b_output,
            wkv_output,
            wkv_normed_output,
            cache,
            cache_lens,
            active,
        )

    qkv_w8a8.inline = qkv_w8a8_inline

    return qkv_w8a8


def make_decode_qkv_with_state_guards(capacity: int, config: Mapping[str, Any] | QKVShape):
    """Build the standalone layer-0 QKV W8A8 PyPTO entry point.

    ``cache`` and ``cache_lens`` are opaque state guards here.  The QKV leaf
    does not write SWA state; cache publication is intentionally performed by
    :mod:`swa` so its slot mapping can be validated independently.
    """

    shape = qkv_shape(config)
    if capacity <= 0 or capacity % 16:
        raise ValueError("QKV capacity must be a positive multiple of 16 for the INT8 GEMM tile")
    if shape.qk_rope_head_dim % 2 or shape.qk_rope_head_dim > shape.head_dim:
        raise ValueError("qk_rope_head_dim must be even and no larger than head_dim")

    hidden = shape.hidden_size
    latent = shape.q_lora_rank
    heads = shape.num_attention_heads
    head_dim = shape.head_dim
    query_width = shape.query_width
    rope_pairs = shape.rope_pairs
    rope_dim = shape.qk_rope_head_dim

    quant_x = decode_make_dynamic_int8_quantize(capacity, hidden)
    quant_q = decode_make_dynamic_int8_quantize(capacity, latent)
    dot_qa = make_int8_gemm(capacity, hidden, latent)
    dot_qb = make_int8_gemm(capacity, latent, query_width)
    dot_kv = make_int8_gemm(capacity, hidden, head_dim)
    dequant_qa = make_fp32_dequantize(capacity, latent)
    dequant_qb = make_fp32_dequantize(capacity, query_width)
    dequant_kv = make_fp32_dequantize(capacity, head_dim)
    norm_q = decode_make_rmsnorm(capacity, latent, shape.rms_norm_eps)
    norm_kv = decode_make_rmsnorm(capacity, head_dim, shape.rms_norm_eps)
    rope_q = decode_qkv_make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = decode_qkv_make_rope(capacity, 1, head_dim, rope_dim)

    @pl.jit.inline
    def qkv_w8a8_inline(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        sine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wq_a_output: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        wq_b_output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        wkv_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wkv_normed_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        x_i8 = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_i8, x_scale, active)

        qa_i32 = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, wq_a, qa_i32, active)
        dequant_qa(qa_i32, x_scale, wq_a_scale, wq_a_output, active)
        norm_q(wq_a_output, q_norm_weight, query_latent, active)

        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_i32 = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, wq_b, qb_i32, active)
        dequant_qb(qb_i32, q_scale, wq_b_scale, wq_b_output, active)
        rope_q(wq_b_output, cosine, sine, query, active)

        kv_i32 = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, wkv, kv_i32, active)
        dequant_kv(kv_i32, x_scale, wkv_scale, wkv_output, active)
        norm_kv(wkv_output, kv_norm_weight, wkv_normed_output, active)
        rope_kv(wkv_normed_output, cosine, sine, window_kv, active)
        return (
            positions,
            query_latent,
            query,
            window_kv,
            wq_a_output,
            wq_b_output,
            wkv_output,
            wkv_normed_output,
            cache,
            cache_lens,
        )

    @pl.jit
    def qkv_w8a8(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden * (latent // 64), 64], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent * (query_width // 64), 64], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden * (head_dim // 64), 64], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        sine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wq_a_output: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        wq_b_output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        wkv_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wkv_normed_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        return qkv_w8a8_inline(
            value,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            cosine,
            sine,
            positions,
            query_latent,
            query,
            window_kv,
            wq_a_output,
            wq_b_output,
            wkv_output,
            wkv_normed_output,
            cache,
            cache_lens,
            active,
        )

    qkv_w8a8.inline = qkv_w8a8_inline

    return qkv_w8a8


def golden_bf16_projection(tensors):
    """Evaluate FP32 matrix multiplication and one BF16 rounding boundary."""
    active = int(tensors["num_tokens"])
    source = tensors["source"][:active].float()
    weight = tensors["weight"].float()
    tensors["output"][:active] = (source @ weight).to(tensors["output"].dtype)


def golden_prefix_compare(active, *, rtol=0.05, atol=0.05):
    """Compare the numeric prefix and require an exact inactive sentinel tail."""
    from golden.validation import ratio_allclose

    numeric = ratio_allclose(rtol=rtol, atol=atol, max_error_ratio=0.0)
    exact = ratio_allclose(rtol=0.0, atol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        for part, selection in (("active", slice(None, active)), ("inactive", slice(active, None))):
            comparator = numeric if part == "active" else exact
            if actual[selection].numel():
                passed, detail = comparator(actual[selection], expected[selection], **context)
                if not passed:
                    return False, f"{part}: {detail}"
        return True, ""

    return compare


def qkv_reference(
    value: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cosine: torch.Tensor,
    sine: torch.Tensor,
    config: Mapping[str, Any] | QKVShape,
    *,
    weight_layout: str = "blocked",
) -> dict[str, torch.Tensor]:
    """CPU/PyTorch golden for the same boundaries as :func:`make_qkv_w8a8`.

    The returned tensors include the three projected BF16 values and the two
    RoPE outputs.  All matrix products use an explicit INT32 accumulator and
    FP32 token/channel dequantization, matching ``W8A8_DYNAMIC``.
    Weights use the loader's blocked layout; ``weight_layout="kn"`` selects
    logical ``[K, N]`` operands for direct-tile callers.
    """

    shape = qkv_shape(config)
    if weight_layout not in ("blocked", "kn"):
        raise ValueError("QKV weight_layout must be 'blocked' or 'kn'")
    if weight_layout == "blocked":
        weights = dict(weights)
        for name, input_width, output_width in (
            ("wq_a", shape.hidden_size, shape.q_lora_rank),
            ("wq_b", shape.q_lora_rank, shape.query_width),
            ("wkv", shape.hidden_size, shape.head_dim),
        ):
            weights[name] = unpack_blocked_weight(weights[name], input_width, output_width)
    x_i8, x_scale = quantize_per_token_int8(value)
    qa_i32 = int8_matmul(x_i8, weights["wq_a"].transpose(-2, -1).contiguous())
    qa = (qa_i32.float() * x_scale.float() * weights["wq_a_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    latent = rms_norm_reference(qa, weights["q_norm_weight"], shape.rms_norm_eps)

    q_i8, q_scale = quantize_per_token_int8(latent)
    qb_i32 = int8_matmul(q_i8, weights["wq_b"].transpose(-2, -1).contiguous())
    qb = (qb_i32.float() * q_scale.float() * weights["wq_b_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    query = rope_reference(
        qb,
        cosine,
        sine,
        shape.num_attention_heads,
        shape.head_dim,
        shape.qk_rope_head_dim,
    )

    kv_i32 = int8_matmul(x_i8, weights["wkv"].transpose(-2, -1).contiguous())
    kv_linear = (kv_i32.float() * x_scale.float() * weights["wkv_scale"].float().reshape(1, -1)).to(
        torch.bfloat16
    )
    kv_normed = rms_norm_reference(kv_linear, weights["kv_norm_weight"], shape.rms_norm_eps)
    window_kv = rope_reference(kv_normed, cosine, sine, 1, shape.head_dim, shape.qk_rope_head_dim)
    return {
        "query_latent": latent,
        "query": query,
        "window_kv": window_kv,
        "wq_a_output": qa,
        "wq_b_output": qb,
        "wkv_output": kv_linear,
        "wkv_normed_output": kv_normed,
    }


def decode_qkv_reference(
    value: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cosine: torch.Tensor,
    sine: torch.Tensor,
    config: Mapping[str, Any] | QKVShape,
    *,
    weight_layout: str = "blocked",
) -> dict[str, torch.Tensor]:
    """CPU/PyTorch golden for the same boundaries as :func:`make_qkv_w8a8`.

    The returned tensors include the three projected BF16 values and the two
    RoPE outputs.  All matrix products use an explicit INT32 accumulator and
    FP32 token/channel dequantization, matching ``W8A8_DYNAMIC``.
    Weights use the loader's blocked layout; ``weight_layout="kn"`` selects
    logical ``[K, N]`` operands for direct-tile callers.
    """

    shape = qkv_shape(config)
    if weight_layout not in ("blocked", "kn"):
        raise ValueError("QKV weight_layout must be 'blocked' or 'kn'")
    if weight_layout == "blocked":
        weights = dict(weights)
        for name, input_width, output_width in (
            ("wq_a", shape.hidden_size, shape.q_lora_rank),
            ("wq_b", shape.q_lora_rank, shape.query_width),
            ("wkv", shape.hidden_size, shape.head_dim),
        ):
            weights[name] = unpack_blocked_weight(weights[name], input_width, output_width)
    x_i8, x_scale = quantize_per_token_int8(value)
    qa_i32 = int8_matmul(x_i8, weights["wq_a"].transpose(-2, -1).contiguous())
    qa = (qa_i32.float() * x_scale.float() * weights["wq_a_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    latent = decode_rms_norm_reference(qa, weights["q_norm_weight"], shape.rms_norm_eps)

    q_i8, q_scale = quantize_per_token_int8(latent)
    qb_i32 = int8_matmul(q_i8, weights["wq_b"].transpose(-2, -1).contiguous())
    qb = (qb_i32.float() * q_scale.float() * weights["wq_b_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    query = decode_rope_reference(
        qb,
        cosine,
        sine,
        shape.num_attention_heads,
        shape.head_dim,
        shape.qk_rope_head_dim,
    )

    kv_i32 = int8_matmul(x_i8, weights["wkv"].transpose(-2, -1).contiguous())
    kv_linear = (kv_i32.float() * x_scale.float() * weights["wkv_scale"].float().reshape(1, -1)).to(
        torch.bfloat16
    )
    kv_normed = decode_rms_norm_reference(kv_linear, weights["kv_norm_weight"], shape.rms_norm_eps)
    window_kv = decode_rope_reference(kv_normed, cosine, sine, 1, shape.head_dim, shape.qk_rope_head_dim)
    return {
        "query_latent": latent,
        "query": query,
        "window_kv": window_kv,
        "wq_a_output": qa,
        "wq_b_output": qb,
        "wkv_output": kv_linear,
        "wkv_normed_output": kv_normed,
    }


def bf16_projection_golden_cases():
    """Bound dynamic/static extents, empty work, partial tiles and BF16 rounding."""
    import torch
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(731)
    for static, active in ((False, 0), (False, 17), (True, 1), (True, 32)):
        capacity, width, output_width = 32, 512, 128
        source = torch.empty((capacity, width), dtype=torch.bfloat16).uniform_(-0.5, 0.5, generator=generator)
        weight = torch.empty((width, output_width), dtype=torch.bfloat16).uniform_(
            -0.5, 0.5, generator=generator
        )
        extent = capacity if static else None
        specs = [
            TensorSpec("source", [capacity, width], torch.bfloat16, init_value=source),
            TensorSpec("weight", [width, output_width], torch.bfloat16, init_value=weight),
            TensorSpec("output", [capacity, output_width], torch.bfloat16, init_value=-12.5),
            ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
        ]
        yield dict(
            name=f"projection-{'static' if static else 'dynamic'}-{active}",
            fn=make_bf16_projection_staged(width, output_width, capacity=extent),
            specs=specs,
            golden_fn=golden_bf16_projection,
            compare_fn={"output": golden_prefix_compare(active)},
        )


def validate_bf16_projection(*, device=0, compile_only=False, platform="a2a3", sim=False):
    """Run bounded owning projection cases through the Golden Harness."""
    from golden import run

    for case in bf16_projection_golden_cases():
        name = case.pop("name")
        result = run(
            **case,
            config={"backend_type": platform, "execution_mode": sim, "device_id": device},
            compile_only=compile_only,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    validate_bf16_projection(
        device=args.device, compile_only=args.compile_only, platform=args.platform, sim=args.sim
    )
