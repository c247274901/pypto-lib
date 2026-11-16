# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rotary embeddings, tables, goldens and phase-specific rotations."""

import math
import pypto.language as pl
import torch as decode_torch
import torch


def make_decode_adjacent_rope(capacity, heads, head_dim, rope_dim):
    """Rotate the final channels of each head in adjacent pairs."""
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, output_width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="a2_adjacent_rope"):
            t = block // heads
            base = (block % heads) * head_dim
            output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
            tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[t : t + 1, :]
            s = sine[t : t + 1, :]
            even_c = pl.mul(even, c)
            odd_s = pl.mul(odd, s)
            odd_c = pl.mul(odd, c)
            even_s = pl.mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def make_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    if rope_dim % 2 or capacity <= 0:
        raise ValueError("RoPE capacity and dimension must be positive and even")
    width = heads * head_dim
    nope = head_dim - rope_dim
    pairs = rope_dim // 2

    @pl.jit.inline
    def rope(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="dsv41_a2a3_rope"):
            token = block // heads
            base = (block % heads) * head_dim
            output[token : token + 1, base : base + nope] = value[token : token + 1, base : base + nope]
            tail = pl.cast(value[token : token + 1, base + nope : base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[token : token + 1, :]
            s = sine[token : token + 1, :]
            real = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            imag = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token : token + 1, base + nope : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def make_rope_tables(capacity, rope_dim, theta, scaling=None):
    """Build plain or YaRN RoPE tables from actual token positions."""
    pairs = rope_dim // 2
    exponent_scale = 2.0 * math.log(theta) / rope_dim
    factor, low, high = 1.0, 0, 1
    if scaling is not None:
        factor = float(scaling["factor"])
        original = scaling["original_max_position_embeddings"]
        fast = scaling.get("beta_fast", 32)
        slow = scaling.get("beta_slow", 1)
        low = max(0, math.floor(rope_dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(theta))))
        high = min(
            rope_dim - 1,
            math.ceil(rope_dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(theta))),
        )
    use_yarn = scaling is not None
    ramp_start = float(low)
    inverse_ramp_width = 1.0 / max(high - low, 0.001)
    interpolation_delta = 1.0 / factor - 1.0

    @pl.jit
    def rope_tables(
        positions: pl.Tensor[[capacity], pl.INT64],
        cosine: pl.InOut[pl.Tensor[[capacity, pairs], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[capacity, pairs], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="prefill_rope_tables"):
            dimension_ids = pl.arange(0, [1, pairs], dtype=pl.INT32)
            dimensions = pl.cast(dimension_ids, pl.FP32)
            powers = pl.exp(pl.mul(dimensions, exponent_scale))
            ones = pl.full([1, pairs], dtype=pl.FP32, value=1.0)
            frequencies = pl.div(ones, powers, high_precision=True)
            if use_yarn:
                ramp_unbounded = pl.mul(pl.sub(dimensions, ramp_start), inverse_ramp_width)
                ramp = pl.minimum(pl.maximum(ramp_unbounded, 0.0), 1.0)
                multiplier = pl.add(pl.mul(ramp, interpolation_delta), 1.0)
                frequencies = pl.mul(frequencies, multiplier)
            position = pl.cast(pl.read(positions, [token]), pl.FP32)
            angles = pl.mul(frequencies, position)
            cosine[token : token + 1, :] = pl.cos(angles)
            sine[token : token + 1, :] = pl.sin(angles)
        return cosine, sine

    return rope_tables


def make_adjacent_rope(capacity, heads, head_dim, rope_dim):
    """Rotate the final channels of each head in adjacent pairs."""
    output_width = heads * head_dim
    nope_dim = head_dim - rope_dim
    pair_width = rope_dim // 2
    head_tile = math.gcd(heads, 8)

    @pl.jit.inline
    def rope(
        x: pl.Tensor[[capacity, output_width], pl.BF16],
        cosine: pl.Tensor[[capacity, pair_width], pl.FP32],
        sine: pl.Tensor[[capacity, pair_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        x_heads = pl.reshape(x, [capacity * heads, head_dim])
        output_heads = pl.reshape(output, [capacity * heads, head_dim])
        for block in pl.spmd(active * (heads // head_tile), name_hint="a2_adjacent_rope"):
            t = block // (heads // head_tile)
            h0 = block * head_tile
            output_heads[h0 : h0 + head_tile, :nope_dim] = x_heads[h0 : h0 + head_tile, :nope_dim]
            tail = pl.cast(x_heads[h0 : h0 + head_tile, nope_dim:head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[t : t + 1, :]
            s = sine[t : t + 1, :]
            even_c = pl.col_expand_mul(even, c)
            odd_s = pl.col_expand_mul(odd, s)
            odd_c = pl.col_expand_mul(odd, c)
            even_s = pl.col_expand_mul(even, s)
            real = pl.sub(even_c, odd_s)
            imag = pl.add(odd_c, even_s)
            rotated = pl.full([head_tile, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output_heads[h0 : h0 + head_tile, nope_dim:head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def make_rope_kernel(capacity: int, heads: int, head_dim: int, rope_dim: int):
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    pairs = rope_dim // 2
    width = heads * head_dim

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        rotate(value, cosine, sine, output, active)
        return output

    return kernel


def make_decode_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    if rope_dim % 2 or capacity <= 0:
        raise ValueError("RoPE capacity and dimension must be positive and even")
    width = heads * head_dim
    nope = head_dim - rope_dim
    pairs = rope_dim // 2

    @pl.jit.inline
    def rope(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="dsv41_a2a3_rope"):
            token = block // heads
            base = (block % heads) * head_dim
            output[token : token + 1, base : base + nope] = value[token : token + 1, base : base + nope]
            tail = pl.cast(value[token : token + 1, base + nope : base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[token : token + 1, :]
            s = sine[token : token + 1, :]
            real = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            imag = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token : token + 1, base + nope : base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


def rope_reference(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim % 2 or value.shape[-1] != heads * head_dim:
        raise ValueError("RoPE dimensions must be even and match heads * head_dim")
    output = value.clone()
    nope = head_dim - rope_dim
    shaped = value.view(value.shape[0], heads, head_dim)
    tail = shaped[..., nope:]
    even, odd = tail[..., 0::2].float(), tail[..., 1::2].float()
    c, s = cosine.float().unsqueeze(1), sine.float().unsqueeze(1)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    output.view(value.shape[0], heads, head_dim)[..., nope:] = rotated.to(torch.bfloat16)
    return output


def decode_rope_reference(
    value: decode_torch.Tensor,
    cosine: decode_torch.Tensor,
    sine: decode_torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> decode_torch.Tensor:
    import torch

    if rope_dim % 2 or value.shape[-1] != heads * head_dim:
        raise ValueError("RoPE dimensions must be even and match heads * head_dim")
    output = value.clone()
    nope = head_dim - rope_dim
    shaped = value.view(value.shape[0], heads, head_dim)
    tail = shaped[..., nope:]
    even, odd = tail[..., 0::2].float(), tail[..., 1::2].float()
    c, s = cosine.float().unsqueeze(1), sine.float().unsqueeze(1)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    output.view(value.shape[0], heads, head_dim)[..., nope:] = rotated.to(torch.bfloat16)
    return output
