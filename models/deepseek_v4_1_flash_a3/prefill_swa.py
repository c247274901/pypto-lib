# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech W8A8 SWA Prefill with BF16 grouped output projections."""

import pypto.language as pl
from .gemm import make_ecotech_dynamic_linear
from .hc import make_hc_pre_inline
from .rmsnorm import make_blocked_rmsnorm as make_norm, make_prefill_norm
from .rope import make_adjacent_rope as make_rope
from .swa import make_swa_inline, make_swa_leaves, make_swa_read


# model config
GEOMETRY = dict(
    hidden_size=5120,
    num_attention_heads=64,
    num_key_value_heads=1,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-20,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
)


def make_builders(capacity=16, config=None, cache_blocks=2, *, tp=1):
    """Build twelve Prefill stages with explicit predecessor and cache buffers."""
    from . import qkv_proj_rope as grouped
    from . import qkv_proj_rope as shared

    config = dict(GEOMETRY) if config is None else config
    if any(config.get(key) != value for key, value in GEOMETRY.items()):
        raise ValueError("Retain complete Eco-Tech SWA geometry")
    if capacity <= 0 or capacity % 16:
        raise ValueError("Capacity must cover complete 16-row tiles")
    grouped_output = grouped.make_prefill_grouped_output(tp)
    publish, _gather, _attend = make_swa_leaves(capacity, cache_blocks, config)
    pre = make_hc_pre_inline(capacity, config)
    norm = pl.jit.inline(make_prefill_norm(capacity, 5120)._func)

    @pl.jit
    def collapse_norm(
        x: pl.Tensor[[capacity, 4, 5120], pl.BF16],
        hc_function: pl.Tensor[[24, 20480], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        incoming: pl.Tensor[[capacity, 4], pl.FP32],
        norm_weight: pl.Tensor[[5120], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, 4, 4], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        pre(x, hc_function, hc_scale, hc_base, incoming, next_pre, post, comb, collapsed, active)
        norm(collapsed, norm_weight, normed, active)
        return next_pre, post, comb, collapsed, normed

    def standalone(entry):
        return pl.jit(entry._func)

    return dict(
        collapse_norm=collapse_norm,
        wq_a=make_ecotech_dynamic_linear(capacity, 5120, 1280, combine_scales=True),
        q_norm=standalone(make_norm(capacity, 1280, 1e-20)),
        wq_b=make_ecotech_dynamic_linear(capacity, 1280, 32768, combine_scales=True),
        q_rope=standalone(make_rope(capacity, 64, 512, 64)),
        wkv=make_ecotech_dynamic_linear(capacity, 5120, 512),
        kv_norm=standalone(make_norm(capacity, 512, 1e-20)),
        kv_rope=standalone(make_rope(capacity, 1, 512, 64)),
        cache_publish=standalone(publish),
        swa_read=make_swa_read(capacity, cache_blocks, config),
        swa=standalone(make_swa_inline(capacity, cache_blocks, config)),
        inverse_rope=standalone(shared.make_rope(64, inverse=True, head_dim=512, rope_dim=64, max_workers=64)),
        grouped_wo_a=standalone(grouped_output),
        wo_b=shared.make_bf16_projection_staged(8192 // tp, 5120, capacity=capacity),
    )
