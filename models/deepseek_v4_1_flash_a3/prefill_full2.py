# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech Full2 Prefill stages with actual intermediate observation ports."""

import pypto.language as pl

from .prefill_cache_publish import make_compressed_cache_publish
from .prefill_full2_observed import (
    make_c2_program, make_indexer_projection_observed, make_indexer_query_quantize_observed,
)
from .prefill_full2_readback import make_readback_entries
from .prefill_swa import GEOMETRY, make_builders
from .hc import make_hc_post_inline
from .prefill_indexer import make_empty_selection, make_sparse_controls
from .prefill_norm import make_prefill_norm
from .prefill_sparse_consumer import make_consumer_sparse_core


def make_entries(active, *, compressor_projected=False):
    """Build complete query, Full2 producer, sparse and BF16 output stages."""
    if type(active) is not int or active not in (1, 7, 16):
        raise ValueError("Original bounded active1/7/16 required")
    if type(compressor_projected) is not bool:
        raise ValueError("Select the actual product compressor projection boundary explicitly")
    config = dict(GEOMETRY)
    base = make_builders(16, config)
    names = (
        "wq_a", "q_norm", "wq_b", "q_rope", "wkv", "kv_norm", "kv_rope",
        "inverse_rope", "grouped_wo_a", "wo_b",
    )
    entries = {name: base[name] for name in names}

    @pl.jit
    def narrow_sparse_BF16(
        source: pl.InOut[pl.Tensor[[16, 32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * 128, name_hint="L2_sparse_native_BF16_boundary"):
            token = block // 128
            column = block % 128 * 256
            values = source[token : token + 1, column : column + 256]
            output[token : token + 1, column : column + 256] = pl.cast(values, pl.BF16, mode="rint")
        return source, output

    entries.update(
        attention_norm=make_prefill_norm(16, 5120),
        c2_project_pool_publish=make_c2_program(config, project_inputs=not compressor_projected),
        swa_publish=pl.jit(make_compressed_cache_publish(16, 2, 512)._func),
        index_projection=make_indexer_projection_observed(16, config),
        index_quantize=make_indexer_query_quantize_observed(16),
        empty_selection=make_empty_selection(16),
        sparse_controls=make_sparse_controls(16, 1, 2, 2, long_page_rows=64, compress_ratio=2),
        sparse_core=make_consumer_sparse_core(active, config, compressed_blocks=1),
        narrow_sparse_BF16=narrow_sparse_BF16,
        HC_post=pl.jit(make_hc_post_inline(16, config)._func),
    )
    return make_readback_entries(entries, active)


def make_physical_compressed_entries(capacity, config, cache_blocks, requests, table_pages, *,
                                     compress_ratio=2, cache_layout=None, layer_id=2, network_cache=False):
    """Build paged physical-layer C2, index packing and full-head compressed attention entries."""
    from .indexer_select import KEY_CAPACITY, make_indexer_select
    from .norm_rope import make_rope_tables
    from .rmsnorm import make_blocked_rmsnorm as make_norm
    from .norm_rope import make_adjacent_rope as make_rope
    from .prefill_c2_pool import make_c2_pool_observed
    from .prefill_cache_publish import make_index_cache_publish_observed
    from .prefill_indexer import make_index_key_gather, make_physical_index_query_entries
    from .prefill_indexer import make_sparse_index_pack
    from .prefill_layer2_c1c2 import make_c2_projection
    from .qkv_proj_rope import make_bf16_projection_staged
    from .swa import make_swa_leaves, make_swa_read

    publish, _gather, _attend = make_swa_leaves(capacity, cache_blocks, config)
    sparse_config = dict(config, sliding_window=640)
    theta, scaling = config["compress_rope_theta"], config.get("rope_scaling")
    entries = dict(
        rope_tables=make_rope_tables(capacity, 64, theta, scaling),
        source_rope_tables=make_rope_tables(capacity, 64, theta, scaling),
        c2_projection=make_c2_projection(capacity, config["hidden_size"]),
        c2_pool=make_c2_pool_observed(capacity, cache_blocks, requests),
        compressed_rope=pl.jit(make_rope(capacity, 1, 512, 64)._func),
        compressed_publish=pl.jit(publish._func),
        index_wk=make_bf16_projection_staged(512, 128, capacity=capacity),
        index_norm=pl.jit(make_norm(capacity, 128, config["rms_norm_eps"])._func),
        index_rope=pl.jit(make_rope(capacity, 1, 128, 64)._func),
        index_publish=make_index_cache_publish_observed(capacity, cache_blocks),
        sparse_index_pack=make_sparse_index_pack(capacity),
        compressed_attention=make_swa_read(capacity, cache_blocks, sparse_config),
    )
    entries.update(make_physical_index_query_entries(capacity, config, cache_blocks, requests, table_pages))
    if compress_ratio not in (1, 2):
        raise ValueError("Physical compressed attention supports ratio1 or ratio2")
    if compress_ratio == 1:
        from .candidate_pool import make_candidate_block_pool, make_candidate_pool_publish
        from .prefill_cache_layout import make_paged_cache_copy, physical_page_geometry

        geometry = (None if network_cache else
                    physical_page_geometry(cache_layout, f"model.layers.{layer_id}.self_attn.", cache_blocks))
        del entries["c2_projection"]
        del entries["c2_pool"]
        entries.update(
            compressor_projection=make_bf16_projection_staged(5120, 512, capacity=capacity),
            compressor_norm=pl.jit(make_norm(capacity, 512, config["rms_norm_eps"])._func),
            index_publish=make_index_cache_publish_observed(capacity, cache_blocks, page_rows=128),
            index_key_gather=make_index_key_gather(cache_blocks, requests, table_pages, page_rows=128),
            index_select=make_indexer_select(capacity, KEY_CAPACITY, compress_ratio=1),
            sparse_controls=make_sparse_controls(capacity, requests, table_pages, table_pages,
                                                long_page_rows=128, compress_ratio=1),
            candidate_pool=make_candidate_block_pool(capacity),
            candidate_publish=make_candidate_pool_publish(capacity),
        )
        dtypes = {"torch.bfloat16": pl.BF16, "torch.int8": pl.INT8, "torch.float16": pl.FP16}
        for name, descriptor in (geometry or {}).items():
            payload, stride, offset = descriptor["payload"], descriptor["stride"], descriptor["offset"]
            dtype = dtypes[descriptor["dtype"]]
            entries["cache_load_" + name] = make_paged_cache_copy(cache_blocks, payload, stride, payload,
                                                                 source_offset=offset, dtype=dtype)
            entries["cache_store_" + name] = make_paged_cache_copy(cache_blocks, payload, payload, stride,
                                                                  destination_offset=offset, dtype=dtype)
    if network_cache:
        from .prefill_layer_state import resolve_layer_role
        from .prefill_network_cache import make_network_cache_entries, network_cache_geometry

        role = resolve_layer_role(layer_id)
        if compress_ratio != role["ratio"]:
            raise ValueError("Network producer ratio differs from its original physical role")
        geometry = network_cache_geometry(cache_layout, role, cache_blocks)
        entries.update(make_network_cache_entries(capacity, config, geometry, role))
    return entries
