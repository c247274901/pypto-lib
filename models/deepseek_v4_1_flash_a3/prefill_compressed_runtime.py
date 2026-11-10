# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Schedule physical compressed Prefill with aliased raw caches and shared index state."""

import torch

from .runtime_utils import tensor_sha


DTYPES = {str(dtype): dtype for dtype in (torch.int8, torch.float16, torch.float32, torch.bfloat16)}


def physical_cache_views(raw, layout):
    """Recover the original typed shapes, strides and offsets on each shared byte backing."""
    def view(descriptor):
        key = descriptor["storage_key"]
        dtype = DTYPES[descriptor["dtype"]]
        size = torch.empty((), dtype=dtype).element_size()
        offset = descriptor["storage_offset_bytes"] - layout["raw"][key]["storage_offset_bytes"]
        if offset < 0 or offset % size:
            raise ValueError("Invalid physical cache byte offset")
        return torch.as_strided(raw[key].view(dtype), descriptor["shape"], descriptor["stride"], offset // size)

    return {name: [view(item) for item in descriptor] if isinstance(descriptor, list) else view(descriptor)
            for name, descriptor in layout["views"].items()}


def compressed_geometry(contract):
    """Bound the physical cache and request arenas from actual input controls."""
    originals = [row for rows in contract["cases"].values() for row in rows]
    requests = max(1, max(len(row["metadata"]["seq_lens"]) for row in originals))
    pages = max(item["block_table"].shape[1] for row in originals
                for item in row["cache_metadata"]["actual_group_requests"])
    return contract["cache_pages"], requests, pages


def run_compressed_attention(state, full_norm, q_input, latent, query, cosine, sine, operands,
                             execute, rank, case, capacity, cache_blocks, requests, table_pages, layer_id):
    """Execute C2, quantized index selection, shared publication and full-head compressed attention."""
    role = state["role"]
    if role["physical_layer"] != layer_id or role["mode"] != "full":
        raise ValueError("Require the original physical producer role")
    prefix = f"model.layers.{layer_id}.self_attn."
    ratio = role["ratio"]
    original, total, nodes = state["original"], state["total"], state["nodes"]
    start, end, _per_rank, _padded = original["cp_token_range"]
    query_rows = max(0, min(end, total) - min(start, total))
    metadata = original["cache_metadata"]["actual_metadata"]
    global_meta = {name: item["global_metadata"] for name, item in metadata.items()}
    live = state["live"]
    raw = live["raw"]
    views = physical_cache_views(raw, live["layout"])
    network = live.get("network_geometry")
    if network is not None:
        from .prefill_network_cache import load_network_index, network_cache_geometry, owned_cache_pages
        from .prefill_network_cache import raw_cache_matrix, store_network_index, validate_network_sparse_indices

        if network != network_cache_geometry(live["layout"], role, cache_blocks):
            raise ValueError("Network producer geometry differs from the actual live raw aliases")
        long_pages = owned_cache_pages(original, prefix + "long_kv_cache", network["long"])
        index_pages = owned_cache_pages(original, prefix + "indexer.k_cache", network["codes"])
        own_pages = owned_cache_pages(original, prefix + "swa_cache", network["own"])
        if long_pages != index_pages:
            raise ValueError("Original long and index cache owner groups differ")
        if ratio == 2:
            ring_pages = owned_cache_pages(original, prefix + "compressor.state_cache", network["ring"])
            if (network["ring"]["storage_key"] == network["long"]["storage_key"]
                    and set(ring_pages).intersection(long_pages)):
                raise ValueError("C2 ring and long-cache owner pages overlap in their shared arena")
            if (network["ring"]["storage_key"] == network["own"]["storage_key"]
                    and set(ring_pages).intersection(own_pages)):
                raise ValueError("C2 ring and own-SWA owner pages overlap in their shared arena")
        if (network["long"]["storage_key"] == network["own"]["storage_key"]
                and set(long_pages).intersection(own_pages)):
            raise ValueError("Long and own-SWA owner pages overlap in their shared arena")

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def pad(value, rows=capacity, fill=0):
        result = torch.full((rows, *value.shape[1:]), fill, dtype=value.dtype)
        result[:len(value)].copy_(value)
        return result

    def table(name):
        value = global_meta[prefix + name]["block_table"]
        target = torch.full((requests, table_pages), -1, dtype=torch.int32)
        target[:value.shape[0], :value.shape[1]].copy_(value)
        return target

    def slots(name, dtype=torch.int32):
        return pad(global_meta[prefix + name]["slot_mapping"].to(dtype), fill=-1)

    # Preserve the full raw backing in a contiguous physical 128-row page view.
    swa_descriptor = live["layout"]["views"][prefix + "swa_cache"]
    swa_key = swa_descriptor["storage_key"]
    if network is not None:
        cache = raw_cache_matrix(raw, network["long"])
        own_cache = raw_cache_matrix(raw, network["own"])
        dense = load_network_index(raw, network, index_pages, execute, rank, case)
        codes, scales = dense["codes"], dense["scales"]
        if ratio == 2:
            ring = raw_cache_matrix(raw, network["ring"]).reshape(cache_blocks, 32, 1024)
    elif ratio == 1:
        from .prefill_cache_layout import physical_page_geometry

        geometry = physical_page_geometry(live["layout"], prefix, cache_blocks)
        dense = {}
        for name, descriptor in geometry.items():
            dtype = DTYPES[descriptor["dtype"]]
            source = raw[descriptor["storage_key"]].view(dtype).reshape(cache_blocks, descriptor["stride"])
            value = torch.empty((cache_blocks, descriptor["payload"]), dtype=dtype)
            execute("cache_load_" + name, rank, case, [source, value], {"payload": value}, "physical_cache_load")
            dense[name] = value
        cache = dense["long"].reshape(cache_blocks, 128, 1, 512)
        codes = dense["codes"].reshape(cache_blocks, 128, 1, 128)
        scales = dense["scales"].reshape(cache_blocks, 128, 1, 1)
    else:
        cache = raw[swa_key].view(torch.bfloat16).reshape(cache_blocks, 128, 1, 512)
        ring = views[prefix + "compressor.state_cache"].reshape(cache_blocks, 32, 1024)
    starts = torch.full((requests + 1,), total, dtype=torch.int32)
    actual_starts = original["metadata"]["query_start_loc"]
    starts[:len(actual_starts)].copy_(actual_starts)
    request_ids = torch.full((capacity,), -1, dtype=torch.int32)
    for request, (begin, finish) in enumerate(zip(actual_starts.tolist(), actual_starts.tolist()[1:])):
        request_ids[begin:finish].fill_(request)
    positions = pad(original["positions"])
    positions32 = positions.to(torch.int32)
    if ratio == 2:
        state_blocks = table("compressor.state_cache")[:, :1].contiguous()
        kv, gate = zeros(capacity, 512, dtype=torch.float32), zeros(capacity, 512, dtype=torch.float32)
        execute("c2_projection", rank, case, [full_norm, operands["compressor_wkv"], operands["compressor_wgate"],
                kv, gate, total], dict(kv=kv, gate=gate), "c2_projection")
        pooled, normalized = zeros(capacity, 512), zeros(capacity, 512)
        execute("c2_pool", rank, case, [kv, gate, starts, positions32, request_ids, state_blocks,
                operands["compressor_norm"], ring, pooled, normalized, total],
                dict(ring=ring, pooled=pooled, normalized=normalized), ("c2_pool", "c2_ring"))
        source_positions = zeros(capacity, dtype=torch.int64)
        source_metadata = global_meta[prefix + "compressor.state_cache"]
        if total:
            source_positions[:total].copy_(source_metadata["c2_source_positions"])
    else:
        kv, normalized = zeros(capacity, 512), zeros(capacity, 512)
        execute("compressor_projection", rank, case, [full_norm, operands["compressor_wkv"], kv, total],
                {"projected": kv}, "ratio1_projection")
        execute("compressor_norm", rank, case, [kv, operands["compressor_norm"], normalized, total],
                {"normalized": normalized}, "ratio1_norm")
        source_positions = positions
    source_cos, source_sin = zeros(capacity, 32, dtype=torch.float32), zeros(capacity, 32, dtype=torch.float32)
    execute("source_rope_tables", rank, case, [source_positions, source_cos, source_sin, total],
            dict(cosine=source_cos, sine=source_sin), "rope")
    rotated = zeros(capacity, 512)
    execute("compressed_rope", rank, case, [normalized, source_cos, source_sin, rotated, total],
            dict(rotated=rotated), "rope")
    long_slots = slots("long_kv_cache", torch.int64)
    publish = "network_compressed_publish" if network is not None else "compressed_publish"
    execute(publish, rank, case, [rotated, long_slots, cache, total], dict(cache=cache), "cache_publish")
    wk, k_norm, key = zeros(capacity, 128), zeros(capacity, 128), zeros(capacity, 128)
    execute("index_wk", rank, case, [normalized, operands["index_wk"], wk, total], dict(wk=wk), "index_key")
    execute("index_norm", rank, case, [wk, operands["index_norm"], k_norm, total], dict(k_norm=k_norm), "index_key")
    execute("index_rope", rank, case, [k_norm, source_cos, source_sin, key, total], dict(key=key), "index_key")
    cache_codes, cache_scales = views[prefix + "indexer.k_cache"]
    if ratio == 2 and network is None:
        codes, scales = cache_codes.contiguous(), cache_scales.contiguous()
    payload, quant_scale = zeros(capacity, 128, dtype=torch.int8), zeros(capacity, 8, dtype=torch.float32)
    execute("index_publish", rank, case, [key, slots("indexer.k_cache"), codes, scales, payload, quant_scale, total],
            dict(codes=codes, scales=scales, payload=payload, quant_scale=quant_scale), ("index_quant", "cache_publish"))
    if network is not None:
        store_network_index(raw, network, index_pages, dense, execute, rank, case)
    elif ratio == 2:
        cache_codes.copy_(codes)
        cache_scales.copy_(scales)
    swa_slots = slots("swa_cache", torch.int64)
    if network is not None:
        execute("network_swa_publish", rank, case, [state["kv_rope"], swa_slots, own_cache, total],
                dict(cache=own_cache), "cache_publish")
    else:
        execute("cache_publish", rank, case, [state["kv_rope"], swa_slots, cache, total], dict(cache=cache), "cache_publish")

    if ratio == 1 and network is None:
        for name, descriptor in geometry.items():
            dtype = DTYPES[descriptor["dtype"]]
            target = raw[descriptor["storage_key"]].view(dtype).reshape(cache_blocks, descriptor["stride"])
            execute("cache_store_" + name, rank, case, [dense[name], target], {"raw": target}, "physical_cache_store")

    from .prefill_indexer_runtime import run_index_query_selection

    selection = run_index_query_selection(state, q_input, latent, cosine, sine, operands,
        execute, rank, case, capacity, requests, table_pages, prefix, prefix, codes, scales)
    selected, lengths = selection["selected"], selection["lengths"]
    compressed_indices, window_indices = selection["compressed"], selection["window"]
    candidate_scores = selection["scores"]
    indices = torch.full((capacity, 640), -1, dtype=torch.int32)
    execute("sparse_index_pack", rank, case, [window_indices, compressed_indices, indices, query_rows],
            dict(indices=indices), "compressed_attention")
    context = zeros(capacity, 32768)
    if network is not None:
        validate_network_sparse_indices(indices, original, network, query_rows)
        execute("network_compressed_attention", rank, case,
                [query, indices, operands["sink"], cache, own_cache, context, query_rows],
                dict(context=context), ("swa", "compressed_attention"))
    else:
        execute("compressed_attention", rank, case, [query, indices, operands["sink"], cache, context, query_rows],
                dict(context=context), ("swa", "compressed_attention"))

    after = live["shared"]
    before = state["shared_state_before"]
    execute("shared_index_publish", rank, case, [selected, lengths, after["topk_indices"], after["topk_lengths"], query_rows],
            dict(indices=after["topk_indices"], lengths=after["topk_lengths"]), "index_select")
    if ratio == 1:
        from .candidate_pool import POOL_CAPACITY

        candidate_blocks = torch.full((capacity, POOL_CAPACITY), -1, dtype=torch.int32)
        candidate_counts = zeros(capacity, dtype=torch.int32)
        execute("candidate_pool", rank, case, [candidate_scores, selection["visible_lengths"],
                candidate_blocks, candidate_counts, query_rows],
                dict(blocks=candidate_blocks, counts=candidate_counts), "candidate_pool")
        execute("candidate_publish", rank, case, [candidate_blocks, candidate_counts,
                after["candidates"], after["candidate_lengths"], query_rows],
                dict(candidates=after["candidates"], lengths=after["candidate_lengths"]), "candidate_publication")
    for name, value in after.items():
        count = query_rows if ratio == 1 or name in ("topk_indices", "topk_lengths") else 0
        nodes["indexer.shared." + name] = value[:count].clone()
        if before is not None:
            nodes["indexer.untouched." + name] = dict(start=count, before_sha256=tensor_sha(before[name][count:]),
                                                       after_sha256=tensor_sha(value[count:]))
    nodes.update({"compressor.wkv": kv[:total].clone(), "compressor.norm": normalized[:total].clone(),
                  "indexer.wk": wk[:total].clone(), "indexer.k_norm": k_norm[:total].clone(),
                  "indexer.key_before_dynamic_quant": key[:total].clone(),
                  "indexer.dynamic_quant_key_and_scale": (payload[:total].clone(), quant_scale[:total, 0].clone())})

    if ratio == 1:
        nodes.update({"compressor.projected_bf16": kv[:total].clone(),
                      "indexer.key_rope_cos": source_cos[:total].unsqueeze(-1).expand(total, 32, 2).reshape(total, 1, 1, 64).clone(),
                      "indexer.key_rope_sin": source_sin[:total].unsqueeze(-1).expand(total, 32, 2).reshape(total, 1, 1, 64).clone()})
    else:
        nodes.update({"compressor.wgate": gate[:total].clone(),
                      "compressor.pooled_bf16": pooled[:total].clone()})
    state.update(raw_cache_after=raw, shared_state_after=after)
    return context, views[prefix + "swa_cache"] if network is not None else cache
