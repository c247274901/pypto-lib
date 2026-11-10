# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Schedule index queries and selection against existing physical key caches."""

import torch

from .indexer_select import KEY_CAPACITY


def validate_refresh_candidates(contract):
    """Require the bounded route where candidate blocks cover every visible source key."""
    if contract["physical_layer_id"] != 24 or contract["source_physical_layer_id"] != 20:
        raise ValueError("Require original physical24 refresh of source20")
    for originals in contract["cases"].values():
        for original in originals:
            candidates = original["shared_state_initial"]["candidates"]["prefix"]
            validate_live_refresh_candidates(original, {"candidates": candidates}, 20)


def validate_live_refresh_candidates(original, shared, source_id):
    """Preserve the short full-coverage guard against the actual live candidate buffer."""
    starts = original["metadata"]["query_start_loc"].tolist()
    metadata = original["cache_metadata"]["actual_metadata"]
    prefix = f"model.layers.{source_id}.self_attn."
    lengths = metadata[prefix + "indexer.k_cache"]["global_metadata"]["cache_seq_lens"].tolist()
    if any(length < 0 or length > KEY_CAPACITY for length in lengths):
        raise ValueError("Refresh history exceeds the supported index-key capacity")
    start = original["cp_token_range"][0]
    candidates = shared["candidates"]
    for local in range(original["actual_local_query_rows"]):
        token = start + local
        request = next(i for i in range(len(starts) - 1) if starts[i] <= token < starts[i + 1])
        visible = min(lengths[request], int(original["positions"][token]) + 1)
        available = set(candidates[local].reshape(-1).tolist())
        if not set(range((visible + 7) // 8)).issubset(available):
            raise ValueError("This input requires candidate pruning; no dense substitute is supported")


def run_index_query_selection(state, q_input, latent, cosine, sine, operands,
                              execute, rank, case, capacity, requests, table_pages,
                              source_prefix, own_prefix, codes, scales):
    """Compute quantized index queries, paged-key scores and sparse controls with PyPTO."""
    original, total, nodes = state["original"], state["total"], state["nodes"]
    start, end, _per_rank, _padded = original["cp_token_range"]
    query_rows = max(0, min(end, total) - min(start, total))
    metadata = original["cache_metadata"]["actual_metadata"]
    global_meta = {name: item["global_metadata"] for name, item in metadata.items()}

    def zeros(*shape, dtype=torch.bfloat16):
        return torch.zeros(shape, dtype=dtype)

    def pad(value, rows=capacity, fill=0):
        result = torch.full((rows, *value.shape[1:]), fill, dtype=value.dtype)
        result[:len(value)].copy_(value)
        return result

    def table(name):
        value = global_meta[name]["block_table"]
        target = torch.full((requests, table_pages), -1, dtype=torch.int32)
        target[:value.shape[0], :value.shape[1]].copy_(value)
        return target

    positions = pad(original["positions"])
    actual_starts = original["metadata"]["query_start_loc"]
    request_ids = torch.full((capacity,), -1, dtype=torch.int32)
    for request, (begin, finish) in enumerate(zip(actual_starts.tolist(), actual_starts.tolist()[1:])):
        request_ids[begin:finish].fill_(request)
    index_qb, weights_proj = zeros(capacity, 4096), zeros(capacity, 32)
    execute("index_wqb", rank, case, [latent, operands["index_wqb"], operands["index_wqb_scale"], index_qb, query_rows],
            dict(projected=index_qb), "index_key")
    execute("index_weights_proj", rank, case, [q_input, operands["index_weights"], weights_proj, query_rows],
            dict(projected=weights_proj), "index_key")
    q_cos, q_sin = pad(cosine[start:start + query_rows]), pad(sine[start:start + query_rows])
    query_key = zeros(capacity, 4096)
    execute("index_query_rope", rank, case, [index_qb, q_cos, q_sin, query_key, query_rows],
            dict(query=query_key), "rope")
    query_codes = zeros(capacity, 32, 128, dtype=torch.int8)
    query_scales, weights = zeros(capacity, 32, dtype=torch.float16), zeros(capacity, 32, dtype=torch.float16)
    execute("index_query_quant", rank, case, [query_key.reshape(capacity, 32, 128), query_codes, query_scales, query_rows],
            dict(codes=query_codes, scales=query_scales), "index_quant")
    execute("index_weights_scale", rank, case, [weights_proj, weights, query_rows], dict(weights=weights), "index_key")
    candidate_scores = zeros(capacity, KEY_CAPACITY, dtype=torch.float32)
    raw_selected = torch.full((capacity, 512), -1, dtype=torch.int32)
    q_positions, q_requests = pad(positions[start:start + query_rows]), pad(request_ids[start:start + query_rows], fill=-1)
    long_table, window_table = table(source_prefix + "long_kv_cache"), table(own_prefix + "swa_cache")
    cache_lengths = global_meta[source_prefix + "indexer.k_cache"]["cache_seq_lens"].tolist()
    if any(length < 0 or length > KEY_CAPACITY for length in cache_lengths):
        raise ValueError("Physical QLI history exceeds the supported index-key capacity")
    for request, (begin, finish) in enumerate(zip(actual_starts.tolist(), actual_starts.tolist()[1:])):
        left, right = max(begin, start), min(finish, start + query_rows)
        if left >= right:
            continue
        offset, count = left - start, right - left
        packed_key = zeros(KEY_CAPACITY, 128, dtype=torch.int8)
        packed_scale = zeros(KEY_CAPACITY, dtype=torch.float16)
        execute("index_key_gather", rank, case + f"-request-{request}",
                [codes, scales, long_table, packed_key, packed_scale, request, cache_lengths[request]],
                dict(key=packed_key, scales=packed_scale), "index_select")
        scores = zeros(capacity, KEY_CAPACITY, dtype=torch.float32)
        picked = torch.full((capacity, 512), -1, dtype=torch.int32)
        selection_args = [pad(query_codes[offset:offset + count]), pad(query_scales[offset:offset + count]),
                          packed_key, packed_scale, pad(weights[offset:offset + count]),
                          pad(q_positions[offset:offset + count]).to(torch.int32), scores, picked, count]
        execute("index_select", rank, case + f"-request-{request}", selection_args,
                dict(scores=scores, selected=picked), "index_select")
        raw_selected[offset:offset + count].copy_(picked[:count])
        candidate_scores[offset:offset + count].copy_(scores[:count])
    if not query_rows:
        # Empty CP ranks retain a recorded index-selection boundary.
        execute("index_weights_scale", rank, case + "-empty-selection", [weights_proj, weights, 0],
                dict(weights=weights), "index_select")
    selected = torch.full((capacity, 512), -1, dtype=torch.int32)
    lengths = zeros(capacity, 1, dtype=torch.int32)
    compressed_indices, window_indices = selected.clone(), torch.full((capacity, 128), -1, dtype=torch.int32)
    controls_args = [raw_selected, q_positions, q_requests, long_table, window_table,
                     selected, lengths, compressed_indices, window_indices, query_rows]
    execute("sparse_controls", rank, case, controls_args,
            dict(selected=selected, lengths=lengths, compressed=compressed_indices, window=window_indices), "index_select")
    nodes.update({"indexer.wq_b": index_qb[:query_rows].clone(),
                  "indexer.weights_proj": weights_proj[:query_rows].clone(),
                  "indexer.query_before_quantize": query_key[:query_rows].reshape(query_rows, 32, 128).clone(),
                  "indexer.quantized_query_and_scale": (query_codes[:query_rows].clone(), query_scales[:query_rows].clone()),
                  "indexer.selected_indices": selected[:query_rows].clone()})
    visible_lengths = zeros(capacity, dtype=torch.int32)
    ratio = state["role"]["ratio"]
    visible_lengths[:query_rows] = ((q_positions[:query_rows] + 1) // ratio).int()
    return dict(selected=selected, lengths=lengths, compressed=compressed_indices,
                window=window_indices, scores=candidate_scores, visible_lengths=visible_lengths)
