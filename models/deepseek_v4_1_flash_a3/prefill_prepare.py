# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare single-request chunk inputs and persistent paged-cache coordinates."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from .prefill_layer_state import resolve_layer_role
from .prefill_network_guard import build_state_regions
from .runtime_utils import file_sha, write_json


MAX_CONTEXT = 2048
PAGE_ROWS = 128


def tokenize_prompt(inputs, checkpoint, vocabulary):
    """Encode raw completion text or validate explicit original token IDs."""
    if isinstance(inputs, str):
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        inputs = tokenizer.encode(inputs).ids
    if (not isinstance(inputs, (list, tuple)) or not 1 <= len(inputs) <= MAX_CONTEXT
            or any(type(token) is not int or not 0 <= token < vocabulary for token in inputs)):
        raise ValueError("Require 1..2048 integer token IDs within the checkpoint vocabulary")
    return list(inputs)


def reserve_cache_layouts(templates, context_length):
    """Resize fresh byte arenas and reserve disjoint group pages for the complete prompt."""
    if len(templates) != 16 or not 1 <= context_length <= MAX_CONTEXT:
        raise ValueError("Require all 16 cache-layout templates and a bounded complete prompt")
    pages = (context_length + PAGE_ROWS - 1) // PAGE_ROWS
    layouts, group_pages, maximum = [], [], 0
    for template in templates:
        layout, occupied, allocations, covered = deepcopy(template["layout"]), {}, [], set()
        for group in template["groups"]:
            names = group["names"]
            if not names or covered.intersection(names):
                raise ValueError("Cache groups must partition the original typed cache names")
            covered.update(names)
            keys = {plane["storage_key"] for name in names for plane in
                    (layout["views"][name] if isinstance(layout["views"][name], list) else [layout["views"][name]])}
            first = max((occupied.get(key, 1) for key in keys), default=1)
            owned = list(range(first, first + pages))
            for key in keys:
                occupied[key] = first + pages
            allocations.append(owned)
            maximum = max(maximum, first + pages)
        if covered != set(layout["views"]):
            raise ValueError("Cache groups do not cover the original typed cache layout")
        layouts.append(layout)
        group_pages.append(allocations)
    blocks = maximum + 1
    for layout in layouts:
        strides = {}
        for descriptor in layout["views"].values():
            for plane in descriptor if isinstance(descriptor, list) else [descriptor]:
                size = torch.empty((), dtype=getattr(torch, plane["dtype"].split(".")[-1])).element_size()
                page_bytes = plane["stride"][0] * size
                key = plane["storage_key"]
                if key in strides and strides[key] != page_bytes:
                    raise ValueError("Aliased typed planes disagree on the original byte-page stride")
                strides[key] = page_bytes
                plane["shape"][0] = blocks
                plane["storage_bytes"] = blocks * page_bytes
        for key, raw in layout["raw"].items():
            raw["shape"] = [blocks * strides[key]]
            raw["storage_bytes"] = raw["shape"][0]
    return layouts, group_pages, blocks - 1


def build_chunk_metadata(template, layout, allocations, *, tokens, context, tp_rank, active, reserved):
    """Build CPU controls for SWA, completed compression pairs and per-request ring state."""
    total = len(tokens) if active else 0
    positions = torch.arange(context, context + total, dtype=torch.int64)
    physical = (total + 3) // 4
    start, end = tp_rank * physical, (tp_rank + 1) * physical
    span = [start, end, physical, physical * 4]
    valid = max(0, min(end, total) - min(start, total))
    lengths = torch.tensor([context + total] if active else [], dtype=torch.int32)
    starts = torch.tensor([0, total] if active else [0], dtype=torch.int32)
    metadata, groups = {}, []
    for source, pages in zip(template["groups"], allocations, strict=True):
        table = torch.tensor([pages], dtype=torch.int32) if active else torch.empty((0, len(pages)), dtype=torch.int32)
        raw_slots = [[pages[int(position) // PAGE_ROWS], int(position) % PAGE_ROWS] for position in positions]
        group = dict(group_id=source["group_id"], names=list(source["names"]), block_table=table,
            slot_mapping=torch.tensor(raw_slots, dtype=torch.int64).reshape(total, 2),
            live_block_ids=list(pages) if active else [], reserved_untouched_block_id=reserved)
        groups.append(group)
        for name in source["names"]:
            layer = int(name.split(".")[2])
            role = resolve_layer_role(layer)
            compressed = name.endswith(("long_kv_cache", "indexer.k_cache"))
            ring = name.endswith("compressor.state_cache")
            ratio = role["ratio"] if compressed else 1
            slots = []
            for position in positions.tolist():
                if ring or (position + 1) % ratio:
                    slots.append([-1, -1])
                else:
                    slots.append([pages[position // PAGE_ROWS], position % PAGE_ROWS // ratio])
            global_metadata = dict(block_table=table[:, :1] if ring else table,
                query_start_loc=starts.clone(), seq_lens=lengths.clone(), positions=positions.clone(),
                slot_mapping=torch.tensor(slots, dtype=torch.int64).reshape(total, 2),
                cache_seq_lens=lengths // ratio, cmp_residual=lengths % ratio if ratio == 2 else None,
                num_actual_tokens=total, num_reqs=len(lengths), max_query_len=total,
                max_cache_seq_len=int((lengths // ratio).max()) if active else 0)
            if ring:
                global_metadata["c2_source_positions"] = torch.where(positions % 2 == 1, positions - 1, 0)
            metadata[name] = dict(cp_token_range=span, num_actual_tokens=valid, global_metadata=global_metadata)
    return dict(layout=layout, metadata=metadata, groups=groups, positions=positions, lengths=lengths,
                query_start_loc=starts, hash_constants=template["hash_constants"])


def prepare_token_chunks(inputs, *, checkpoint, cache, config, output, chunk_size=16, dp_rank=0):
    """Yield prepared chunks lazily while retaining a complete fixed page reservation."""
    topology = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)
    if (config["topology"] != topology or type(chunk_size) is not int or not 1 <= chunk_size <= 16
            or type(dp_rank) is not int or not 0 <= dp_rank < 4):
        raise ValueError("Automatic chunks require DP4/TP4/EP16, chunk_size 1..16 and dp_rank 0..3")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    model = json.loads((checkpoint / "config.json").read_text())
    if model != config["model_config"]:
        raise ValueError("Prepared model configuration differs from the selected checkpoint")
    tokens = tokenize_prompt(inputs, checkpoint, model["text_config"]["vocab_size"])
    layouts, allocations, reserved = reserve_cache_layouts(cache["cache_ABI"], len(tokens))
    identities = {name: file_sha(checkpoint / name) for name in
                  ("config.json", "quant_model_weights.safetensors.index.json", "tokenizer.json")
                  if (checkpoint / name).is_file()}
    output.mkdir(parents=True, exist_ok=False)
    predecessor = None
    for epoch, begin in enumerate(range(0, len(tokens), chunk_size)):
        directory = output / f"chunk-{epoch}"
        directory.mkdir()
        current = tokens[begin:begin + chunk_size]
        request = dict(request_slot=0, token_ids=current, position_ids=list(range(begin, begin + len(current))),
                       live_token_mask=[1] * len(current), initial_context_length=begin, history_tokens=tokens[:begin])
        engines = [dict(dp_rank=rank, active=len(current) if rank == dp_rank else 0,
                        requests=[request] if rank == dp_rank else []) for rank in range(4)]
        actual_input = dict(topology=topology, checkpoint=str(checkpoint), checkpoint_sha256=identities,
            epoch=epoch, predecessor_input_sha256=predecessor, DP_engines=engines)
        input_path = directory / "input-contract.json"
        write_json(input_path, actual_input)
        identity = file_sha(input_path)
        payloads, sources = [], []
        for rank, template in enumerate(cache["cache_ABI"]):
            payload = build_chunk_metadata(template, layouts[rank], allocations[rank], tokens=current, context=begin,
                tp_rank=rank % 4, active=rank // 4 == dp_rank, reserved=reserved)
            path = directory / f"rank-{rank}-metadata.pt"
            torch.save(payload, path)
            sources.append(dict(rank=rank, path=str(path), sha256=file_sha(path)))
            payloads.append(payload)
        regions = build_state_regions(payloads, identity)
        region_path = directory / "state-regions.json"
        write_json(region_path, regions)
        contract = dict(config, checkpoint=str(checkpoint), input_contract_path=str(input_path),
            input_contract_sha256=identity, state_regions=dict(path=str(region_path), sha256=file_sha(region_path)))
        prepared_inputs = dict(input_contract=actual_input, prepare_input_stage=True)
        prepared_cache = dict(cache, cache_ABI=payloads,
            cache_ABI_sources=[dict(rank=row["rank"], payload_sha256=row["sha256"], report_sha256=row["sha256"])
                               for row in sources])
        yield dict(inputs=prepared_inputs, contract=contract, metadata_sources=sources, cache=prepared_cache)
        predecessor = identity


def validate_host():
    """Check 2048-token planning, all layer coordinates and exact state write boundaries."""
    from tempfile import TemporaryDirectory

    from .prefill_network_state import NetworkCacheProvider, _check_layout
    from .prefill_network_cache import network_cache_geometry, owned_cache_pages
    from .prefill_indexer_runtime import validate_live_refresh_candidates

    views = {}
    groups = []
    for layer in range(40):
        root = f"model.layers.{layer}.self_attn."
        groups.append(dict(group_id=len(groups), names=[root + "swa_cache"]))
        views[root + "swa_cache"] = (128, 512, "torch.bfloat16")
        role = resolve_layer_role(layer)
        if role["mode"] == "full":
            compressed = [root + "long_kv_cache", root + "indexer.k_cache"]
            groups.append(dict(group_id=len(groups), names=compressed))
            views[compressed[0]] = (128 // role["ratio"], 512, "torch.bfloat16")
            views[compressed[1]] = [(128 // role["ratio"], 128, "torch.int8"),
                                    (128 // role["ratio"], 1, "torch.float16")]
            if role["ratio"] == 2:
                groups.append(dict(group_id=len(groups), names=[root + "compressor.state_cache"]))
                views[root + "compressor.state_cache"] = (32, 1024, "torch.float32")
    descriptors, aliases = {}, {"shared": [], "index": []}
    for name, dimensions in views.items():
        planes = dimensions if isinstance(dimensions, list) else [dimensions]
        items = []
        for plane_id, (rows, width, dtype) in enumerate(planes):
            size = torch.empty((), dtype=getattr(torch, dtype.split(".")[-1])).element_size()
            key = "index" if name.endswith("indexer.k_cache") and rows == 128 else "shared"
            offset = (65536 + (8192 if plane_id else 0)) if name.endswith("indexer.k_cache") and rows == 64 else 0
            if key == "index" and plane_id:
                offset = 16384
            items.append(dict(storage_key=key, dtype=dtype, shape=[2, rows, 1, width],
                stride=[131072 // size, width, width, 1], storage_offset_bytes=offset, storage_bytes=262144))
            if name not in aliases[key]:
                aliases[key].append(name)
        descriptors[name] = items if isinstance(dimensions, list) else items[0]
    raw = {key: dict(storage_key=key, dtype="torch.int8", shape=[262144], stride=[1],
                    storage_offset_bytes=0, storage_bytes=262144) for key in aliases}
    layout = dict(raw=raw, views=descriptors, aliases=aliases)
    template = dict(layout=layout, groups=groups, hash_constants={})
    cache = dict(cache_ABI=[deepcopy(template) for _ in range(16)], engram_contracts={1: {}, 14: {}})
    topology = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)
    model = dict(text_config=dict(vocab_size=4096))
    provider = object.__new__(NetworkCacheProvider)
    provider.topology, provider.capacity = topology, 16
    checked, consumed = 0, []
    with TemporaryDirectory(prefix="a3-prefill-prepare-") as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        write_json(checkpoint / "config.json", model)
        config = dict(topology=topology, model_config=model)
        chunks = prepare_token_chunks(list(range(2048)), checkpoint=checkpoint, cache=cache,
                                      config=config, output=root / "prepared")
        previous = None
        for epoch, packet in enumerate(chunks):
            actual = packet["inputs"]["input_contract"]
            request = actual["DP_engines"][0]["requests"][0]
            if actual["predecessor_input_sha256"] != previous or request["history_tokens"] != consumed:
                raise RuntimeError("Automatic chunks lost the immediate input lineage or history")
            consumed.extend(request["token_ids"])
            previous = packet["contract"]["input_contract_sha256"]
            if epoch not in (0, 7, 8, 31, 32, 63, 64, 127):
                continue
            payloads = packet["cache"]["cache_ABI"]
            provider.layouts = [{"fresh": payload["layout"]} for payload in payloads]
            for rank, payload in enumerate(payloads):
                _check_layout(payload["layout"])
                for layer in range(40):
                    record = provider._record("fresh", layer, rank, payload, actual, [16, 0, 0, 0])
                    role = resolve_layer_role(layer)
                    if role["mode"] != "swa":
                        blocks = payload["layout"]["views"][f"model.layers.{layer}.self_attn.swa_cache"]["shape"][0]
                        geometry = network_cache_geometry(payload["layout"], role, blocks)
                        source_root = f"model.layers.{role['kv_source']}.self_attn."
                        for name, plane in (("long_kv_cache", "long"), ("indexer.k_cache", "codes")):
                            owned_cache_pages(record, source_root + name, geometry[plane])
                    checked += 1
            payload = payloads[0]
            for source in packet["metadata_sources"]:
                if file_sha(source["path"]) != source["sha256"]:
                    raise RuntimeError("Prepared rank metadata identity changed")
            own = payload["metadata"]["model.layers.0.self_attn.swa_cache"]["global_metadata"]
            slots = own["slot_mapping"].tolist()
            table = own["block_table"][0].tolist()
            expected = [[table[position // 128], position % 128] for position in range(epoch * 16, epoch * 16 + 16)]
            if slots != expected:
                raise RuntimeError("A chunk crossed a page boundary with the wrong physical slots")
            regions = json.loads(Path(packet["contract"]["state_regions"]["path"]).read_text())
            intervals = regions["ranks"]["0"]["layers"][0]["raw_readonly_untouched_complement"]["shared"]
            for page, row in slots:
                offset = page * 131072 + row * 1024
                if any(begin <= offset < end for begin, end in intervals):
                    raise RuntimeError("Writable SWA bytes were marked readonly")
            if not any(begin == 0 and end >= 131072 for begin, end in intervals):
                raise RuntimeError("Reserved page zero lost exact untouched protection")
        if consumed != list(range(2048)) or epoch != 127:
            raise RuntimeError("Automatic chunks skipped or duplicated prompt tokens")
        record = provider._record("fresh", 24, 0, packet["cache"]["cache_ABI"][0], actual, [16, 0, 0, 0])
        candidates = torch.full((16, 1, 2048), -1, dtype=torch.int32)
        candidates[:, 0, :256] = torch.arange(256, dtype=torch.int32)
        validate_live_refresh_candidates(record, {"candidates": candidates}, 20)
        candidates[:, 0, 64] = -1
        try:
            validate_live_refresh_candidates(record, {"candidates": candidates}, 20)
        except ValueError:
            pass
        else:
            raise RuntimeError("Layer24 accepted a candidate pool missing a visible historical block")
        if cache["cache_ABI"][0]["layout"] != layout:
            raise RuntimeError("Reserving automatic pages mutated the caller's cache template")
        layouts, allocations, reserved = reserve_cache_layouts(cache["cache_ABI"], 2048)
        for begin in (0, 1, 7, 15, 31, 127, 128, 511, 512, 1023, 1024, 2047):
            current = list(range(begin, min(2048, begin + 7)))
            payload = build_chunk_metadata(template, layouts[12], allocations[12], tokens=current,
                context=begin, tp_rank=0, active=True, reserved=reserved)
            root_name = "model.layers.2.self_attn."
            compressed = payload["metadata"][root_name + "long_kv_cache"]["global_metadata"]
            table = compressed["block_table"][0].tolist()
            expected = [[table[pos // 128], pos % 128 // 2] if pos % 2 else [-1, -1] for pos in current]
            if compressed["slot_mapping"].tolist() != expected:
                raise RuntimeError("An odd chunk boundary published an incomplete compression pair")
            if compressed["cache_seq_lens"].tolist() != [(begin + len(current)) // 2]:
                raise RuntimeError("C2 completed history length changed at an odd chunk boundary")
            ring = payload["metadata"][root_name + "compressor.state_cache"]["global_metadata"]
            if ring["c2_source_positions"].tolist() != [pos - 1 if pos % 2 else 0 for pos in current]:
                raise RuntimeError("C2 RoPE positions lost the preceding token at an odd chunk boundary")
            regions = build_state_regions([payload] * 16, "boundary-input")
            if begin:
                own = payload["metadata"]["model.layers.0.self_attn.swa_cache"]["global_metadata"]
                previous_page = int(own["block_table"][0, (begin - 1) // 128])
                offset = previous_page * 131072 + ((begin - 1) % 128) * 1024
                untouched = regions["ranks"]["0"]["layers"][0]["raw_readonly_untouched_complement"]["shared"]
                if not any(left <= offset and offset + 1024 <= right for left, right in untouched):
                    raise RuntimeError("Preparing a chunk allowed an earlier SWA history row to change")
        chunks = list(prepare_token_chunks(list(range(129)), checkpoint=checkpoint, cache=cache,
            config=config, output=root / "odd-chunks", chunk_size=15, dp_rank=3))
        if len(chunks) != 9 or chunks[-1]["inputs"]["input_contract"]["DP_engines"][3]["active"] != 9:
            raise RuntimeError("Automatic Prefill lost the partial final chunk")
        for packet in chunks:
            for rank, payload in enumerate(packet["cache"]["cache_ABI"]):
                provider.layouts = [{"fresh": row["layout"]} for row in packet["cache"]["cache_ABI"]]
                active = packet["inputs"]["input_contract"]["DP_engines"][3]["active"]
                provider._record("fresh", 2, rank, payload, packet["inputs"]["input_contract"], [0, 0, 0, active])
        for invalid in ([], [True], [-1], [4096], list(range(2049))):
            try:
                tokenize_prompt(invalid, checkpoint, 4096)
            except ValueError:
                continue
            raise RuntimeError("Invalid automatic input was accepted")
    return dict(tokens=2048, chunks=128, layer_rank_coordinates=checked, odd_boundaries="checked", final_tail="checked",
                layer24_candidates="checked",
                NPU_numerics="not_run")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-host", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate_host(), sort_keys=True))
