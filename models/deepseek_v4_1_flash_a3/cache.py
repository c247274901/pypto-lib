# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cache layouts, source ownership, geometry and legal state lifetimes."""

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import pypto.language as pl
import torch
from .swa import make_swa_leaves
from .indexer import INDEXER_SELECT_KEY_CAPACITY as KEY_CAPACITY


# tiling
COPY_TILE = 128


def make_paged_cache_copy(
    blocks,
    payload,
    source_stride,
    destination_stride,
    source_offset=0,
    destination_offset=0,
    dtype=pl.BF16,
    first_page=0,
    copy_tile=COPY_TILE,
):
    """Copy one contiguous payload per page while preserving unrelated bytes."""
    if blocks < 1 or payload < 1 or copy_tile not in (64, 128) or payload % copy_tile:
        raise ValueError("Require positive pages with aligned contiguous payloads")
    if not 0 <= source_offset <= source_stride - payload:
        raise ValueError("Source cache payload exceeds its physical page")
    if not 0 <= destination_offset <= destination_stride - payload:
        raise ValueError("Destination cache payload exceeds its physical page")

    if not 0 <= first_page < blocks:
        raise ValueError("Require a nonempty owned page range")

    @pl.jit
    def paged_cache_copy(
        source: pl.Tensor[[blocks, source_stride], dtype],
        destination: pl.InOut[pl.Tensor[[blocks, destination_stride], dtype]],
    ):
        for block in pl.spmd(
            (blocks - first_page) * (payload // copy_tile), name_hint="prefill_paged_cache_copy"
        ):
            page = first_page + block // (payload // copy_tile)
            offset = block % (payload // copy_tile) * copy_tile
            source_column = source_offset + offset
            destination_column = destination_offset + offset
            value = source[page : page + 1, source_column : source_column + copy_tile]
            destination[page : page + 1, destination_column : destination_column + copy_tile] = value
        return destination

    return paged_cache_copy


def physical_page_geometry(layout, prefix, blocks):
    """Validate contiguous page payloads in the original shared raw cache ABI."""
    views = layout["views"]
    long = views[prefix + "long_kv_cache"]
    if views[prefix + "swa_cache"] != long:
        raise ValueError("Ratio1 long and SWA must retain the same typed physical arena")
    descriptors = dict(
        long=long, codes=views[prefix + "indexer.k_cache"][0], scales=views[prefix + "indexer.k_cache"][1]
    )
    geometry = {}
    for name, descriptor in descriptors.items():
        width = 512 if name == "long" else 128 if name == "codes" else 1
        dtype = "torch.bfloat16" if name == "long" else "torch.int8" if name == "codes" else "torch.float16"
        size = 1 if name == "codes" else 2
        shape, stride = descriptor["shape"], descriptor["stride"]
        key = descriptor["storage_key"]
        backing = layout["raw"][key]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        if shape != [blocks, 128, 1, width] or stride[1:] != [width, width, 1]:
            raise ValueError("Cache payload must retain its original 128-row typed page")
        if descriptor["dtype"] != dtype or offset < 0 or offset % size:
            raise ValueError("Cache payload dtype or byte offset differs")
        if backing["shape"] != [blocks * stride[0] * size] or backing["stride"] != [1]:
            raise ValueError("Raw cache backing does not cover the complete strided pages")
        payload = 128 * width
        if offset // size + payload > stride[0]:
            raise ValueError("Cache payload escapes its page")
        geometry[name] = dict(
            storage_key=key,
            dtype=dtype,
            stride=stride[0],
            payload=payload,
            offset=offset // size,
            width=width,
        )
    return geometry


def physical_consumer_geometry(layout, own_prefix, source_prefix, blocks):
    """Bind the shared ratio1 payload while preserving every source-owned raw page."""
    own = layout["views"][own_prefix + "swa_cache"]
    source = layout["views"][source_prefix + "long_kv_cache"]
    key = own["storage_key"]
    backing = layout["raw"][key]
    stride = own["stride"][0]
    if (
        source["storage_key"] != key
        or own["shape"] != [blocks, 128, 1, 512]
        or own["dtype"] != source["dtype"]
        or own["dtype"] != "torch.bfloat16"
        or own["stride"] != source["stride"]
        or own["stride"][1:] != [512, 512, 1]
        or own["storage_offset_bytes"] != source["storage_offset_bytes"]
        or source["shape"][1:] != [128, 1, 512]
    ):
        raise ValueError("Require the original shared source and own BF16 page payloads")
    offset = own["storage_offset_bytes"] - backing["storage_offset_bytes"]
    first_owned = source["shape"][0]
    if (
        offset < 0
        or offset % 2
        or offset // 2 + 65536 > stride
        or not 0 < first_owned < blocks
        or backing["shape"] != [blocks * stride * 2]
        or backing["stride"] != [1]
    ):
        raise ValueError("Missing readonly source prefix or complete raw page backing")
    return dict(storage_key=key, stride=stride, payload=65536, offset=offset // 2, first_owned=first_owned)


def physical_index_geometry(layout, prefix, blocks):
    """Validate readonly source index payloads in the shared consumer backing."""
    geometry = {}
    for name, descriptor in zip(("codes", "scales"), layout["views"][prefix + "indexer.k_cache"]):
        width, size = (128, 1) if name == "codes" else (1, 2)
        dtype = "torch.int8" if name == "codes" else "torch.float16"
        backing = layout["raw"][descriptor["storage_key"]]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        stride = descriptor["stride"][0]
        if (
            descriptor["dtype"] != dtype
            or (not 0 < descriptor["shape"][0] <= blocks or descriptor["shape"][1:] != [128, 1, width])
            or descriptor["stride"][1:] != [width, width, 1]
            or offset < 0
            or offset % size
            or backing["shape"] != [blocks * stride * size]
            or backing["stride"] != [1]
            or offset // size + 128 * width > stride
        ):
            raise ValueError("Readonly source index page geometry differs")
        geometry[name] = dict(
            storage_key=descriptor["storage_key"],
            dtype=dtype,
            stride=stride,
            payload=128 * width,
            offset=offset // size,
        )
    return geometry


@lru_cache(maxsize=1)
def _layer_plan():
    with Path(__file__).with_name("prefill_layer_plan.json").open(encoding="utf-8") as stream:
        plan = json.load(stream)
    roles = plan["layers"]
    if [role["physical_layer"] for role in roles] != list(range(40)):
        raise ValueError("Require the complete original physical-layer plan")
    return roles


def resolve_layer_role(layer_id, role=None):
    """Resolve original pure-data semantics without importing a reference model."""
    if type(layer_id) is not int or not 0 <= layer_id < 40:
        raise ValueError("Require an original physical layer in 0..39")
    expected = _layer_plan()[layer_id]
    if role is not None and any(role.get(name) != value for name, value in expected.items()):
        raise ValueError("Supplied role differs from the original physical-layer plan")
    return dict(expected)


def validate_live_state(role, original, live, epoch, token_map_id):
    """Reject stale selections and missing actual arenas before any layer mutation."""
    if role["engram_slot"] is not None:
        prepared = live.get("network_engram")
        if prepared is not None:
            if (
                prepared["shared_history_owner"] != 1
                or prepared["physical_layer"] != role["physical_layer"]
                or prepared["slot"] != role["engram_slot"]
                or prepared["epoch"] != epoch
                or prepared["token_map_id"] != token_map_id
            ):
                raise ValueError(
                    "The network layer requires its current slot from shared model-level history"
                )
        elif "hash_cache" not in live or live.get("hash_owner") != role["physical_layer"]:
            raise ValueError("Engram requires its own persistent request hash-cache state")
    if role["mode"] == "swa":
        if "cache" not in live:
            raise ValueError("SWA requires its request-owned live cache")
        return
    if not all(name in live for name in ("raw", "layout", "shared", "versions")):
        raise ValueError(
            "Compressed attention requires actual live backing, layout, selection and owner versions"
        )
    metadata = original["cache_metadata"]
    if metadata.get("physical_layer_id", role["physical_layer"]) != role["physical_layer"]:
        raise ValueError("Cache metadata differs from the requested physical layer")
    if role["mode"] != "full":
        if metadata["source_physical_layer_id"] != role["kv_source"]:
            raise ValueError("Readonly cache metadata differs from the original KV source")
        required = [("kv", role["kv_source"])]
        if role["mode"] != "reindex":
            required.append(("index", role["index_source"]))
        if role["candidate_filter"]:
            required.append(("candidate", role["candidate_source"]))
        for name, owner in required:
            if tuple(live["versions"].get(name, ())) != (owner, epoch, token_map_id):
                raise ValueError("Missing or stale live " + name + " ownership for this token map")
    if role["candidate_filter"]:
        validate_live_refresh_candidates(original, live["shared"], role["kv_source"])


def publish_live_versions(role, live, epoch, token_map_id):
    """Publish owner versions only after the layer has completed its real computation."""
    for name, writes, owner in (
        ("kv", role["kv_write"], role["kv_source"]),
        ("index", role["index_selection_write"], role["index_source"]),
        ("candidate", role["candidate_write"], role["candidate_source"]),
    ):
        if writes:
            live["versions"][name] = (owner, epoch, token_map_id)


def invalidate_live_versions(role, live):
    """Clear pending publications so a failed retry cannot retain a valid owner tag."""
    for name, writes in (
        ("kv", role["kv_write"]),
        ("index", role["index_selection_write"]),
        ("candidate", role["candidate_write"]),
    ):
        if writes:
            live["versions"].pop(name, None)


DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}
PTO_DTYPES = {"torch.bfloat16": pl.BF16, "torch.int8": pl.INT8, "torch.float16": pl.FP16}


def network_cache_geometry(layout, role, blocks):
    """Bind complete raw strides and original typed plane offsets without alias remapping."""
    role = resolve_layer_role(role["physical_layer"], role)
    if role["mode"] not in ("full", "reuse", "reindex") or role["ratio"] not in (1, 2):
        raise ValueError("The network cache adapter requires an original compressed role")
    root = f"model.layers.{role['kv_source']}.self_attn."
    own_root = f"model.layers.{role['physical_layer']}.self_attn."
    views = layout["views"]
    descriptors = dict(
        long=views[root + "long_kv_cache"],
        own=views[own_root + "swa_cache"],
        codes=views[root + "indexer.k_cache"][0],
        scales=views[root + "indexer.k_cache"][1],
    )
    result = {}
    for name, descriptor in descriptors.items():
        rows = 128 if name == "own" else 128 // role["ratio"]
        width = 512 if name in ("long", "own") else 128 if name == "codes" else 1
        dtype = (
            "torch.bfloat16"
            if name in ("long", "own")
            else "torch.int8"
            if name == "codes"
            else "torch.float16"
        )
        size = 1 if name == "codes" else 2
        shape, stride = descriptor["shape"], descriptor["stride"]
        backing = layout["raw"][descriptor["storage_key"]]
        offset = descriptor["storage_offset_bytes"] - backing["storage_offset_bytes"]
        if (
            descriptor["dtype"] != dtype
            or shape != [blocks, rows, 1, width]
            or stride[1:] != [width, width, 1]
            or offset < 0
            or offset % size
            or backing["shape"] != [blocks * stride[0] * size]
            or backing["stride"] != [1]
            or offset // size + rows * width > stride[0]
        ):
            raise ValueError("Original full-network " + name + " plane geometry is unsupported")
        result[name] = dict(
            storage_key=descriptor["storage_key"],
            dtype=dtype,
            blocks=blocks,
            stride=stride[0],
            offset=offset // size,
            rows=rows,
            width=width,
            payload=rows * width,
        )
    if role["ratio"] == 2 and role["mode"] == "full":
        ring = views[root + "compressor.state_cache"]
        backing = layout["raw"][ring["storage_key"]]
        if (
            ring["dtype"] != "torch.float32"
            or ring["shape"] != [blocks, 32, 1, 1024]
            or ring["stride"] != [32768, 1024, 1024, 1]
            or ring["storage_offset_bytes"] != 0
            or backing["shape"] != [blocks * 131072]
        ):
            raise ValueError("C2 requires its actual contiguous32x1024 FP32 ring plane")
        result["ring"] = dict(
            storage_key=ring["storage_key"],
            dtype="torch.float32",
            blocks=blocks,
            stride=32768,
            offset=0,
            rows=32,
            width=1024,
            payload=32768,
        )
    return result


def raw_cache_matrix(raw, descriptor):
    """Expose one complete contiguous raw arena in the plane's scalar dtype."""
    value = raw[descriptor["storage_key"]]
    return value.view(DTYPES[descriptor["dtype"]]).reshape(descriptor["blocks"], descriptor["stride"])


def owned_cache_pages(original, cache_name, plane):
    """Validate actual owner pages, slots and the reserved untouched page."""
    source = original["cache_metadata"]
    groups = [group for group in source["actual_group_requests"] if cache_name in group["layer_names"]]
    if len(groups) != 1:
        raise ValueError("Require one original scheduler group per cache plane")
    group = groups[0]
    pages = sorted(set(group["live_block_ids"]))
    if group["reserved_untouched_block_id"] in pages or any(
        type(page) is not int or not 0 < page < plane["blocks"] for page in pages
    ):
        raise ValueError("Cache owner pages escape the real allocation or include the untouched reservation")
    metadata = source["actual_metadata"][cache_name]["global_metadata"]
    table_pages = {int(page) for row in metadata["block_table"].tolist() for page in row if int(page) > 0}
    if not table_pages.issubset(set(pages)):
        raise ValueError("Native block table refers to a foreign cache-owner page")
    if cache_name.endswith("compressor.state_cache"):
        return pages
    slots = metadata["slot_mapping"]
    if slots.dtype not in (torch.int32, torch.int64) or list(slots.shape) != [len(original["positions"]), 2]:
        raise ValueError("Network publication requires actual native page/offset slot pairs")
    for page, offset in slots.tolist():
        if page < 0 or offset < 0:
            if (page, offset) != (-1, -1):
                raise ValueError("A padded native slot must retain both negative fields")
        elif page not in pages or not 0 <= offset < plane["rows"]:
            raise ValueError("Publication would write a foreign page or leave its typed payload")
    return pages


def make_network_cache_publish(capacity, plane):
    """Publish current BF16 rows into their real page stride and payload offset."""
    blocks, stride, offset = plane["blocks"], plane["stride"], plane["offset"]
    width = plane["width"]

    @pl.jit
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        raw: pl.InOut[pl.Tensor[[blocks, stride], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="network_cache_plane_publish"):
            page = pl.read(slots, [token, 0])
            row = pl.read(slots, [token, 1])
            if page >= 0:
                page_index = pl.cast(page, pl.INDEX)
                column = pl.cast(offset + row * width, pl.INDEX)
                raw[page_index : page_index + 1, column : column + width] = value[token : token + 1, :]
        return raw

    return publish


def make_network_sparse_read(capacity, config, geometry):
    """Gather buffered groups from distinct cache planes and attend640 slots."""
    long, own = geometry["long"], geometry["own"]
    blocks, long_stride, own_stride = long["blocks"], long["stride"], own["stride"]
    long_offset, own_offset = long["offset"], own["offset"]
    heads, head_dim = config["num_attention_heads"], config["head_dim"]
    width = heads * head_dim
    sparse_config = dict(config, sliding_window=640)
    _publish, _gather, attend = make_swa_leaves(capacity, blocks, sparse_config)

    @pl.jit
    def sparse_read(
        query: pl.Tensor[[capacity, width], pl.BF16],
        indices: pl.Tensor[[capacity, 640], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        long_raw: pl.Tensor[[blocks, long_stride], pl.BF16],
        own_raw: pl.Tensor[[blocks, own_stride], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        selected = pl.create_tensor([capacity * 640, head_dim], dtype=pl.BF16)
        for group in pl.spmd(active * 10, name_hint="network_distinct_cache_gather", allow_early_resolve=True):
            token = group // 10
            start = group % 10 * 64
            group_kv = pl.full([64, head_dim], dtype=pl.BF16, value=0.0)
            for within in pl.pipeline(64, stage=2):
                lane = start + within
                index = pl.read(indices, [token, lane])
                if index >= 0:
                    page = pl.cast(index // 128, pl.INDEX)
                    row = pl.cast(index % 128, pl.INDEX)
                    if lane < 128:
                        column = own_offset + row * head_dim
                        group_kv[within : within + 1, :] = own_raw[page : page + 1, column : column + head_dim]
                    else:
                        column = long_offset + row * head_dim
                        group_kv[within : within + 1, :] = long_raw[page : page + 1, column : column + head_dim]
            group_row = token * 640 + start
            selected[group_row : group_row + 64, :] = group_kv
        attend(query, selected, indices, sink, output, active)
        return output

    return sparse_read


def make_network_cache_entries(capacity, config, geometry, role):
    """Build only actual cache-addressing and owner-page transfer entries."""
    role = resolve_layer_role(role["physical_layer"], role)
    entries = dict(
        network_swa_publish=make_network_cache_publish(capacity, geometry["own"]),
        network_compressed_attention=make_network_sparse_read(capacity, config, geometry),
    )
    if role["mode"] == "full":
        entries["network_compressed_publish"] = make_network_cache_publish(capacity, geometry["long"])
    if role["mode"] == "reuse":
        return entries
    for name in ("codes", "scales"):
        plane = geometry[name]
        payload, stride, offset = plane["payload"], plane["stride"], plane["offset"]
        tile = min(128, payload)
        dtype = PTO_DTYPES[plane["dtype"]]
        entries["network_load_" + name] = make_paged_cache_copy(
            1, payload, stride, payload, source_offset=offset, dtype=dtype, copy_tile=tile
        )
        if role["mode"] == "full":
            entries["network_store_" + name] = make_paged_cache_copy(
                1, payload, payload, stride, destination_offset=offset, dtype=dtype, copy_tile=tile
            )
    return entries


def load_network_index(raw, geometry, pages, execute, rank, case):
    """Stage actual index-owner pages in their native64/128-row typed shapes."""
    result = {}
    for name in ("codes", "scales"):
        plane = geometry[name]
        dense = torch.empty((plane["blocks"], plane["payload"]), dtype=DTYPES[plane["dtype"]])
        physical = raw_cache_matrix(raw, plane)
        for page in pages:
            value = dense[page : page + 1]
            execute(
                "network_load_" + name,
                rank,
                case + f"-page-{page}",
                [physical[page : page + 1], value],
                {"payload": value},
                "physical_cache_load",
            )
        result[name] = dense.reshape(plane["blocks"], plane["rows"], 1, plane["width"])
    return result


def store_network_index(raw, geometry, pages, dense, execute, rank, case):
    """Store only real index-owner pages and preserve aliased ring/SWA pages."""
    for name in ("codes", "scales"):
        plane = geometry[name]
        physical = raw_cache_matrix(raw, plane)
        source = dense[name].reshape(plane["blocks"], plane["payload"])
        for page in pages:
            value = physical[page : page + 1]
            execute(
                "network_store_" + name,
                rank,
                case + f"-page-{page}",
                [source[page : page + 1], value],
                {"raw": value},
                "physical_cache_store",
            )


def validate_network_sparse_indices(indices, original, geometry, active):
    """Reject physical IDs outside the real request's long or own-SWA page maps."""
    starts = original["metadata"]["query_start_loc"].tolist()
    start = original["cp_token_range"][0]
    layer = original["physical_layer_id"]
    role = resolve_layer_role(layer, original.get("role"))
    root = f"model.layers.{role['kv_source']}.self_attn."
    own_root = f"model.layers.{layer}.self_attn."
    metadata = original["cache_metadata"]["actual_metadata"]
    long_table = metadata[root + "long_kv_cache"]["global_metadata"]["block_table"].tolist()
    own_table = metadata[own_root + "swa_cache"]["global_metadata"]["block_table"].tolist()
    for local, values in enumerate(indices[:active].tolist()):
        token = start + local
        request = next(
            index for index in range(len(starts) - 1) if starts[index] <= token < starts[index + 1]
        )
        for lane, value in enumerate(values):
            if value < 0:
                if value != -1:
                    raise ValueError("Sparse padding ID must remain exactly -1")
                continue
            page, offset = divmod(value, 128)
            kind = "own" if lane < 128 else "long"
            table = own_table if lane < 128 else long_table
            if page not in table[request] or not 0 <= offset < geometry[kind]["rows"]:
                raise ValueError("Sparse read would leave its original request-owned typed cache plane")


def attach_network_cache_geometry(provider):
    """Bind original compressed-role addressing without changing computation or owner versions."""
    if provider.scheduler_capacity != 8192:
        raise ValueError("The existing shared publication entries require the original8192-row ABI")
    for rank in range(provider.topology["ep"]):
        for case, layout in provider.layouts[rank].items():
            for layer in range(40):
                provider.state_handles_for_layer(layer)[rank][case]["shared"] = provider.shared[rank][case]
            for layer in range(2, 40):
                role = resolve_layer_role(layer)
                blocks = layout["views"][f"model.layers.{layer}.self_attn.swa_cache"]["shape"][0]
                geometry = network_cache_geometry(layout, role, blocks)
                provider.state_handles_for_layer(layer)[rank][case]["network_geometry"] = deepcopy(geometry)
                provider.geometry[rank][case][layer] = []
    return dict(
        attached_layers=list(range(2, 40)),
        entry_ABI="distinct_actual_raw_cache_planes",
        roles="original_full_reuse_reindex",
        compile="not_run",
        NPU="not_run",
        numerical="not_run",
    )


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
