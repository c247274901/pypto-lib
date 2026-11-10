# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check metadata-defined raw, shared and history preservation around each complete network layer."""

import hashlib
import json
from pathlib import Path

import torch

from .runtime_utils import file_sha, write_json
from .prefill_layer_state import resolve_layer_role


def build_state_regions(payloads, input_contract_sha256, scheduler_capacity=8192):
    """Derive exact write complements from chunk slots, ring rows and layer roles."""
    def complement(extent, intervals):
        cursor, result = 0, []
        for begin, end in sorted(intervals):
            if not 0 <= begin < end <= extent:
                raise ValueError("A prepared write interval escapes its original byte arena")
            if cursor < begin:
                result.append([cursor, begin])
            cursor = max(cursor, end)
        if cursor < extent:
            result.append([cursor, extent])
        return result

    ranks = {}
    for rank, payload in enumerate(payloads):
        layout, metadata = payload["layout"], payload["metadata"]
        extents = {key: value["shape"][0] for key, value in layout["raw"].items()}
        layers = []
        for layer in range(40):
            role = resolve_layer_role(layer)
            root = f"model.layers.{layer}.self_attn."
            writable = [root + "swa_cache"]
            if role["mode"] == "full":
                writable += [root + "long_kv_cache", root + "indexer.k_cache"]
                if role["ratio"] == 2:
                    writable.append(root + "compressor.state_cache")
            intervals = {key: [] for key in extents}
            for name in writable:
                controls = metadata[name]["global_metadata"]
                slots = controls["slot_mapping"].tolist()
                if name.endswith("compressor.state_cache"):
                    starts = payload["query_start_loc"].tolist()
                    table = controls["block_table"].tolist()
                    slots = [[table[request][0], int(payload["positions"][token]) % 32]
                             for request, (begin, end) in enumerate(zip(starts, starts[1:]))
                             for token in range(begin, end)]
                descriptor = layout["views"][name]
                for plane in descriptor if isinstance(descriptor, list) else [descriptor]:
                    dtype = getattr(torch, plane["dtype"].split(".")[-1])
                    size = torch.empty((), dtype=dtype).element_size()
                    for page, row in slots:
                        if page < 0:
                            continue
                        offset = plane["storage_offset_bytes"] + (page * plane["stride"][0] + row * plane["stride"][1]) * size
                        intervals[plane["storage_key"]].append([offset, offset + plane["shape"][-1] * size])
            active = metadata[root + "swa_cache"]["num_actual_tokens"]
            shared = {}
            for name in ("topk_indices", "topk_lengths", "candidates", "candidate_lengths"):
                writes = role["index_selection_write"] if name.startswith("topk_") else role["candidate_write"]
                if name == "candidate_lengths":
                    writes = False
                shared[name] = [[active if writes else 0, scheduler_capacity]]
            layers.append(dict(layer=layer,
                raw_readonly_untouched_complement={key: complement(extent, intervals[key]) for key, extent in extents.items()},
                shared_readonly_untouched_rows=shared))
        ranks[str(rank)] = dict(raw_extents=extents, layers=layers)
    return dict(complete_rank_metadata=len(payloads) == 16, result_dependent_masks=False,
                input_contract_sha256=input_contract_sha256, ranks=ranks)


def interval_digest(value, intervals, identity):
    """Hash exact CPU row intervals and their identities in the frozen order."""
    if value.device.type != "cpu" or value.ndim == 0:
        raise ValueError("State preservation requires actual CPU staging with an explicit row axis")
    digest = hashlib.sha256()
    description = dict(identity=identity, shape=list(value.shape), dtype=str(value.dtype), intervals=intervals)
    header = json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    previous = 0
    for begin, end in intervals:
        if not 0 <= previous <= begin < end <= len(value):
            raise ValueError("State preservation intervals overlap or escape actual storage")
        payload = value[begin:end].contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        previous = end
    return digest.hexdigest()


class NetworkStatePreservation:
    """Retain compact exact complement checks from independently admitted all-rank metadata."""

    def __init__(self, provider, regions_path, regions_sha256, input_contract_sha256):
        path = Path(regions_path)
        if file_sha(path) != regions_sha256:
            raise ValueError("Frozen network state region identity changed")
        self.regions = json.loads(path.read_text())
        if (not self.regions["complete_rank_metadata"] or self.regions["result_dependent_masks"]
                or self.regions["input_contract_sha256"] != input_contract_sha256):
            raise ValueError("Require actual all-rank regions derived from the identical input before any outputs")
        if set(self.regions["ranks"]) != {str(rank) for rank in range(provider.topology["ep"])}:
            raise ValueError("State preservation must cover every actual allocated rank")
        self.provider, self.regions_sha256 = provider, regions_sha256
        self.pending, self.completed = None, []
        for rank, cases in enumerate(provider.arenas):
            if set(cases) != {"fresh"}:
                raise ValueError("This state contract covers the admitted initial fresh network chunk")
            record = self.regions["ranks"][str(rank)]
            extents = {name: value.numel() for name, value in cases["fresh"].items()}
            if extents != record["raw_extents"]:
                raise ValueError("State regions have a different actual raw allocation")
            if [item["layer"] for item in record["layers"]] != list(range(40)):
                raise ValueError("State preservation lacks an original physical layer")

    def _observe(self, layer):
        result = {}
        for rank, cases in enumerate(self.provider.arenas):
            regions = self.regions["ranks"][str(rank)]["layers"][layer]
            for case, raw in cases.items():
                identity = dict(rank=rank, case=case, layer=layer, regions_sha256=self.regions_sha256)
                for name, value in raw.items():
                    intervals = regions["raw_readonly_untouched_complement"][name]
                    key = f"rank{rank}/{case}/raw/{name}"
                    result[key] = interval_digest(value, intervals, dict(identity, storage=key))
                for name, value in self.provider.shared[rank][case].items():
                    intervals = regions["shared_readonly_untouched_rows"][name]
                    key = f"rank{rank}/{case}/shared/{name}"
                    result[key] = interval_digest(value, intervals, dict(identity, storage=key))
                history = self.provider.history.caches[rank][case]
                key = f"rank{rank}/{case}/singleton_hash_history"
                result[key] = interval_digest(history, [[0, len(history)]], dict(identity, storage=key))
        return result

    def begin_layer(self, layer):
        """Observe every exact complement before this layer can write its own allowed rows."""
        if self.pending is not None or layer != len(self.completed) or not 0 <= layer < 40:
            raise ValueError("State observations require one complete original layer at a time")
        self.pending = (layer, self._observe(layer))

    def finish_layer(self, layer, output):
        """Publish the real before/after result and reject any readonly or untouched mutation."""
        if self.pending is None or self.pending[0] != layer:
            raise ValueError("A complete layer has no matching state observation")
        before, after = self.pending[1], self._observe(layer)
        changed = [key for key in before if after.get(key) != before[key]]
        receipt = dict(layer=layer, regions_sha256=self.regions_sha256, before=before, after=after,
                       exact=not changed, changed=changed,
                       scope="Exact raw/shared complements and post-preparation singleton history only")
        write_json(Path(output), receipt)
        if changed:
            raise RuntimeError("A complete network layer changed readonly or untouched state: " + str(changed))
        self.completed.append(layer)
        self.pending = None
        return receipt
