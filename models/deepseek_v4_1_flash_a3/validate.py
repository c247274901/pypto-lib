# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate import contracts, model roles and bounded HOST controls without device imports."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import ast
import json
from pathlib import Path

from .config import FLASH
from .runtime_config import ExpertParallelConfig
from . import smla_scheduler, swa_scheduler


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    """Fail every invalid subcase rather than reporting a partial pass."""
    if not condition:
        raise RuntimeError(message)


def check_sources():
    """Resolve the maintained local import DAG independently of migration identities."""
    modules = {path.stem: path for path in ROOT.glob("*.py")}
    graph = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        graph[name] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                require(node.level == 1, "Model imports must stay in this flat package")
                imported = [node.module.split(".")[0]] if node.module else [item.name for item in node.names]
                for dependency in imported:
                    require(dependency in modules, "Missing local module: " + dependency)
                    graph[name].add(dependency)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names]
                require(not any(value.startswith("models.") for value in names), "Cross-model coupling is prohibited")
    visited, active = set(), set()

    def visit(name):
        require(name not in active, "Cyclic local import: " + name)
        if name in visited:
            return
        active.add(name)
        for dependency in graph[name]:
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in graph:
        visit(name)
    exports = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    mapping = next(node.value for node in exports.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets))
    for public, (module, symbol) in ast.literal_eval(mapping).items():
        require(module in modules, "Missing public module: " + public)
        tree = ast.parse(modules[module].read_text(encoding="utf-8"))
        defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        defined.update(target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                       for target in ast.walk(node) if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store))
        require(symbol in defined, "Missing public symbol: " + public)
    return dict(modules=len(modules), import_graph="acyclic", public_exports=len(ast.literal_eval(mapping)))



def check_layer_plan():
    """Compare every original physical-layer role with the released model configuration."""
    plan = json.loads((ROOT / "prefill_layer_plan.json").read_text(encoding="utf-8"))["layers"]
    require(len(plan) == FLASH.num_hidden_layers == 40, "All original physical layers are required")
    for layer, role in enumerate(plan):
        expected = FLASH.layer_config(layer)
        require(role["physical_layer"] == layer and role["mode"] == expected.mode.value,
                "Layer numbering or attention mode changed")
        require(role["ratio"] == expected.compression_ratio, "Compression ratio changed")
        require(role["kv_source"] == expected.kv_source_layer_id, "KV ownership changed")
        require(role["index_source"] == expected.index_source_layer_id, "Index ownership changed")
        expected_slot = FLASH.engram_layer_ids.index(layer) if layer in FLASH.engram_layer_ids else None
        require(role["engram_slot"] == expected_slot, "Engram slot ownership changed")
    return dict(original_layers=len(plan), engram_layers=list(FLASH.engram_layer_ids))


def _reject(call, message):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError(message)


def check_controls():
    """Exercise exact EP ownership, empty ranks, causal page mappings and invalid controls."""
    topologies = [(4, 4, 16), (2, 2, 4), (16, 1, 16), (8, 2, 16)]
    for dp, tp, ep in topologies:
        layout = ExpertParallelConfig(dp, tp, ep)
        owned = [expert for rank in range(ep) for expert, slot in layout.ownership(rank)]
        require(owned == list(range(384)), "Global expert coverage changed")
        for rank in range(ep):
            require(all(layout.owner(expert) == rank for expert, slot in layout.ownership(rank)),
                    "Owner and local expert slot disagree")
            require(layout.active_rows({"active_by_rank": [0] * ep}, rank) == 0, "Empty rank became active")
        _reject(lambda: layout.ownership(ep), "Out-of-world expert rank was accepted")
        _reject(lambda: layout.owner(384), "Out-of-range expert was accepted")
        if tp != 1:
            _reject(lambda: layout.active_rows({"active_by_dp": [1] * dp}, 0),
                    "TP partitions must not invent local token counts")
    _reject(lambda: ExpertParallelConfig(4, 4, 8), "Inconsistent EP world was accepted")
    pages = [list(range(request * 65, (request + 1) * 65)) for request in range(16)]
    schedule = swa_scheduler.build([8] * 16, [8192] * 16, pages, cache_blocks=1040, capacity=128)
    require(schedule["active"] == 128 and schedule["query_start_loc"] == list(range(0, 129, 8)),
            "Multiple requests lost their packed row partitions")
    for row, position in enumerate(schedule["positions"]):
        request = row // 8
        logical = list(range(position - 127, position + 1))
        physical = [pages[request][token // 128] * 128 + token % 128 for token in logical]
        require(schedule["logical_window_indices"][row] == logical, "Noncausal SWA window")
        require(schedule["physical_window_indices"][row] == physical, "Cross-request physical page alias")
    fresh = swa_scheduler.build([2, 1], [0, 0], [[1], [0]], cache_blocks=2, capacity=16)
    require(fresh["flat_slot_mapping"] == [128, 129, 0], "Fresh request page permutation changed")
    _reject(lambda: swa_scheduler.build([1, 1], [0, 0], [[0], [0]], cache_blocks=1, capacity=16),
            "Aliased live request pages were accepted")
    _reject(lambda: swa_scheduler.build([True], [0], [[0]], cache_blocks=1, capacity=16),
            "Boolean scheduler control was accepted as an integer")
    selected = [[0] + [-1] * 511]
    compressed = smla_scheduler.from_controls([0, 1], [1], [[0]], [[1]], selected,
                                              ori_blocks=1, cmp_blocks=2, capacity=16)
    require(compressed["physical_compressed_indices"][0][0] == 128, "Compressed selection lost its source page")
    _reject(lambda: smla_scheduler.from_controls([0, 1], [1], [[0]], [[1]], [[1] + [-1] * 511],
                                               ori_blocks=1, cmp_blocks=2, capacity=16),
            "A future compressed token was accepted")
    return dict(expert_topologies=len(topologies), experts=384, scheduling_requests=16, scheduling_active_rows=128)



def check_runtime_configuration():
    """Check explicit TP and device binding without inherited process configuration."""
    import importlib
    import os
    import sys
    from unittest.mock import patch

    from . import runtime_config

    original = sys.argv
    try:
        sys.argv = ["unrelated-tool", "--tp", "not-an-integer"]
        runtime = importlib.reload(runtime_config)
        require(not hasattr(runtime, "TP_SIZE"), "TP must not be frozen at import time")
        require([runtime.validate_tp(tp) for tp in (4, 1, 2, 4)] == [4, 1, 2, 4],
                "Explicit TP specializations depend on import order")
    finally:
        sys.argv = original
    for tp in (True, 0, 3, 8, "4"):
        _reject(lambda value=tp: runtime.validate_tp(value), "Invalid TP accepted")
    require(runtime.validate_devices(["2", "0"], world=2) == ["2", "0"], "Device order was changed")
    for devices in ([0, 0], [0, "00"], [0, -1], [True, 1], [0], "0,1"):
        _reject(lambda value=devices: runtime.validate_devices(value, world=2), "Invalid device mapping accepted")
        with patch.dict(os.environ, {}, clear=True):
            _reject(lambda value=devices: runtime.configure_devices(value, world=2), "Invalid device binding accepted")
            require("ASCEND_RT_VISIBLE_DEVICES" not in os.environ, "Invalid mapping changed device visibility")
    with patch.dict(os.environ, {}, clear=True):
        assigned = runtime.configure_devices([3, 1], world=2)
        require(assigned == ["3", "1"] and os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3,1",
                "Explicit device binding depends on an inherited task environment")
        require(runtime.configure_devices(assigned, world=2) == assigned, "Matching runtime mapping was rejected")
        _reject(lambda: runtime.configure_devices([1, 3], world=2), "Conflicting device order was accepted")
        require(os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3,1", "Conflicting mapping changed device visibility")
    return dict(tp="explicit_per_builder", import_argv="ignored", device_mapping="explicit_and_unique",
                inherited_environment="not_required", conflicting_mapping="rejected_without_mutation")


def check_chunk_continuation():
    """Exercise the actual chunk runner with HOST substitutes for NPU execution and allocations."""
    from collections.abc import Mapping
    from contextlib import ExitStack
    from copy import deepcopy
    import hashlib
    import os
    import sys
    from tempfile import TemporaryDirectory
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch

    from . import prefill_fwd

    source = ast.parse((ROOT / "prefill_network_runtime.py").read_text(encoding="utf-8"))
    runner = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                  and node.name == "run_network_runtime")
    code = compile(ast.Module(body=[runner], type_ignores=[]), "prefill_network_runtime.py", "exec")
    topology = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)
    modes = dict(real_w8a8=True)
    fixtures = 0
    rejected = 0

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    with TemporaryDirectory(prefix="a3-prefill-chunks-") as temporary:
        root = Path(temporary)

        def exercise(*, count=4, legacy=False, empty=False, change=None, fail_epoch=None,
                     corrupt_admission=False, checkpoint=None):
            nonlocal fixtures
            fixtures += 1
            directory = root / str(fixtures)
            directory.mkdir()
            packets, predecessor = [], None
            for epoch in range(count):
                actual_input = dict(epoch=epoch, predecessor_input_sha256=predecessor,
                    tokens=list(range(epoch * 16, (epoch + 1) * 16)), history=list(range(epoch * 16)))
                input_path = directory / f"input-{epoch}.json"
                write(input_path, actual_input)
                contract = dict(topology=deepcopy(topology), checkpoint="original-checkpoint", model_config={},
                    required_modes=deepcopy(modes), capacity=16, scheduler_capacity=8192,
                    fresh_shared_initialization={"topk_indices": -1}, input_contract_path=str(input_path),
                    input_contract_sha256=sha(input_path))
                predecessor = contract["input_contract_sha256"]
                sources = []
                for rank in range(16):
                    path = directory / f"metadata-{epoch}-{rank}.json"
                    write(path, dict(rank=rank, epoch=epoch))
                    sources.append(dict(rank=rank, path=str(path), sha256=sha(path)))
                packets.append(dict(inputs={"input_contract": actual_input}, contract=contract,
                                    metadata_sources=sources))
            if change is not None:
                change(packets)
                for packet in packets:
                    path = packet["contract"]["input_contract_path"]
                    write(path, packet["inputs"]["input_contract"])
                    packet["contract"]["input_contract_sha256"] = sha(path)
            observed = dict(completed=[], consumed=[], providers=[], weights=[], continuity=[], outputs=[])

            class Provider:
                def __init__(self, *args, **kwargs):
                    self.shared = [{"fresh": {"topk_indices": SimpleNamespace(fill_=lambda value: None)}}]
                    self.kv, self.history, self.finished = bytearray(), [], -1
                    observed["providers"].append(self)

                def require_network_geometry(self):
                    return None

                def begin_chunk(self, *, epoch, token_map_id, **kwargs):
                    require(epoch == self.finished + 1, "A chunk advanced before preceding execution completed")
                    if epoch:
                        actual = kwargs["input_by_case"]["fresh"]
                        require(actual["history"] == self.history, "Continuation lost the committed token history")
                        metadata = kwargs["metadata_by_case"]["fresh"]
                        require([row["rank"] for row in metadata] == list(range(16)), "Metadata lost rank order")
                        require(all(row["epoch"] == epoch for row in metadata), "Continuation reused stale metadata")
                        if corrupt_admission:
                            self.kv.append(255)

            class Weights:
                def __init__(self, path, topology, model_config):
                    self.path, self.releases = path, 0
                    self.layout = ExpertParallelConfig(topology["dp"], topology["tp"], topology["ep"])
                    observed["weights"].append(self)

                def release_current(self):
                    self.releases += 1

            def snapshot(provider, storage):
                return dict(kv_owner=id(provider.kv), kv=provider.kv.hex(), history_owner=id(provider.history),
                    history=list(provider.history), table_owner=id(storage), tables=list(storage or {}))

            def run_chunk(inputs, **kwargs):
                epoch, provider = kwargs["epoch"], kwargs["provider"]
                if epoch == fail_epoch:
                    raise RuntimeError("Injected device execution failure")
                require(kwargs["weights"] is observed["weights"][0], "Chunk replaced the weight provider")
                storage = kwargs["table_storage"]
                if epoch and storage is not None:
                    require("retained-engram" in storage, "Chunk lost the retained Engram table")
                if storage is not None:
                    storage["retained-engram"] = True
                provider.kv.append(epoch)
                provider.history.extend(inputs["input_contract"]["tokens"])
                provider.finished = epoch
                observed["completed"].append(epoch)
                observed["outputs"].append(kwargs["output"].name)
                return dict(epoch=epoch)

            def write_result(path, value):
                observed["continuity"].append(value)
                write(path, value)

            def continuations():
                for epoch, packet in enumerate(packets[1:], start=1):
                    require(observed["completed"] == list(range(epoch)), "The stream loaded a future chunk early")
                    observed["consumed"].append(epoch)
                    yield packet

            namespace = dict(__package__=__package__, Mapping=Mapping, ExitStack=ExitStack, Path=Path, json=json,
                NetworkWeights=Weights, NetworkCacheProvider=Provider, file_sha=sha,
                attach_network_cache_geometry=lambda provider: None, _network_owner_snapshot=snapshot,
                _run_network_chunk=run_chunk, write_json=write_result,
                torch=SimpleNamespace(load=lambda path, **kwargs: json.loads(Path(path).read_text())))
            exec(code, namespace)
            module = ModuleType(__package__ + ".prefill_network_runtime")
            module.run_network_runtime = namespace["run_network_runtime"]
            transport = ModuleType(__package__ + ".moe_transport")
            transport.CANNMemory = SimpleNamespace
            cache = dict(cache_ABI=[], cache_ABI_sources=[],
                engram_contracts={layer: dict(topology=topology, engram_shards=[0] * 4) for layer in (1, 14)})
            continuation = packets[1] if legacy else continuations()
            if count == 1:
                continuation = [] if empty else None
            modules = {module.__name__: module, transport.__name__: transport}
            try:
                with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, modules):
                    result = prefill_fwd.run_prefill(packets[0]["inputs"], weights=checkpoint or "original-checkpoint",
                        cache=cache, config=packets[0]["contract"], devices=list(range(16)),
                        output=directory / "output", continuation=continuation)
                expected = list(range(2 if legacy else count))
                require(observed["completed"] == expected, "Chunk execution was skipped or duplicated")
                require(result["epoch"] == expected[-1], "Result does not describe the last completed chunk")
                if len(expected) > 1:
                    require([row["epoch"] for row in result["completed_prefill_chunks"]] == expected,
                            "The aggregate omitted completed chunks")
                    require(observed["outputs"][1:] == [f"continuation-chunk{epoch}" for epoch in expected[1:]],
                            "Continuation output directories collided")
                require(all(row["before"] == row["after"] and row["exact"] for row in observed["continuity"]),
                        "Chunk admission changed persistent state")
                require(observed["providers"][0].history == list(range(len(expected) * 16)),
                        "Final history does not contain the complete token stream")
                return observed
            finally:
                require(len(observed["providers"]) == len(observed["weights"]) == 1,
                        "Chunks replaced session providers")
                require(observed["weights"][0].releases == 1, "Session cleanup was skipped or repeated")
                exercise.observed = observed

        exercise()
        exercise(count=2, legacy=True)
        exercise(count=1)
        exercise(count=1, empty=True)
        overridden = exercise(checkpoint="replacement-checkpoint")
        require(overridden["weights"][0].path == str(Path("replacement-checkpoint").resolve()),
                "Checkpoint override was not propagated to continuations")

        def wrong_predecessor(packets):
            packets[2]["inputs"]["input_contract"]["predecessor_input_sha256"] = packets[0]["contract"]["input_contract_sha256"]

        def skipped_epoch(packets):
            packets[2]["inputs"]["input_contract"]["epoch"] = 3

        def boolean_epoch(packets):
            packets[1]["inputs"]["input_contract"]["epoch"] = True

        def changed_capacity(packets):
            packets[2]["contract"]["capacity"] = 32

        def missing_rank(packets):
            packets[2]["metadata_sources"].pop()

        def changed_metadata(packets):
            write(packets[2]["metadata_sources"][0]["path"], {"tampered": True})

        def boolean_rank(packets):
            packets[2]["metadata_sources"][1]["rank"] = True

        def incomplete_contract(packets):
            del packets[2]["contract"]["model_config"]

        def malformed_source(packets):
            del packets[2]["metadata_sources"][0]["sha256"]

        changes = (wrong_predecessor, skipped_epoch, boolean_epoch, changed_capacity, missing_rank,
                   changed_metadata, boolean_rank, incomplete_contract, malformed_source)
        for change in changes:
            _reject(lambda change=change: exercise(change=change), "Invalid continuation was admitted")
            expected = [0] if change is boolean_epoch else [0, 1]
            require(exercise.observed["completed"] == expected, "Invalid continuation failed at the wrong boundary")
            rejected += 1
        _reject(lambda: exercise(corrupt_admission=True), "Admission changed cache bytes without rejection")
        require(exercise.observed["completed"] == [0], "Execution used corrupted persistent state")
        _reject(lambda: exercise(fail_epoch=1), "Device execution failure was ignored")
        require(exercise.observed["completed"] == [0] and exercise.observed["consumed"] == [1],
                "The stream advanced after failed execution")
        rejected += 2
    return dict(chunks=4, total_tokens=64, legacy_two_chunks="preserved", session_owners="preserved",
                lazy_stream="checked", rejected_controls=rejected, device_execution="stubbed_not_validated")


def check_chunk_configuration():
    """Check CLI chunk descriptors and lazy tensor loading without Torch or device imports."""
    import sys
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch

    from . import prefill_fwd

    with TemporaryDirectory(prefix="a3-prefill-config-") as temporary:
        root = Path(temporary)
        metadata_dir = root / "metadata"
        metadata_dir.mkdir()
        sources = [dict(rank=rank, path=f"rank-{rank}.pt", sha256="captured-identity") for rank in range(16)]
        metadata_path = metadata_dir / "sources.json"
        metadata_path.write_text(json.dumps(sources), encoding="utf-8")
        (root / "contract.json").write_text("{}", encoding="utf-8")
        descriptor = dict(inputs="chunk-inputs.pt", contract="contract.json", metadata_sources="metadata/sources.json")
        settings = dict(checkpoint="checkpoint", inputs="inputs.pt", cache_state="cache.pt", contract="contract.json",
                        devices=list(range(16)), output="output", continuations=[descriptor] * 2)
        config_path = root / "prefill.json"
        config_path.write_text(json.dumps(settings), encoding="utf-8")
        resolved = prefill_fwd.load_configuration(config_path)
        require(resolved["checkpoint"] == root / "checkpoint", "Checkpoint path lost its configuration base")
        require(all(chunk["inputs"] == root / "chunk-inputs.pt" for chunk in resolved["continuations"]),
                "Continuation paths lost their configuration base")
        loaded = []

        def load(path, *, map_location, weights_only):
            require(map_location == "cpu" and weights_only is True, "Chunk loader changed tensor safety controls")
            loaded.append(path)
            return {"input_contract": {"epoch": len(loaded)}}

        with patch.dict(sys.modules, {"torch": SimpleNamespace(load=load)}):
            chunks = prefill_fwd.load_continuations(resolved["continuations"])
            require(not loaded, "Chunk descriptors eagerly loaded tensors")
            first = next(chunks)
            require(len(loaded) == 1, "Loading one continuation prefetched a future chunk")
            require(first["metadata_sources"][0]["path"] == str(metadata_dir / "rank-0.pt"),
                    "Metadata paths lost their descriptor base")
            require(next(chunks)["inputs"]["input_contract"]["epoch"] == 2, "Second continuation was not loaded")
            sources[1]["rank"] = True
            metadata_path.write_text(json.dumps(sources), encoding="utf-8")
            _reject(lambda: next(prefill_fwd.load_continuations(resolved["continuations"])),
                    "Boolean rank was accepted by the CLI chunk loader")
            require(len(loaded) == 2, "Invalid metadata triggered a tensor load")
        for invalid in ({}, [dict(inputs="input.pt")], [dict(descriptor, inputs="")]):
            config_path.write_text(json.dumps(dict(settings, continuations=invalid)), encoding="utf-8")
            _reject(lambda: prefill_fwd.load_configuration(config_path), "Invalid chunk configuration was accepted")
        automatic = dict(settings)
        automatic.pop("inputs")
        automatic.pop("continuations")
        for input_fields in (dict(prompt="Hello"), dict(token_ids=list(range(2048)))):
            config_path.write_text(json.dumps(dict(automatic, **input_fields)), encoding="utf-8")
            resolved = prefill_fwd.load_configuration(config_path)
            require(resolved["chunk_size"] == 16 and resolved["dp_rank"] == 0, "Automatic defaults changed")
        for invalid in (dict(token_ids=[]), dict(token_ids=[True]), dict(token_ids=[-1]),
                        dict(token_ids=list(range(2049))), dict(prompt=""), dict(prompt="Hello", token_ids=[1]),
                        dict(prompt="Hello", inputs="prepared.pt"), dict(prompt="Hello", chunk_size=17),
                        dict(prompt="Hello", chunk_size=True), dict(prompt="Hello", dp_rank=4),
                        dict(prompt="Hello", continuations=[descriptor])):
            config_path.write_text(json.dumps(dict(automatic, **invalid)), encoding="utf-8")
            _reject(lambda: prefill_fwd.load_configuration(config_path), "Invalid automatic configuration was accepted")
    return dict(relative_paths="checked", lazy_tensor_loading="checked", metadata_rank_order="checked",
                automatic_schema="checked")


def check_automatic_entry():
    """Check lazy automatic-input dispatch without importing Torch, PyPTO or devices."""
    import os
    import sys
    from tempfile import TemporaryDirectory
    from types import ModuleType
    from unittest.mock import patch

    from . import prefill_fwd

    observed = []
    prepare = ModuleType(__package__ + ".prefill_prepare")
    runtime = ModuleType(__package__ + ".prefill_network_runtime")

    def chunks(inputs, **kwargs):
        require(inputs == list(range(32)), "Automatic entry changed the token IDs")
        require(kwargs["chunk_size"] == 16 and kwargs["dp_rank"] == 2, "Automatic controls were not forwarded")
        for epoch in range(2):
            require(observed == list(range(epoch)), "Automatic entry prefetched the next chunk")
            yield dict(inputs=dict(input_contract=dict(epoch=epoch), prepare_input_stage=True),
                       cache=dict(cache_ABI="reserved-layout"), contract=kwargs["config"], metadata_sources=[])

    def execute(inputs, **kwargs):
        require(inputs["cache_ABI"] == "reserved-layout", "The first automatic layout was not supplied")
        require(inputs["prepare_input_stage"] is True, "Automatic input embedding was skipped")
        observed.append(inputs["input_contract"]["epoch"])
        for packet in kwargs["continuation"]:
            observed.append(packet["inputs"]["input_contract"]["epoch"])
        return dict(completed=list(observed))

    prepare.prepare_token_chunks, runtime.run_network_runtime = chunks, execute
    with TemporaryDirectory(prefix="a3-automatic-entry-") as temporary:
        config = dict(topology=dict(ep=16), chunk_size=16, dp_rank=2)
        with patch.dict(sys.modules, {prepare.__name__: prepare, runtime.__name__: runtime}), \
                patch.dict(os.environ, {}, clear=True):
            result = prefill_fwd.run_prefill(list(range(32)), weights="checkpoint", cache={}, config=config,
                devices=list(range(16)), output=Path(temporary) / "new-output")
            require(result == {"completed": [0, 1]}, "The automatic public entry lost a chunk")
            _reject(lambda: prefill_fwd.run_prefill([1], weights="checkpoint", cache={}, config=config,
                devices=list(range(16)), output=temporary), "Automatic entry accepted an existing output directory")
            _reject(lambda: prefill_fwd.run_prefill([1], weights="checkpoint", cache={}, config=config,
                devices=list(range(16)), output=Path(temporary) / "another", continuation=[]),
                "Automatic entry accepted conflicting explicit continuations")
    return dict(lazy_preparation="checked", reserved_cache="forwarded", execution="stubbed_not_validated")


def main():
    """Publish a bounded HOST result without claiming numerical device acceptance."""
    result = dict(source=check_sources(), layer_plan=check_layer_plan(), controls=check_controls(),
                  runtime_configuration=check_runtime_configuration(),
                  chunk_continuation=check_chunk_continuation(),
                  chunk_configuration=check_chunk_configuration(),
                  automatic_entry=check_automatic_entry(),
                  device_initialization="not_run", compilation="not_run", NPU_numerics="not_run")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
