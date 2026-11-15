# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain A3 workers, typed device allocations and HCCL groups across a complete generation."""

import os
import struct
from collections import OrderedDict
from contextlib import ExitStack
from threading import Condition, Lock, RLock, Thread
from time import perf_counter
import torch


def _host_bytes(value):
    """View the complete CPU allocation as bytes, including typed aliases."""
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("Resident staging requires contiguous CPU tensors")
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def _device_view(allocation, value, base_offset=0):
    """Bind a contiguous typed view to the allocation's owning Buffer."""
    from pypto.runtime import DeviceTensor

    offset = value.storage_offset() * value.element_size() - base_offset
    if offset < 0:
        raise ValueError("Resident tensor view begins before its allocation")
    if offset + value.numel() * value.element_size() > allocation.nbytes:
        raise ValueError("Resident tensor view exceeds its allocation")
    return DeviceTensor(allocation.data_ptr + offset, value.shape, value.dtype,
                        buffer=allocation.buffer, byte_offset=offset)


class _RankBatch:
    def __init__(self, function, ranks, world):
        self.world = world
        self.results, self.errors = [None] * world, [None] * world
        self.reset(function, ranks)

    def reset(self, function, ranks):
        order = tuple(ranks)
        members = frozenset(order)
        if len(members) != len(order) or any(rank < 0 or rank >= self.world for rank in order):
            raise ValueError("Rank batch requires distinct allocated ranks")
        self.function, self.order, self.ranks = function, order, members
        for rank in range(self.world):
            self.results[rank] = self.errors[rank] = None
        self.remaining = len(order)


class _RankExecutor:
    """Publish complete batches atomically to persistent rank-affine threads."""

    def __init__(self, world):
        self._world = world
        lock = Lock()
        self._condition = Condition(lock)
        self._complete = Condition(lock)
        self._threads = [None] * world
        self._admission = Lock()
        self._slot = _RankBatch(None, (), world)
        self._batch, self._generation = None, 0
        self._closing = self._closed = False

    def start(self):
        for rank in range(self._world):
            thread = Thread(target=self._work, args=(rank,),
                            name=f"ThreadPoolExecutor-a3-transport-{rank}")
            self._threads[rank] = thread
            thread.start()

    def _work(self, rank):
        generation = 0
        while True:
            with self._condition:
                while not self._closed and self._generation == generation:
                    self._condition.wait()
                if self._closed:
                    return
                generation, batch = self._generation, self._batch
                if batch is None or rank not in batch.ranks:
                    continue
            result, error = None, None
            try:
                result = batch.function(rank)
            except BaseException as failure:
                error = failure
            with self._condition:
                batch.results[rank], batch.errors[rank] = result, error
                batch.remaining -= 1
                if not batch.remaining:
                    self._batch = None
                    self._complete.notify_all()

    def run(self, function, ranks):
        with self._admission:
            try:
                return self._run(function, ranks)
            finally:
                self._slot.function = None

    def _run(self, function, ranks):
        batch = self._slot
        batch.reset(function, ranks)
        if not batch.order:
            return []
        with self._condition:
            if self._closing:
                raise RuntimeError("Rank executor is closed")
            while self._batch is not None:
                self._complete.wait()
                if self._closing:
                    raise RuntimeError("Rank executor is closed")
            self._batch = batch
            self._generation += 1
            try:
                self._condition.notify_all()
            except BaseException:
                # Notified threads cannot inspect the batch until this lock
                # is released, so failed admission can roll back atomically.
                self._generation -= 1
                self._batch = None
                raise
            wait_error = None
            while batch.remaining:
                try:
                    self._complete.wait()
                except BaseException as error:
                    if wait_error is None:
                        wait_error = error
        if wait_error is not None:
            raise wait_error
        for rank in batch.order:
            if batch.errors[rank] is not None:
                raise batch.errors[rank]
        return [batch.results[rank] for rank in batch.order]

    def shutdown(self):
        with self._condition:
            self._closing = True
            while self._batch is not None:
                self._complete.wait()
            self._closed = True
            self._condition.notify_all()
        error = None
        for thread in self._threads:
            if thread is not None and thread.ident is not None:
                try:
                    thread.join()
                except BaseException as failure:
                    if error is None:
                        error = failure
        if error is not None:
            raise error
        self._threads.clear()
        with self._admission:
            self._slot.reset(None, ())


class ResidentNetworkRuntime:
    """Own resident compute workers and independent CPU-staged IO workers per logical rank."""

    def __init__(self, devices, topology, *, worker_factory=None):
        begin = perf_counter()
        self.devices, self.topology = tuple(devices), dict(topology)
        os.environ.setdefault("HCCL_EXEC_TIMEOUT", "180")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "180")
        if len(self.devices) != topology["ep"] or len(set(self.devices)) != len(self.devices):
            raise ValueError("Resident runtime requires the exact complete distinct device mapping")
        self.workers, self.pinned, self.channels = [], {}, {}
        self.device_owners = {}
        self.staging_workers = {}
        self.rotations = {}
        self._worker_owners = [None] * len(devices)
        self._staging_owners = [None] * len(devices)
        self._dispatch_locks = [RLock() for _ in devices]
        self._transient_slots = {}
        self._device_views = {}
        self._dispatch_cache = {}
        self._native_dispatch = worker_factory is None
        self._closed = False
        self._exchange_executor = None
        self._exchange_lock = RLock()
        self._transport_channels = []
        self._select = lambda rank: None
        self._memory = None
        self.stats = dict(worker_initializations=0, dispatches=0, h2d_bytes=0, d2h_bytes=0,
                          weight_bytes=[0] * len(devices), state_bytes=[0] * len(devices),
                          transport_calls=0, transport_seconds=0.0,
                          staging_worker_initializations=0, staging_dispatches=0)
        try:
            if worker_factory is None:
                from pypto.backend import BackendType
                from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

                def worker_factory(rank):
                    config = RunConfig(arch=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
                                       device_id=rank)
                    return ChipWorker(config=config)

                from .moe_transport import CANNMemory

                # CANN's current device belongs to the calling thread. Creating
                # several workers leaves that thread on the last initialized
                # device; every later allocation/copy must select its owner.
                memory = CANNMemory(diagnostics=False)
                self._select = memory.select
                self._memory = memory

            self._worker_factory = worker_factory
            for rank in range(len(devices)):
                self._select(rank)
                worker = worker_factory(rank)
                self._worker_owners[rank] = worker
                self.workers.append(worker)
                self.stats["worker_initializations"] += 1
            self.stats["worker_initialization_seconds"] = perf_counter() - begin
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def _close_worker(self, rank, worker):
        self._select(rank)
        worker.close()

    def close(self):
        """Drain every communicator and worker, including partial initialization."""
        with self._exchange_lock, ExitStack() as dispatches:
            for lock in self._dispatch_locks:
                dispatches.enter_context(lock)
            self._closed = True
            # A failed shutdown retains all device owners for a later retry.
            if self._exchange_executor is not None:
                self._exchange_executor.shutdown()
                self._exchange_executor = None
            remaining, errors = [], []
            for channel in self._transport_channels:
                if channel is None:
                    continue
                try:
                    channel.close()
                except BaseException as error:
                    remaining.append(channel)
                    errors.append(error)
            self._transport_channels = remaining
            if errors:
                raise errors[0]
            self.channels.clear()
            for owners in (self._staging_owners, self._worker_owners):
                for rank in reversed(range(len(owners))):
                    worker = owners[rank]
                    if worker is None:
                        continue
                    try:
                        self._close_worker(rank, worker)
                        owners[rank] = None
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise errors[0]
            self.pinned.clear()
            self.device_owners.clear()
            self.rotations.clear()
            self.staging_workers.clear()
            self._transient_slots.clear()
            self._device_views.clear()
            self._dispatch_cache.clear()

    def pin(self, value, ranks, *, readonly, name):
        """Upload one owning allocation per rank and retain all typed aliases."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        full_host = _host_bytes(value)
        for rank in ranks:
            self._select(rank)
            key = (rank, value.untyped_storage()._cdata)
            existing = self.pinned.get(key)
            if existing is not None:
                if existing["readonly"] != readonly:
                    raise ValueError("An allocation cannot be both a weight and mutable state: " + name)
                existing["versions"][id(value)] = (value, value._version)
                offset = value.storage_offset() * value.element_size() - existing["base_offset"]
                if offset < 0 or offset + value.numel() * value.element_size() > existing["allocation"].nbytes:
                    raise ValueError("Resident weight views require one complete owning span per rank")
                continue
            offset = value.storage_offset() * value.element_size() if readonly else 0
            extent = value.numel() * value.element_size() if readonly else full_host.numel()
            host = full_host.narrow(0, offset, extent)
            allocation = self.workers[rank].alloc_tensor(host.shape, host.dtype, init=host)
            self.pinned[key] = dict(host=host, nbytes=extent, source=value, allocation=allocation, readonly=readonly,
                                    base_offset=offset,
                                    name=name, versions={id(value): (value, value._version)})
            kind = "weight_bytes" if readonly else "state_bytes"
            self.stats[kind][rank] += allocation.nbytes
            self.stats["h2d_bytes"] += allocation.nbytes

    def weight(self, value, rank):
        """Upload a complete immutable owned expert operand without retaining host payloads."""
        self._select(rank)
        value = value.contiguous()
        result = self.workers[rank].alloc_tensor(value.shape, value.dtype, init=value)
        self.device_owners[id(result.buffer)] = rank
        self.stats["weight_bytes"][rank] += result.nbytes
        self.stats["h2d_bytes"] += result.nbytes
        return result

    def staging_worker(self, rank):
        """Create a separate public worker whose buffers never alias resident allocations."""
        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        self._select(rank)
        if rank not in self.staging_workers:
            worker = self._worker_factory(rank)
            self._staging_owners[rank] = worker
            self.staging_workers[rank] = worker
            self.stats["worker_initializations"] += 1
            self.stats["staging_worker_initializations"] += 1
        return self.staging_workers[rank]

    def execute(self, compiled, rank, arguments, abi, *, staged=False):
        """Dispatch existing arithmetic with resident constants and exact aliased staging."""
        with self._dispatch_locks[rank]:
            self._execute(compiled, rank, arguments, abi, staged=staged)

    def _execute(self, compiled, rank, arguments, abi, *, staged):
        from pypto.runtime import DeviceTensor

        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        if len(arguments) != len(abi["params"]):
            raise ValueError("Resident dispatch requires the complete compiled parameter ABI")
        worker = self.staging_worker(rank) if staged else self.workers[rank]
        local, bound = {}, []
        self._select(rank)
        writable, transient = {}, []
        slots = self._transient_slots.setdefault((rank, staged), [])
        try:
            for value, parameter, direction in zip(arguments, abi["params"], abi["effective_directions"]):
                if parameter["shape"] is None:
                    if isinstance(value, (torch.Tensor, DeviceTensor)) or direction != "In":
                        raise ValueError("Resident scalar ABI differs")
                    bound.append(value)
                    continue
                if isinstance(value, DeviceTensor):
                    if staged:
                        raise ValueError("Staged IO requires CPU tensors, not resident device operands")
                    if direction != "In":
                        raise ValueError("Owned expert operands must remain read-only")
                    if self.device_owners.get(id(value.buffer)) != rank:
                        raise ValueError("Resident device operand belongs to another rank or runtime")
                    bound.append(value)
                    continue
                if value.device.type != "cpu" or not value.is_contiguous():
                    raise ValueError("Resident staging requires contiguous CPU tensors")
                storage = value.untyped_storage()
                key = (rank, storage._cdata)
                resident_owner = self.pinned.get(key)
                if staged and resident_owner is not None and direction != "In":
                    raise ValueError("Staged IO may not write resident state or immutable weights")
                owner = None if staged else resident_owner
                if owner is not None:
                    if owner["readonly"] and direction != "In":
                        raise ValueError("A compiled entry writes an immutable resident weight")
                    changed = any(tensor._version != version for tensor, version in owner["versions"].values())
                    if changed or (not owner["readonly"] and id(value) not in owner["versions"]):
                        if owner["readonly"]:
                            raise ValueError("Immutable resident weight changed on the host")
                        copy = owner["host"]
                        self._copy(worker, owner["allocation"].data_ptr, copy.data_ptr(), copy.numel(), 1)
                        self.stats["h2d_bytes"] += copy.numel()
                        owner["versions"] = {key: (tensor, tensor._version)
                                             for key, (tensor, _) in owner["versions"].items()}
                    owner["versions"][id(value)] = (value, value._version)
                elif key in local:
                    owner = local[key]
                else:
                    # Retain this call's complete backing without constructing
                    # a second Torch tensor solely for its byte pointer/extent.
                    nbytes = storage.nbytes()
                    slot = len(transient)
                    if slot == len(slots):
                        slots.append(worker.alloc_tensor((nbytes,), torch.uint8))
                    elif slots[slot].shape[0] < nbytes:
                        replacement = worker.alloc_tensor((nbytes,), torch.uint8)
                        worker.free_tensor(slots[slot])
                        self._device_views.pop(id(slots[slot]), None)
                        slots[slot] = replacement
                    allocation = slots[slot]
                    transient.append(allocation)
                    self._copy(worker, allocation.data_ptr, storage.data_ptr(), nbytes, 1)
                    owner = local[key] = dict(host=storage, nbytes=nbytes, allocation=allocation, readonly=False)
                    self.stats["h2d_bytes"] += nbytes
                bound.append(self._view(owner["allocation"], value, owner.get("base_offset", 0)))
                if direction != "In":
                    writable[key] = owner
            self._run_compiled(compiled, rank, staged, worker, bound)
            self.stats["dispatches"] += 1
            self.stats["staging_dispatches"] += int(staged)
            for owner in writable.values():
                host, allocation = owner["host"], owner["allocation"]
                self._copy(worker, host.data_ptr(), allocation.data_ptr, owner["nbytes"], 2)
                self.stats["d2h_bytes"] += owner["nbytes"]
        finally:
            # Device slots retain no host inputs or output observations.
            transient.clear()

    def _view(self, allocation, value, base_offset):
        """Retain bounded immutable typed views on each live device allocation."""
        key = value.storage_offset() * value.element_size() - base_offset, tuple(value.shape), value.dtype
        record = self._device_views.get(id(allocation))
        if record is None:
            record = allocation, {}
            self._device_views[id(allocation)] = record
        views = record[1]
        view = views.get(key)
        if view is None:
            view = _device_view(allocation, value, base_offset)
            if len(views) == 8:
                views.pop(next(iter(views)))
            views[key] = view
        return view

    def _copy(self, worker, destination, source, nbytes, direction):
        """Copy complete staging spans while the selected worker owns their allocations."""
        if self._memory is not None:
            self._memory.copy(destination, source, nbytes, direction)
        elif direction == 1:
            worker.copy_to(destination, source, nbytes)
        else:
            worker.copy_from(destination, source, nbytes)

    def _run_compiled(self, compiled, rank, staged, worker, bound):
        """Reuse validated wire descriptors for the same live device spans and scalar values."""
        from pypto.runtime import DeviceTensor

        if not self._native_dispatch:
            worker.run(compiled, *bound)
            return
        worker._check_binding(compiled)
        runtime_config = compiled.runtime_config
        thread_count = runtime_config.get("aicpu_thread_num")
        host_indices = runtime_config.get("host_tensor_indices", ())
        if (type(host_indices) not in (tuple, list) or host_indices
                or type(thread_count) not in (int, type(None))):
            worker.run(compiled, *bound)
            return
        signature = []
        for value in bound:
            if isinstance(value, DeviceTensor):
                signature.append((id(value.buffer), value.data_ptr, value.shape, value.dtype, value.byte_offset))
            elif type(value) is float:
                signature.append((float, struct.pack("!d", value)))
            elif type(value) in (bool, int):
                signature.append((type(value), value))
            else:
                worker.run(compiled, *bound)
                return
        key = id(compiled), thread_count, tuple(signature)
        cache = self._dispatch_cache.setdefault((rank, staged), OrderedDict())
        record = cache.get(key)
        if record is None:
            arguments, coerced, _ = compiled._build_orch_args(*bound, worker=worker._impl)
            config = compiled._build_call_config()
            record = compiled, coerced, compiled.chip_callable, arguments, config
            cache[key] = record
            if len(cache) > 64:
                cache.popitem(last=False)
        else:
            for value in bound:
                if isinstance(value, DeviceTensor):
                    worker._require_owned_resident_tensor(value, "Resident dispatch argument")
            cache.move_to_end(key)
        worker._run_chip(record[2], record[3], record[4])

    def _group_ranks(self, group):
        world, tp = self.topology["ep"], self.topology["tp"]
        groups = dict(ep=[list(range(world))],
                      tp=[list(range(start, start + tp)) for start in range(0, world, tp)],
                      dp=[list(range(lane, world, tp)) for lane in range(tp)])
        if group not in groups:
            raise ValueError("Unknown resident HCCL group: " + group)
        return groups[group]

    def load_rotation(self, checkpoint, hidden_size):
        """Retain the validated checkpoint rotation for both slots and subsequent tokens."""
        from .engram_host import load_rotation

        key = str(checkpoint), hidden_size
        if key not in self.rotations:
            self.rotations[key] = load_rotation(checkpoint, hidden_size)
        return self.rotations[key]

    def exchange(self, stage, packets, collective, group):
        """Exchange actual CPU boundaries through persistent HCCL without files or subprocesses."""
        with self._exchange_lock:
            return self._exchange(stage, packets, collective, group)

    def _exchange_rank_tasks(self, function, ranks):
        """Quarantine rank resources after a failed batch until close drains them."""
        try:
            return self._exchange_executor.run(function, ranks)
        except BaseException:
            self._closed = True
            raise

    def _exchange(self, stage, packets, collective, group):
        from .moe_transport import CANNMemory, HCCLBytes

        if self._closed:
            raise RuntimeError("Resident runtime is closed")
        world = len(self.workers)
        if len(packets) != world or not packets:
            raise ValueError("Every allocated resident rank must participate in a collective")
        schema = [(case, tuple(packet)) for case, packet in packets[0].items()]
        if any([(case, tuple(packet)) for case, packet in row.items()] != schema for row in packets):
            raise ValueError("Resident collective rank packet orders differ")
        if collective not in ("all_gather", "all_to_all", "reduce_scatter"):
            raise ValueError("Unsupported resident collective")
        groups = self._group_ranks(group)
        for ranks in groups:
            for case, names in schema:
                for name in names:
                    reference = packets[ranks[0]][case][name]
                    if not isinstance(reference, torch.Tensor):
                        raise ValueError("Resident collective requires actual tensor packets")
                    for rank in ranks:
                        value = packets[rank][case][name]
                        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                                or not value.is_contiguous() or value.ndim < 1 or value.numel() == 0
                                or value.shape != reference.shape or value.dtype != reference.dtype):
                            raise ValueError("Resident collective peer tensor ABIs differ")
                        if collective != "all_gather" and value.numel() * value.element_size() % len(ranks):
                            raise ValueError("Resident collective byte count does not divide across peers")
                        if collective == "reduce_scatter" and (
                                value.dtype not in (torch.bfloat16, torch.float32) or value.shape[0] % len(ranks)):
                            raise ValueError("Resident reduce-scatter requires equal BF16/FP32 row buckets")
        if self._exchange_executor is None:
            # Keep each communicator on its own host thread. CANN context and
            # HCCL thread-local state must not migrate between logical ranks.
            self._exchange_executor = _RankExecutor(world)
            try:
                self._exchange_executor.start()
            except BaseException:
                self.close()
                raise
        if group not in self.channels:
            memory = CANNMemory(diagnostics=False)
            channels = {}
            for ranks in groups:
                root_info = HCCLBytes.root_info(memory)
                first_slot = len(self._transport_channels)
                slots = {rank: first_slot + index for index, rank in enumerate(ranks)}
                # Reserve ownership slots before creating any native channel.
                self._transport_channels.extend([None] * len(ranks))

                def connect(rank):
                    def retain(channel):
                        self._transport_channels[slots[rank]] = channel

                    channel = HCCLBytes(memory, rank, len(ranks), ranks, root_info=root_info,
                                        _register_owner=retain)
                    return rank, channel

                channels.update(self._exchange_rank_tasks(connect, ranks))
            self.channels[group] = channels
        channels, outputs = self.channels[group], [{} for _ in packets]
        begin = perf_counter()

        def transfer(rank):
            channel = channels[rank]
            for case, names in schema:
                outputs[rank][case] = {}
                for name in names:
                    value = packets[rank][case][name]
                    outputs[rank][case][name] = channel.exchange(value, stage + "/" + case + "/" + name, collective)

        self._exchange_rank_tasks(transfer, range(world))
        seconds = perf_counter() - begin
        self.stats["transport_calls"] += 1
        self.stats["transport_seconds"] += seconds
        receipts = [dict(rank=rank, backend="persistent CANN/HCCL", group_ranks=channels[rank].group_ranks,
                         collective=collective, stage=stage, seconds=seconds, files_used=False,
                         memory=dict(scope="Transport buffers only; not total HBM or a peak",
                                     samples=[], intervals=[], transport_buffer_peak_bytes=channels[rank].buffers_peak))
                    for rank in range(world)]
        return outputs, receipts

    def summary(self):
        """Report actual allocations and dispatches separately from numerical acceptance."""
        return dict(self.stats, assigned_devices=list(self.devices),
                    execution_model="persistent_A3_ChipWorkers_with_host_layer_scheduler",
                    single_device_graph=False, cache_readback="retained for current host consumers",
                    numerical_acceptance="not_established_by_residency")
