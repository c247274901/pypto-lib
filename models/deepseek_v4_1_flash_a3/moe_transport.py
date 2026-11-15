# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transport contiguous PyPTO route buckets through CANN memory and HCCL bytes."""

import ctypes
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import RLock

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"
import torch


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} returned CANN/HCCL status {status}")


def _bind(library, name, argument_types):
    function = getattr(library, name)
    function.argtypes = argument_types
    function.restype = ctypes.c_int
    return function


class CANNMemory:
    """Allocate/copy CANN memory and retain raw device-memory samples."""

    def __init__(self, *, diagnostics=False):
        self.diagnostics = diagnostics
        self.library = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
        pointer, size, integer = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        _bind(self.library, "aclInit", [ctypes.c_char_p])
        _bind(self.library, "aclrtSetDevice", [integer])
        _bind(self.library, "aclrtGetCurrentContext", [ctypes.POINTER(pointer)])
        _bind(self.library, "aclrtCreateStream", [ctypes.POINTER(pointer)])
        _bind(self.library, "aclrtDestroyStream", [pointer])
        _bind(self.library, "aclrtSynchronizeStream", [pointer])
        _bind(self.library, "aclrtMalloc", [ctypes.POINTER(pointer), size, integer])
        _bind(self.library, "aclrtFree", [pointer])
        _bind(self.library, "aclrtMallocHost", [ctypes.POINTER(pointer), size])
        _bind(self.library, "aclrtFreeHost", [pointer])
        _bind(self.library, "aclrtMemcpy", [pointer, size, pointer, size, integer])
        _bind(self.library, "aclrtMemcpyAsync", [pointer, size, pointer, size, integer, pointer])
        _bind(self.library, "aclrtGetMemInfo", [integer, ctypes.POINTER(size), ctypes.POINTER(size)])
        status = self.library.aclInit(None)
        # A persistent ChipWorker can already own this process's ACL initialization.
        if status != 100002:  # ACL_ERROR_REPEAT_INITIALIZE
            _check(status, "aclInit")
        self.samples = []
        self.intervals = []
        self._empty_interval = nullcontext()

    def interval(self, rank, phase, stage):
        """Record a stage on the allocation observer's common host clock."""
        if not self.diagnostics:
            return self._empty_interval
        return self._interval(rank, phase, stage)

    @contextmanager
    def _interval(self, rank, phase, stage):
        begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        try:
            yield
        finally:
            end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            self.intervals.append({"rank": rank, "phase": phase, "stage": stage,
                                   "begin_ns": begin, "end_ns": end})

    def select(self, rank):
        _check(self.library.aclrtSetDevice(rank), "aclrtSetDevice")

    def copy(self, destination, source, nbytes, direction):
        """Synchronously transfer a live owned span in the selected device context."""
        if nbytes <= 0 or direction not in (1, 2):
            raise ValueError("CANN staging requires positive bytes and H2D/D2H direction")
        arguments = [destination, nbytes, source, nbytes, direction]
        _check(self.library.aclrtMemcpy(*arguments), "aclrtMemcpy/staging")

    def sample(self, rank, stage):
        if not self.diagnostics:
            return None
        self.select(rank)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        _check(self.library.aclrtGetMemInfo(1, ctypes.byref(free), ctypes.byref(total)), "aclrtGetMemInfo")
        sample = {"rank": rank, "stage": stage, "time_ns": time.time_ns(),
                  "free_bytes": free.value, "total_bytes": total.value, "used_bytes": total.value - free.value}
        self.samples.append(sample)
        return sample


class HCCLBytes:
    """Exchange equal contiguous peer blocks without interpreting model tensors."""

    @staticmethod
    def root_info(memory):
        """Create communicator bootstrap bytes in the coordinating process."""
        library = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
        function = _bind(library, "HcclGetRootInfo", [ctypes.c_void_p])
        root = ctypes.create_string_buffer(4108)
        _check(function(root), "HcclGetRootInfo")
        return root.raw

    def __init__(self, memory, rank, world, group_ranks=None, *, root_info, _register_owner=None):
        self.memory, self.rank, self.world = memory, rank, world
        self._lock = RLock()
        self._closed = False
        self._pending = False
        self.send_buffer, self.recv_buffer = ctypes.c_void_p(), ctypes.c_void_p()
        self.send_capacity = self.recv_capacity = 0
        self._spare_buffers = []
        self.host_buffer, self.host_capacity = ctypes.c_void_p(), 0
        self._host_spare_buffers = []
        self.group_ranks = list(range(world)) if group_ranks is None else group_ranks
        local_rank = self.group_ranks.index(rank)
        memory.select(rank)
        self._context = ctypes.c_void_p()
        _check(memory.library.aclrtGetCurrentContext(ctypes.byref(self._context)), "aclrtGetCurrentContext")
        self.library = ctypes.CDLL("libhccl.so", mode=ctypes.RTLD_GLOBAL)
        pointer, u64, integer = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        _bind(self.library, "HcclGetRootInfo", [pointer])
        _bind(self.library, "HcclCommInitRootInfo", [ctypes.c_uint32, pointer, ctypes.c_uint32, ctypes.POINTER(pointer)])
        _bind(self.library, "HcclCommDestroy", [pointer])
        _bind(self.library, "HcclAlltoAll", [pointer, u64, integer, pointer, u64, integer, pointer, pointer])
        _bind(self.library, "HcclAllGather", [pointer, pointer, u64, integer, pointer, pointer])
        _bind(self.library, "HcclReduceScatter", [pointer, pointer, u64, integer, integer, pointer, pointer])
        self.stream, self.comm = pointer(), pointer()
        if _register_owner is not None:
            _register_owner(self)
        try:
            _check(memory.library.aclrtCreateStream(ctypes.byref(self.stream)), "aclrtCreateStream")
            root = ctypes.create_string_buffer(4108)
            if len(root_info) != ctypes.sizeof(root):
                raise ValueError("Invalid in-memory HCCL root information size")
            ctypes.memmove(root, root_info, len(root_info))
            arguments = [world, root, local_rank, ctypes.byref(self.comm)]
            _check(self.library.HcclCommInitRootInfo(*arguments), "HcclCommInitRootInfo")
        except BaseException:
            self.close()
            raise
        self.buffers_peak = 0
        self.transfers = []

    def exchange(self, value, stage, collective="all_to_all"):
        with self._lock:
            if self._closed:
                raise RuntimeError("HCCL transport is closed")
            if self._pending:
                raise RuntimeError("HCCL transport has undrained work; close it before reuse")
            return self._exchange(value, stage, collective)

    def _reserve(self, name, required):
        """Register replacements before allocation so every failure retains ownership."""
        if required <= getattr(self, name + "_capacity"):
            return
        acl = self.memory.library
        replacement = ctypes.c_void_p()
        spares = self._host_spare_buffers if name == "host" else self._spare_buffers
        slot = len(spares)
        spares.append((replacement, required))
        free = acl.aclrtFreeHost if name == "host" else acl.aclrtFree
        if name == "host":
            _check(acl.aclrtMallocHost(ctypes.byref(replacement), required), "aclrtMallocHost/staging")
        else:
            _check(acl.aclrtMalloc(ctypes.byref(replacement), required, 0), "aclrtMalloc/" + name)
            self.buffers_peak = max(self.buffers_peak, self.send_capacity + self.recv_capacity
                                    + sum(size for _, size in self._spare_buffers))
        previous = getattr(self, name + "_buffer")
        try:
            if previous.value:
                _check(free(previous), "free/" + name)
        except BaseException:
            if not free(replacement):
                spares.pop(slot)
            raise
        setattr(self, name + "_buffer", replacement)
        setattr(self, name + "_capacity", required)
        spares.pop(slot)

    def _exchange(self, value, stage, collective):
        current = ctypes.c_void_p()
        status = self.memory.library.aclrtGetCurrentContext(ctypes.byref(current))
        # Never infer selection from the last rank: another owner can change
        # this thread's actual context between transport calls.
        if status or not current.value or current.value != self._context.value:
            self.memory.select(self.rank)
        if value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("HCCL byte transport requires a contiguous CPU staging container")
        byte_count = value.numel() * value.element_size()
        if not byte_count or (collective != "all_gather" and byte_count % self.world):
            raise ValueError("Every rank must provide an equal nonempty block for every peer")
        if collective == "all_gather":
            received = torch.empty((self.world * value.shape[0], *value.shape[1:]), dtype=value.dtype)
        elif collective == "reduce_scatter":
            if value.dtype not in (torch.bfloat16, torch.float32) or value.shape[0] % self.world:
                raise ValueError("Reduce-scatter requires equal BF16/FP32 row buckets")
            received = torch.empty((value.shape[0] // self.world, *value.shape[1:]), dtype=value.dtype)
        elif collective == "all_to_all":
            received = torch.empty_like(value)
        else:
            raise ValueError("Unsupported basic collective: " + collective)
        received_bytes = received.numel() * received.element_size()
        acl = self.memory.library
        self._reserve("send", byte_count)
        self._reserve("recv", received_bytes)
        receive_offset = (byte_count + 63) // 64 * 64
        self._reserve("host", receive_offset + received_bytes)
        host_send = self.host_buffer.value
        host_receive = host_send + receive_offset
        send_buffer, recv_buffer = self.send_buffer, self.recv_buffer
        self.buffers_peak = max(self.buffers_peak, self.send_capacity + self.recv_capacity)
        try:
            self.memory.sample(self.rank, stage + "/load")
            with self.memory.interval(self.rank, "load", stage + "/staging"):
                ctypes.memmove(host_send, value.data_ptr(), byte_count)
                self._pending = True
                arguments = [send_buffer, byte_count, host_send, byte_count, 1, self.stream]
                _check(acl.aclrtMemcpyAsync(*arguments), "aclrtMemcpyAsync/H2D")
            peer_bytes = byte_count // self.world
            with self.memory.interval(self.rank, "communication", stage):
                self._pending = True
                if collective == "all_to_all":
                    arguments = [send_buffer, peer_bytes, 7, recv_buffer, peer_bytes, 7, self.comm, self.stream]
                    status = self.library.HcclAlltoAll(*arguments)
                elif collective == "all_gather":
                    status = self.library.HcclAllGather(send_buffer, recv_buffer, byte_count, 7, self.comm, self.stream)
                else:
                    dtype = 11 if value.dtype == torch.bfloat16 else 4
                    count = received.numel()
                    status = self.library.HcclReduceScatter(send_buffer, recv_buffer, count, dtype, 0, self.comm, self.stream)
                _check(status, "HCCL/" + collective)
                arguments = [host_receive, received_bytes, recv_buffer, received_bytes, 2, self.stream]
                _check(acl.aclrtMemcpyAsync(*arguments), "aclrtMemcpyAsync/D2H")
                _check(acl.aclrtSynchronizeStream(self.stream), "aclrtSynchronizeStream")
                self._pending = False
            self.memory.sample(self.rank, stage + "/execute")
            ctypes.memmove(received.data_ptr(), host_receive, received_bytes)
            self.transfers.append({"stage": stage, "bytes_per_peer": peer_bytes, "world": self.world,
                                   "shape": list(value.shape), "dtype": str(value.dtype), "collective": collective,
                                   "transport_type": "BF16/FP32 SUM" if collective == "reduce_scatter" else "UINT8"})
        finally:
            if self._pending:
                _check(acl.aclrtSynchronizeStream(self.stream), "aclrtSynchronizeStream/failure")
                self._pending = False
        return received

    def close(self):
        with self._lock:
            self._closed = True
            self.memory.select(self.rank)
            if self.stream.value:
                # Failed synchronization cannot establish that DMA/HCCL stopped.
                # Retain every owner so a later close can drain and release it.
                self._pending = True
                _check(self.memory.library.aclrtSynchronizeStream(self.stream), "aclrtSynchronizeStream/close")
                self._pending = False
            errors = []
            for name, library, operation in (
                    ("comm", self.library, "HcclCommDestroy"),
                    ("stream", self.memory.library, "aclrtDestroyStream"),
                    ("host_buffer", self.memory.library, "aclrtFreeHost"),
                    ("recv_buffer", self.memory.library, "aclrtFree"),
                    ("send_buffer", self.memory.library, "aclrtFree")):
                pointer = getattr(self, name)
                if pointer.value:
                    try:
                        _check(getattr(library, operation)(pointer), operation)
                        pointer.value = None
                    except BaseException as error:
                        errors.append(error)
            for spares, operation in ((self._spare_buffers, "aclrtFree"),
                                      (self._host_spare_buffers, "aclrtFreeHost")):
                for index in range(len(spares) - 1, -1, -1):
                    pointer, size = spares[index]
                    try:
                        if pointer.value:
                            _check(getattr(self.memory.library, operation)(pointer), "free/spare")
                            pointer.value = None
                        spares.pop(index)
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise errors[0]
