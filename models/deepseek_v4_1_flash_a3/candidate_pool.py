# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reduce 2048 index scores into eight-row candidate blocks and publish persistent IDs."""

from __future__ import annotations

import pypto.language as pl


# model config
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 2048
SCORE_WIDTH = 2048
POOL_CAPACITY = SCORE_WIDTH // CANDIDATE_BLOCK_SIZE


def make_candidate_block_pool(
    capacity: int,
    *,
    score_width: int = SCORE_WIDTH,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
):
    """Build the fixed-bucket candidate publisher used by Prefill."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("candidate pool capacity must be a positive multiple of 16")
    if score_width != SCORE_WIDTH or topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError("A2/A3 candidate pool uses a fixed 2048-row score bucket")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("A2/A3 candidate blocks contain eight compressed rows")

    @pl.jit
    def kernel(
        scores: pl.Tensor[[capacity, score_width], pl.FP32],
        compressed_lengths: pl.Tensor[[capacity], pl.INT32],
        pool_block_ids: pl.Out[pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32]],
        pool_block_counts: pl.Out[pl.Tensor[[capacity], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_candidate_pool"):
            token = pl.tile.get_block_idx()
            if token < active:
                raw = pl.load(scores, [token, 0], [1, score_width])
                grouped = pl.reshape(raw, [POOL_CAPACITY, CANDIDATE_BLOCK_SIZE])
                reduce_tmp = pl.tile.create(
                    [POOL_CAPACITY, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32
                )
                maxima = pl.reshape(
                    pl.row_max(grouped, tmp_tile=reduce_tmp), [1, POOL_CAPACITY]
                )
                length = pl.max(pl.read(compressed_lengths, [token]), 0)
                visible_blocks = pl.min(
                    (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    POOL_CAPACITY,
                )
                lanes = pl.tile.arange(0, [1, POOL_CAPACITY], dtype=pl.INT32)
                # Build masks and IDs with vector operations. Scalar tile writes
                # can race the following sort on A3.
                last_block = pl.cast(visible_blocks - 1, pl.INT32)
                invalid = pl.minimum(pl.maximum(pl.sub(lanes, last_block), 0), 1)
                one = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=1)
                valid = pl.sub(one, invalid)
                maxima = pl.add(pl.mul(maxima, pl.cast(valid, pl.FP32)),
                                pl.mul(pl.cast(invalid, pl.FP32), -1.0e30))
                delta = pl.sub(lanes, last_block)
                distance = pl.maximum(delta, pl.mul(delta, -1))
                pinned = pl.sub(one, pl.minimum(distance, 1))
                maxima = pl.add(pl.mul(maxima, pl.cast(pl.sub(one, pinned), pl.FP32)),
                                pl.mul(pl.cast(pinned, pl.FP32), 1.0e30))
                logical_ids = pl.sub(lanes, pl.mul(pl.add(lanes, 1), invalid))
                sort_scores = pl.tile.full([1, POOL_CAPACITY], dtype=pl.FP32, value=-1.0e30)
                sort_scores = pl.tile.assemble(sort_scores, maxima, [0, 0])
                sort_ids = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                sort_ids = pl.tile.assemble(sort_ids, logical_ids, [0, 0])
                sort_tmp = pl.tile.create([1, 2 * POOL_CAPACITY], dtype=pl.FP32)
                unsigned_ids = pl.reinterpret_view(sort_ids, pl.UINT32)
                pairs = pl.tile.sort32(sort_scores, unsigned_ids, tmp=sort_tmp)
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
                leading_pairs = pl.tile.slice(pairs, [1, 2 * POOL_CAPACITY], [0, 0])
                sorted_ids = pl.tile.gather_mask(
                    leading_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                )
                # Native A3 preserves score order, with the newest block first.
                pl.store(sorted_ids, [token, 0], pool_block_ids)
            else:
                empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                pl.store(empty, [token, 0], pool_block_ids)
        # Scalar INT32 stores from one worker per token can share a cache line
        # on A2/A3.  Publish the compact count vector from a single worker.
        with pl.spmd(1, name_hint="dsv41_a2a3_candidate_counts"):
            worker = pl.tile.get_block_idx()
            if worker == 0:
                for token in pl.range(capacity):
                    length = pl.max(pl.read(compressed_lengths, [token]), 0)
                    visible_blocks = pl.min(
                        (length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                        POOL_CAPACITY,
                    )
                    count = pl.min(visible_blocks, POOL_CAPACITY)
                    if token >= active:
                        count = 0
                    pl.write(pool_block_counts, [token], pl.cast(count, pl.INT32))
        return pool_block_ids, pool_block_counts

    return kernel


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "CANDIDATE_TOPK_BLOCKS",
    "POOL_CAPACITY",
    "SCORE_WIDTH",
    "make_candidate_block_pool",
]


def make_candidate_pool_publish(capacity: int, shared_rows: int = 8192):
    """Publish compact candidate blocks into active persistent-state rows."""

    @pl.jit
    def candidate_pool_publish(
        blocks: pl.Tensor[[capacity, POOL_CAPACITY], pl.INT32],
        counts: pl.Tensor[[capacity], pl.INT32],
        shared_blocks: pl.InOut[pl.Tensor[[shared_rows, 1, CANDIDATE_TOPK_BLOCKS], pl.INT32]],
        shared_counts: pl.InOut[pl.Tensor[[shared_rows, 1], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(shared_blocks, [shared_rows, CANDIDATE_TOPK_BLOCKS])
        for section in pl.spmd(capacity * (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY),
                               name_hint="prefill_candidate_blocks_publish"):
            token = section // (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY)
            column = section % (CANDIDATE_TOPK_BLOCKS // POOL_CAPACITY) * POOL_CAPACITY
            if token < active:
                if column == 0:
                    ids = pl.load(blocks, [token, 0], [1, POOL_CAPACITY])
                    pl.store(ids, [token, 0], flat)
                else:
                    empty = pl.tile.full([1, POOL_CAPACITY], dtype=pl.INT32, value=-1)
                    pl.store(empty, [token, column], flat)
        # The pinned A3 native QLI returns candidates but leaves this persistent
        # length buffer unchanged on its nonempty source path.
        return shared_blocks, shared_counts

    return candidate_pool_publish


def golden_candidate_block_pool(args):
    """Order visible logical blocks by maximum score, with the newest block first."""
    import torch

    args["pool_block_ids"].fill_(-1)
    args["pool_block_counts"].zero_()
    maxima = args["scores"].reshape(-1, POOL_CAPACITY, CANDIDATE_BLOCK_SIZE).amax(dim=-1)
    for token in range(int(args["active"])):
        count = min(POOL_CAPACITY, max(0, (int(args["compressed_lengths"][token]) + 7) // 8))
        if count:
            ordered = torch.argsort(maxima[token, :count - 1], descending=True)
            ids = torch.cat((torch.tensor([count - 1]), ordered)).int()
            args["pool_block_ids"][token, :count] = ids
            args["pool_block_counts"][token] = count


def golden_candidate_pool_publish(args):
    """Replace active candidate rows and preserve persistent count bytes and inactive rows."""
    active = int(args["active"])
    args["shared_blocks"][:active].fill_(-1)
    args["shared_blocks"][:active, 0, :POOL_CAPACITY] = args["blocks"][:active]


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Check empty rows, partial blocks, 512/1024 boundaries and the 2048-row limit."""
    import torch
    from golden import ScalarSpec, TensorSpec, run

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    capacity, shared_rows = 16, 32
    lengths = torch.tensor([0, 1, 7, 8, 9, 511, 512, 513, 1023, 1024, 1025, 2040, 2041, 2047, 2048, -1]).int()
    block_scores = (torch.arange(POOL_CAPACITY) * 73) % POOL_CAPACITY
    scores = (block_scores[:, None] + torch.arange(8)[None, :] / 16).flatten().repeat(capacity, 1)
    records = []
    for active in (0, 1, 7, capacity):
        pool_specs = [
            TensorSpec("scores", [capacity, SCORE_WIDTH], torch.float32, init_value=scores),
            TensorSpec("compressed_lengths", [capacity], torch.int32, init_value=lengths),
            TensorSpec("pool_block_ids", [capacity, POOL_CAPACITY], torch.int32, init_value=-53),
            TensorSpec("pool_block_counts", [capacity], torch.int32, init_value=-59),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        pool = {spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
                for spec in pool_specs}
        golden_candidate_block_pool(pool)
        for token in range(capacity):
            count = int(pool["pool_block_counts"][token])
            expected = min(POOL_CAPACITY, max(0, (int(lengths[token]) + 7) // 8)) if token < active else 0
            ids = pool["pool_block_ids"][token]
            if count != expected or set(ids[:count].tolist()) != set(range(count)) or (ids[count:] != -1).any():
                raise RuntimeError("Candidate pool lost a visible block or published an invalid block")
            if count and int(ids[0]) != count - 1:
                raise RuntimeError("Candidate pool did not pin the newest block")
        publish_specs = [
            TensorSpec("blocks", [capacity, POOL_CAPACITY], torch.int32, init_value=pool["pool_block_ids"]),
            TensorSpec("counts", [capacity], torch.int32, init_value=pool["pool_block_counts"]),
            TensorSpec("shared_blocks", [shared_rows, 1, CANDIDATE_TOPK_BLOCKS], torch.int32, init_value=-67),
            TensorSpec("shared_counts", [shared_rows, 1], torch.int32, init_value=-71),
            ScalarSpec("active", torch.int32, active, compile_runtime=True),
        ]
        cases = [("pool", pool_specs, golden_candidate_block_pool, make_candidate_block_pool),
                 ("publish", publish_specs, golden_candidate_pool_publish, make_candidate_pool_publish)]
        for name, specs, golden_fn, builder in cases:
            if cpu_only:
                values = {spec.name: spec.create_tensor() if hasattr(spec, "create_tensor") else spec.to_python()
                          for spec in specs}
                golden_fn(values)
                if name == "publish":
                    if ((values["shared_blocks"][active:] != -67).any()
                            or (values["shared_counts"] != -71).any()
                            or (values["shared_blocks"][:active, :, POOL_CAPACITY:] != -1).any()):
                        raise RuntimeError("Candidate publication changed readonly bytes or retained stale tail IDs")
            else:
                fn = builder(capacity) if name == "pool" else builder(capacity, shared_rows)
                result = run(
                    fn=fn,
                    specs=specs,
                    golden_fn=golden_fn,
                    config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
                    compile_only=compile_only,
                    rtol=0,
                    atol=0,
                )
                if not result.passed:
                    raise RuntimeError(f"candidate-{name}-active{active}: {result.error}")
            records.append(dict(case=f"{name}-active{active}", cpu_golden="passed" if cpu_only else "not_run",
                                kernel="not_run" if cpu_only or compile_only else "passed",
                                compiled=not cpu_only))
    return records


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    result = validate(device=args.device, sim=args.sim, compile_only=args.compile_only, cpu_only=args.cpu_only)
    print(json.dumps(result))
