# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Score INT8 index keys and select a causal Top-512 from up to 2048 keys."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import torch
import pypto.language as pl


# model config
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
KEY_CAPACITY = 2048
NEGATIVE_INF = -3.0e38

# tiling
KEY_TILE = 32
HEAD_TILE = 16
SORT_TILE = 512


def indexer_select_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """Ideal visibility-aware oracle without the executable kernel's FP16 boundaries.

    ``query`` is ``[T, H, 128]`` INT8, ``key`` is ``[K, 128]`` INT8, and the
    two scale tensors are FP16.  ``positions`` is the raw token position used
    by vLLM to derive the visible compressed-key prefix.
    """

    if query.ndim != 3 or tuple(query.shape[1:]) != (INDEX_HEADS, INDEX_HEAD_DIM):
        raise ValueError("query must have shape [T, 32, 128]")
    if key.ndim != 2 or key.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("key must have shape [K, 128]")
    tokens, heads, _ = query.shape
    key_rows = key.shape[0]
    if query_scale.shape != (tokens, heads) or weights.shape != (tokens, heads):
        raise ValueError("query_scale and weights must have shape [T, 32]")
    if key_scale.numel() != key_rows or positions.shape != (tokens,):
        raise ValueError("key_scale must be [K] and positions must be [T]")
    if not 1 <= topk <= key_rows:
        raise ValueError("topk must be in [1, key.shape[0]]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, key_rows))
    scores = torch.relu(dots)
    scores.mul_(weights.float().unsqueeze(-1))
    scores = scores.sum(dim=1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key_rows)
    for token in range(tokens):
        count = min(int(visible[token]), topk)
        if count:
            # QLI returns score order.  ``sorted=False`` matches the native
            # operator's contract; ties are not expected for random fixtures.
            picked = torch.topk(scores[token, : int(visible[token])], count, sorted=False).indices
            indices[token, :count] = picked.to(torch.int32)
    return {"scores": scores, "indices": indices}


def make_indexer_select(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket PyPTO QLI score/Top-K kernel."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    if key_capacity not in (512, 1024, 2048):
        raise ValueError("A2/A3 QLI key_capacity must be 512, 1024 or 2048")
    if topk != INDEX_TOPK:
        raise ValueError(f"A2/A3 QLI uses fixed topk={INDEX_TOPK}")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit.inline
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ):
        token = pl.tile.get_block_idx()
        position = pl.read(positions, [token])
        visible = pl.min(key_capacity, pl.max(0, (position + 1) // compress_ratio))
        for k0 in pl.range(0, key_capacity, KEY_TILE):
            key_tile = key[k0:k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0:k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            q_tile = pl.reshape(query[token:token + 1, :, :], [heads, head_dim])
            dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
            dots_fp32 = pl.cast(dots, pl.FP32, mode="none")
            dots_scaled = pl.mul(pl.maximum(dots_fp32, 0.0), 0.0009765625)
            dots_half = pl.cast(dots_scaled, pl.FP16, mode="rint")
            q_scale_row = pl.cast(query_scale[token:token + 1, :], pl.FP32)
            weight_row = pl.cast(weights[token:token + 1, :], pl.FP32)
            combined = pl.mul(q_scale_row, weight_row)
            combined_half = pl.cast(combined, pl.FP16, mode="rint")
            weight_ones = pl.full([HEAD_TILE, heads], dtype=pl.FP16, value=1.0)
            weight_matrix = pl.col_expand_mul(weight_ones, combined_half)
            reduced = pl.matmul(weight_matrix, dots_half, out_dtype=pl.FP32)
            reduced_vector = pl.mul(reduced, 1.0)
            reduced_row = pl.slice(reduced_vector, [1, KEY_TILE], [0, 0])
            total = pl.mul(reduced_row, scale_tile)
            scores[token:token + 1, k0:k0 + KEY_TILE] = total

        # Fixed 512-key leaves, causal masking and running Top-512 pairs.
        for leaf0 in pl.range(0, key_capacity, SORT_TILE):
            row_raw = pl.load(scores, [token, leaf0], [1, SORT_TILE])
            ramp = pl.tile.arange(0, [1, SORT_TILE], dtype=pl.INT32)
            absolute = pl.add(ramp, pl.cast(leaf0, pl.INT32))
            distance = pl.mul(pl.sub(absolute, pl.cast(visible, pl.INT32)), -1)
            valid = pl.minimum(pl.maximum(distance, 0), 1)
            invalid = pl.sub(pl.tile.full([1, SORT_TILE], dtype=pl.INT32, value=1), valid)
            valid_scores = pl.mul(row_raw, pl.cast(valid, pl.FP32))
            invalid_scores = pl.mul(pl.cast(invalid, pl.FP32), NEGATIVE_INF)
            row = pl.add(valid_scores, invalid_scores)
            leaf_indices = pl.sub(absolute, pl.mul(pl.add(absolute, 1), invalid))
            sort_tmp = pl.tile.create([1, 2 * SORT_TILE], dtype=pl.FP32)
            leaf_pairs = pl.tile.sort32(row, pl.reinterpret_view(leaf_indices, pl.UINT32), tmp=sort_tmp)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
            if leaf0 == 0:
                pl.store(leaf_pairs, [token, 0], running_pairs)
            else:
                previous = pl.load(running_pairs, [token, 0], [1, 2 * topk])
                merge_tmp = pl.tile.create([1, 4 * topk], dtype=pl.FP32)
                merged = pl.tile.mrgsort(previous, leaf_pairs, tmp=merge_tmp)
                leading = pl.tile.slice(merged, [1, 2 * topk], [0, 0])
                pl.store(leading, [token, 0], running_pairs)
        pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
        selected = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(selected, [token, 0], indices)
        return scores, indices

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        for _worker in pl.spmd(active, name_hint="dsv41_a2a3_qli_select"):
            select_one(
                query, query_scale, key, key_scale, weights, positions,
                running_pairs, scores, indices,
            )
        return scores, indices

    return kernel


def golden_indexer_select(tensors, *, compress_ratio=2):
    """Score all keys and select the causal Top-512 with both FP16 boundaries."""
    active = int(tensors["active"])
    if not 0 <= active <= tensors["query"].shape[0]:
        raise ValueError("Indexer active rows must fit the actual capacity")
    if not active:
        return
    key = tensors["key"].to(torch.int32)
    for token in range(active):
        dots = tensors["query"][token].to(torch.int32) @ key.T
        positive = dots.float().clamp_min(0.0) * 0.0009765625
        narrowed_dots = positive.to(torch.float16).float()
        combined = tensors["query_scale"][token].float() * tensors["weights"][token].float()
        narrowed_weight = combined.to(torch.float16).float()
        reduced = narrowed_weight[None] @ narrowed_dots
        scores = reduced[0] * tensors["key_scale"].float()
        tensors["scores"][token] = scores
        visible = min(len(key), max(0, (int(tensors["positions"][token]) + 1) // compress_ratio))
        count = min(INDEX_TOPK, visible)
        tensors["indices"][token].fill_(-1)
        tensors["indices"][token, :count] = torch.argsort(scores[:visible], descending=True)[:count].int()


def indexer_active_compare(active, *, exact=False):
    """Compare the defined prefix; pure Out inactive rows have no preservation contract."""
    from golden.validation import ratio_allclose

    tolerance = 0.0 if exact else 0.05
    comparator = ratio_allclose(rtol=tolerance, atol=tolerance, max_error_ratio=0.0)

    def compare(actual, expected, **context):
        if not active:
            return True, "No active pure-Out rows; inactive rows are undefined"
        return comparator(actual[:active], expected[:active], **context)

    return compare


def indexer_select_golden_cases(*, build=True, key_capacity=KEY_CAPACITY):
    """Cover short and long histories, causal boundaries and negative-dot heads."""
    from functools import partial
    from golden import ScalarSpec, TensorSpec

    capacity = 16
    rows = torch.arange(KEY_CAPACITY, dtype=torch.int32)
    key = torch.zeros((KEY_CAPACITY, INDEX_HEAD_DIM), dtype=torch.int8)
    key[:, 0] = (rows // 16).to(torch.int8)
    key[:, 1] = (rows % 16).to(torch.int8)
    query = torch.ones((capacity, INDEX_HEADS, INDEX_HEAD_DIM), dtype=torch.int8)
    query[:, :, 0] = 16
    query[:, ::3] *= -1
    heads = torch.arange(INDEX_HEADS, dtype=torch.float32)
    query_scale = ((heads % 7 + 8.0) / 17.0).repeat(capacity, 1).to(torch.float16)
    weights = ((heads % 5 + 3.0) / 11.0).repeat(capacity, 1).to(torch.float16)
    boundaries = [-1, 0, 1, 31, 127, 511, 512, 513, 1023, 1024, 1025, 2046, 2047, 2048, 4095, 7]
    positions = torch.tensor(boundaries, dtype=torch.int32)
    key = key[:key_capacity]
    for ratio in (1, 2):
        for active in (0, 1, 7, capacity):
            specs = [
                TensorSpec("query", [capacity, INDEX_HEADS, INDEX_HEAD_DIM], torch.int8, init_value=query),
                TensorSpec("query_scale", [capacity, INDEX_HEADS], torch.float16, init_value=query_scale),
                TensorSpec("key", [key_capacity, INDEX_HEAD_DIM], torch.int8, init_value=key),
                TensorSpec("key_scale", [key_capacity], torch.float16, init_value=0.75),
                TensorSpec("weights", [capacity, INDEX_HEADS], torch.float16, init_value=weights),
                TensorSpec("positions", [capacity], torch.int32, init_value=positions),
                TensorSpec("scores", [capacity, key_capacity], torch.float32, init_value=-53.0),
                TensorSpec("indices", [capacity, INDEX_TOPK], torch.int32, init_value=-59),
                ScalarSpec("active", torch.int32, active, compile_runtime=True),
            ]
            case = dict(name=f"indexer-keys{key_capacity}-top512-ratio{ratio}-active{active}", specs=specs,
                        golden_fn=partial(golden_indexer_select, compress_ratio=ratio),
                        compare_fn=dict(scores=indexer_active_compare(active),
                                        indices=indexer_active_compare(active, exact=True)))
            if build:
                case["fn"] = make_indexer_select(capacity, key_capacity, compress_ratio=ratio)
            yield case


def validate(*, device=0, sim=False, compile_only=False, cpu_only=False):
    """Check causal selection at supported key capacities and active-row boundaries."""
    from golden import run
    from itertools import chain

    if cpu_only and compile_only:
        raise ValueError("CPU golden checks and kernel compilation are separate modes")
    records = []
    cases = (indexer_select_golden_cases(build=not cpu_only, key_capacity=keys) for keys in (512, 1024, 2048))
    for case in chain.from_iterable(cases):
        name = case.pop("name")
        if cpu_only:
            values = {spec.name: spec.create_tensor().clone() if hasattr(spec, "create_tensor") else spec.to_python()
                      for spec in case["specs"]}
            before = {key: value.clone() for key, value in values.items() if isinstance(value, torch.Tensor)}
            case["golden_fn"](values)
            active = int(values["active"])
            for key in before:
                if key not in ("scores", "indices") and not torch.equal(values[key], before[key]):
                    raise RuntimeError("Indexer golden mutated input " + key)
            if active:
                scores = values["scores"][:active]
                ordered = torch.sort(scores, dim=1).values
                if not (torch.diff(ordered, dim=1) > 0).all():
                    raise RuntimeError("The exact-index fixture must contain no tied scores")
                ratio = int(name.split("ratio")[1].split("-")[0])
                expected = torch.full((active, INDEX_TOPK), -1, dtype=torch.int32)
                for token in range(active):
                    visible = min(len(values["key"]), max(0, (int(values["positions"][token]) + 1) // ratio))
                    count = min(visible, INDEX_TOPK)
                    expected[token, :count] = torch.arange(visible - 1, visible - count - 1, -1).int()
                if not torch.equal(values["indices"][:active], expected):
                    raise RuntimeError("Causal Top-512 selection changed its exact score ordering")
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Indexer golden produced nonfinite scores")
            records.append(dict(name=name, cpu_golden="passed", device="not_run", inactive_Out_rows="undefined"))
            continue
        result = run(
            **case,
            config=dict(backend_type="a2a3", execution_mode=bool(sim), device_id=device),
            compile_only=compile_only,
            rtol=0.05,
            atol=0.05,
        )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
        records.append(dict(name=name, compiled=True, device="not_run" if compile_only else "passed",
                            inactive_Out_rows="undefined"))
    return records


__all__ = ["INDEX_HEADS", "INDEX_HEAD_DIM", "INDEX_TOPK", "indexer_select_reference", "make_indexer_select",
           "golden_indexer_select", "indexer_select_golden_cases", "validate"]


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
