# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded precision and device-latency validation for the complete Prefill Indexer."""

import argparse
import ast
import contextlib
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shutil

import torch
from golden import ScalarSpec, TensorSpec, run
from .gemm import prepare_ecotech_dynamic_weight
from .indexer import golden_indexer_select, indexer_active_compare, indexer_select_golden_cases
from .prefill_indexer import golden_index_key_gather, make_prefill_indexer_chain
from .qkv_proj_rope import golden_prefix_compare
from .utils import unpack_blocked_weight
from .weights import DecodeSafetensorsCheckpoint


CAPACITY = 16
OUTPUT_NAMES = (
    "projected", "head_projected", "rotated", "query", "query_scale", "weights",
    "key", "key_scale", "scores", "indices",
)
PERFORMANCE_CASES = (
    ("ratio2-k512-first", 512, 2, 4, 8, 0),
    ("ratio2-k512-continuation", 512, 2, 4, 512, 1008),
    ("ratio2-k1024-first", 1024, 2, 4, 8, 0),
    ("ratio2-k1024-continuation", 1024, 2, 4, 1024, 2032),
    ("ratio1-k2048-first", 2048, 1, 4, 16, 0),
    ("ratio1-k2048-continuation", 2048, 1, 4, 2048, 2032),
)
BOUNDARY_CASES = (
    ("ratio2-empty", 512, 2, 0, 0, 0),
    ("ratio2-single", 512, 2, 1, 1, 1),
    ("ratio2-partial-page", 1024, 2, 4, 513, 1008),
    ("ratio1-tail-position", 2048, 1, 4, 2048, 2044),
)


@functools.lru_cache(maxsize=2)
def checkpoint_operands(checkpoint):
    """Prepare one real checkpoint's fixed Indexer operands on the host."""
    reader = DecodeSafetensorsCheckpoint(checkpoint)
    names = [name for name in reader.weight_map if name.endswith(".indexer.wq_b.weight")]
    name = min(names, key=lambda value: int(value.split(".")[1]))
    prefix = name.removesuffix("wq_b.weight")
    raw = reader.tensor(name)
    scale = reader.tensor(prefix + "wq_b.weight_scale")
    offset = reader.tensor(prefix + "wq_b.weight_offset")
    packed, channel_scale = prepare_ecotech_dynamic_weight(raw, scale, offset)
    head_weight = reader.tensor(prefix + "weights_proj.weight").T.contiguous()
    if head_weight.dtype != torch.bfloat16 or head_weight.shape != (5120, 32):
        raise ValueError("Indexer head weights must be BF16 [5120,32]")
    if not torch.equal(unpack_blocked_weight(packed, 1280, 4096), raw.T):
        raise RuntimeError("Packed Indexer loader changed checkpoint bytes")
    return packed, channel_scale, head_weight


def golden_prefill_indexer_chain(values, *, compress_ratio):
    """Evaluate projections, adjacent RoPE, rounded query scales, gather and QLI in Torch."""
    active = int(values["active"])
    latent = values["latent"][:active].float()
    maximum = latent.abs().amax(dim=-1, keepdim=True)
    inverse = 127.0 / maximum.clamp_min(1e-30)
    payload = torch.round(latent * inverse).clamp(-127, 127).to(torch.int32)
    token_scale = maximum * (1.0 / 127.0)
    weight = unpack_blocked_weight(values["query_weight"], 1280, 4096).to(torch.int32)
    accumulator = (payload @ weight).float()
    combined = token_scale * values["channel_scale"][None]
    projected = (accumulator * combined).to(torch.bfloat16)
    values["projected"][:active] = projected
    head_projected = (values["hidden"][:active].float() @ values["head_weight"].float()).bfloat16()
    values["head_projected"][:active] = head_projected
    shaped = projected.reshape(active, 32, 128).float()
    rotated = shaped.clone()
    even, odd = shaped[..., 64::2], shaped[..., 65::2]
    cosine, sine = values["cosine"][:active, None], values["sine"][:active, None]
    rotated[..., 64::2] = even * cosine - odd * sine
    rotated[..., 65::2] = odd * cosine + even * sine
    rotated = rotated.bfloat16()
    values["rotated"][:active] = rotated.flatten(1)
    raw_scale = rotated.float().abs().amax(dim=-1) / 127.0
    query_scale = raw_scale.half().clamp_min(2.0 ** -24)
    query = torch.round(rotated.float() / query_scale.float()[..., None]).clamp(-127, 127).to(torch.int8)
    values["query"][:active] = query
    values["query_scale"][:active] = query_scale
    values["weights"][:active] = (head_projected.float() / 64.0).half()
    gather_values = dict(
        cache=values["cache"], scales=values["cache_scale"], table=values["table"],
        key=values["key"], key_scale=values["key_scale"], request=values["request"], length=values["length"],
    )
    golden_index_key_gather(gather_values)
    golden_indexer_select(values, compress_ratio=compress_ratio)


def chain_case(case, checkpoint):
    """Build deterministic activations, permuted physical cache pages and real packed weights."""
    name, keys, ratio, active, length, start = case
    rows, pages = 128 // ratio, keys // (128 // ratio)
    blocks = pages + 2
    generator = torch.Generator().manual_seed(314159)
    latent = torch.randn(CAPACITY, 1280, generator=generator).bfloat16()
    hidden = torch.randn(CAPACITY, 5120, generator=generator).bfloat16()
    packed, channel_scale, head_weight = checkpoint_operands(checkpoint)
    positions = torch.arange(start, start + CAPACITY, dtype=torch.int32)
    frequency = torch.exp(-torch.arange(32).float() * (math.log(10000.0) / 32))
    angles = positions.float()[:, None] * frequency[None]
    physical = torch.arange(blocks * rows)
    cache = torch.zeros(blocks * rows, 128, dtype=torch.int8)
    cache[:, 0] = ((physical // 16) % 128).to(torch.int8)
    cache[:, 1] = (physical % 16).to(torch.int8)
    cache[:, 2:] = torch.randint(-8, 9, (blocks * rows, 126), generator=generator, dtype=torch.int8)
    cache = cache.reshape(blocks, rows, 1, 128)
    cache_scale = ((physical.float() % 19 + 16) / 4096).half().reshape(blocks, rows, 1, 1)
    table = torch.stack((torch.arange(1, pages + 1), torch.arange(pages, 0, -1))).int()
    specs = [
        TensorSpec("latent", [CAPACITY, 1280], torch.bfloat16, init_value=latent),
        TensorSpec("hidden", [CAPACITY, 5120], torch.bfloat16, init_value=hidden),
        TensorSpec("query_weight", [1280 * 64, 64], torch.int8, init_value=packed),
        TensorSpec("channel_scale", [4096], torch.float32, init_value=channel_scale),
        TensorSpec("head_weight", [5120, 32], torch.bfloat16, init_value=head_weight),
        TensorSpec("cosine", [CAPACITY, 32], torch.float32, init_value=angles.cos()),
        TensorSpec("sine", [CAPACITY, 32], torch.float32, init_value=angles.sin()),
        TensorSpec("cache", [blocks, rows, 1, 128], torch.int8, init_value=cache),
        TensorSpec("cache_scale", [blocks, rows, 1, 1], torch.float16, init_value=cache_scale),
        TensorSpec("table", [2, pages], torch.int32, init_value=table),
        TensorSpec("positions", [CAPACITY], torch.int32, init_value=positions),
        TensorSpec("projected", [CAPACITY, 4096], torch.bfloat16, init_value=-53),
        TensorSpec("head_projected", [CAPACITY, 32], torch.bfloat16, init_value=-53),
        TensorSpec("rotated", [CAPACITY, 4096], torch.bfloat16, init_value=-53),
        TensorSpec("query", [CAPACITY, 32, 128], torch.int8, init_value=-53),
        TensorSpec("query_scale", [CAPACITY, 32], torch.float16, init_value=-53),
        TensorSpec("weights", [CAPACITY, 32], torch.float16, init_value=-53),
        TensorSpec("key", [keys, 128], torch.int8, init_value=-53),
        TensorSpec("key_scale", [keys], torch.float16, init_value=-53),
        TensorSpec("scores", [CAPACITY, keys], torch.float32),
        TensorSpec("indices", [CAPACITY, 512], torch.int32),
        ScalarSpec("request", torch.int32, 1, compile_runtime=True),
        ScalarSpec("length", torch.int32, length, compile_runtime=True),
        ScalarSpec("active", torch.int32, active, compile_runtime=True),
    ]
    compare = {key: golden_prefix_compare(active) for key in ("projected", "head_projected", "rotated", "weights")}
    compare.update({key: golden_prefix_compare(active, rtol=0, atol=0) for key in ("query", "query_scale")})
    compare.update({key: indexer_active_compare(active, exact=key == "indices") for key in ("scores", "indices")})
    compare.update({key: golden_prefix_compare(keys, rtol=0, atol=0) for key in ("key", "key_scale")})
    return dict(name=name, specs=specs, compare_fn=compare,
                golden_fn=functools.partial(golden_prefill_indexer_chain, compress_ratio=ratio))


def cpu_checks(checkpoint):
    """Check immutable inputs, active tails, causal bounds and independent reference identities."""
    from .indexer import validate_prefill_select
    from .prefill_indexer import validate_key_gather

    validate_prefill_select(cpu_only=True)
    validate_key_gather(cpu_only=True)
    records = []
    for case in PERFORMANCE_CASES + BOUNDARY_CASES:
        entry = chain_case(case, checkpoint)
        values = {spec.name: spec.create_tensor() if isinstance(spec, TensorSpec) else spec.to_python()
                  for spec in entry["specs"]}
        before = {key: value.clone() for key, value in values.items() if isinstance(value, torch.Tensor)}
        entry["golden_fn"](values)
        active, ratio = case[3], case[2]
        for key in before.keys() - set(OUTPUT_NAMES):
            if not torch.equal(before[key], values[key]):
                raise RuntimeError(f"Independent golden mutated input {key}")
        for key in OUTPUT_NAMES[:6]:
            if not torch.equal(values[key][active:], before[key][active:]):
                raise RuntimeError(f"Independent golden changed inactive {key}")
        for token in range(active):
            visible = min(case[1], max(0, (int(values["positions"][token]) + 1) // ratio))
            picked = values["indices"][token]
            count = min(512, visible)
            if (picked[:count] < 0).any() or (picked[:count] >= visible).any() or (picked[count:] != -1).any():
                raise RuntimeError("Independent golden violated causal selection bounds")
            if len(torch.unique(picked[:count])) != count:
                raise RuntimeError("Independent golden repeated Top-K indices")
        if not torch.isfinite(values["scores"][:active]).all():
            raise RuntimeError("Independent golden emitted nonfinite scores")
        records.append(dict(name=case[0], cpu_golden="passed"))
    return records


@contextlib.contextmanager
def precision_only():
    previous = os.environ.get("PYPTO_BENCH")
    os.environ["PYPTO_BENCH"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PYPTO_BENCH", None)
        else:
            os.environ["PYPTO_BENCH"] = previous


def existing_device_gates(device, *, include_select=True):
    """Run every existing score/Top-K case and paged gather boundary with build reuse."""
    from .prefill_indexer import make_index_key_gather

    builds = {}
    records = []
    with precision_only():
        for keys in ((512, 1024, 2048) if include_select else ()):
            for entry in indexer_select_golden_cases(key_capacity=keys):
                ratio = int(entry["name"].split("ratio")[1].split("-")[0])
                identity = (keys, ratio)
                result = run(
                    fn=entry["fn"],
                    specs=entry["specs"],
                    golden_fn=entry["golden_fn"],
                    compare_fn=entry["compare_fn"],
                    runtime_dir=builds.get(identity),
                    config=dict(backend_type="a2a3", execution_mode=False, device_id=device),
                )
                if not result.passed:
                    raise RuntimeError(f"{entry['name']}: {result.error}")
                builds[identity] = result.work_dir
                records.append(dict(name=entry["name"], kernel="passed", work_dir=str(result.work_dir)))
        for rows in (64, 128):
            pages, blocks = 2048 // rows, 2048 // rows + 2
            physical = torch.arange(blocks * rows)
            codes = ((physical[:, None] + torch.arange(128)[None]) % 127).to(torch.int8)
            codes = codes.reshape(blocks, rows, 1, 128)
            scales = ((physical.float() + 1) / 4096).half().reshape(blocks, rows, 1, 1)
            table = torch.stack((torch.arange(1, pages + 1), torch.arange(pages, 0, -1))).int()
            build = None
            for length in (0, 1, 511, 512, 513, 1024, 2048):
                specs = [
                    TensorSpec("cache", [blocks, rows, 1, 128], torch.int8, init_value=codes),
                    TensorSpec("scales", [blocks, rows, 1, 1], torch.float16, init_value=scales),
                    TensorSpec("table", [2, pages], torch.int32, init_value=table),
                    TensorSpec("key", [2048, 128], torch.int8, init_value=-53),
                    TensorSpec("key_scale", [2048], torch.float16, init_value=-59),
                    ScalarSpec("request", torch.int32, 1, compile_runtime=True),
                    ScalarSpec("length", torch.int32, length, compile_runtime=True),
                ]
                result = run(
                    fn=make_index_key_gather(blocks, 2, pages, page_rows=rows),
                    specs=specs,
                    golden_fn=golden_index_key_gather,
                    runtime_dir=build,
                    config=dict(backend_type="a2a3", execution_mode=False, device_id=device),
                    rtol=0,
                    atol=0,
                )
                if not result.passed:
                    raise RuntimeError(f"gather rows={rows} length={length}: {result.error}")
                build = result.work_dir
                records.append(dict(name=f"gather-rows{rows}-length{length}", kernel="passed", work_dir=str(build)))
    return records


def validate_chain(args):
    """Validate all chain cases and sum the six current-shape median device times."""
    if args.cpu_only:
        return dict(outcome="valid", cpu=cpu_checks(args.checkpoint), device="not_run")
    if args.formal:
        required = {"PYPTO_BENCH": "1", "PYPTO_BENCH_WARMUP": "5",
                    "PYPTO_BENCH_ROUNDS": "100", "PYPTO_BENCH_RAW": "1"}
        if any(os.environ.get(key) != value for key, value in required.items()):
            raise ValueError("Formal Indexer verification requires the frozen 5/100 raw benchmark environment")
    records, builds = [], {}
    cases = PERFORMANCE_CASES + BOUNDARY_CASES
    if args.case:
        cases = tuple(case for case in cases if case[0] == args.case)
        if not cases:
            raise ValueError("Unknown Indexer chain case")
    for case in cases:
        name, keys, ratio, _active, _length, _start = case
        print(f"[INDEXER] case={name}", flush=True)
        entry = chain_case(case, args.checkpoint)
        identity = (keys, ratio)
        snapshot = Path(args.golden_data) / name if args.golden_data else None
        build = args.runtime_dir or builds.get(identity)
        config = dict(backend_type="a2a3", execution_mode=False, device_id=args.device)
        config["enable_chip_swimlane"] = args.enable_chip_swimlane
        measured = case in PERFORMANCE_CASES
        with contextlib.nullcontext() if measured else precision_only():
            result = run(
                fn=make_prefill_indexer_chain(CAPACITY, keys, ratio, keys // (128 // ratio) + 2,
                                              2, keys // (128 // ratio)),
                specs=entry["specs"],
                golden_fn=entry["golden_fn"],
                compare_fn=entry["compare_fn"],
                golden_data=snapshot,
                save_data=args.save_data,
                runtime_dir=build,
                compile_only=args.compile_only,
                config=config,
            )
        if not result.passed:
            raise RuntimeError(f"{name}: {result.error}")
        if args.save_data and args.snapshot_root:
            destination = Path(args.snapshot_root) / name
            shutil.copytree(Path(result.work_dir) / "data", destination, dirs_exist_ok=True)
        builds[identity] = result.work_dir
        record = dict(name=name, kernel="compiled" if args.compile_only else "passed", work_dir=str(result.work_dir))
        if measured and not args.compile_only:
            samples = result.bench.per_round("effective") if result.bench is not None else []
            if args.formal and (len(samples) != 100 or any(not math.isfinite(value) or value <= 0 for value in samples)):
                raise RuntimeError("Missing or invalid 100-round device-domain timing")
            if samples:
                record.update(effective_us_median=statistics.median(samples), effective_us_raw=samples)
        records.append(record)
    gates = existing_device_gates(args.device) if args.existing_gates and not args.compile_only else []
    medians = [record["effective_us_median"] for record in records if "effective_us_median" in record]
    return dict(outcome="valid", effective_us_sum_medians=sum(medians), cases=records, existing_gates=gates)


def verify_integrity(manifest_path, golden_root):
    """Verify frozen reference functions, composition, and replay tensor bytes."""
    manifest = json.loads(Path(manifest_path).read_text())
    for relative, functions in manifest["protected_functions"].items():
        source = Path(relative).read_text()
        nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        for name, expected in functions.items():
            actual = hashlib.sha256(ast.get_source_segment(source, nodes[name]).encode()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Frozen reference/composition changed: {relative}::{name}")
    if golden_root:
        for relative, expected in manifest["golden_files"].items():
            actual = hashlib.sha256((Path(golden_root) / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Frozen replay data changed: {relative}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--existing-gates", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0)
    parser.add_argument("--report")
    parser.add_argument("--integrity-manifest")
    args = parser.parse_args(argv)
    if args.integrity_manifest:
        verify_integrity(args.integrity_manifest, args.golden_data)
    result = validate_chain(args)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
