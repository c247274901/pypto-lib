# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay the official TP1 Attention CLI with frozen inputs and comparisons."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("models/deepseek_v4_flash_dspark")
REFERENCE_REVISION = "7cabb8ba022929bf09cc3cffd8cdadc3351de1fd"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ast_text(node):
    return ast.dump(node, include_attributes=False)


def protected_nodes(source):
    protected = {}
    signatures = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = ast_text(node.args)
            is_kernel = "value=Name(id='pl'" in annotations or any(
                "Name(id='pl'" in ast_text(decorator) for decorator in node.decorator_list
            )
            if is_kernel:
                signatures[node.name] = (annotations, ast_text(node.returns) if node.returns else None)
            else:
                protected[node.name] = ast_text(node)
        elif isinstance(node, ast.ClassDef):
            protected[node.name] = ast_text(node)
        elif isinstance(node, ast.If) and "__name__" in ast_text(node.test):
            protected["__main__"] = ast_text(node)
    return protected, signatures


def check_source_contract():
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", REFERENCE_REVISION, "--", str(MODEL)], cwd=ROOT, text=True,
    ).splitlines()
    for filename in changed:
        if not filename.endswith(".py"):
            continue
        original = subprocess.run(
            ["git", "show", f"{REFERENCE_REVISION}:{filename}"], cwd=ROOT, text=True,
            capture_output=True, check=False,
        )
        if original.returncode:
            continue
        before_host, before_signatures = protected_nodes(original.stdout)
        after_host, after_signatures = protected_nodes((ROOT / filename).read_text())
        for name, value in before_host.items():
            if after_host.get(name) != value:
                raise ValueError(f"frozen host/reference/input/CLI definition changed: {filename}:{name}")
        for name, value in before_signatures.items():
            if name in after_signatures and after_signatures[name] != value:
                raise ValueError(f"kernel parameter contract changed: {filename}:{name}")


def spec_manifest(specs):
    return [dict(name=spec.name, shape=list(spec.shape), dtype=str(spec.dtype)) for spec in specs]


def snapshot_manifest(path):
    return {str(item.relative_to(path)): digest(item) for item in sorted(path.rglob("*.pt"))}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def check_device_idle():
    physical = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not physical.isdigit():
        raise ValueError("device verification requires exactly one assigned physical device")
    status = subprocess.check_output(["npu-smi", "info"], text=True)
    rows = re.findall(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|", status, re.MULTILINE)
    busy = [dict(device=device, pid=pid) for device, chip, pid, name in rows if device == physical]
    if busy:
        raise RuntimeError(f"assigned NPU has another process; wait for resources: {busy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["swa", "csa", "hca"])
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--max-effective-us", type=float)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    check_source_contract()
    if args.static_only:
        for tp in (1, 2, 4):
            code = (
                "import sys; "
                f"sys.path.insert(0, {str(ROOT / MODEL)!r}); "
                f"sys.argv=['attention-import-check', '--tp', '{tp}']; "
                "import decode_swa, decode_csa, decode_hca, config; "
                f"assert config.TP == {tp}; print('TP{tp} imports passed')"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
        print(json.dumps(dict(outcome="valid", static_contract=True)))
        return
    if args.snapshot is None:
        parser.error("--snapshot is required for device verification")
    check_device_idle()
    args.snapshot = args.snapshot.resolve()
    assets_path = args.assets or Path(__file__).parent / f"assets_{args.stage}.json"
    expected = None
    if not args.capture:
        expected = json.loads(assets_path.read_text())
        if snapshot_manifest(args.snapshot) != expected["tensors"]:
            raise ValueError("frozen input/golden snapshot identity mismatch")
    import golden
    import torch

    if Path(golden.__file__).resolve().parent != ROOT / "golden":
        raise ValueError("golden must import from the actual source under verification")
    if (os.environ.get("PYPTO_BENCH"), os.environ.get("PYPTO_BENCH_WARMUP"),
            os.environ.get("PYPTO_BENCH_ROUNDS")) != ("1", "5", "100"):
        raise ValueError("official timing requires PYPTO_BENCH=1, warmup=5, rounds=100")
    origin_cache = Path(__file__).parent / f"baseline_{args.stage}.json"
    if os.environ.get("GOAL_PLUS_VERIFIER_PHASE") == "origin" and origin_cache.exists():
        cached = json.loads(origin_cache.read_text())
        current_tree = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{MODEL}"], text=True,
        ).strip()
        if cached["source_model_tree"] == current_tree:
            if expected is None or cached["specs"] != expected["specs"]:
                raise ValueError("cached origin tensor contract mismatch")
            cached.update(baseline_reused=True, verification_device=os.environ["ASCEND_RT_VISIBLE_DEVICES"])
            print(json.dumps(cached, sort_keys=True), flush=True)
            return
    torch.manual_seed(0)
    original_run = golden.run
    reports = []

    def official_run(**kwargs):
        specs = spec_manifest(kwargs["specs"])
        by_name = {spec.name: spec for spec in kwargs["specs"]}
        if by_name["x_hc"].shape[0] != 8 or by_name["x_out"].shape[0] != 8:
            raise ValueError("the fixed workload requires exactly eight query/output rows")
        positions = by_name["position_ids"].create_tensor().reshape(-1).tolist()
        if positions != list(range(8192, 8200)):
            raise ValueError(f"unexpected active query positions: {positions}")
        if expected is not None and specs != expected["specs"]:
            raise ValueError("official tensor names/shapes/dtypes changed")
        kwargs["save_data"] = args.capture
        kwargs["golden_data"] = None if args.capture else str(args.snapshot)
        if args.runtime_dir is not None:
            kwargs["runtime_dir"] = str(args.runtime_dir.resolve())
        result = original_run(**kwargs)
        report = dict(
            stage=args.stage, outcome="error", reference_revision=REFERENCE_REVISION,
            source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            specs=specs, positions=positions, work_dir=str(result.work_dir),
            snapshot=str(args.snapshot), golden_replayed=not args.capture,
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            logical_device=0, platform="a2a3", tp=1, batch=1, k=7, query_rows=8,
            warmup=5, rounds=100, error=result.error,
        )
        if result.passed and result.bench is not None:
            stats = result.bench
            samples = list(stats.per_round("effective"))
            if (stats.all_zero_device or stats.fallback_flattened or len(samples) != 100
                    or stats.rounds != 100 or stats.warmup != 5
                    or any(not math.isfinite(x) or x <= 0 for x in samples)):
                raise ValueError("invalid or missing real-device benchmark samples")
            report.update(outcome="valid", effective_us=statistics.median(samples), samples_us=samples)
            blocks = [statistics.median(samples[index:index + 25]) for index in range(0, 100, 25)]
            report["block_medians_us"] = blocks
            report["relative_block_range"] = (max(blocks) - min(blocks)) / report["effective_us"]
            if report["relative_block_range"] > 0.03:
                report.update(outcome="error", error="nonstationary device timings exceed three percent noise")
            if args.max_effective_us is not None and report["effective_us"] > args.max_effective_us:
                if report["outcome"] == "valid":
                    report.update(outcome="invalid", error="full-path performance exceeds the frozen limit")
        elif not result.passed and "validation failed" in str(result.error).lower():
            report["outcome"] = "invalid"
        if args.capture and report["outcome"] == "valid":
            shutil.copytree(Path(result.work_dir) / "data", args.snapshot, dirs_exist_ok=False)
            assets = dict(reference_revision=REFERENCE_REVISION, specs=specs,
                          tensors=snapshot_manifest(args.snapshot))
            if not any(name.startswith("out/") for name in assets["tensors"]):
                raise ValueError("a correctness snapshot must contain official golden outputs")
            write_json(assets_path, assets)
            for tensor in args.snapshot.rglob("*.pt"):
                tensor.chmod(0o444)
        reports.append(report)
        if args.report is not None:
            write_json(args.report, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return result

    golden.run = official_run
    entry = ROOT / MODEL / f"decode_{args.stage}.py"
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), "-p", "a2a3", "--tp", "1", "--start-pos", "8192", "-d", "0"]
    runpy.run_path(str(entry), run_name="__main__")
    if len(reports) != 1:
        raise RuntimeError("official entry did not produce one valid full-path measurement")
    if reports[0]["outcome"] != "valid":
        raise SystemExit(1 if reports[0]["outcome"] == "invalid" else 2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        print(json.dumps(dict(outcome="error", error=f"{type(error).__name__}: {error}")), flush=True)
        raise SystemExit(2) from error
