# Decode HCA Attention validation contract

This task validates the existing full TP1 HCA CLI entry on real a2a3 hardware;
promotion additionally replays the completed SWA and CSA paths. Each invocation uses one request, positions 8192 through
8199, DSpark K7, and eight query rows. The official tensor layouts and pool
capacities remain unchanged.

`verify.sh` sources `PYPTO_ENV_SCRIPT` with the actual source root, isolates
build output under `GOAL_PLUS_VERIFIER_TMPDIR`, and pins benchmarking to
5 warmup launches and 100 measured rounds. Scoring uses the median of the
official `BenchmarkStats.per_round("effective")` values. Additional profiling
is disabled during these measurements.

Preparation captures the official inputs and golden outputs once, with
PyTorch seed 0. The frozen `assets_*.json` files record every tensor file's
SHA256 and every argument's name, shape, and dtype. Replays verify these
identities, import `golden` from the current source root, and execute the
unchanged official CLI with its original comparisons for every output and
in-place KV, cache, and state tensor. Reference computation is skipped only
because its frozen expected outputs are replayed. AST checks protect host
input construction, golden helpers, entry comparisons, and existing kernel
parameter contracts against the initial source revision.

During the origin bootstrap only, a cached official measurement may be reused
when the entire model source tree and tensor contract still match its recorded
identity. This preserves valid evidence across an authorized physical-device
change. Workspace relocation, candidate checks, and promotion execute the
actual source on device. A cached origin never substitutes for those checks.

Goal Plus process and promotion verifiers acquire the declared pool resource
lock. Local diagnostics use `locked.py` with the installed Goal Plus Python;
it resolves the assigned physical device from single-device visibility and
uses the same official lock implementation. Do not wrap a formal verifier
in this additional lock. A visible physical device is addressed as logical
device 0. Device migration belongs to Main and the public Goal Plus workflow.

Each Search keeps two persistent gpt-6.1-sol candidates with xhigh effort.
Main uses gpt-6.1-sol with max effort and tracks an effective optimization
window of exactly 10800 seconds per stage after actual dispatch. Preparation,
infrastructure recovery, device waiting, selection, and closeout are excluded.
Early returns continue the same candidate and native session.

SWA, CSA, and HCA run sequentially, with each promoted revision becoming the
next baseline. A three percent relative noise allowance is frozen for
performance decisions after inspecting baseline raw samples. Promotion checks
all affected paths; previous completed paths must stay inside their frozen
regression limits. Main also compares fresh baseline and candidate results
before applying a performance change. TP2/TP4 import checks preserve interface
coverage, but this contract does not claim multi-device numerical validation,
MoE validation, a full decode layer, or complete network execution.
