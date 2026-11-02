# A2 operator optimization migration

This change ports six operator optimizations onto PR #2 revision
`addb655f7b4f4a4ce56af875724a04d79d3fe325`. Measurements below compare that
unmodified revision with the migrated implementation on the same device and
frozen inputs. They are PyPTO-to-PyPTO comparisons, not vLLM speedups.

## Implementation scope

| Operator | Migrated changes |
| --- | --- |
| QKV | Whole-row quantization, earlier KV scheduling, pipelined QB columns, fused normalization/quantization/RoPE epilogues; optional prepacked weights. |
| SWA | Fused gather, QK, softmax and PV for one-window cases, retaining the general fallback. |
| O-proj | Batched inverse RoPE, grouped Wo-A, larger output tiles, partial-max quantization and Wo-B epilogue fusion. |
| HC | Fused FP32 projection/norm and revised collapse/post tiling; all 20 Sinkhorn iterations remain. |
| Shared expert | BF16-path packing/GEMM/activation/quantization fusion and tiling; the distinct FP32 fused-activation semantics remain available. |
| Routed expert | Tiling only: receive rows 64 to 16, gate inner tiles 3 to 1, output columns 256 to 320. PR quantization, rounding, padding and scale order remain. |

The migrated factories serve the canonical W8A8 `attention_block`, the
corresponding `layer0_single_card` components, and shared/routed expert APIs.
The separate raw-prefill `ecotech_linear` and `ecotech_shared` paths were not
replaced. Native HC-pre callers still execute `_C_ascend.npu_hc_pre_v3`.
The separate diagnostic `qkv.py` package export was not replaced either.
This change does not establish performance gains for every historical
Prefill/Decode entry or for the eleven independently aligned layers.

## Measurement contract

- Hardware: one Ascend 910B3, `a2a3`, physical device 3.
- Environment: conda `pypto`; PyPTO 0.1.0, simpler 0.1.0, PTOAS 0.67,
  CANN 9.1.1, Torch 2.10.0+cpu. Installed wheel metadata did not expose PyPTO
  or simpler source revisions; package versions are not revision pins.
- `PYPTO_BENCH=1`, five warmup rounds and 100 measured rounds in one process
  per operator and implementation. The local adapter calls
  `pypto.runtime.benchmark`, as used by the golden harness.
- Metric: median `effective` microseconds, following the
  [performance guide](../../debug-and-tune/performance-tuning.md). This is the
  runtime device-domain interval metric, not an msprof kernel-duration sum
  or synchronized CPU wall time.
- Boundary: the complete canonical operator entry, including per-call weight
  packing where present. Compilation, host transfers, mutable-buffer
  restoration and accuracy checks are outside the device metric. QKV's
  optional prepacked API is not used for the performance table.
- Golden inputs and expected outputs are replayed from frozen native
  fixtures. The baseline and candidate use identical shapes, dtypes,
  buffers, warmup and sampling settings. All 100 samples and raw runtime
  spans are retained, along with source SHA256 receipts.

## Performance

Capacity is the allocated row count; active is the number of rows consumed
by the measured operator. Speedup is baseline median divided by candidate
median. Each row passed its native-fixture check before and after timing.

| Operator | Capacity / active | PR baseline (us) | Migrated (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| QKV | 16 / 1 | 698.92 | 232.01 | 3.01x |
| QKV | 16 / 7 | 804.29 | 233.17 | 3.45x |
| QKV | 16 / 16 | 1118.33 | 235.61 | 4.75x |
| QKV | 32 / 17 | 1664.02 | 244.37 | 6.81x |
| SWA | 16 / 1 | 56.55 | 19.38 | 2.92x |
| SWA | 16 / 7 | 266.54 | 26.61 | 10.02x |
| SWA | 16 / 16 | 589.49 | 30.81 | 19.13x |
| SWA | 32 / 17 | 624.65 | 31.60 | 19.77x |
| O-proj | 16 / 1 | 484.55 | 149.13 | 3.25x |
| O-proj | 16 / 7 | 590.90 | 203.44 | 2.90x |
| O-proj | 16 / 16 | 781.27 | 184.78 | 4.23x |
| O-proj | 32 / 17 | 1038.90 | 188.41 | 5.51x |
| HC | 16 / 1 | 114.47 | 33.91 | 3.38x |
| HC | 16 / 7 | 227.26 | 44.27 | 5.13x |
| HC | 16 / 16 | 442.34 | 59.00 | 7.50x |
| Shared expert | 16 / 1 | 292.72 | 79.22 | 3.69x |
| Shared expert | 16 / 7 | 295.10 | 79.99 | 3.69x |
| Shared expert | 16 / 16 | 342.43 | 84.56 | 4.05x |
| Shared expert | 32 / 17 | 512.32 | 86.53 | 5.92x |
| Routed expert | 64 / 2 | 116.24 | 86.70 | 1.34x |

The routed fixture is named `active7-expert50`, but expert 50 receives two
rows. That measurement covers local expert computation, excluding routing,
dispatch, combination and communication. SWA/O-proj prefix lengths are
0, 0, 112 and 120 for their four rows, respectively. These are independent
operator results; their speedups cannot be combined into a network speedup.

## Accuracy

All 20 frozen native-fixture cases passed: QKV four, SWA four, O-proj four,
HC three, shared expert four and routed expert one. The original floating
policy remains `rtol=0.05, atol=0.05`, with zero violating elements;
readonly inputs, state and inactive regions use their original exact checks.
Native fixtures were captured with vLLM-Ascend
`87f2b060adff3079bd63f995c6c628820fec0f9d` and vLLM
`db9527a46873454610df6dbedf79a36d6bf1a7f6`.

Two complete independent L0 regressions also passed against the unmodified
PR, at capacity 16 with active 1 and 7. Actual device outputs flowed through
attention HC-pre/norm, QKV, SWA, O-proj, HC-post, FFN HC-pre/norm, shared
expert, top-six routed experts and FFN HC-post. The runs used real L0 weights
and fixed fixture hidden inputs; router/dispatch/combine orchestration ran
on CPU. Router IDs matched exactly, every floating comparison had zero
violations, and final hidden maximum absolute differences were 0.015625 and
0.0234375. This is a PR regression check, not a fresh eleven-layer native
acceptance run or a communication benchmark.

The native QKV comparator treats `query_latent` as diagnostic, checking
shape, dtype, finite values and inactive tails. Its numerical regression
against the PR was additionally checked within the complete L0 runs.
The QKV prepacked API passed four cases, and the retained FP32 shared
`fused_activation=True` API passed four cases against the PR's FP32 path.
Every device argument was also read back in a separate resident audit.
For routed expert only, that extra audit used a local schema-3 NZ logical
shape validation adapter; the timed host API and complete L0 runs did not
use this adapter. No shared runtime installation was modified.

## Local evidence

Generated artifacts are intentionally untracked under
`build_output/pr2-migration/`. Frozen native fixtures and checkpoint files
are external inputs and are not distributed with this report.

| Evidence | Relative artifact path |
| --- | --- |
| Timing adapter and driver | `bench.py`, `batch.py` |
| Performance samples and before/after native checks | `bench-pypto/{baseline,candidate}-{qkv,swa,o-proj,hc,moe-shared}/report.json` |
| Routed performance, final run | `bench-pypto/{baseline,candidate}-moe-routed-v3/report.json` |
| Full argument device readback, 19 cases | `resident-pypto/report.json` (19 per-case passes; run ended before the separate routed retry) |
| Routed device readback retry | `resident-routed-pypto/report.json` |
| Complete L0 baseline/candidate | `layer-pypto/{baseline,candidate}-active{1,7}/{report.json,checks.json,actual.pt}` |
| QKV prepacked checks | `resident-qkv-prepacked-pypto/report.json` |
| FP32 shared checks | `fused-pypto/{baseline,candidate}/report.json` |

Only the paths listed above support the results in this report. Earlier
failed attempts and results from other environments or timing methods are
not part of the comparison.
