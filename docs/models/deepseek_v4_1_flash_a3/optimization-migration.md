# A3 Prefill operator optimization migration

This migration adapts applicable optimizations from A2 revision
`f1fc8c564f92e7705900434887d3756759daf774`. The current optimization baseline
rebases the original migration onto A3 revision
`60783e6ae0da024663716ba3678ce9523c54afb9`, including consolidated revision
`9fb52df7e28adc458d63a83a0fca3215d7fdee1a`, prepacked-weight
revision `21cac9478f4b02048af1de4c4c91787c5e9ebe05` and its packed-loader
Torch reference fixes. Prefill is the first optimization target; Decode
tuning follows separately.

The latest upstream fixes accept tuple Engram layer identifiers and remove
obsolete Indexer tensor-collector access. The migrated optimization patch is
unchanged by this rebase; no new device accuracy or latency results are claimed.

## Current prepacked baseline

The QKV and expert optimizations now live in the consolidated `gemm.py`
and `quant.py` primitives and their actual Prefill callers in `prefill_swa.py`
and `prefill_moe.py`. Fixed INT8
weights retain the new CPU-prepacked `[N/64,K,64]` ABI and generation-scoped
residency. The former per-call device packing optimization is obsolete and
is not restored. The fused 128-window SWA reader and widened HC tiles remain.
FP32 activation quantization, channel-scale rounding, BF16 output projection
and the existing Decode contracts are preserved.

The device measurements below belong to the original canonical-weight
migration on A3 revision `c8297f4d1f747c82a93528f5e35a81e4327789c1`.
They are historical evidence, not accuracy or speedup claims for this rebase.
Weight preparation, packing and upload must be outside new device timing;
dynamic activation quantization stays inside. QKV and expert measurements
that included per-call weight packing cannot serve as the new baseline.
Other measurements can be reused only after verifying the same input,
implementation, environment and measurement boundary.

Before the latest upstream fixes, the prepacked baseline passed compile-only
checks with PTOAS 0.72 for the actual
TP4 QKV composition, fused 128-window SWA, HC, prepacked shared FP32-channel
and routed FP16-channel experts, and the K=8192 blocked fallback. These six
checks generate code without executing device kernels. Device accuracy and
latency have not been rerun for the prepacked baseline.

## Scope

| Component | Adaptation |
| --- | --- |
| QKV | Whole-row BF16 input quantization, full-K input reuse, and QB column pipelining with prepacked weights. |
| SWA | Fused cache-run gather, QK, sink softmax and PV for the 128-token window. Publication and CP query counts remain independent. Standalone SWA uses the same reader after publication. |
| HC | Wider collapse/post tiles and reuse of residual streams across two output streams. Cube projection, weight rounding, high-precision RMS and all 20 Sinkhorn iterations remain. |
| Shared/routed expert | Reuse quantization and GEMM primitives with prepacked weights, retaining FP32 activation values and the respective FP32/FP16 channel boundaries. |
| Output projection | Retain the current BF16 grouped Wo-A and BF16 Wo-B. The legacy INT8 Wo-B optimization does not implement this contract. |

QKV retains combined-scale QA/QB dequantization and channel-first KV
dequantization, with the existing BF16 normalization and RoPE boundaries.
Whole-row quantization retains zero scales for zero rows. FP32 activation
quantization keeps its existing implementation. Wider K dimensions retain
blocked quantization/GEMM; the migrated full-K tiles cover K up to 5120.

The 640-token compressed-attention reader retains its sparse maximum scan
and BF16 probability rounding. No compressor, index scoring, TopK, routing,
communication, network schedule or new Decode performance optimization is
included. Shared HC callers use the updated common implementation; the
measurements below cover Prefill components.

## Historical shapes and measurement boundary

The current network configuration is DP4 / TP4 / EP16 with B=1, allocated
capacity 16 and chunks of at most 16 tokens, targeting a 2048-token context.
These are local component measurements for one chunk, not latency for an
entire 2048-token prompt or a 40-layer forward.

| Component | Measured geometry |
| --- | --- |
| QKV | Capacity 16; query rows 4, KV rows 16; full 64 query heads of dimension 512; hidden 5120, latent 1280. |
| SWA | Capacity 16, active 4; 64 heads, head dimension 512, window 128; prefix lengths 0 and 112; three cache pages. |
| Local TP4 output projection | Capacity/active 16/16; two local output groups; 8192 input channels, 2048 intermediate channels, 5120 output channels. |
| HC | Capacity/active 16/4; four residual streams of width 5120. |
| Shared expert | Capacity/active 16/4; hidden 5120, intermediate 2304; fused activation, FP32 channel boundary. |
| Routed expert | Capacity 64, active 1 or 4; one local expert, hidden 5120, intermediate 2304; fused activation, FP16 channel boundary. |

The QKV measurement composes the actual Prefill Q/KV projections,
normalization and RoPE. SWA measures reading an already published cache.
The local output projection excludes inverse RoPE and TP communication.
Routed-expert timing excludes router, dispatch, combination and EP communication.
HC timing includes coefficient generation, collapse and post expansion.

## Historical measurement contract

- Device: `a2a3`, reported SoC `Ascend910_9382`, physical device 0.
- Environment: conda `pypto`; PyPTO 0.1.0, simpler 0.1.0, PTOAS 0.72,
  CANN 9.1.0, Torch 2.10.0+cpu. Package versions alone do not establish
  compiler/runtime source revision pins.
- Metric: median device-domain `effective_us`, with `PYPTO_BENCH=1`,
  five warmup rounds and 100 measured rounds in one process per case.
- Per-call weight packing is included. Compilation, checkpoint preparation,
  host transfers, reference computation and validation are outside this metric.
- Inputs use real L0 checkpoint weights and deterministic synthetic
  activations/cache values. These are not captured vLLM fixtures.
- Each baseline freezes inputs and CPU Torch reference outputs with
  `save_data=True`. The migrated run replays the same snapshot, with both
  input and output cache hits and validation enabled.

The previously completed QKV/SWA baselines were retained on the user's
instruction: they ran on node 1, device 0. All migrated measurements and
the remaining baselines ran on node 0, device 0. Both report the same SoC
and software versions. The first three comparisons are therefore cross-host
references; the HC and expert comparisons are on the same host/device.
No baseline was rerun merely to collect more samples.

## Historical results

Speedup is baseline median divided by migrated median. All eight baseline
and eight migrated accuracy checks passed.

| Component | Baseline (us) | Migrated (us) | Ratio | Comparison |
| --- | ---: | ---: | ---: | --- |
| QKV, query 4 / KV 16 | 716.81 | 426.91 | 1.68x | Cross-host |
| SWA, prefix 0 | 115.04 | 23.26 | 4.95x | Cross-host |
| SWA, prefix 112 | 115.63 | 23.14 | 5.00x | Cross-host |
| BF16 local output projection | 359.22 | 372.24 | 0.97x | Unchanged control |
| HC, active 4 | 379.46 | 316.81 | 1.20x | Same host/device |
| Shared expert, active 4 | 262.36 | 206.28 | 1.27x | Same host/device |
| Routed expert, active 1 | 259.74 | 204.27 | 1.27x | Same host/device |
| Routed expert, active 4 | 261.83 | 204.96 | 1.28x | Same host/device |

The unchanged output projection is a control, not a migrated optimization;
its measured variation establishes no benefit. These ratios are independent
operator results and cannot be multiplied or summed into a network speedup.
CPU Torch is the accuracy reference, not the performance baseline. No vLLM
performance comparison was performed.

## Historical additional accuracy coverage

Seventeen additional device checks passed without benchmarking:

- QKV empty inputs, zero singleton input with a three-row KV tail, and a
  fifteen-row KV tail.
- Shared expert empty and zero singleton input; routed expert empty input
  and seventeen active rows crossing a GEMM tile boundary.
- Full-width HC at active counts 0, 1, 7 and 16 with nonuniform incoming
  stream coefficients, plus the smaller existing HC validation geometry.
- Full-geometry SWA with active counts 0, 1 and 7, all-masked rows,
  noncontiguous/cross-page indices and invalid inactive-row indices.
- The existing 640-window fallback and publication followed by the fused
  128-window reader, including exact cache-write validation.

Floating active outputs use the existing `rtol=0.05, atol=0.05` policy with
zero violating elements. Untouched output tails, published cache values and
all-masked SWA outputs use exact comparisons. Reference arithmetic and
acceptance thresholds were not changed to accommodate a migrated kernel.
These checks do not establish native independent-layer alignment or
40-layer network accuracy, generation quality, or Decode performance.

After measurement, the full-K optimization was bounded to K <= 5120.
All 15 generated QKV PTO modules match the measured build after removing
source-location metadata. A compile-only check also passed for the blocked
fallback at capacity 16, K=8192 and N=64; this is not a device accuracy or
performance result for that larger K.

## Local evidence

Generated artifacts remain untracked under
`build_output/a3-prefill-migration/`; no checkpoint or tensor snapshot is
included in the source change.

| Evidence | Relative artifact path |
| --- | --- |
| Benchmark driver | `bench.py` |
| Baseline builds, source receipts and all raw timing spans | `baseline/compile.json`, `baseline/results.json` |
| Migrated builds, source receipts and all raw timing spans | `migrated-final/compile.json`, `migrated-final/results.json` |
| Frozen input/reference data | Each baseline result's `work_dir/data/{in,out}` |
| Additional correctness driver and receipts | `checks.py`, `checks-compile.json`, `checks-results.json` |
| Prepacked rebase compile-only driver and receipt | `rebase_compile.py`, `rebase-compile.json` |

The benchmark driver skips completed cases and reuses compatible compiled
work directories. Correctness checks reuse the migrated runtime builds with
runtime active-row scalars. Raw samples are retained in the structured
benchmark results; no additional timing runs are needed to recover medians.
