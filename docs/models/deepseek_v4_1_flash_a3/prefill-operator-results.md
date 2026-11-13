# A3 Prefill operator optimization

The reporting baseline is A3 revision
`60783e6ae0da024663716ba3678ce9523c54afb9`, before optimization migration.
That revision already supports resident CPU-prepacked weights. These results
compare the original implementation with the combined operator changes;
the historical canonical-weight results in
[optimization migration](optimization-migration.md) use a different boundary.

## Measurement boundary

Measurements use an A3 `a2a3` device, PyPTO/simpler 0.1.0 from the paired
`5ab1ff12` build, PTOAS 0.72, CANN 9.1.0 and Torch 2.10.0+cpu.
Each case uses `PYPTO_BENCH=1`, five discarded warmup rounds, 100 measured
rounds and the median device-domain `effective_us`. Baseline and final
measurements use the same physical device, input, CPU Torch reference and
comparison thresholds. Baseline receipts are reused where the migration
did not change the invoked implementation.

Fixed weight loading, preparation, packing and upload are outside timing.
Dynamic activation quantization, intermediate scratch and the measured
operator computation remain inside. CPU Torch supplies accuracy references;
it is not the performance baseline. No vLLM timing is used.

These are local operator or composed-operator measurements for one chunk.
They exclude distributed communication and do not measure a complete
2048-token prompt, a layer, text generation or Decode.

## Combined results

Values are microseconds. Attention rows average three case medians;
Compressor averages eight medians. Indexer sums six different workload
medians; its total is not a single invocation's latency.

| Entry | Original | Combined final | Speedup |
| --- | ---: | ---: | ---: |
| QKV, query 4 / KV 16 | 665.8000 | 109.9300 | 6.0566x |
| Shared expert, active 4 | 210.5495 | 65.5900 | 3.2101x |
| Routed expert, active 1 | 210.9295 | 63.4795 | 3.3228x |
| Routed expert, active 4 | 214.3695 | 65.6600 | 3.2648x |
| SWA128 | 116.7767 | 22.7730 | 5.1279x |
| Sparse640, ratio1 | 504.5868 | 50.9932 | 9.8952x |
| Sparse640, ratio2 | 533.3163 | 49.8300 | 10.7027x |
| Indexer, six-case sum | 1451.7485 | 404.9085 | 3.5854x |
| Compressor, eight-case mean | 105.5997 | 68.2639 | 1.5469x |
| BF16 local output projection | 358.1100 | 62.9195 | 5.6916x |
| HC, active 4 | 385.0195 | 37.6795 | 10.2183x |

The combined Compressor result includes the optimized shared BF16
projection. Its isolated-task result was 86.4900 us; combining the shared
projection reduces it to 68.2639 us. Independent task improvements must
not be added together or multiplied into an estimated network speedup.

## Workloads and changes

The model uses B=1, DP4/TP4/EP16, capacity 16, chunks up to 16 tokens and
context reservation up to 2048. Each measurement runs the corresponding
local calculation on one device.

| Component | Workload and implementation |
| --- | --- |
| QKV | Hidden 5120, latent 1280, 64 heads of width 512; query active4 and KV active16. Reuse full-K INT8 operands, fuse projection scaling and batch adjacent RoPE heads. |
| Shared/routed experts | Hidden5120, intermediate2304; shared capacity16/active4, routed capacity64/active1 or4. Fuse activation quantization and retain FP32/FP16 channel boundaries. |
| Attention | Active4, 64x512; SWA128 and distinct-plane sparse640, first/continuous/context-tail cases. Batch cache gathers, reuse sparse scores and widen QK/PV work. |
| Compressor | Active16, hidden5120, compressed width512, Index-K128; ratio1/2 first/continuous/odd/wrap. Partition FP32 projection accumulation and reuse predecessor loads while preserving ring/cache writes. |
| Indexer | Active4, 32x128; key capacities512/1024/2048 and TopK512. Fuse query projection scaling, batch RoPE/gather and reduce scoring/sort overhead. |
| Output projection | Capacity/active16/16; grouped BF16 8192->2048->5120. Pipeline Cube slabs and fuse BF16 stores while retaining both rounding boundaries. |
| HC | Active4, four streams of width5120; share FP32 activation conversion, partition FP32 Cube projection and keep coefficients local. All24 coefficients, high-precision RMS, exact weight rounding and20 Sinkhorn iterations remain intact. |

## Accuracy and replay

All six groups passed combined-source regression: Compressor28
cases, W8A8 17, Attention24, Indexer48, and all six output-projection
boundary cases plus its existing projection and grouped TP1/TP2 gates.
The shared projection's Engram observed-readonly composition also passed.
HC passed22 full/caller checks on the combined source. Its independently
accepted source also passed11 additional output-head, compressed and C2A
caller checks.

Checks retain CPU references and existing thresholds, including exact
inactive tails, untouched cache/state, read-only inputs, causal visibility
and Indexer's exact Top-512 IDs. Input/reference snapshots, source hashes,
raw timing samples and queue task receipts remain with the individual
optimization and combined-regression task artifacts.

The complete Indexer validation entry is included in the model:

```bash
PYPTO_BENCH=1 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_RAW=1 \
  python -m models.deepseek_v4_1_flash_a3.prefill_indexer_validation \
  --checkpoint /path/to/DeepSeek-V4.1-Flash-w8a8 --formal --existing-gates -d 0
```

Run NPU commands through the site's device allocator or task queue.
Operator defaults provide bounded correctness checks; reproducing the
target-shape timing rows requires the corresponding frozen task contracts
and snapshots. A compile-only run does not establish device accuracy or
performance. Full-network numerical alignment remains a separate task.
