# DeepSeek V4.1 Flash A3 Prefill and Decode

A3 Eco-Tech W8A8 operators, the original 40-layer Prefill composition and
persistent runtime, and bounded Decode operators. See [Decode](decode.md)
for the implemented Decode scope and remaining work.
See [Prefill optimization migration](optimization-migration.md) for the A2-to-A3
operator adaptations, historical measurements and current prepacked baseline.
See [Prefill operator results](prefill-operator-results.md) for the combined
operator changes, accuracy coverage and measurements against the original A3.

## Development priorities

- Performance target: complete-network correctness and continuous generation
  with generation-scoped workers, compute weights, caches and communication,
  retaining Eco-Tech real W8A8, 384 routed experts, Top6 and original Engram state.
- Deferred performance work: fused device layer scheduling and final throughput
  tuning. Residency is an active target; elapsed-time improvements require
  measurements on an allocated device group.
- Compatibility requirements: preserve original layer roles, byte aliases,
  cache/source lifetime, quantization, dtypes, and explicit DP/TP/EP mappings.
  TP1/TP2 fixed-16 full-network compatibility remains unvalidated.

## Structure

| Responsibility | Owner |
| --- | --- |
| Whole-network Prefill, compilation reuse and runtime | `prefill_fwd.py` |
| Generic layer composition and L2 bindings | `prefill_layer.py`, `prefill_layer_plan.json` |
| Bounded SWA Decode | `decode_fwd.py` |
| Vocabulary embedding, initial HC and output head | `input_pack.py`, `prefill_fwd.py`, `lm_head.py` |
| Attention, compression and Indexer | `swa.py`, `sparse_attention.py`, `compressor.py`, `indexer.py`, phase compositions |
| Phase MoE routing, dispatch and shared experts | `prefill_moe.py`, `decode_moe.py` |
| Grouped routed-expert computation | `expert_routed.py` |
| Linear, quantization, Norm, RoPE and HC | `gemm.py`, `quant.py`, `rmsnorm.py`, `rope.py`, `hc.py` |
| Checkpoint operands and phase layouts | `weights.py` |
| Cache ownership and device lifetime | `cache.py`, `state.py`, `prefill_resident.py` |
| CANN/HCCL communication | `moe_transport.py` |
| Host Engram tables and row preparation | `engram_host.py` |
| Device Engram projection and gating | `engram.py`, `prefill_network_engram.py` |

The [ownership map](../../../models/deepseek_v4_1_flash_a3/MODULE_ALIGNMENT.md)
describes shared primitives and phase-specific numerical variants.
Historical audits, independent-layer runners, reference-capture workers,
state replay, tensor collectors and identity-lock manifests are removed.
Their source remains in Git history. Operators retain their local Torch
goldens and standard Golden Harness entries.

MoE phase files own their kernels, Torch goldens and validation entries.
Shared GEMM and quantization primitives remain in independent child modules;
the phase modules do not import each other. Prefill controlled routing and
FP16/FP32 fused-channel variants remain distinct from Decode FP32 router
inputs and activation boundaries. Standalone validation uses the same kernels
as composition:

```bash
python -m models.deepseek_v4_1_flash_a3.prefill_moe -p a2a3 --compile-only
python -m models.deepseek_v4_1_flash_a3.decode_moe -p a2a3 --compile-only
```

These commands compile bounded operator cases; they do not execute a network.
Omit `--compile-only` on an allocated A3 device to run operator comparisons.

## Whole-network Prefill

`prefill_fwd.prefill_fwd` executes embedding, initial HC, layers 0 through 39,
final HC collapse, RMSNorm and the vocabulary head. Each layer consumes the
previous layer's actual activations. The implemented topology is
DP4/TP4/EP16, PP1/PCP1/DCP1, with capacity 16. Physical devices are explicit
and ordered by logical rank. Duplicate IDs, rank-count mismatches and a
conflicting `ASCEND_RT_VISIBLE_DEVICES` mapping are rejected before initialization.

```python
from models.deepseek_v4_1_flash_a3.prefill_fwd import compile_prefill, prefill_fwd

program = compile_prefill(
    inputs, cache=cache_state, config=network_contract,
    output="build_output/ds41-compiled", weights=checkpoint,
)
result = prefill_fwd(
    inputs, weights=checkpoint, cache=cache_state, config=network_contract,
    devices=list(range(16)), compiled=program, output="build_output/ds41-run",
)
logits = result["logits"][0]["fresh"]
```

For automatic inputs, `inputs` is raw completion text or 1..2048 token IDs.
The entry tokenizes with the checkpoint tokenizer, reserves pages for the
complete prompt and processes chunks of up to 16 tokens. `chunk_size` can
select 1..16 and `dp_rank` can select 0..3. Other DP engines remain empty.
Positions, sequence lengths, page tables, slot mappings, C2 cross-chunk state
and Engram history persist across chunks. Indexer searches up to 2048
positions, keeps TopK 512 and excludes future positions. This automatic path
handles one request; it is not a multi-request serving scheduler.

`cache_state` supplies canonical cache-layout and both Engram templates.
A checkpoint and prompt alone do not define cache aliases or source ownership.
The cache dictionary contains `cache_ABI`, `cache_ABI_sources` and
`engram_contracts` with integer slots 1 and 14. The network contract contains
`topology`, `model_config`, `required_modes`, `capacity`, `scheduler_capacity`,
`fresh_shared_initialization`, `input_contract_path` and
`input_contract_sha256`. Prepared dictionary inputs contain `input_contract`.
These are request/storage interfaces, without private task admission receipts.
Embedding always runs in the forward; captured backbone activation files
are no longer accepted as an alternative input path.

### Compilation and execution

`compile_prefill` specializes canonical metadata with storage-free meta tensors.
It compiles all stages without allocating full caches, initializing CANN or
executing kernels. Automatic inputs require `weights=checkpoint` for checkpoint
metadata. A `PrefillProgram` is reusable within the same process; structural
equality, argument names, shapes, constants and cache geometry must match.
`compilation.json` records unique entries, stage bindings and compiler time.
Omitting `compiled` compiles inside the forward.

| L2 region | Existing computation |
| --- | --- |
| Input | Embedding summation and initial SP/HC streams |
| QKV | Q/KV projection, normalization and RoPE |
| Attention packing | Inverse RoPE and head packet packing |
| Output projection | Head unpack, grouped projection and reduction packing |
| FFN dispatch | HC, normalization, router, shared expert and EP packing |
| MoE finish | Combine, valid-row masking and FFN HC post |
| Head | SP unpack, RMSNorm and vocabulary projection |

L2 denotes PyPTO orchestration, not physical layer 2. The network retains host
scheduling and HCCL boundaries; it is not a single fused 40-layer device graph.
No loading, compilation or token-latency guarantee follows from this organization.

Execution uses persistent ChipWorkers, preloaded compute weights, cache
allocations and HCCL groups for the complete stream. Input/output staging
workers keep registration slots separate from resident expert operands.
The streaming-weight diagnostic alternative and task-directory compiler reuse
manifests are removed. Runtime dtype, shape, rank, alias, write-direction,
source-version and lifecycle checks remain in force.

### Fixed INT8 weight layout

Initialization packs QKV, Indexer query and Prefill shared/routed expert
weights on the CPU before uploading their final compute operands. Decode
shared-expert and packed-QKV builders accept the same blocked operands.
For logical `[K, N]` INT8 weights, the layout is `[N/64, K, 64]`, flattened
to `[K*(N/64), 64]`. `utils.pack_blocked_weight` performs only a byte reorder.
`prepare_ecotech_dynamic_weight` and the expert loaders return this packed
shape; callers must not pass the original `[K, N]` matrix to these builders.
QKV channel scales retain the existing BF16-to-FP32 conversion, while expert
loaders retain their original FP32 scales. Tail padding and scale multiplication
order remain unchanged.

The QKV and shared-expert Torch reference interfaces accept these loader
outputs directly and unpack only for reference computation. The QKV references
also accept explicit `weight_layout="kn"` for the bounded Decode direct-tile
entry. Shared-expert references restore the logical intermediate extent from
the packed down operand without changing activation rounding. Routed-expert
references continue to take the documented output-major checkpoint tensors;
their separate NZ or blocked device operands are not interchangeable.

`ResidentNetworkWeights.preload` prepares these operands once per generation.
Layer and chunk calls reuse the pinned projection tensors and each EP owner's
resident expert handles, without repacking or uploading the fixed weights.
Activation quantization and routed-token dispatch packing still run per call.
Decode routed experts retain their separate FRACTAL_NZ format, including down
projection padding. The bounded `decode_fwd` direct-tile GEMMs retain logical
`[K, N]` weights and do not perform a whole-weight pack. A5 FP8/FP4 layouts are
not used by these A3 W8A8 kernels.

Changing this operand ABI requires fresh compilation and reloading weights;
compiled programs created for the former canonical-weight ABI are incompatible.
Device accuracy and latency require separate A3 validation; CPU byte checks
and compilation alone do not establish a speedup.

### Engram and continuation

Engram lookup runs on the host before the prepared forward call. `HostEngram`
owns both complete CPU INT8/FP32 tables, hashes request history, gathers selected
entries and applies group32 dequantization with BF16 rounding. Device kernels
consume only rank-local `[capacity, 6144]` BF16 rows for projection and gating.
The NPU does not dereference the host table. Tables are not registered with
CANN or counted as full HBM model residency.

Automatic inputs and the CLI prepare these rows. For prepared Python inputs:

```python
from models.deepseek_v4_1_flash_a3.engram_host import HostEngram

with HostEngram(checkpoint, cache_state["engram_contracts"], "generation") as host:
    rows = host.lookup(inputs, cache=cache_state, config=network_contract)
    result = prefill_fwd(
        inputs, weights=checkpoint, cache=cache_state, config=network_contract,
        devices=list(range(16)), output="build_output/ds41-host-rows",
        engram_lookup=rows,
    )
```

Each prepared continuation supplies fresh `inputs["engram_lookup"]`, its next
epoch and predecessor input identity, and ordered metadata sources for all
ranks. The cache provider retains allocations and publishes layer/source
versions. Failed or completed chunks cannot be replayed in place.

## Commands

Copy [the configuration example](../../../models/deepseek_v4_1_flash_a3/prefill.example.json)
and supply checkpoint, request/cache templates, contract, devices and a new
output directory. JSON paths resolve relative to the configuration file.
For automatic inputs, replace `inputs` with `prompt` or `token_ids` and
optionally add `chunk_size` and `dp_rank`.

```bash
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --check-config
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --compile-only
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json -p a2a3
```

`--checkpoint`, `--devices` and `--output` override the JSON settings.
`--check-config` checks only configuration and device mapping. Compile-only
does not execute NPU work. Output includes final logits, compilation accounting,
execution progress and failures, without per-layer reference-comparison captures.

`generate.py` provides greedy feedback through the same complete Prefill path:

```bash
python -m models.deepseek_v4_1_flash_a3.generate \
  --checkpoint weights/deepseek-v4.1-flash-w8a8 \
  --assets inputs/generation-templates --prompt inputs/prompt.json \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --output build_output/ds41-generation --result build_output/ds41-answer
```

Assets contain `input-contract.json`, `cache-ABI.pt` and `engram-contracts.pt`.
The prompt JSON supplies messages, thinking mode, rendered text, token IDs,
EOS and checkpoint/tokenizer identities. Generation verifies these identities,
feeds each actual last-row greedy token into the next chunk and stops at EOS
or its 256-token bound. It runs a complete Prefill forward per generated token;
it does not use the bounded Decode kernel as an integrated Decode network.

## Correctness scope

Operator arithmetic, quantization, rounding and cache layouts remain unchanged.
Operator-local checks use the [Golden Harness](../../run-and-validate/golden-harness.md).
CPU goldens, imports and IR specialization do not establish NPU accuracy.
Complete-network accuracy, dataset scores, full-model HBM residency and
performance still require device evidence. A complete EP16 Decode network,
Prefill-to-Decode state handoff and serving scheduler remain unimplemented.
