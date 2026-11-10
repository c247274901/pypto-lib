# DeepSeek V4.1 Flash A3 Prefill

This A3 implementation contains real PyPTO operators and the runtime wiring
for the original 40-layer text backbone. The exported product snapshot is
`ebf19ad0ae7d668b4f7f2b04b526c9dcff2785a9`. Its historical source directory
was `models/deepseek_v4_1_flash_a2`; this delivery uses the separate
`models/deepseek_v4_1_flash_a3` package and actual A3 task sources.

## Development priorities

- Performance target: correctness of the A3 complete Prefill path, with
  Eco-Tech real W8A8, 384 routed experts, Top6, and original Engram state.
- Deferred performance work: final throughput and latency tuning until
  complete-network accuracy is independently established.
- Compatibility requirements: preserve original layer roles, byte aliases,
  cache/source lifetime, quantization, dtypes, and explicit DP/TP/EP mappings.
  TP1/TP2 fixed-16 full-network compatibility remains unvalidated.

## Evidence and scope

| Item | Evidence scope |
| --- | --- |
| Original A3 independent complete layers | 11/11: physical layers 0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25 |
| Independent-layer admitted source | `afef9572cfe9c518f4e60d5ea7b00368b7297fbb` |
| Early four layers | DP4 / TP4 / EP16, PP1 / PCP1 / DCP1 |
| Late seven layers | DP2 / TP2 / EP4, PP1 / PCP1 / DCP1 |
| Exported latest network implementation | Full original 40-layer composition, DP4 / TP4 / EP16, capacity 16 |
| Full-network accuracy and actual L39 output | Not established at export; original task validation was in progress |
| Fixed-16 TP1/TP2 full-network compatibility | Not established |
| Full real-weight and both Engram tables in HBM | Not established by code or host weight-byte counts |
| Dataset score / Decode / serving integration | Not established by this delivery |
| Relocated package device execution | Not run; HOST checks do not replace NPU validation |

Independent alignment uses the same complete-layer inputs and legal source/cache
state for product and reference. Floating tensors retain the original
`rtol=0.05, atol=0.05` criterion; discrete indices, controls, and untouched state
remain exact. An independent-layer pass does not imply the original 40-layer
chain passes. In particular, late-layer EP4 results do not establish EP16
full-network correctness.

`source_manifest.json` records the exported Git identity, every original file's
Git blob and SHA256, its identity at independent-layer admission, and normalized
definition hashes. It distinguishes later network changes from admitted files.
The original source names in this manifest are intentional provenance.

## Code structure and entry points

| Responsibility | Implementation |
| --- | --- |
| Complete layer | `prefill_layer.make_prefill_layer_entries`, `run_prefill_layer_inmemory` |
| Original 40-layer composition | `prefill_layer.run_prefill_network_inmemory` |
| Actual checkpoint, persistent state, HCCL and L39 head | `prefill_network_runtime.run_network_runtime` |
| Input embedding / HC and final output kernels | `prefill_embedding.py`, `prefill_network_input.py`, `prefill_output.py` |
| SWA and C2/C1 attention | `prefill_swa.py`, `prefill_full2*.py`, `prefill_consumer*.py`, `prefill_ratio1.py` |
| Persistent source/cache ownership | `prefill_network_state.py`, `prefill_network_cache.py`, `prefill_layer_state.py` |
| Two Engram slots and singleton history | `prefill_network_engram.py`, `prefill_engram_runtime.py`, `engram_storage.py` |
| EP MoE and actual communication | `prefill_moe_ep.py`, `moe_dispatch.py`, `moe_transport.py` |
| Shared primitives | `quant.py`, `ecotech_linear*.py`, `qkv*.py`, `hc.py`, `gemm.py` |

The public entry is `prefill_fwd.run_prefill`. It explicitly accepts inputs,
checkpoint weights, canonical cache state, numerical configuration, physical
device IDs and a new output directory. It has no dependency on historical
`formal_layer_receipts`, task releases, `TASK_DEVICE` or `DS41_JOB_ROOT`.
The runtime forwards those IDs in `assigned_devices` to both HCCL workers and
Engram preparation. Each uses the same explicit mapping in logical rank order;
duplicate IDs, rank-count mismatches and conflicting existing
`ASCEND_RT_VISIBLE_DEVICES` mappings are rejected before device initialization.
Historical release and continuation admission can be checked separately with
`audit_task.audit_task_release` and `audit_task.audit_task_continuation`.
Shape, dtype, topology, legal cache aliases and exact state preservation remain
runtime checks. Input/cache identity hashes describe the supplied operands;
they are not credentials from a private task.

```python
from models.deepseek_v4_1_flash_a3.prefill_fwd import run_prefill

result = run_prefill(
    inputs, weights=checkpoint, cache=cache_state, config=network_contract,
    devices=list(range(16)), output="build_output/ds41-prefill-new-run",
)
```

For the CLI, copy `models/deepseek_v4_1_flash_a3/prefill.example.json`, supply
actual files, and run from the repository root:

```bash
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json --check-config
python -m models.deepseek_v4_1_flash_a3.prefill_fwd --config prefill.json -p a2a3
```

Paths in the JSON are relative to that JSON file. `--checkpoint`, `--devices`
and `--output` override the corresponding fields. `--check-config` checks the
configuration and mapping only; it does not load weights, prove input/cache
compatibility, compile or execute. Output directories must be new.

The standalone `prefill_moe_ep` driver also requires `--devices`, for example
`--devices 3,1` for a matching EP2 contract.
It writes that ordered mapping to `runtime_config.json` for its HCCL workers.

`inputs` and `cache_state` are Torch `weights_only=True` dictionaries. The first
contains `input_contract`, `product_input_paths` and `product_input_files`.
The second contains `cache_ABI`, `cache_ABI_sources` and `engram_contracts`
with integer Engram slots 1 and 14. Their canonical rank/page metadata and
physical aliases are interpreted by `NetworkCacheProvider`; metadata must
match the actual allocated tensors. The JSON `contract` supplies `topology`,
`model_config`, `required_modes`, `capacity`, `scheduler_capacity`,
`fresh_shared_initialization`, `input_contract_sha256` and `state_regions`
(path and SHA256). These remain explicit state/operand contracts. This entry
accepts an already prepared canonical backbone/cache boundary; it does not
create that boundary from only a checkpoint and a text prompt.

The current runtime supports the original 40-layer plan with DP4/TP4/EP16,
PP1/PCP1/DCP1 and capacity 16. Device IDs are runtime devices, not a claim of
16 separate physical cards. Multi-chunk continuation additionally retains
exact own-state identities, metadata lineage and unchanged numerical settings.
Dataset accuracy remains a separate check after execution.

### Continuous Prefill chunks

Pass one prepared continuation mapping, or an iterable of mappings, through
the existing `continuation` argument:

```python
result = run_prefill(
    first_inputs, weights=checkpoint, cache=cache_state, config=first_contract,
    devices=devices, output="build_output/ds41-prefill-chunks",
    continuation=iter([chunk1, chunk2, chunk3]),
)
```

Each continuation has `inputs`, `contract`, and `metadata_sources`. `inputs`
uses the same prepared backbone input boundary as the first chunk. Its input
contract must have the next integer `epoch` (1, 2, 3, ...) and a
`predecessor_input_sha256` matching the immediately preceding completed chunk.
The numerical configuration, model and topology stay fixed. The optional
checkpoint override applies to the entire stream.

`metadata_sources` is an ordered list of 16 `{rank, path, sha256}` descriptors
for the next chunk's original-builder metadata. Every rank, including empty
ranks, remains represented. The runner checks the input identity and each
metadata file before admission. Existing cache/provider checks enforce request
history, page ownership and completion of all 40 layers before advancing.
The same weight provider, raw KV backing, shared index state, Engram history
and retained Engram tables span the entire stream. An execution failure stops
iteration; later chunks are not loaded. This does not change the weight-loading
policy or establish that all weights reside in HBM.

The iterable can load or prepare its next item lazily after the preceding
chunk returns. First-chunk outputs retain their original location; subsequent
outputs are in `continuation-chunk1`, `continuation-chunk2`, and so on, each
with `own-state-continuity.json`. The returned result describes the final
chunk and includes `completed_prefill_chunks` when a continuation completed.
The original single-chunk and one-mapping/two-chunk calls remain supported.

For the CLI, add prepared continuation descriptors to the main configuration:

```json
{
  "continuations": [
    {
      "inputs": "inputs/chunk1-inputs.pt",
      "contract": "inputs/chunk1-contract.json",
      "metadata_sources": "inputs/chunk1-metadata-sources.json"
    },
    {
      "inputs": "inputs/chunk2-inputs.pt",
      "contract": "inputs/chunk2-contract.json",
      "metadata_sources": "inputs/chunk2-metadata-sources.json"
    }
  ]
}
```

These descriptor paths are relative to the main configuration. Each
`metadata_sources` JSON file contains the ordered rank descriptors above;
their `path` fields are relative to that descriptor file. Paths embedded in
prepared Torch dictionaries and numerical/input contracts retain their
explicit runtime meaning and must resolve from the execution directory.
`--check-config` checks descriptor structure and path resolution without
opening continuation tensors or initializing devices.

Each chunk still has at most 16 query tokens per DP group. The existing QLI
path still supports at most 512 cached index rows per request (therefore at
most 512 original positions for ratio1 layers). Streaming does not remove
that bound, construct metadata from raw text, implement request eviction, or
add an autoregressive generation driver. A 2048-token prompt still requires
the long-context Indexer implementation. These controls are independent of
subsequent L2 composition and resident-weight changes.

The HOST validator exercises four chunks with substituted device execution,
including lazy loading, persistent state ownership, checkpoint overrides,
the original two-chunk call, and rejection of stale lineage, skipped epochs,
changed configuration, invalid rank metadata and failed execution. These are
scheduler/control checks; they do not establish NPU numerical correctness or
normal text generation.

TP is passed to builders explicitly, for example
`make_prefill_layer_entries(..., ep=16, tp=4)` or
`make_prefill_grouped_output(tp=2)`. Importing a module does not inspect
`sys.argv` or specialize the process globally. TP1/TP2 factories do not
establish TP1/TP2 full-network acceptance.

Prefill-specific files have a `prefill_` prefix; shared primitives retain general
names. Fixed-layer CLI wrappers for L1/L3 and misleading `decode_*` facades
were removed. Use `prefill_layer --layer-id ...`; L0/L2 retain their actual
computation/reference definitions without duplicate CLI wrappers. Explicit
`*_reference` APIs resolve directly to their owning operator modules.

## Dependencies and execution

Use the repository's PyPTO installation and `golden` harness, with compatible
Torch, NumPy, safetensors, tokenizers, and the PyPTO runtime. Actual A3 execution
also requires the installed CANN and HCCL shared libraries and an explicit
healthy device allocation. The Python transport calls CANN/HCCL through ctypes;
there is no omitted custom product `.cpp` or `.h` implementation. Compiler
outputs such as generated C++, PTO, and shared objects are build artifacts.

Native/reference helper modules additionally use Torch-NPU, vLLM-Ascend, and
custom native operators when explicitly invoked. They are not required by the
device-free checks and do not substitute for the A3 PyPTO network arithmetic.
`runtime.example.json` describes only this retained native-helper configuration.

Run bounded host controls and operator checks from the repository root:

```bash
python -B -m models.deepseek_v4_1_flash_a3.validate
python -m models.deepseek_v4_1_flash_a3.hc -p a2a3 --cpu-only
python -m models.deepseek_v4_1_flash_a3.swa -p a2a3 --cpu-only
python -m models.deepseek_v4_1_flash_a3.indexer_select -p a2a3 --cpu-only
python -m models.deepseek_v4_1_flash_a3.prefill_embedding -p a2a3 -d 0
```

`validate` uses only the standard library. It checks the maintained import DAG,
public exports, all 40 roles, scheduler/EP ownership, TP import independence and
device mappings. It deliberately does not require historical AST equality.
`audit_source` performs the separate opt-in source identity audit.

Executable operator entries accept `-p/--platform a2a3`, `-d/--device`,
`--sim` and `--compile-only`. HC, SWA and QLI additionally support `--cpu-only`;
that mode checks Torch reference fixtures without running kernels and cannot
be combined with `--compile-only`. Default device runs use `golden.run` with
the same compute factories as model composition. HC/SWA compare floating
outputs at `rtol=0.05, atol=0.05`, and untouched state exactly. SWA also checks
the entire updated cache exactly. QLI checks active discrete indices exactly;
its inactive pure-`Out` rows are undefined and are not compared. The QLI golden
models both FP16 rounding boundaries and the fixed 512-candidate score order;
`indexer_select_reference` remains the separate ideal logical visibility oracle.

CI owns this directory on `a2a3` and resolves relative imports for downstream
selection. Drivers needing external weights, cache state or whole-group device
allocation use `# ci: manual`, keeping their dependencies visible while excluding
them from automatic bounded execution.

HOST validation passed 19 CI-selection tests, all 114 module imports and 84
public exports, 20 CPU golden cases, and TP4/TP1/TP2/TP4 factory construction.
All 20 HC/SWA/QLI cases and eight grouped-output/sparse TP specializations
compiled with PTOAS 0.70. Simulator dispatch was checked
to resolve to `a2a3sim`, but simulation could not execute because the validation
host lacks `g++-15`. Device accuracy, full simulator CI and full-network execution
of the refactored package remain unverified; earlier A3 acceptance does not
certify this refactor. An AST comparison preserved all 200 original JIT definitions;
this structural check does not replace numerical regression.
No original task source, Goal or device job was changed for these checks.
See [migration notes](prefill_migration.md) for provenance and helper ownership.
