#!/usr/bin/env bash
set -eo pipefail
attention_source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PYPTO_ENV_SCRIPT:?PYPTO_ENV_SCRIPT must identify the activation script}" "$attention_source_root"
cd "$attention_source_root"
attention_tmp="${GOAL_PLUS_VERIFIER_TMPDIR:-$attention_source_root/.tmp/attention-verifier}"
mkdir -p "$attention_tmp"
export PYPTO_PROG_BUILD_DIR="$attention_tmp/pypto-build"
export TMPDIR="$attention_tmp"
export PYPTO_BENCH=1 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_RAW=1
export PYPTO_GOLDEN_NUM_THREADS=16
exec python .verification/attention/verify.py "$@"
