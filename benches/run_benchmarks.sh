#!/usr/bin/env bash
#
# Run all Pie benchmarks and collect results.
#
# Usage:
#   ./benches/run_benchmarks.sh                          # Run all benchmarks
#   ./benches/run_benchmarks.sh --server ws://host:port  # Custom server
#   ./benches/run_benchmarks.sh --only determinism       # Run one benchmark
#   ./benches/run_benchmarks.sh --quick                  # Fast subset
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="${SERVER:-ws://127.0.0.1:8080}"
RESULTS_DIR="${SCRIPT_DIR}/results/$(date +%Y%m%d_%H%M%S)"
ONLY=""
QUICK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server) SERVER="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Pie Benchmark Suite"
echo "============================================"
echo "Server:  $SERVER"
echo "Results: $RESULTS_DIR"
echo ""

# Track pass/fail
TOTAL=0
PASSED=0
FAILED=0
FAILED_NAMES=""

run_bench() {
    local name="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")

    # Skip if --only is set and doesn't match
    if [[ -n "$ONLY" && "$name" != "$ONLY" ]]; then
        return 0
    fi

    TOTAL=$((TOTAL + 1))
    echo "--- [$name] ---"

    # Build command args — skip --output-json for scripts that don't support it
    local cmd_args=(--server "$SERVER")
    if python "$SCRIPT_DIR/$script" --help 2>&1 | grep -q -- '--output-json'; then
        cmd_args+=(--output-json "$RESULTS_DIR/${name}.json")
    fi
    cmd_args+=("${extra_args[@]}")

    if python "$SCRIPT_DIR/$script" "${cmd_args[@]}" 2>&1; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_NAMES="$FAILED_NAMES $name"
    fi
    echo ""
}

# ---- Correctness benchmarks ----

run_bench "max_tokens" "bench_max_tokens.py"

run_bench "determinism" "bench_determinism.py" \
    --runs 5

run_bench "batch_position" "bench_batch_position.py"

# ---- Performance benchmarks ----

run_bench "cold_warm_start" "bench_cold_warm_start.py" \
    --warm-runs 10

if [[ "$QUICK" == "false" ]]; then
    run_bench "concurrent_scaling" "bench_concurrent_scaling.py" \
        --max-concurrency 64 --requests-per-level 16

    run_bench "long_context" "bench_long_context.py" \
        --max-context 4096
else
    run_bench "concurrent_scaling" "bench_concurrent_scaling.py" \
        --max-concurrency 8 --requests-per-level 8

    run_bench "long_context" "bench_long_context.py" \
        --max-context 1024
fi

# ---- Resilience benchmarks ----

run_bench "client_disconnect" "bench_client_disconnect.py"

if [[ "$QUICK" == "false" ]]; then
    run_bench "stress" "bench_stress.py" \
        --max-instances 128
else
    run_bench "stress" "bench_stress.py" \
        --max-instances 16
fi

# ---- Existing throughput benchmark ----

run_bench "throughput" "tput.py" \
    --num-requests 32 --concurrency 16 --max-tokens 50

# ---- Summary ----

echo "============================================"
echo "  SUMMARY"
echo "============================================"
echo "Total:  $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
if [[ -n "$FAILED_NAMES" ]]; then
    echo "Failed benchmarks:$FAILED_NAMES"
fi
echo "Results: $RESULTS_DIR"
echo "============================================"

exit $FAILED
