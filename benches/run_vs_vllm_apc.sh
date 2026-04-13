#!/usr/bin/env bash
# Pie vs vLLM-with-APC Benchmark Runner
#
# Runs the same benchmark suite as run_vs_vllm.sh but against a vLLM server
# that was started with automatic prefix caching enabled. Results are tagged
# with a "apc_on" label suffix so they can be compared side-by-side with an
# APC-off run in the same JSON file.
#
# Prerequisites:
#   - Pie server running on ws://127.0.0.1:8080
#   - vLLM server on http://localhost:8000 started with:
#       vllm serve <model> --port 8000 --enable-prefix-caching
#   - wasm32-wasip2 target installed: rustup target add wasm32-wasip2
#
# Usage:
#   ./benches/run_vs_vllm_apc.sh                          # Full suite, apc_on tag
#   ./benches/run_vs_vllm_apc.sh --tiers 2a,2b            # Tier 2 only
#   ./benches/run_vs_vllm_apc.sh --label-suffix apc_off   # Override tag
#
# Recommended workflow for APC comparison:
#   1. Start vLLM WITHOUT --enable-prefix-caching.
#      Run: ./benches/run_vs_vllm.sh --label-suffix apc_off
#   2. Stop vLLM, restart WITH --enable-prefix-caching.
#      Run: ./benches/run_vs_vllm_apc.sh
#   3. Diff/merge the two JSON result files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERLETS_DIR="$SCRIPT_DIR/inferlets"
RESULTS_DIR="$SCRIPT_DIR/results/$(date +%Y-%m-%d_%H-%M-%S)_apc"

echo "=== Pie vs vLLM-with-APC Benchmark Suite ==="
echo ""
echo "IMPORTANT: this script assumes vLLM was started with"
echo "           --enable-prefix-caching. If it wasn't, the"
echo "           'apc_on' label will be misleading."
echo ""

# Default label; allow override via first matching --label-suffix arg.
LABEL="apc_on"
PASSTHROUGH_ARGS=()
SAW_LABEL=0
for arg in "$@"; do
    if [[ "$arg" == "--label-suffix" || "$arg" == --label-suffix=* ]]; then
        SAW_LABEL=1
    fi
    PASSTHROUGH_ARGS+=("$arg")
done

# Step 1: Build standard text-completion inferlet
echo "--- Building std/text-completion ---"
(cd "$REPO_ROOT/std/text-completion" && cargo build --target wasm32-wasip2 --release 2>&1) || {
    echo "Error: Failed to build std/text-completion"
    echo "Make sure wasm32-wasip2 target is installed: rustup target add wasm32-wasip2"
    exit 1
}
echo "  Done."

# Step 2: Build benchmark inferlets
echo "--- Building benchmark inferlets ---"
(cd "$INFERLETS_DIR" && cargo build --target wasm32-wasip2 --release 2>&1) || {
    echo "Error: Failed to build benchmark inferlets"
    exit 1
}
echo "  Done."
echo ""

# Step 3: Run benchmarks
echo "--- Running benchmarks ---"
mkdir -p "$RESULTS_DIR"

if [[ "$SAW_LABEL" -eq 0 ]]; then
    python3 "$SCRIPT_DIR/bench_vs_vllm.py" \
        --output-json "$RESULTS_DIR/vs_vllm_apc.json" \
        --label-suffix "$LABEL" \
        "${PASSTHROUGH_ARGS[@]}"
else
    python3 "$SCRIPT_DIR/bench_vs_vllm.py" \
        --output-json "$RESULTS_DIR/vs_vllm_apc.json" \
        "${PASSTHROUGH_ARGS[@]}"
fi

echo ""
echo "Results saved to: $RESULTS_DIR/vs_vllm_apc.json"
