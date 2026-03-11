#!/usr/bin/env bash
# Pie vs SGLang Benchmark Runner
#
# Usage:
#   ./benches/run_vs_sglang.sh                          # Full suite
#   ./benches/run_vs_sglang.sh --tiers 1a,1b            # Tier 1 only
#   ./benches/run_vs_sglang.sh --pie-only               # Skip SGLang
#   ./benches/run_vs_sglang.sh --sglang-only            # Skip Pie
#
# Prerequisites:
#   - Pie server running on ws://127.0.0.1:8080
#   - SGLang server running on http://localhost:30000
#   - wasm32-wasip2 target installed: rustup target add wasm32-wasip2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFERLETS_DIR="$SCRIPT_DIR/inferlets"
RESULTS_DIR="$SCRIPT_DIR/results/$(date +%Y-%m-%d_%H-%M-%S)"

echo "=== Pie vs SGLang Benchmark Suite ==="
echo ""

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

python3 "$SCRIPT_DIR/bench_vs_sglang.py" \
    --output-json "$RESULTS_DIR/vs_sglang.json" \
    "$@"

echo ""
echo "Results saved to: $RESULTS_DIR/vs_sglang.json"
