#!/usr/bin/env bash
#
# Pie Accuracy Benchmark — end-to-end automation
#
# Runs the full eval pipeline on a fresh machine (e.g. RunPod):
#   1. Build inferlets (WASM)
#   2. Install Python deps (eval extras)
#   3. Configure and start Pie server
#   4. Optionally start vLLM for comparison
#   5. Run lm-eval-harness benchmarks
#   6. Shut down servers, print results
#
# Usage:
#   ./evals/run_benchmark.sh                              # defaults: Qwen3-0.6B, all datasets, pie only
#   ./evals/run_benchmark.sh --engines pie,vllm           # pie + vllm comparison
#   ./evals/run_benchmark.sh --engines vllm               # vllm only (no pie)
#   ./evals/run_benchmark.sh --engines pie,vllm,sglang    # all three
#   ./evals/run_benchmark.sh --model Qwen/Qwen3-8B       # different model
#   ./evals/run_benchmark.sh --datasets arc_easy,math500  # subset of datasets
#   ./evals/run_benchmark.sh --limit 20                   # quick smoke test
#   ./evals/run_benchmark.sh --device "cuda:0,cuda:1"     # multi-GPU tensor parallel
#   ./evals/run_benchmark.sh --backend custom              # use custom backend instead of harness
#
# Environment:
#   PIE_REPO     — repo root (auto-detected from script location)
#   VLLM_PORT    — vLLM port (default: 8000)
#   SGLANG_PORT  — SGLang port (default: 30000)
#   PIE_PORT     — Pie server port (default: 8080)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIE_REPO="${PIE_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PIE_DIR="$PIE_REPO/pie"

MODEL="Qwen/Qwen3-0.6B"
DATASETS=""
ENGINES="pie"
LIMIT=""
BACKEND="harness"
DEVICE="cuda:0"
VLLM_PORT="${VLLM_PORT:-8000}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
PIE_PORT="${PIE_PORT:-8080}"
VERBOSE=""
NO_THINK=""
SKIP_BUILD=false
SKIP_INSTALL=false
OUTPUT_JSON=""

# PIDs to clean up
PIE_PID=""
VLLM_PID=""
SGLANG_PID=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[pie-eval]${NC} $*"; }
ok()   { echo -e "${GREEN}[pie-eval]${NC} $*"; }
warn() { echo -e "${YELLOW}[pie-eval]${NC} $*"; }
die()  { echo -e "${RED}[pie-eval]${NC} $*" >&2; exit 1; }

cleanup() {
    log "Cleaning up..."
    if [[ -n "$PIE_PID" ]] && kill -0 "$PIE_PID" 2>/dev/null; then
        log "Stopping Pie server (PID $PIE_PID)"
        kill "$PIE_PID" 2>/dev/null || true
        wait "$PIE_PID" 2>/dev/null || true
    fi
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "Stopping vLLM server (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    if [[ -n "$SGLANG_PID" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
        log "Stopping SGLang server (PID $SGLANG_PID)"
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --model MODEL         HuggingFace model repo (default: $MODEL)
  --engines LIST        Comma-separated engines: pie,vllm,sglang (default: $ENGINES)
  --datasets LIST       Comma-separated datasets (default: all in config)
  --limit N             Max questions per dataset (quick test)
  --device DEVICE       GPU device(s), e.g. "cuda:0" or "cuda:0,cuda:1" (default: $DEVICE)
  --backend TYPE        harness (default) or custom
  --vllm-port PORT      vLLM port (default: $VLLM_PORT)
  --sglang-port PORT    SGLang port (default: $SGLANG_PORT)
  --pie-port PORT       Pie server port (default: $PIE_PORT)
  --output-json PATH    Save results JSON to this path
  --no-think            Disable thinking mode (Qwen3, etc.)
  --verbose             Verbose output
  --skip-build          Skip inferlet WASM builds (if already built)
  --skip-install        Skip Python dependency install
  -h, --help            Show this help
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --engines)      ENGINES="$2"; shift 2 ;;
        --datasets)     DATASETS="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --backend)      BACKEND="$2"; shift 2 ;;
        --vllm-port)    VLLM_PORT="$2"; shift 2 ;;
        --sglang-port)  SGLANG_PORT="$2"; shift 2 ;;
        --pie-port)     PIE_PORT="$2"; shift 2 ;;
        --output-json)  OUTPUT_JSON="$2"; shift 2 ;;
        --no-think)     NO_THINK="--no-think"; shift ;;
        --verbose|-v)   VERBOSE="-v"; shift ;;
        --skip-build)   SKIP_BUILD=true; shift ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        -h|--help)      usage ;;
        *)              die "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Parse engine list
# ---------------------------------------------------------------------------

IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"

has_engine() {
    local target="$1"
    for e in "${ENGINE_LIST[@]}"; do
        [[ "$e" == "$target" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

log "${BOLD}Pie Accuracy Benchmark${NC}"
log "  Repo:     $PIE_REPO"
log "  Model:    $MODEL"
log "  Device:   $DEVICE"
log "  Backend:  $BACKEND"
log "  Engines:  $ENGINES"
log "  Datasets: ${DATASETS:-all}"
log "  Limit:    ${LIMIT:-all}"
echo ""

[[ -d "$PIE_DIR" ]] || die "Pie directory not found at $PIE_DIR"

# Check prerequisites
command -v cargo &>/dev/null || die "cargo not found — install Rust via rustup"
command -v uv &>/dev/null    || die "uv not found — install from https://docs.astral.sh/uv/"

# Ensure WASM target is installed
if ! rustup target list --installed 2>/dev/null | grep -q wasm32-wasip2; then
    log "Installing wasm32-wasip2 target..."
    rustup target add wasm32-wasip2
fi

# ---------------------------------------------------------------------------
# Step 1: Build inferlets
# ---------------------------------------------------------------------------

if has_engine pie; then
    if [[ "$SKIP_BUILD" == false ]]; then
        log "${BOLD}Building inferlets...${NC}"

        log "  text-completion..."
        cargo build --target wasm32-wasip2 --release \
            --manifest-path "$PIE_REPO/std/text-completion/Cargo.toml" --quiet

        log "  loglikelihood..."
        cargo build --target wasm32-wasip2 --release \
            --manifest-path "$PIE_REPO/std/loglikelihood/Cargo.toml" --quiet

        ok "Inferlets built"
    else
        warn "Skipping inferlet builds (--skip-build)"
    fi

    # Verify WASM files exist
    TC_WASM="$PIE_REPO/std/text-completion/target/wasm32-wasip2/release/text_completion.wasm"
    LL_WASM="$PIE_REPO/std/loglikelihood/target/wasm32-wasip2/release/loglikelihood.wasm"
    [[ -f "$TC_WASM" ]] || die "text-completion WASM not found at $TC_WASM — run without --skip-build"
    [[ -f "$LL_WASM" ]] || die "loglikelihood WASM not found at $LL_WASM — run without --skip-build"
else
    log "Skipping inferlet builds (pie not in engine list)"
fi

# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies
# ---------------------------------------------------------------------------

if [[ "$SKIP_INSTALL" == false ]]; then
    log "${BOLD}Installing Python dependencies...${NC}"
    cd "$PIE_DIR"

    # Detect CUDA extra from nvidia-smi
    CUDA_EXTRA="cu126"
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true)
        log "  NVIDIA driver: ${CUDA_VERSION:-unknown}"
    fi

    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    uv sync --extra "$CUDA_EXTRA" --extra eval --quiet 2>&1 | tail -5 || true
    ok "Dependencies installed"
else
    warn "Skipping dependency install (--skip-install)"
    cd "$PIE_DIR"
fi

# ---------------------------------------------------------------------------
# Step 3: Download model (if not cached)
# ---------------------------------------------------------------------------

log "${BOLD}Ensuring model is available...${NC}"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
uv run python -c "
from huggingface_hub import try_to_load_from_cache, snapshot_download
import os
# Quick check: see if config.json is cached (proxy for 'model downloaded')
cached = try_to_load_from_cache('$MODEL', 'config.json')
if cached is None or cached is False:
    print('  Downloading $MODEL...')
    snapshot_download('$MODEL')
    print('  Done.')
else:
    print('  $MODEL already cached.')
"

MAX_RETRIES=120  # 2 minutes (model loading can be slow)

# ---------------------------------------------------------------------------
# Step 4: Start requested engines
# ---------------------------------------------------------------------------

# --- Pie ---
if has_engine pie; then
    log "${BOLD}Configuring Pie...${NC}"

    # Download Python runtime if not already present (needed by pie serve)
    if [[ ! -d "$HOME/.pie" ]] || [[ ! -d "$HOME/.pie/lib" ]]; then
        uv run pie config init
    fi

    # Generate a local Pie server config inside evals/ so everything is self-contained.
    PIE_SERVE_CONFIG="$PIE_REPO/evals/pie_serve_config.toml"

    # Build device array: "cuda:0,cuda:1" → ["cuda:0", "cuda:1"]
    IFS=',' read -ra DEV_ARRAY <<< "$DEVICE"
    DEVICE_LIST=""
    for d in "${DEV_ARRAY[@]}"; do
        [[ -n "$DEVICE_LIST" ]] && DEVICE_LIST+=", "
        DEVICE_LIST+="\"$d\""
    done

    cat > "$PIE_SERVE_CONFIG" <<TOML
# Auto-generated by run_benchmark.sh — do not edit manually
host = "127.0.0.1"
port = $PIE_PORT
enable_auth = false
python_snapshot = true

[[model]]
hf_repo = "$MODEL"
device = [$DEVICE_LIST]
tensor_parallel_size = ${#DEV_ARRAY[@]}
activation_dtype = "bfloat16"
weight_dtype = "bfloat16"
kv_page_size = 16
max_batch_tokens = 10240
max_dist_size = 64
max_num_embeds = 128
TOML

    log "  Config written to evals/pie_serve_config.toml"

    log "Starting Pie server..."
    uv run pie serve --config "$PIE_SERVE_CONFIG" &
    PIE_PID=$!

    log "  Waiting for Pie server (port $PIE_PORT)..."
    RETRIES=0
    while ! uv run python -c "
import asyncio
from pie_client import PieClient
async def check():
    c = PieClient('ws://127.0.0.1:$PIE_PORT')
    await c.connect()
    await c.close()
asyncio.run(check())
" 2>/dev/null; do
        RETRIES=$((RETRIES + 1))
        if [[ $RETRIES -ge $MAX_RETRIES ]]; then
            die "Pie server failed to start after ${MAX_RETRIES}s"
        fi
        if ! kill -0 "$PIE_PID" 2>/dev/null; then
            die "Pie server process died during startup"
        fi
        sleep 1
    done
    ok "Pie server ready (PID $PIE_PID)"
fi

# --- vLLM ---
if has_engine vllm; then
    log "${BOLD}Starting vLLM...${NC}"

    if ! python -m vllm.entrypoints.openai.api_server --help &>/dev/null 2>&1; then
        die "vLLM not found. Install it in a separate venv:
  python -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate && pip install vllm"
    fi

    vllm serve "$MODEL" --port "$VLLM_PORT" &
    VLLM_PID=$!

    log "  Waiting for vLLM (port $VLLM_PORT)..."
    RETRIES=0
    while ! curl -s "http://localhost:$VLLM_PORT/v1/models" &>/dev/null; do
        RETRIES=$((RETRIES + 1))
        if [[ $RETRIES -ge $MAX_RETRIES ]]; then
            die "vLLM failed to start after ${MAX_RETRIES}s"
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            die "vLLM process died during startup"
        fi
        sleep 1
    done
    ok "vLLM ready (PID $VLLM_PID)"
fi

# --- SGLang ---
if has_engine sglang; then
    log "${BOLD}Starting SGLang...${NC}"

    if ! python -m sglang.launch_server --help &>/dev/null 2>&1; then
        die "SGLang not found. Install it in a separate venv:
  python -m venv ~/.venvs/sglang && source ~/.venvs/sglang/bin/activate && pip install 'sglang[all]'"
    fi

    python -m sglang.launch_server --model "$MODEL" --port "$SGLANG_PORT" &
    SGLANG_PID=$!

    log "  Waiting for SGLang (port $SGLANG_PORT)..."
    RETRIES=0
    while ! curl -s "http://localhost:$SGLANG_PORT/v1/models" &>/dev/null; do
        RETRIES=$((RETRIES + 1))
        if [[ $RETRIES -ge $MAX_RETRIES ]]; then
            die "SGLang failed to start after ${MAX_RETRIES}s"
        fi
        if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
            die "SGLang process died during startup"
        fi
        sleep 1
    done
    ok "SGLang ready (PID $SGLANG_PID)"
fi

# ---------------------------------------------------------------------------
# Step 6: Run evaluations
# ---------------------------------------------------------------------------

log "${BOLD}Running evaluations...${NC}"
echo ""

# Generate a temp config with correct ports (eval_config.toml hardcodes defaults)
EVAL_CONFIG="$PIE_REPO/evals/eval_config.toml"
if [[ "$PIE_PORT" != "8080" ]] || [[ "$VLLM_PORT" != "8000" ]] || [[ "$SGLANG_PORT" != "30000" ]]; then
    EVAL_CONFIG=$(mktemp /tmp/pie_eval_config.XXXXXX.toml)
    sed -e "s|ws://127.0.0.1:8080|ws://127.0.0.1:$PIE_PORT|g" \
        -e "s|http://localhost:8000|http://localhost:$VLLM_PORT|g" \
        -e "s|http://localhost:30000|http://localhost:$SGLANG_PORT|g" \
        "$PIE_REPO/evals/eval_config.toml" > "$EVAL_CONFIG"
    log "  Using patched config (pie=$PIE_PORT, vllm=$VLLM_PORT, sglang=$SGLANG_PORT)"
fi

# Build the eval command
EVAL_CMD=(
    uv run python -m evals.run_evals
    --config "$EVAL_CONFIG"
    --backend "$BACKEND"
)

# Override engines in config to match what we started
EVAL_CMD+=(--engines "$ENGINES")

if [[ -n "$DATASETS" ]]; then
    EVAL_CMD+=(--datasets "$DATASETS")
fi

if [[ -n "$LIMIT" ]]; then
    EVAL_CMD+=(--limit "$LIMIT")
fi

if [[ -n "$VERBOSE" ]]; then
    EVAL_CMD+=("$VERBOSE")
fi

if [[ -n "$NO_THINK" ]]; then
    EVAL_CMD+=("$NO_THINK")
fi

if [[ -n "$OUTPUT_JSON" ]]; then
    EVAL_CMD+=(--output-json "$OUTPUT_JSON")
fi

# PYTHONPATH must include repo root so `evals` package is importable
export PYTHONPATH="$PIE_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

"${EVAL_CMD[@]}"
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    ok "${BOLD}Benchmark complete.${NC}"
else
    die "Benchmark failed (exit code $EXIT_CODE)"
fi
