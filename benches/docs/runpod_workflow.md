# RunPod Benchmark Workflow

End-to-end guide: building the Docker image, deploying to RunPod, and running
Pie vs vLLM vs SGLang benchmarks.

---

## 1. Build & Push the Docker Image (local machine)

```bash
cd /Users/vojtech/Desktop/ecl/runpod-dev

# Point the image at the benchmark branch
PIE_REPO_REF=benchmarks/vs-sglang ./deploy_image.sh
```

What this does:
- Builds `linux/amd64` Docker image with Rust, uv, Tailscale, zsh
- Clones the Pie repo at the specified branch
- Runs `uv sync`, installs bakery + componentize-py, builds text-completion.wasm
- Pushes to Docker Hub as `kysilkavojtech/runpod-dev:<tag>`

The script prints the full image name at the end — copy it.

## 2. Create / Update RunPod Pod

1. Go to [runpod.io](https://runpod.io) → Pods → Create/Edit
2. Paste the image name (e.g. `kysilkavojtech/runpod-dev:v20260405-1430`)
3. Set environment variables (under "Environment Variables"):
   - `RUNPOD_SECRET_TAILSCALE_AUTH_KEY` — for SSH access via Tailscale
   - `RUNPOD_SECRET_SSH_KEY` — base64-encoded GitHub SSH key for private repos
4. Select GPU (RTX 4000 Ada for 0.6B, A100/H100 for 8B+)
5. Start the pod

## 3. Pod Boot Sequence (automatic)

When the pod starts, `start_dev.sh` runs automatically:

1. Starts Tailscale daemon + authenticates (SSH access)
2. Sets up GitHub SSH keys
3. Runs `bootstrap_once.sh`:
   - Checks if repo exists, optionally `git pull`
   - Hash-checks `pyproject.toml` + `uv.lock` → skips `uv sync` if unchanged
   - Downloads model if missing (default: `Qwen/Qwen3-0.6B`)
   - Builds `text-completion.wasm` if missing
4. Hands off to RunPod's `/start.sh` (JupyterLab, etc.)

## 4. SSH into the Pod

```bash
# Via Tailscale (if configured)
ssh root@<pod-id>

# Or via RunPod web terminal
```

## 5. Switch to Benchmark Branch (if not already)

```bash
cd /root/Workspace/pie
git fetch origin benchmarks/vs-sglang
git checkout benchmarks/vs-sglang
```

## 6. Install Extra Dependencies

The benchmark scripts need `httpx` (for vLLM HTTP calls) and `pie_client`:

```bash
cd /root/Workspace/pie/pie
uv pip install --python .venv/bin/python httpx
uv pip install --python .venv/bin/python -e /root/Workspace/pie/client/python
```

vLLM must be installed separately (it's not in Pie's deps):

```bash
uv pip install --python .venv/bin/python vllm
```

---

## Running Benchmarks

### A. Pie vs vLLM (automated)

The `run_bench_vs_vllm.sh` script handles everything: starts both servers,
builds inferlets, runs benchmarks, saves results.

```bash
# Full suite — all tiers, both engines
run_bench_vs_vllm.sh

# Just overhead isolation (fastest, ~2 min)
BENCH_TIERS=0 run_bench_vs_vllm.sh

# Tier 1 only (latency + throughput + TTFT, ~10 min)
BENCH_TIERS=0,1a,1b,1c run_bench_vs_vllm.sh

# Tier 2 only (multi-step workflows, ~5 min)
BENCH_TIERS=2a,2b,2c run_bench_vs_vllm.sh

# With vLLM prefix caching (APC) enabled
VLLM_ENABLE_APC=1 run_bench_vs_vllm.sh

# Only Pie (skip vLLM server)
BENCH_PIE_ONLY=1 run_bench_vs_vllm.sh

# Only vLLM (skip Pie server)
BENCH_VLLM_ONLY=1 run_bench_vs_vllm.sh

# Larger model
PIE_MODEL=meta-llama/Llama-3.1-8B run_bench_vs_vllm.sh

# More runs for statistical confidence
BENCH_RUNS=10 run_bench_vs_vllm.sh

# Leave servers running after benchmark (for manual follow-up)
BENCH_KEEP_SERVERS=1 run_bench_vs_vllm.sh

# Higher concurrency ceiling
BENCH_MAX_CONCURRENCY=128 BENCH_TIERS=1b run_bench_vs_vllm.sh

# Pass extra args directly to the Python script
run_bench_vs_vllm.sh --vllm-metrics --tiers 0,1a
```

#### With Pie Runtime Profiling

To see where time is spent inside Pie (WASM instantiation breakdown):

```bash
# 1. Build Pie runtime with profiling enabled
cd /root/Workspace/pie/runtime
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --features ipc-profiling

# 2. Rebuild the Python module (maturin)
cd /root/Workspace/pie/pie
uv run --extra cu128 --frozen --no-sync maturin develop --release \
    --manifest-path ../runtime/Cargo.toml \
    --features ipc-profiling

# 3. Run benchmarks
BENCH_TIERS=0 run_bench_vs_vllm.sh

# 4. Check the profiling output
grep LAUNCH-PROFILE /root/Workspace/benchmark-results/*/pie-server.log
```

The `[LAUNCH-PROFILE]` output shows:
- `store_ms` — `Store::new()` creation
- `linker_ms` — WASI + API linker setup
- `deps_ms` — dependency instantiation (dynamic linking)
- `instantiate_ms` — main component instantiation
- `resolve_ms` — export resolution
- `execution_ms` — actual inferlet execution (includes GPU work)

### B. Pie vs SGLang (manual server setup)

SGLang must be started manually since it's a separate Python process:

```bash
# 1. Install SGLang
pip install sglang[all]

# 2. Start SGLang server (in tmux or background)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 &

# Wait for "Server started" message

# 3. Start Pie server (in another tmux pane)
cd /root/Workspace/pie/pie
uv run --extra cu128 --frozen --no-sync pie serve --no-auth &

# 4. Build inferlets and run
cd /root/Workspace/pie
./benches/run_vs_sglang.sh

# Or run specific tiers
./benches/run_vs_sglang.sh --tiers 1a,1b
./benches/run_vs_sglang.sh --pie-only
./benches/run_vs_sglang.sh --sglang-only
```

### C. Pie-Only Benchmarks (correctness + performance)

These don't need vLLM or SGLang — just a running Pie server:

```bash
# Start Pie server
cd /root/Workspace/pie/pie
uv run --extra cu128 --frozen --no-sync pie serve --no-auth &

# Run the full internal benchmark suite
cd /root/Workspace/pie
./benches/run_benchmarks.sh          # full suite
./benches/run_benchmarks.sh --quick  # fast subset
./benches/run_benchmarks.sh --only determinism  # single test
```

### D. Throughput-Only (original benchmark)

```bash
# Automated (starts server, runs tput.py, stops server)
run_benchmark_updated.sh

# Or with custom parameters
BENCH_NUM_REQUESTS=256 BENCH_CONCURRENCY=64 run_benchmark_updated.sh
```

---

## Results

All results go to `/root/Workspace/benchmark-results/<timestamp>/`:

```
/root/Workspace/benchmark-results/20260405T143000Z/
├── vs_vllm.json          # Structured benchmark results (Pie vs vLLM)
├── benchmark.log         # Console output
├── pie-server.log        # Pie server log (grep for LAUNCH-PROFILE)
├── vllm-server.log       # vLLM server log
├── pie-config.toml       # Pie config used for this run
└── context.env           # All parameters for reproducibility
```

SGLang results go to `benches/results/<timestamp>/vs_sglang.json`.

To copy results off the pod:

```bash
# From local machine, via Tailscale
scp -r root@<pod-id>:/root/Workspace/benchmark-results/ ./results/

# Or tar and download
ssh root@<pod-id> "tar czf /tmp/results.tar.gz /root/Workspace/benchmark-results/"
scp root@<pod-id>:/tmp/results.tar.gz .
```

---

## Benchmark Tiers Reference

| Tier | Name | What It Measures | Time |
|------|------|-----------------|------|
| 0 | Overhead Isolation | Pure framework overhead (WASM vs HTTP) | ~2 min |
| 1A | Single-Request Latency | Raw latency at various input/output sizes | ~3 min |
| 1B | Throughput Scaling | req/s at concurrency 1→128 | ~5 min |
| 1C | Time-to-First-Token | Prefill + first decode step latency | ~3 min |
| 2A | Chain-of-Generations | Multi-turn (draft→critique→revise), KV reuse | ~3 min |
| 2B | Best-of-N | Shared prefix, N parallel generations, fork | ~3 min |
| 2C | Constrained Retry | JSON validation with rollback on failure | ~3 min |

---

## Investigation Plan

### Phase 1: Establish Baseline (do first)

1. Deploy image with `PIE_REPO_REF=benchmarks/vs-sglang`
2. Run `BENCH_TIERS=0 run_bench_vs_vllm.sh` on 0.6B
   - This directly answers: **how much does WASM instantiation cost?**
   - Compare noop (pure WASM), flush-only (WASM + 1 IPC), one-token (WASM + IPC + GPU)
   - vs vLLM's one-token overhead
3. Run full suite: `run_bench_vs_vllm.sh`
   - Get baseline Pie vs vLLM numbers on same hardware as SGLang tests
4. Enable Pie profiling (see instructions above)
   - Check `[LAUNCH-PROFILE]` to see Store/linker/deps/instantiate breakdown
   - Identify which part of the ~500ms overhead dominates

### Phase 2: Investigate the Overhead Discrepancy

5. Run with APC on: `VLLM_ENABLE_APC=1 BENCH_TIERS=2a,2b run_bench_vs_vllm.sh`
   - See how much vLLM's prefix caching helps in multi-step workflows
   - Compare with and without — does it close the gap with Pie?
6. **Ask your advisor**: What hardware, model, Pie version, and concurrency level
   did he test at? His claim that WASM overhead isn't costly might be because:
   - Different CPU (WASM instantiation is CPU-bound)
   - Different Pie version (instantiation path may have changed)
   - High concurrency (overhead amortized — you showed Pie wins at c=32)
   - Different measurement (wall time vs throughput vs TTFT)
7. Test 8B: `PIE_MODEL=meta-llama/Llama-3.1-8B run_bench_vs_vllm.sh`
   - Prefill becomes expensive (~100ms for 2K tokens vs ~10ms on 0.6B)
   - Pie's prefill savings should become visible in wall time

### Phase 3: Higher Concurrency + Three-Way Report

8. Concurrency ceiling: `BENCH_TIERS=1b BENCH_MAX_CONCURRENCY=128 run_bench_vs_vllm.sh`
   - Find where Pie overtakes vLLM (you found c=32 for SGLang)
9. Run SGLang on same pod for three-way comparison
   - Key table: Pie vs SGLang vs vLLM at each tier
   - Key numbers: overhead, crossover point, prefill redundancy ratio

### Phase 4: Deeper Analysis (optional)

10. vLLM metrics: `run_bench_vs_vllm.sh --vllm-metrics`
    - Scrapes Prometheus `/metrics` for queue time, TTFT, ITL breakdowns
11. Tree-of-thought (Tier 3): Write an inferlet that forks 8-15 times
    - Pie's fork advantage should be multiplicative here
12. vLLM `best_of=N`: Test vLLM's native parameter vs Pie's fork approach
13. Profile WASM instantiation deeper: Which component of the Wasmtime
    instantiation (Store, linker, memory init, dependency linking) dominates?
    This identifies optimization targets for the Pie team.

---

## Troubleshooting

**Pie server won't start:**
```bash
# Check the log
tail -50 /root/Workspace/benchmark-results/*/pie-server.log

# Common issue: model not downloaded
cd /root/Workspace/pie/pie
uv run --extra cu128 --frozen --no-sync pie model download Qwen/Qwen3-0.6B
```

**vLLM server won't start:**
```bash
# Check if vLLM is installed
python -c "import vllm; print(vllm.__version__)"

# Check the log
tail -50 /root/Workspace/benchmark-results/*/vllm-server.log

# Common issue: OOM on large models
# Use --max-model-len to limit context size
# Or use --gpu-memory-utilization 0.8
```

**Inferlets fail to build:**
```bash
# Check wasm32-wasip2 target is installed
rustup target list --installed | grep wasm

# If missing:
rustup target add wasm32-wasip2
```

**"pie_client not found" or "httpx not found":**
```bash
cd /root/Workspace/pie/pie
uv pip install --python .venv/bin/python httpx
uv pip install --python .venv/bin/python -e /root/Workspace/pie/client/python
```

**Benchmark results look wrong (all latencies identical, zero throughput):**
- Check that both servers are actually running: `ps aux | grep -E "pie serve|vllm"`
- Check that the model is loaded: look for "Engine running" in pie-server.log
- Check warmup: the first request is always slow (WASM compilation). Benchmarks
  should handle this automatically, but verify with `BENCH_RUNS=1 BENCH_TIERS=0`
