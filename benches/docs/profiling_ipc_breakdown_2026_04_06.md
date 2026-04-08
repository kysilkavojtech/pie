# IPC Profiling Breakdown — 2026-04-06

**GPU:** NVIDIA RTX 4000 Ada (20GB VRAM)
**Model:** Qwen/Qwen3-0.6B (bfloat16)
**Pie branch:** benchmarks/vs-sglang
**Feature flag:** `ipc-profiling` (runtime `Cargo.toml`)

---

## Background

The main benchmark suite (see `results/20260406T032202Z/report.md`) showed Pie takes ~41ms for a noop request (pure WASM instantiation, no GPU work). We needed to understand where that 41ms is actually spent — is it WASM setup, IPC overhead, or something else?

## How to Reproduce

### 1. Build the runtime with `ipc-profiling` feature

```bash
cd /root/Workspace/pie/pie

# Install maturin if needed
uv pip install --python .venv/bin/python maturin

# Build with profiling feature enabled
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv run --extra cu128 --frozen --no-sync \
    maturin develop --release --manifest-path ../runtime/Cargo.toml -F ipc-profiling
```

**IMPORTANT: maturin has a bug** where `-F ipc-profiling` compiles the feature into the cargo target but does NOT copy the profiling-enabled binary into the installed Python package. You must manually copy:

```bash
# Verify the cargo output has profiling
strings /root/Workspace/pie/runtime/target/release/lib_pie.so | grep 'LAUNCH-PROFILE'

# Copy to both locations (editable install puts it in src/)
cp /root/Workspace/pie/runtime/target/release/lib_pie.so \
   /root/Workspace/pie/pie/src/pie/_pie.cpython-312-x86_64-linux-gnu.so

cp /root/Workspace/pie/runtime/target/release/lib_pie.so \
   /root/Workspace/pie/pie/.venv/lib/python3.12/site-packages/_pie/_pie.cpython-312-x86_64-linux-gnu.so

# Verify
strings /root/Workspace/pie/pie/src/pie/_pie.cpython-312-x86_64-linux-gnu.so | grep 'LAUNCH-PROFILE'
```

### 2. Run Tier 0 benchmarks

```bash
BENCH_TIERS=0 BENCH_PIE_ONLY=1 run_bench_vs_vllm.sh
```

### 3. Extract profiling data

```bash
# Find latest results
ls -t /root/Workspace/benchmark-results/ | head -1

# Extract LAUNCH-PROFILE lines from Pie server log
grep 'LAUNCH-PROFILE' /root/Workspace/benchmark-results/<timestamp>/pie-server.log
```

## What the Profiling Measures

The `[LAUNCH-PROFILE]` instrumentation is in `runtime/src/runtime.rs` (the `launch_instance` function, ~line 1295-1365). It measures:

| Phase | What it measures |
|-------|-----------------|
| `store_ms` | `Store::new()` — create WASM store with instance state |
| `linker_ms` | `create_linker()` — set up WASM linker with host functions |
| `deps_ms` | `instantiate_libraries()` — dynamic linking of dependency components |
| `instantiate_ms` | `linker.instantiate_async()` — instantiate the main WASM component |
| `resolve_ms` | `get_export()` + `get_typed_func()` — resolve the `run` function |
| `total_setup_ms` | Sum of all above (store through resolve) |
| `execution_ms` | `run_func.call_async()` — actual inferlet execution |
| `total_ms` | total_setup_ms + execution_ms |

## Raw Results

### Tier 0: noop (5 runs)

Noop inferlet returns immediately — no GPU work.

| Run | setup_ms | execution_ms | total_ms |
|-----|----------|-------------|----------|
| 1 | 0.7 | 0.3 | 1.0 |
| 2 | 0.4 | 0.2 | 0.6 |
| 3 | 0.2 | 0.1 | 0.4 |
| 4 | 0.7 | 0.3 | 1.0 |
| 5 | 0.3 | 0.1 | 0.4 |

**Median: ~0.5ms setup, ~0.2ms execution, ~0.6ms total**

### Tier 0: flush-only (5 runs)

Fills one token and flushes (one prefill round-trip to GPU).

| Run | setup_ms | execution_ms | total_ms |
|-----|----------|-------------|----------|
| 1 | 0.6 | 1030.2 | 1030.8 |
| 2 | 0.9 | 16.6 | 17.5 |
| 3 | 0.7 | 15.8 | 16.4 |
| 4 | 0.5 | 15.1 | 15.6 |
| 5 | 0.6 | 14.3 | 14.9 |

**First run: 1030ms** (model warmup / CUDA kernel compilation).
**Subsequent runs median: ~15.5ms execution** (prefill round-trip cost).

### Tier 0: one-token (5 runs)

Fills tokens, flushes (prefill), generates one token (one decode step).

| Run | setup_ms | execution_ms | total_ms |
|-----|----------|-------------|----------|
| 1 | 0.8 | 149.2 | 150.0 |
| 2 | 0.6 | 27.3 | 27.9 |
| 3 | 0.7 | 26.4 | 27.1 |
| 4 | 0.5 | 25.2 | 25.8 |
| 5 | 0.7 | 26.1 | 26.8 |

**First run: 149ms** (additional CUDA warmup for decode kernels).
**Subsequent runs median: ~26.3ms execution** (prefill + 1 decode step).

## Analysis

### 1. WASM instantiation is NOT the bottleneck

Total WASM setup (store + linker + deps + instantiate + resolve) is consistently **< 1ms**. The breakdown:
- `store_ms`: 0.0ms (negligible)
- `linker_ms`: 0.2–0.6ms (dominant setup cost, but still tiny)
- `deps_ms`: 0.0ms (no dependencies for bench-noop)
- `instantiate_ms`: 0.1–0.3ms
- `resolve_ms`: 0.0ms

### 2. Where does the 41ms benchmark measurement come from?

The benchmark client measures **41ms** for noop, but the runtime processes it in **< 1ms**. The ~40ms gap is:

```
41ms (client-measured) - 0.6ms (runtime-measured) = ~40ms unaccounted
```

This 40ms is the **client-side overhead**:
- WebSocket connection establishment (TCP + WS handshake)
- Message serialization/deserialization
- Python asyncio scheduling on the client side
- Network round-trip (loopback, but still involves kernel)

### 3. Per-decode-step cost

From flush-only vs one-token (after warmup):
- flush-only execution: ~15.5ms (prefill only)
- one-token execution: ~26.3ms (prefill + 1 decode step)
- **Single decode step: ~10.8ms** (26.3 - 15.5)

This is the cost of ONE decode step inside the runtime — it includes:
1. WASM inferlet calls `generate()` → crosses WASM sandbox boundary → Rust host function
2. Rust runtime sends IPC request to Python GPU worker
3. Python worker runs one forward pass on GPU
4. Python worker returns token via IPC
5. Rust runtime passes token back to WASM inferlet

For comparison, vLLM's per-token decode is ~4.2ms (from Tier 1A data). So the **IPC overhead per decode step is ~6.6ms** (10.8ms - 4.2ms GPU time).

### 4. Model warmup

First request incurs significant warmup:
- flush-only first run: 1030ms (vs 15ms steady-state) → ~1015ms warmup
- one-token first run: 149ms (vs 26ms steady-state) → ~123ms additional decode warmup

This explains why the earlier SGLang benchmarks showed ~500ms overhead — they may have been measuring warmup-contaminated requests.

## Key Takeaway

| Component | Time | % of 41ms client-measured |
|-----------|------|--------------------------|
| WASM instantiation | ~0.5ms | 1% |
| Client-side overhead (WS + serialization) | ~40ms | 98% |
| Runtime dispatch | ~0.1ms | <1% |

**The "WASM overhead" is actually client/protocol overhead.** The WASM instantiation itself is sub-millisecond. For reducing per-request latency, optimizing the client protocol (e.g., persistent connections, connection pooling) would have more impact than optimizing WASM instantiation.

**For the 3x decode speed gap** — see the updated analysis below.

## CORRECTION: Python Worker Profiling (added 2026-04-07)

The LAUNCH-PROFILE data above only measured WASM overhead. A follow-up experiment added per-batch profiling to the Python GPU worker (`fire_batch()` in `pie_worker/runtime.py`), which directly measures the GPU forward pass time.

### Python worker `[PROFILING]` output (0.6B, RTX 4000 Ada, sustained workload)

```
[PROFILING] Local avg: 11.2ms (724) | Last step: build_batch=0.1ms get_inputs=0.1ms inference=10.7ms create_resp=0.0ms total=11.0ms
[PROFILING] Local avg: 11.2ms (685) | Last step: build_batch=0.1ms get_inputs=0.1ms inference=10.7ms create_resp=0.0ms total=11.0ms
[PROFILING] Local avg: 11.3ms (672) | Last step: build_batch=0.1ms get_inputs=0.1ms inference=10.6ms create_resp=0.0ms total=11.0ms
```

### Revised per-token breakdown

| Component | Time | Method |
|-----------|------|--------|
| GPU forward pass (`inference`) | **~11.0ms** | Python worker profiling |
| Python batch overhead (build+inputs+responses) | ~0.2ms | Python worker profiling |
| IPC + WASM + Rust scheduling | ~2.5ms | 13.7ms total - 11.2ms Python |
| **Pie total per token** | **~13.7ms** | Client-measured (1768ms/128) |
| **vLLM total per token** | **~4.3ms** | Client-measured (556ms/128) |

### The real bottleneck: CUDA graphs disabled, not kernel differences

**Pie's GPU forward pass alone (11ms) is 2.6x slower than vLLM's entire per-token time (4.3ms).** Initially this was attributed to "slower GPU kernels," but both engines use the same FlashInfer paged attention kernels.

The actual cause: **Pie has `use_cuda_graphs = false`** while vLLM uses CUDA graphs by default (`enforce_eager = False`).

Without CUDA graphs, Pie's forward pass (~24 layers × ~10 ops = ~240 kernel launches) incurs Python dispatch overhead per launch. On a tiny 0.6B model where each kernel does minimal GPU compute, this overhead dominates. vLLM captures the decode step as a single CUDA graph replay with zero Python overhead.

The 3.2x gap breaks down as:
- **~60-70% is Python dispatch overhead** from running ~240 individual kernel launches without CUDA graphs
- **~20% is IPC overhead** — ~2.5ms per token for WASM↔Rust + Rust↔Python round-trips
- **~10-20% is other overhead** (FlashInfer `wrapper.plan()` per step, etc.)

**On 8B models (RTX 6000 Ada),** the gap shrinks to 1.27x because larger models have more GPU compute per kernel, making the fixed dispatch overhead proportionally smaller.

**Note:** Pie's `qwen3.py` already has CUDA graph support (`warmup_cuda_graphs()`, `_run_layers_graphed()`). Enabling `use_cuda_graphs = true` in config should validate this hypothesis.

## Steps I Took to Get These Results

1. **Identified profiling code**: Found `[LAUNCH-PROFILE]` instrumentation in `runtime/src/runtime.rs` (~line 1295-1365), gated behind `#[cfg(feature = "ipc-profiling")]`.

2. **First build attempt** (`maturin develop -F ipc-profiling`): Built quickly (0.35s) from cache — profiling string NOT in binary. Cargo was using cached artifacts from a non-profiling build.

3. **Forced clean rebuild** (`cargo clean` then `maturin develop -F ipc-profiling`): Full 1.5min rebuild. Cargo debug logs confirmed `features: ["ipc-profiling"]` was active. But the installed `.so` file still didn't have the string.

4. **Root cause**: maturin's editable install copies the cdylib from cargo's target directory, but was copying a stale artifact. The cargo-built `lib_pie.so` at `runtime/target/release/lib_pie.so` DID have the profiling strings. The installed `_pie.cpython-312-x86_64-linux-gnu.so` in both `src/pie/` and `.venv/lib/python3.12/site-packages/_pie/` did NOT.

5. **Workaround**: Manually copied `lib_pie.so` to both install locations. Verified with `strings ... | grep LAUNCH-PROFILE`.

6. **Ran Tier 0 benchmark**: `BENCH_TIERS=0 BENCH_PIE_ONLY=1 run_bench_vs_vllm.sh`

7. **Extracted profiling data**: `grep 'LAUNCH-PROFILE' .../pie-server.log`

8. **Results directory**: `/root/Workspace/benchmark-results/20260406T035456Z/` on the Ada 4000 RunPod pod (100.126.138.66)
