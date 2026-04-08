# Pie vs vLLM: 8B Model Comparison — 2026-04-06

**GPU:** NVIDIA RTX 6000 Ada (48GB VRAM)
**Model:** Qwen/Qwen3-8B (bfloat16)
**GPU Memory:** Each engine ran alone with 0.8 utilization (not shared)
**vLLM version:** 0.19.0
**Pie branch:** benchmarks/vs-sglang

---

## Side-by-Side: 0.6B vs 8B

### Tier 0: Overhead

| Test | 0.6B Pie | 0.6B vLLM | 8B Pie | 8B vLLM |
|------|----------|-----------|--------|---------|
| one-token | 42ms | 17ms | 41ms | 29ms |

Overhead is model-independent for Pie (~41ms, dominated by client-side WebSocket). vLLM's overhead increases slightly with model size (17ms → 29ms).

### Tier 1A: Single-Request Latency

| Configuration | 0.6B Pie | 0.6B vLLM | 0.6B Ratio | **8B Pie** | **8B vLLM** | **8B Ratio** |
|---------------|----------|-----------|------------|------------|-------------|-------------|
| 128 in / 128 out | 1,755ms | 556ms | 3.2x | **2,844ms** | **2,237ms** | **1.27x** |
| 512 in / 128 out | 1,775ms | 566ms | 3.1x | **2,837ms** | **2,235ms** | **1.27x** |
| 2048 in / 256 out | 3,729ms | 1,210ms | 3.1x | **5,843ms** | **4,508ms** | **1.30x** |

**The gap shrinks from 3.2x to 1.3x on the larger model.**

### Per-Token Decode Cost

| Model | Pie ms/token | vLLM ms/token | IPC overhead |
|-------|-------------|---------------|-------------|
| 0.6B | ~13.4ms | ~4.2ms | ~9.2ms (69% of Pie's time) |
| 8B | ~22.2ms | ~17.5ms | ~4.7ms (21% of Pie's time) |

The IPC overhead per token is roughly constant (~5-9ms). On small models where GPU time is tiny, IPC dominates. On 8B models where GPU time is ~17ms, IPC becomes a smaller fraction.

### Tier 2A: Chain-of-Generations

| Engine | 0.6B p50 | 0.6B ratio | **8B p50** | **8B ratio** |
|--------|----------|------------|------------|-------------|
| Pie | 10,716ms | — | **16,810ms** | — |
| vLLM | 3,633ms | 2.9x faster | **13,539ms** | **1.24x faster** |

**Pie's KV cache persistence is starting to matter.**

On 0.6B: vLLM re-prefills 2,404 tokens per step (~50ms) — negligible.
On 8B: vLLM re-prefills 2,404 tokens per step — this now costs meaningful GPU time, narrowing the gap.

Pie only prefills once (KV cache persists across all 3 generation steps), while vLLM prefills 3x.

---

## Analysis

### Why the gap narrowed

Both Pie and vLLM use FlashInfer for paged attention (same kernels). The key difference:

- **Pie: `use_cuda_graphs = false`** — every decode step runs the full Python layer loop (~240 kernel launches on 0.6B)
- **vLLM: CUDA graphs ON** (`enforce_eager = False`) — captures entire decode as a single graph replay

Without CUDA graphs, Python dispatch overhead (~5-10μs per kernel launch) dominates on small models where each kernel does very little GPU compute. On larger models, each kernel does more work, so dispatch overhead becomes proportionally smaller.

### Per-token breakdown (0.6B, from Python worker profiling)

```
Pie per-token = GPU_forward_pass + python_overhead + IPC_overhead
vLLM per-token = CUDA_graph_replay (near-zero Python overhead)

On 0.6B:  11.0ms + 0.2ms + 2.5ms = 13.7ms  vs  4.3ms  → 3.2x
On 8B:    ~20ms  + 0.2ms + 2.0ms = 22.2ms  vs  17.5ms → 1.27x
```

| Model | Pie GPU pass | Pie IPC+overhead | Pie total | vLLM total | Ratio |
|-------|-------------|-----------------|-----------|-----------|-------|
| 0.6B | ~11.0ms | ~2.7ms | ~13.7ms | ~4.3ms | 3.2x |
| 8B | ~20ms (est.) | ~2.2ms (est.) | ~22.2ms | ~17.5ms | 1.27x |

The ~11ms Pie GPU pass on 0.6B includes significant Python dispatch overhead from ~240 individual kernel launches (24 layers × ~10 ops). CUDA graphs would eliminate this, bringing the forward pass closer to vLLM's.

### What this means for larger models

On larger models, actual GPU compute per kernel grows but dispatch overhead stays constant:

| Model size | Pie GPU pass (est.) | IPC overhead | Pie total | vLLM total (est.) | Ratio |
|-----------|-------------------|-------------|-----------|------------------|-------|
| 0.6B | 11ms | ~2.5ms | 13.7ms | 4.3ms | 3.2x |
| 8B | ~20ms | ~2.2ms | 22.2ms | 17.5ms | 1.27x |
| 70B | ~95ms | ~2ms | ~97ms | ~90ms | ~1.08x |

### Pie's value proposition becomes clear at 8B+

| Advantage | 0.6B | 8B | Projected 70B |
|-----------|------|-----|-------------|
| Single-request gap | 3.2x slower | 1.3x slower | ~1.05x slower |
| Multi-step savings | Negligible | Meaningful | Significant |
| High-concurrency throughput | Wins at c=128 | Not tested yet | Should win earlier |

**Low-hanging fruit:** Enabling CUDA graphs (`use_cuda_graphs = true`) should significantly close the 0.6B gap and likely improve 8B too. Pie's `qwen3.py` already has CUDA graph capture/replay code.

---

## Raw Data

### Pie (8B, Ada 6000)
- Tier 0 noop: p50=41ms
- Tier 0 flush-only: p50=41ms
- Tier 0 one-token: p50=41ms (DECODE_MS=25.5ms from inferlet)
- 128in/128out: p50=2,844ms (σ=76ms)
- 512in/128out: p50=2,837ms (σ=12ms)
- 2048in/256out: p50=5,843ms (σ=59ms)
- Chain-of-gen: p50=16,810ms (1x prefill)

### vLLM (8B, Ada 6000)
- Tier 0 one-token: p50=29ms
- 128in/128out: p50=2,237ms (σ=120ms)
- 512in/128out: p50=2,235ms (σ=5ms)
- 2048in/256out: p50=4,508ms (σ=43ms)
- Chain-of-gen: p50=13,539ms (3x prefill, avg 2,404 tokens)

---

## How to Reproduce

### On RunPod with RTX 6000 Ada (48GB)

```bash
# Prerequisites: Same as 0.6B setup (see standup_2026_04_06.md)
# After bootstrap + dependency fixes:

# 1. Download model
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"

# 2. IMPORTANT: Can't share GPU with both engines at 8B
#    Run them sequentially, each with gpu_mem=0.8

# 3. Pie first
PIE_MODEL=Qwen/Qwen3-8B PIE_GPU_MEM=0.8 BENCH_TIERS=0,1a,2a BENCH_PIE_ONLY=1 \
    run_bench_vs_vllm.sh

# 4. Kill Pie, then vLLM
pkill -9 python; sleep 5
PIE_MODEL=Qwen/Qwen3-8B VLLM_GPU_MEM=0.8 BENCH_TIERS=0,1a,2a BENCH_VLLM_ONLY=1 \
    run_bench_vs_vllm.sh
```

**Important:** Unlike the 0.6B benchmarks, 8B models cannot share the GPU (8B bf16 = ~16GB weights + KV cache). Must run engines sequentially with full GPU access.
