# Pie Benchmarking Results — Consolidated

This is the single up-to-date record of every Pie benchmark we've run, what
exactly each one tests, and what we found. For how to reproduce any of these
runs on a fresh pod, see `runpod-dev/BENCHMARKS.md`.

Dated snapshots of the original standup notes and reports live in
`benches/docs/archive/` — they preserve superseded numbers, wrong hypotheses,
and the reasoning trail. This file is the ground truth.

**Last updated:** 2026-04-07 (consolidation pass)

---

## Table of Contents

1. [Engines and Versions](#engines-and-versions)
2. [Hardware and Models](#hardware-and-models)
3. [What Each Engine Actually Runs](#what-each-engine-actually-runs)
4. [Benchmark Tier Definitions](#benchmark-tier-definitions)
5. [Latest Results](#latest-results)
6. [Profiling Methodology](#profiling-methodology)
7. [Overhead Taxonomy](#overhead-taxonomy)
8. [Priority-Ordered Improvements](#priority-ordered-improvements)
9. [Open Questions / Not Yet Run](#open-questions--not-yet-run)
10. [Progression Timeline](#progression-timeline)

---

## Engines and Versions

| Engine | Version | Role |
|--------|---------|------|
| **Pie** | branch `benchmarks/vs-sglang` (post 2026-04-06 merge with `upstream/main`) | Primary system under test |
| **vLLM** | 0.19.0 | Baseline #1 — most widely deployed |
| **SGLang** | snapshot used in 2026-03 runs | Baseline #2 — closest competitor (fork/join + RadixAttention) |

All three serve via their standard APIs:

- Pie: WebSocket, WASM inferlet launches
- vLLM: HTTP OpenAI-compatible `/v1/chat/completions`
- SGLang: HTTP OpenAI-compatible `/v1/chat/completions`

---

## Hardware and Models

| Rig | GPU | VRAM | Model | Pod IP | Used for |
|-----|-----|------|-------|--------|----------|
| Ada-4000 | NVIDIA RTX 4000 Ada | 20 GB | Qwen/Qwen3-0.6B (bf16) | 100.126.138.66 | vs-SGLang (2026-03), full 0.6B vs-vLLM (2026-04-06), WASM+Python profiling |
| Ada-6000 | NVIDIA RTX 6000 Ada | 48 GB | Qwen/Qwen3-8B (bf16) | 100.104.196.95 | 8B vs-vLLM (2026-04-06), Python profiling, CUDA-graphs experiment |

**GPU sharing:**
- 0.6B: both engines run simultaneously at `gpu_mem_utilization = 0.4`
- 8B: engines run **sequentially**, each with `gpu_mem_utilization = 0.8` (8B bf16 weights + KV cache do not fit twice on 48 GB)

---

## What Each Engine Actually Runs

Exact command lines and code paths exercised — so future runs can confirm
they are measuring the same thing.

### Pie

**Inferlet:** `std/text-completion` (WASM) for Tier 0/1; `benches/inferlets/bench-*` for Tier 0 specials and Tier 2.

Launched via `pie_client.launch_instance(...)` with:

```
--prompt <text>
--max-tokens <N>
--temperature 0
```

Every generated token crosses this pipeline:

```
WASM inferlet ctx.generate()
  → Rust host function (WASM sandbox boundary)
    → Rust batch scheduler (runtime/src/model.rs, ~line 470-570)
      → msgpack serialize ForwardPassRequest
        → ipc-channel (OS pipe, Rust → Python, runtime/src/model/ffi_ipc.rs)
          → Python ipc_queue.recv() + msgpack deserialize
            → fire_batch() in pie_worker/runtime.py:
                → Batch() constructor
                → batch.get_model_inputs(device)
                → batch.get_sampling_metadata(device, dtype)
                → _run_step():
                    → engine.embed_inputs()
                    → engine.transform()   (24 or 36 layers, FlashInfer paged attention)
                    → engine.sample()      (lm_head + softmax + sample)
                → batch.create_responses()
            → msgpack serialize response
              → ipc-channel (Python → Rust)
                → Rust deserializes
                  → tokio oneshot → back into the WASM instance
```

Attention uses **FlashInfer** (`BatchDecodeWithPagedKVCacheWrapper`,
`BatchPrefillWithPagedKVCacheWrapper`) — same kernels as vLLM. Pie imports it
as `import flashinfer as ops` in `pie_worker/model/qwen3.py`.

**Crucial config:** `use_cuda_graphs = false` in the default Pie config
(`pie/src/pie/config.py`). As of 2026-04-07 the qwen3 init path also contains
a bug where `warmup_cuda_graphs()` is never called even when
`use_cuda_graphs = true` — see the [followup plan](pie_fixes_followup.md).

### vLLM

Launched via:

```
python -m vllm.entrypoints.openai.api_server \
    --model <HF_REPO> \
    --host 127.0.0.1 --port 8000 \
    --dtype auto --max-model-len 4096 \
    --gpu-memory-utilization <0.4|0.8>
    [--enable-prefix-caching]   # only when VLLM_ENABLE_APC=1
```

Benchmarks hit `POST /v1/chat/completions` with the same prompt text, same
`max_tokens`, `temperature=0`. Key defaults left in place:

- `enforce_eager = False` — CUDA graphs ON for decode
- FlashInfer paged attention (same kernels as Pie)
- Continuous batching

### SGLang

Launched via (2026-03 runs):

```
python -m sglang.launch_server --model-path <HF_REPO> --port 30000
```

Driven through SGLang's OpenAI-compatible endpoint by
`benches/bench_vs_sglang.py`. Multi-step tests (Tier 2) use SGLang programs
with `gen()` / `fork()` to opt into RadixAttention prefix caching.

---

## Benchmark Tier Definitions

| Tier | Name | What it measures | Pie harness | vLLM/SGLang harness |
|------|------|-----------------|-------------|---------------------|
| 0 | Framework overhead | Per-request cost with little/no GPU work | `bench-noop`, `bench-flush-only`, `bench-one-token` inferlets | `max_tokens=1` chat completion |
| 1A | Single-request latency | Wall time, 1 request at a time | `text-completion` inferlet | `/v1/chat/completions` |
| 1B | Throughput scaling | req/s vs concurrency (1 → 128) | `text-completion` inferlet, concurrent launches | concurrent HTTP requests |
| 1C | Time-to-first-token | Prefill latency isolated from decode | streaming token timestamps | streaming SSE, measure first chunk |
| 2A | Chain-of-generations | Draft → Critique → Revise, KV cache across 3 steps | `bench-chain-of-gen` inferlet | 3 sequential HTTP calls, each sending growing history |
| 2B | Best-of-N | Shared 2K prefix + 4 parallel forks | `bench-best-of-n` inferlet (`flush` + `fork`×4) | 4 concurrent requests with the same prompt |
| 2C | Constrained retry | Generate JSON, validate, retry on failure | `bench-constrained-retry` inferlet, fork from checkpoint | Restart from scratch each attempt |

Each test runs 5 times after a warmup request. Results are reported as p50 /
p99 / mean.

---

## Latest Results

### Tier 0 — Framework Overhead

| Test | 0.6B Pie | 0.6B vLLM | 8B Pie | 8B vLLM |
|------|----------|-----------|--------|---------|
| one-token | 42 ms | 17 ms | 41 ms | 29 ms |

Pie's ~41 ms is dominated by client-side WebSocket + JSON serialization. The
**actual WASM instantiation inside the Rust runtime is <1 ms** (see
[profiling](#profiling-methodology) below).

### Tier 1A — Single-Request Latency (p50)

| Configuration | 0.6B Pie | 0.6B vLLM | 0.6B ratio | 8B Pie | 8B vLLM | 8B ratio |
|---------------|----------|-----------|------------|--------|---------|----------|
| 128 in / 128 out | 1,755 ms | 556 ms | 3.2× | 2,844 ms → **2,716 ms\*** | 2,237 ms | 1.27× → **1.22×\*** |
| 512 in / 128 out | 1,775 ms | 566 ms | 3.1× | 2,837 ms | 2,235 ms | 1.27× |
| 2048 in / 256 out | 3,729 ms | 1,210 ms | 3.1× | 5,843 ms | 4,508 ms | 1.30× |

\* **Corrected 8B 128/128 result** after applying the lazy `warmup_cuda_graphs()`
workaround on the Ada-6000 pod. Saved ~130 ms on that one configuration,
tightening the ratio from 1.27× → 1.22×. Larger configs not re-run before the
pod was killed. See [progression](#progression-timeline).

### Tier 1B — Throughput (0.6B)

| Concurrency | Pie req/s | vLLM req/s | Winner |
|-------------|-----------|------------|--------|
| c=1  | 1.5  | 3.4 | vLLM |
| c=16 | 12.5 | 23.0 | vLLM |
| c=32 | 20.3 | 27.6 | vLLM |
| c=64 | 29.4 | 30.0 | ~tie |
| **c=128** | **52.3** | **31.7** | **Pie** |

Pie's batch scheduler amortizes the per-request WASM cost and overtakes vLLM
around c=64–128. Not yet tested on 8B.

### Tier 1B — Throughput (0.6B, vs SGLang, 2026-03)

| Concurrency | Pie req/s | SGLang req/s |
|-------------|-----------|--------------|
| c=1  | 1.5  | 3.7  |
| c=16 | 13.4 | 13.9 |
| c=32 | 23.6 | 19.8 |

Pie overtakes SGLang earlier (c=32 vs c=128 for vLLM) because SGLang doesn't
capture decode as aggressively as vLLM's CUDA graph path.

### Tier 2A — Chain-of-Generations (p50 wall time)

| Engine | 0.6B | 8B |
|--------|------|-----|
| Pie | 10,716 ms | 16,810 ms |
| vLLM | 3,633 ms (2.9× faster) | 13,539 ms (1.24× faster) |

Pie prefills the ~2K system prompt **once**; vLLM re-prefills it on every
call (average 2,404 tokens per step). On 0.6B this cost is negligible
(~50 ms total). On 8B it's meaningful, which is why the gap narrows.

### Tier 2B — Best-of-4 (p50 wall time, 0.6B)

| Engine | Wall time | Prefill strategy |
|--------|-----------|-----------------|
| Pie | 5,942 ms | 1× prefill, fork shares KV cache across 4 branches |
| vLLM | 1,539 ms (3.9× faster) | 4× prefill (APC disabled) |

Per-token decode speed still dominates on 0.6B.

### Tier 2C — Constrained Retry (p50 wall time, 0.6B)

| Engine | Wall time | Avg attempts |
|--------|-----------|-------------|
| Pie | 26,324 ms | 6.0 |
| vLLM | 8,941 ms (2.9× faster) | 6.0 |

Both exhausted retries, which means the test was measuring "how fast can you
fail 6 times" — same story as the decode gap. Tune the prompt to get ~3-5
attempts to actually show the rollback advantage.

### Per-Token Decode Cost

Derived from Tier 1A wall time and per-batch Python worker profiling
(see next section).

| Model | Pie ms/token | vLLM ms/token | Gap | Gap composition |
|-------|-------------|---------------|-----|-----------------|
| 0.6B  | ~13.7 ms | ~4.3 ms | 9.4 ms | IPC 2.5 ms (27%) + forward pass 6.7 ms (73%) |
| 8B    | ~22.0 ms | ~17.4 ms | 4.6 ms | IPC 2.8 ms (61%) + forward pass 1.4 ms (30%) + batch 0.4 ms (9%) |

**Key observation:** as the model grows, the IPC portion stays roughly
constant (~2.5–2.8 ms) while the forward-pass-dispatch portion shrinks. On
larger models IPC becomes the dominant gap.

---

## Profiling Methodology

Three independent layers of instrumentation, all driven through the same
benchmark harness so the numbers are directly comparable.

### Layer 1 — Rust WASM instantiation (`[LAUNCH-PROFILE]`)

- **Where:** `runtime/src/runtime.rs` around `launch_instance()` (~line 1295-1365), gated behind the `ipc-profiling` Cargo feature.
- **How to enable:** `setup_bench_env.sh --with-profiling` (handles the maturin `-F` bug automatically).
- **What it logs:** per-request JSON with `store_ms`, `linker_ms`, `deps_ms`, `instantiate_ms`, `resolve_ms`, `total_setup_ms`, `execution_ms`, `total_ms`.
- **How to read:** `grep LAUNCH-PROFILE /root/Workspace/benchmark-results/<ts>/pie-server.log`.
- **Finding:** every field is <1 ms. Linker at ~0.2–0.6 ms dominates, which is
  still negligible.

### Layer 2 — Python GPU worker (`[PROFILING]`)

- **Where:** an ad-hoc print block added inside `fire_batch()` in
  `pie_worker/runtime.py`, in the **local** (non-IPC-spawned-worker)
  code path. Accumulates running averages and prints every 10 seconds.
- **What it measures:** per-batch `build_batch`, `get_inputs`, `inference` (the full `_run_step()`), `create_resp`, and `total`.
- **Why `inference` is meaningful:** it wraps `_run_step()` which runs
  `embed_inputs` + `transform` (all transformer layers, FlashInfer paged
  attention) + `sample` (lm_head + softmax + sample). This is the closest
  apples-to-apples to vLLM's internal per-step time.
- **How to enable:** currently a manual patch — **not yet committed**. See
  the [followup plan](pie_fixes_followup.md#profiling-helpers-uncommitted).

Raw output, 0.6B on Ada-4000, sustained Tier 1A workload:

```
[PROFILING] Local avg: 11.2ms (724) | Last step: build_batch=0.1ms get_inputs=0.1ms inference=10.7ms create_resp=0.0ms total=11.0ms
```

Raw output, 8B on Ada-6000, sustained Tier 1A workload:

```
[PROFILING] Local avg: 19.7ms (453) | Last step: build_batch=0.2ms get_inputs=0.2ms inference=18.7ms create_resp=0.0ms total=19.2ms
[PROFILING] Local avg: 19.8ms (443) | Last step: build_batch=0.1ms get_inputs=0.1ms inference=18.8ms create_resp=0.0ms total=19.2ms
```

### Layer 3 — Client-side wall time

`bench_vs_vllm.py` measures `time.perf_counter()` around the WebSocket call
(Pie) or `httpx` call (vLLM). Wall time minus Layer-2 `total` gives the
IPC + WASM + client overhead per batch, which divided by tokens gives the
per-token IPC cost.

### Why the three layers matter together

Each layer alone is misleading:

- Layer 1 alone made it look like Pie had negligible overhead (<1 ms) while the client saw 41 ms.
- Layer 3 alone made it look like Pie used slower GPU kernels ("3× gap").
- Only when Layer 2 revealed that Pie's GPU forward pass was **itself** ~11 ms on 0.6B vs ~4 ms for vLLM did we realize the issue was Python kernel-launch dispatch, not IPC.

---

## Overhead Taxonomy

Every generated token in Pie pays these five taxes. Reducing the ones that
don't scale with model size is the highest-leverage work.

### 1. IPC serialization + transport — ~1.0–1.5 ms/token (constant)

**What:** msgpack encode the `ForwardPassRequest`, push through `ipc-channel`
OS pipe, kernel context switch to the Python worker, msgpack decode; mirror
for the response.

**Scales with model size?** **No.** The payload is small (token IDs + page
indices + sampler params = a few hundred bytes). This cost is identical on
0.6B and 70B.

**How to reduce:**
- Shared-memory ring buffer + futex instead of OS pipes → removes kernel context switches (~0.1 ms possible).
- Batch multiple decode steps into one IPC call → amortizes the round-trip over N tokens.

### 2. Rust async scheduler — ~0.5–1.0 ms/token (constant)

**What:** tokio mpsc channel into the batch scheduler, decision to fire,
`tokio::spawn` of `execute_forward_pass_batch`, oneshot back to the WASM
host function.

**Scales with model size?** **No.**

**How to reduce:**
- Fast path for batch_size=1: bypass the accumulation/firing loop and go directly to IPC.
- Move the decode loop out of WASM entirely: WASM calls `generate(max_tokens=N)`, Rust/Python runs the loop, returns all tokens at once.

### 3. Python batch construction — ~0.4 ms/token

**What:** `Batch()` constructor (unpack msgpack kwargs, BRLE mask decode),
`get_model_inputs(device)` (tensor creation on GPU), `get_sampling_metadata(...)`.

**Scales with model size?** Very weakly (larger KV cache → more page
indices), essentially constant.

**How to reduce:** reuse batch objects across steps with delta updates,
pre-allocate input tensors.

### 4. Forward pass dispatch — ~1.4 ms (8B) to ~6.7 ms (0.6B)

**What:** Pie's `engine.transform()` runs as a Python for-loop over all
layers, launching ~10 kernels per layer × 24 layers (0.6B) or 36 layers (8B).
Each launch eats ~5-10 μs of Python-to-CUDA dispatch. FlashInfer's
`wrapper.plan()` is also called every step.

vLLM uses CUDA graphs: the whole decode step is captured once and replayed
with near-zero Python overhead.

**Scales with model size?** Inversely. Per-layer GPU compute grows with
`hidden_dim²`, so on 8B each kernel does ~16× more work than on 0.6B while
dispatch stays constant — the relative cost shrinks.

**How to reduce:** enable CUDA graphs. Pie's `qwen3.py` already has
`warmup_cuda_graphs()` and `_run_layers_graphed()`, but there's a bug where
the qwen3 init path never calls `warmup_cuda_graphs()` — see the
[followup plan](pie_fixes_followup.md).

**Experiment status (8B, with lazy warmup fix applied on pod):**

| Configuration | Pie no graphs | Pie lazy warmup | vLLM |
|---------------|---------------|----------------|------|
| 128 in / 128 out | 2,844 ms | **2,716 ms** | 2,237 ms |

~130 ms savings (~4.5% on that configuration). Modest because on 8B most
forward-pass time is already GPU compute.

**Experiment status (0.6B):** NOT RUN. The Ada-4000 pod was busy and the
Ada-6000 pod was killed before 0.6B could be downloaded and re-run. The
prediction, based on the dispatch analysis, is CUDA graphs would save ~4-5 ms
per token on 0.6B, bringing the Tier 1A 128/128 ratio from 3.2× down to
~1.8–2.0×. **This is the single most valuable experiment still pending.**

### 5. Client protocol overhead — ~40 ms/request (NOT per token)

**What:** WebSocket TCP handshake + message serialization + Python asyncio
scheduling on the client side + loopback round-trip.

**Amortized:** over a 128-token generation, this is ~0.3 ms/token —
negligible for generation workloads, dominant for Tier 0 overhead numbers.

**Scales with model size?** No.

**How to reduce:** connection pooling, persistent WebSockets (already fine
for real workloads).

### Summary table

| Overhead | Time/token | Constant? | Dominates on | Reducible? |
|----------|-----------|-----------|--------------|------------|
| IPC ser/deser + transport | ~1.0–1.5 ms | yes | 8B+ | yes — shared memory |
| Rust async scheduler | ~0.5–1.0 ms | yes | 8B+ | yes — fast path |
| Python batch construction | ~0.4 ms | mostly | 8B+ | yes — reuse/delta |
| Forward pass dispatch | 1.4 ms (8B), 6.7 ms (0.6B) | shrinks | 0.6B | yes — CUDA graphs |
| Client protocol | ~0.3 ms amortized | yes | never | already fine |

### Which one is "most painful / least scalable"?

**IPC + scheduler (~2.8 ms/token combined).** It:

1. Is constant per token regardless of model size — won't shrink on 70B/405B.
2. Is per-token, not per-batch — every single generated token pays it.
3. Is already 61% of the 8B gap. On 70B (forward pass ~90 ms/token), it would
   be ~3% of total — acceptable, but it means **Pie can never match vLLM's
   single-request latency, only approach it asymptotically**.

The client-protocol overhead looks huge at Tier 0 but vanishes on real
workloads. The dispatch overhead is painful on small models but goes away on
large ones. IPC is the one that stays painful forever.

---

## Priority-Ordered Improvements

Ordered by "biggest win for the least scalable problem, first".

1. **Move the decode loop out of WASM into Rust/Python.** Eliminates the
   per-token IPC round-trip entirely. Instead of
   `WASM.generate() → IPC → Python → IPC → WASM`, let the Python worker run
   N decode steps internally and return all N tokens at once. Expected saving:
   ~2.8 ms × N tokens. On 8B, 128-token generation saves ~358 ms (~12% of
   wall time). **Trade-off:** loses per-token WASM programmability (the
   inferlet cannot inspect/rewrite tokens mid-generation). Roadmap question.

2. **Shared-memory IPC.** Replace `ipc-channel` OS pipes with a shared-memory
   ring buffer + futex notification. Eliminates the two kernel context
   switches per round-trip. Could bring the IPC portion from ~1.5 ms down to
   ~0.1 ms, saving ~1.4 ms/token (~5% on 8B). Keeps per-token
   programmability. Medium effort.

3. **Enable CUDA graphs on 0.6B (and validate on 8B).** After fixing the
   qwen3 init bug, just flip `use_cuda_graphs = true`. Expected saving:
   ~4-5 ms/token on 0.6B (huge), confirmed ~1 ms/token on 8B (modest). **The
   single cheapest experiment still pending.**

4. **Scheduler fast path for batch_size=1.** Bypass the accumulation loop
   when there's only one in-flight request. Expected saving:
   ~0.3–0.5 ms/token.

5. **Reuse batch objects / pre-allocate tensors.** Small, ~0.2–0.4 ms/token.

---

## Open Questions / Not Yet Run

These are experiments that would close the remaining narrative gaps but
weren't done before the Ada pods were released.

| Experiment | Expected outcome | Why it matters |
|------------|------------------|---------------|
| CUDA graphs on 0.6B | Tier 1A 128/128 from 1,755 ms → ~1,200 ms (ratio 3.2× → ~2.1×) | Validates the dispatch-overhead hypothesis, closes the biggest "easy win" |
| Shared-memory IPC prototype | Per-token IPC ~1.5 ms → ~0.1 ms | Quantifies #2 above |
| Decode-loop-in-Rust prototype | Eliminates per-token IPC, ~2.8 ms/token savings | Quantifies #1 above |
| 8B at high concurrency (c=64, c=128) | Should show the same crossover as 0.6B | Confirms Pie's throughput advantage generalizes |
| 70B single-request latency | Gap should be ~1.05× | Confirms IPC becomes irrelevant on large models |
| vLLM with `--enable-prefix-caching` on Tier 2A/2B | Should close some of the prefill-savings gap | Quantifies how much of Pie's design advantage is neutralized by APC |
| Tree-of-thought (Tier 3) | Pie should win decisively (15 branches sharing prefix) | The one scenario where Pie's fork model compounds multiplicatively |
| TTFT as a separate metric | Pie ~constant at ~41 ms, vLLM grows with context | May look favorable for Pie on long prompts |

---

## Progression Timeline

How the story changed run-to-run. Each entry links to the original snapshot
in `archive/`.

### 2026-02-23 — Legacy internal benchmark suite

Eight benchmarks (correctness, performance, resilience) using a
`bench_utils.py` framework. 32/32 passed on RunPod + Qwen3-0.6B. These
measure things like determinism, batch-position independence, stress, client
disconnect cleanup. Not directly comparable to vLLM/SGLang.
→ `archive/2026-02-23_setup_guide_legacy.md`

### 2026-03-23 — First Pie vs SGLang comparison

Tier 1 + Tier 2, Qwen3-0.6B on RTX 4000 Ada. **Conclusion at the time:**
"Pie has ~500-600 ms per-request WASM overhead". **Revised later:** this was
never actually WASM — it was decode speed dominated by Python kernel
dispatch. SGLang numbers are still valid as a data point.
→ `archive/2026-03-23_report_vs_sglang.md`,
  `archive/2026-03-23_standup_vs_sglang.md`

Key finding that survived: **Pie overtakes SGLang at c=32 in Tier 1B
throughput.**

### 2026-04-06 — Pie vs vLLM, 0.6B

Full Tier 0/1/2 suite vs vLLM on Ada-4000. **Headline:** Pie 3.1–3.2× slower
on single requests, overtakes at c=128. The Tier 0 "WASM overhead" measured
41 ms — much lower than the 500 ms of the SGLang era, which hinted the
earlier number was misattributed.
→ `archive/2026-04-06_0.6b_full_suite_report.md`,
  `archive/2026-04-06_standup_vs_vllm.md`

### 2026-04-06 — WASM profiling (Layer 1)

Added `ipc-profiling` feature to `runtime/src/runtime.rs`, measured
`launch_instance()` phase-by-phase. **Found:** WASM setup <1 ms.
**Conclusion:** the 41 ms is client protocol (WebSocket + serialization),
not WASM.
→ `archive/2026-04-06_profiling_ipc_breakdown.md`

At this point the story was still "Pie has slow kernels" — we hadn't
profiled the Python worker yet.

### 2026-04-06 — Python worker profiling (Layer 2) + 8B run

Added the ad-hoc `[PROFILING]` print inside `fire_batch()`. Ran the full
Tier 0/1/2 suite on 8B on Ada-6000 (sequential mode, 0.8 GPU each).
**Headlines:**

- 8B 1A gap drops from 3.2× to 1.27× — Pie scales better on bigger models.
- 8B Tier 2A gap drops from 2.9× to 1.24× — KV persistence starts to pay.
- GPU forward pass on 0.6B is 11 ms; on 8B it's ~19 ms.
- IPC portion stays ~2.5–2.8 ms regardless of model size.

This is when the "Python dispatch overhead, not kernel quality" story
crystallized: both engines use FlashInfer, so the kernels are identical.
The gap must be dispatch.
→ `archive/2026-04-06_8b_comparison.md`

### 2026-04-06/07 — CUDA graphs experiment on 8B

**First attempt:** set `use_cuda_graphs = true` in config, ran 8B.
**Result:** 2,844 ms → 2,789 ms. Basically no change. Confusing.

**Investigation:** `grep` of `pie-server.log` showed no "Capturing CUDA
graphs" message. Traced `pie_worker/runtime.py` and found
`warmup_cuda_graphs()` is only called in the `llama3` init branch — **the
qwen3 branch never calls it.** With the warmup buffer empty,
`_run_layers_graphed()` silently falls back to the non-graphed path.

**Second attempt:** added `warmup_cuda_graphs()` to the qwen3 `__init__`
path. **Startup timeout** — the warmup (~0.75 s for 13 bins) takes long
enough that the Rust server's `IpcOneShotServer::accept()` times out and
kills the worker.

**Third attempt:** lazy warmup on first `fire_batch()` call. Works.
**Corrected 8B 128/128 result: 2,716 ms (p50), ratio 1.22× vs vLLM.**
~130 ms saved, ~4.5% on that config. Modest but real.

The Ada-6000 pod was killed before 0.6B could be downloaded to re-run there,
so the big predicted 0.6B win is still hypothetical.

### 2026-04-07 — Consolidation pass (this document)

Merged all dated docs into this file + `runpod-dev/BENCHMARKS.md` +
`pie_fixes_followup.md`. Archived originals under `benches/docs/archive/`.
Pulled `upstream/main` into `main` and `benchmarks/vs-sglang` (single fix:
PR #309, Python-component snapshot fix in `runtime/src/runtime.rs`).

---

## Related Documents

- [How to reproduce any of this on a fresh RunPod instance](../../runpod-dev/BENCHMARKS.md)
- [Pie bugs / missing features discovered while benchmarking](pie_fixes_followup.md)
- `archive/` — every dated snapshot that fed into this summary
- `ideas.md`, `plan.md` — original benchmark-suite design (still mostly valid)
