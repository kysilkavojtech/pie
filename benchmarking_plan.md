# Pie Competitive Benchmark Suite

## Engines Under Test

| Engine | Why |
|--------|-----|
| **Pie** | Programmable inference via WASM inferlets |
| **vLLM** | Industry standard, highest adoption |
| **SGLang** | Closest competitor — fork/join, RadixAttention |
| **TGI** | Same Rust+Python architecture, production-proven |

## Test Configuration

```
Models:        Qwen3-0.6B (fast iteration), Llama-3.1-8B (realistic)
Prompts:       ShareGPT (realistic), synthetic fixed-length (controlled)
Hardware:      Single GPU (A100/H100 or equivalent)
Repetitions:   5 runs per test, report median + p95
```

## Metrics (Collected Across All Tiers)

| Metric | What It Measures |
|--------|-----------------|
| **TTFT** | Time to first token (ms), p50/p95/p99 |
| **ITL** | Inter-token latency (ms), p50/p95/p99 |
| **Throughput** | Output tokens/sec |
| **Total prefill tokens** | Total input tokens processed across all forward passes — the cost/efficiency metric |
| **GPU memory peak** | High-water mark VRAM usage |
| **Wall time** | End-to-end task completion time |

---

## Tier 1: Baseline Serving

**Goal:** Establish that pie is competitive at standard workloads. Pie doesn't need to win here — it needs to not lose badly.

### Test 1A — Single-Request Latency

Send one request at a time, measure TTFT and ITL across varying input/output lengths.

| Input tokens | Output tokens |
|-------------|--------------|
| 128 | 128 |
| 512 | 128 |
| 2048 | 256 |
| 4096 | 256 |

**Pie inferlet:** `std/text-completion`
**Competitors:** `/v1/completions` endpoint

**Primary metric:** TTFT, ITL distribution
**Secondary:** Throughput

### Test 1B — Throughput Scaling

Fixed workload (512 input, 128 output), increase concurrency: 1, 4, 8, 16, 32, 64.

**Pie inferlet:** `std/text-completion`
**Competitors:** `/v1/completions` with concurrent clients

**Primary metric:** Throughput (tok/s) vs concurrency curve
**Secondary:** Latency degradation under load, GPU memory peak

---

## Tier 2: Multi-Step Workflows

**Goal:** Show the "Chat API Tax." Same task, same result, but pie avoids redundant prefill by keeping control logic server-side.

### Test 2A — Chain-of-Generations

**Task:** Draft → Critique → Revise. Three sequential generations where each step sees all prior output.

| Engine | Approach |
|--------|----------|
| Pie | `bench/chain-of-gen` inferlet — 3 `generate()` calls, KV cache persists across steps |
| vLLM | 3 sequential `/v1/completions` calls, re-sending growing context each time |
| SGLang | Single SGLang program with 3 `gen()` calls |
| TGI | 3 sequential API calls, same as vLLM |

**Workload:** 1K token system prompt + few-shot examples. Each generation step: 256 tokens.

**Primary metric:** Wall time, total prefill tokens
**Secondary:** TTFT per step (step 2 and 3 should be near-zero for pie/SGLang)

### Test 2B — Best-of-N

**Task:** Generate N=4 completions from a shared 2K-token prefix, return the one with highest average log-prob.

| Engine | Approach |
|--------|----------|
| Pie | `bench/best-of-n` inferlet — fill prefix once, fork KV cache 4 times via `export_resources`, generate in parallel, score, return best |
| vLLM | 4 parallel `/v1/completions` with `best_of=4` or 4 requests (APC may share prefix) |
| SGLang | `fork` into 4 branches, RadixAttention shares prefix automatically |
| TGI | 4 parallel requests |

**Workload:** 2K token prefix, 256 token generations, N=4.

**Primary metric:** Wall time, total prefill tokens (pie/SGLang should prefill the 2K prefix once; vLLM/TGI may prefill it 4 times)
**Secondary:** GPU memory peak

### Test 2C — Constrained Retry

**Task:** Generate JSON output. If output fails validation (bracket matching), roll back to the point of failure and retry with different sampling. Up to 5 attempts.

| Engine | Approach |
|--------|----------|
| Pie | `bench/constrained-retry` inferlet — generates token-by-token, on constraint failure uses `import_resources` to restore KV cache to branch point, retries |
| vLLM | Restart generation from scratch each retry (or use outlines integration) |
| SGLang | Regex-constrained decoding via `gen(regex=...)`, or retry from scratch |
| TGI | Restart from scratch each retry |

**Workload:** 1K token prompt, target 512 token JSON output, intentionally tricky schema that triggers ~3 retries on average.

**Primary metric:** Wall time, total prefill tokens
**Secondary:** Number of retries needed (should be equal across engines — same model, same task), wasted generation tokens

---

## Tier 3: Tree-Structured Generation

**Goal:** Pie vs SGLang head-to-head on branching workloads. vLLM/TGI are expected to fall behind here — this tier is about proving pie matches or beats SGLang's fork/join.

### Test 3A — Tree-of-Thought

**Task:** Solve a multi-step reasoning problem. At each level, branch 3 ways, generate 128 tokens per branch, score, prune to top 2. Three levels deep.

```
Level 0:  [root prompt]
Level 1:  branch 3 → score → keep 2
Level 2:  branch 3 each (6 total) → score → keep 2
Level 3:  branch 3 each (6 total) → score → keep 1 (final answer)
```

| Engine | Approach |
|--------|----------|
| Pie | `bench/tree-of-thought` inferlet — explicit KV cache fork/prune at each level |
| SGLang | Nested `fork`/`join` with RadixAttention |
| vLLM | External orchestration, N separate requests per level |
| TGI | External orchestration, N separate requests per level |

**Total generations:** 3 + 6 + 6 = 15, all sharing progressively longer prefixes.

**Primary metric:** Wall time, total prefill tokens
**Secondary:** GPU memory peak (KV cache pressure from 15 active branches)

### Test 3B — Parallel Exploration with Shared Context

**Task:** Given a long document (4K tokens), answer 8 different questions about it in parallel. Each answer is a separate generation sharing the same document prefix.

| Engine | Approach |
|--------|----------|
| Pie | `bench/parallel-qa` inferlet — fill document prefix once, fork 8 times, generate answers in parallel |
| SGLang | `fork` into 8 branches after shared prefix |
| vLLM | 8 parallel requests (APC may cache the prefix) |
| TGI | 8 parallel requests |

**Primary metric:** Wall time, total prefill tokens (pie/SGLang: 4K once; vLLM/TGI: up to 4K × 8)
**Secondary:** Throughput, TTFT per question

---

## Tier 4: Resource Control

**Goal:** Capabilities unique to pie. No direct competitor comparison — this is about demonstrating what's possible.

### Test 4A — Pinned Prefix Stability

**Task:** Serve 100 sequential requests that share a 2K-token system prompt + few-shot examples. Measure whether TTFT stays stable (prefix always cached) or degrades (prefix evicted and recomputed).

| Engine | Approach |
|--------|----------|
| Pie | `bench/pinned-prefix` inferlet — `export_resources` + pin on first request, `import_resources` on subsequent requests. Guaranteed no recomputation. |
| vLLM | APC — automatic, but prefix may be evicted under memory pressure |
| SGLang | RadixAttention — automatic, same eviction risk |
| TGI | Prefix caching if supported, otherwise recomputed |

**Workload:** 2K prefix, 128 token responses, 100 requests in sequence, run at 70% GPU memory utilization to create eviction pressure.

**Primary metric:** TTFT over time (should be flat for pie, may spike for others as cache is evicted)
**Secondary:** Total prefill tokens across 100 requests

### Test 4B — Memory Pressure Graceful Degradation

**Task:** Gradually increase concurrent requests until GPU memory is exhausted. Measure how each engine handles it.

**Pie inferlet:** `std/text-completion`
**Workload:** 2K input, 256 output, concurrency ramping from 1 to engine failure.

**Primary metric:** Max concurrency before failure/OOM
**Secondary:** Latency curve as memory fills, error behavior (graceful queue vs crash)

---

## Inferlet Build List

| Inferlet | Tier | Complexity | Core SDK APIs Used |
|----------|------|------------|-------------------|
| `std/text-completion` | 1 | Exists | — |
| `std/chat` | 1 | Exists | — |
| `bench/chain-of-gen` | 2A | Low | `generate()` × 3 |
| `bench/best-of-n` | 2B | Medium | `generate()`, `export_resources`, `import_resources`, log-prob scoring |
| `bench/constrained-retry` | 2C | Medium | `forward_pass`, `import_resources`, JSON validation |
| `bench/tree-of-thought` | 3A | Medium | `generate()`, `export_resources`, `import_resources`, scoring/pruning |
| `bench/parallel-qa` | 3B | Low | `generate()`, `export_resources`, fork pattern |
| `bench/pinned-prefix` | 4A | Low | `export_resources`, `import_resources`, pin |

6 new inferlets to build. All are compositions of the same 4-5 SDK primitives.

---

## Reporting

For each test, produce:

1. **Table:** Engine × metric, with median and p95 values
2. **Chart: Throughput curve** (Tier 1B) — tok/s vs concurrency, one line per engine
3. **Chart: Prefill efficiency** (Tiers 2-3) — bar chart of total prefill tokens per engine per task. This is the cost story.
4. **Chart: TTFT stability** (Tier 4A) — TTFT over 100 sequential requests, one line per engine
5. **Summary: Redundancy ratio** — `total prefill tokens / minimum necessary prefill tokens` per engine. Pie should approach 1.0; vLLM/TGI will be higher.

The redundancy ratio is the single most compelling number. It directly translates to "for every dollar pie spends on compute, engine X spends $Y."
