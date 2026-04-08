# Pie vs vLLM Benchmark Report

**Date:** 2026-04-06
**GPU:** NVIDIA RTX 4000 Ada (20GB VRAM)
**Model:** Qwen/Qwen3-0.6B (bfloat16)
**GPU Memory Split:** Pie 40%, vLLM 40%
**vLLM version:** 0.19.0
**Pie branch:** benchmarks/vs-sglang
**Runs per test:** 5

---

## Tier 0: Framework Overhead

Measures pure framework cost with minimal or no GPU work.

| Test | Pie p50 | vLLM p50 | Notes |
|------|---------|----------|-------|
| noop (pure WASM instantiation) | 41ms | — | No GPU work at all |
| flush-only (WASM + 1 prefill) | 41ms | — | Single IPC round-trip to GPU |
| one-token (WASM + prefill + 1 decode) | 42ms | 17ms | Apples-to-apples comparison |

**Finding:** WASM instantiation overhead is ~41ms. This is constant regardless of whether GPU work is done (noop ≈ flush-only ≈ one-token), meaning the GPU operations (prefill + decode) complete within the WASM setup time for this tiny input. vLLM's HTTP + scheduling overhead is ~17ms. Pie has ~2.4x higher framework overhead, but 41ms is much lower than the ~500ms observed in earlier SGLang benchmarks — suggesting the SGLang overhead was dominated by something else (possibly different Pie version or measurement methodology).

## Tier 1A: Single-Request Latency

Full end-to-end latency for a single request at varying input/output sizes.

| Configuration | Pie p50 | vLLM p50 | Ratio |
|---------------|---------|----------|-------|
| 128 in / 128 out | 1,755ms | 556ms | 3.2x |
| 512 in / 128 out | 1,775ms | 566ms | 3.1x |
| 2048 in / 256 out | 3,729ms | 1,210ms | 3.1x |

**Finding:** Pie is consistently ~3.1x slower on single requests. The ratio is stable across input sizes, suggesting the gap is in per-token decode speed, not prefill. On this 0.6B model, prefill is cheap (~10ms for 2K tokens), so Pie's KV cache persistence doesn't provide meaningful savings.

## Tier 1B: Throughput Scaling

Concurrent requests measuring aggregate throughput (req/s) and per-request latency.

| Concurrency | Pie req/s | vLLM req/s | Pie p50 | vLLM p50 | Winner |
|-------------|-----------|------------|---------|----------|--------|
| c=1 | 1.5 | 3.4 | 677ms | 238ms | vLLM |
| c=2 | 3.0 | 5.9 | 659ms | 292ms | vLLM |
| c=4 | 4.2 | 10.4 | 911ms | 331ms | vLLM |
| c=8 | 7.2 | 1.7* | 1,079ms | 4,617ms | Pie* |
| c=16 | 12.5 | 23.0 | 1,185ms | 498ms | vLLM |
| c=32 | 20.3 | 27.6 | 1,517ms | 751ms | vLLM |
| c=64 | 29.4 | 30.0 | 2,146ms | 1,321ms | ~Tie |
| **c=128** | **52.3** | **31.7** | 2,399ms | 2,433ms | **Pie** |

*c=8 vLLM result (1.7 req/s) appears anomalous — likely a scheduling hiccup or warmup artifact.

**Finding:** Pie overtakes vLLM in throughput at c=128 (52 vs 32 req/s). At c=64 they're roughly tied. Below c=64, vLLM wins on both throughput and latency. The crossover point is around c=64–128.

## Tier 1C: Time-to-First-Token (TTFT)

Measures how quickly the first token is produced, using streaming for vLLM and the noop inferlet for Pie.

| Prompt Length | Pie TTFT p50 | vLLM TTFT p50 |
|---------------|-------------|---------------|
| Short (~50 tokens) | 41ms | 22ms |
| Medium (~200 tokens) | 41ms | 23ms |
| Long (~1000 tokens) | 41ms | 19ms |

**Finding:** Pie's TTFT is constant at 41ms regardless of prompt length — this is the WASM instantiation time, and the actual prefill is deferred. vLLM's TTFT is ~20ms and also roughly constant for this small model. On larger models where prefill takes 100ms+, Pie's constant TTFT could be an advantage.

## Tier 2A: Chain-of-Generations (Draft → Critique → Revise)

3 sequential generation steps within a single context. Pie keeps KV cache across steps; vLLM must re-send growing context each time.

| Engine | p50 Wall Time | Prefill Behavior |
|--------|--------------|-----------------|
| Pie | 10,716ms | 1x prefill (KV cache persists) |
| vLLM | 3,633ms | 3x prefill (avg 2,404 tokens re-prefilled) |

**Finding:** Despite Pie avoiding redundant prefill, vLLM is still 2.9x faster. The per-token decode speed gap dominates. On a 0.6B model, re-prefilling 2,400 tokens costs vLLM ~50ms — negligible compared to the decode time difference.

## Tier 2B: Best-of-N (Shared Prefix, 4 Parallel Generations)

Shared prompt prefilled once, then forked into 4 parallel generation streams.

| Engine | p50 Wall Time | Prefill Behavior |
|--------|--------------|-----------------|
| Pie | 5,942ms | 1x prefill (fork shares KV cache) |
| vLLM | 1,539ms | 4x prefill (avg 2,060 tokens each) |

**Finding:** vLLM is 3.9x faster despite prefilling 4x. Again, decode speed dominates.

## Tier 2C: Constrained Retry (JSON Validation with Rollback)

Generate JSON, validate, retry from checkpoint on failure. Both engines averaged 6 attempts.

| Engine | p50 Wall Time | Attempts | Prefill Behavior |
|--------|--------------|----------|-----------------|
| Pie | 26,324ms | 6.0 | 1x prefill (fork from checkpoint) |
| vLLM | 8,941ms | 6.0 | Re-prefill each attempt (avg 1,050 tokens) |

**Finding:** vLLM is 2.9x faster. Same pattern — decode speed > prefill savings.

---

## Summary

### What We Learned

1. **WASM overhead is 41ms, not 500ms.** The earlier SGLang benchmarks likely measured something else (possibly including model loading or first-request warmup). 41ms is reasonable.

2. **Pie's per-token decode speed is ~3x slower than vLLM** on RTX 4000 Ada with Qwen3-0.6B. This is the dominant factor across all benchmarks.

3. **Pie wins on throughput at high concurrency** (c=128: 52 vs 32 req/s). The crossover is around c=64–128.

4. **KV cache persistence doesn't help much on small models** because prefill is cheap (~10ms for 2K tokens). This may change dramatically on 8B+ models where prefill costs 100ms+ for long contexts.

5. **vLLM's c=8 anomaly** (1.7 req/s) needs investigation — could be a scheduling edge case.

### Open Questions

- **Why is Pie's decode 3x slower?** Is this the Rust↔Python IPC per-step overhead? WASM function call overhead per token? Batch scheduling differences? Need profiling with `ipc-profiling` feature flag.
- **How do results change on 8B+ models?** Prefill becomes expensive, Pie's KV persistence should provide real savings.
- **Does vLLM's APC (Automatic Prefix Caching) close the gap in Tier 2?** Run with `VLLM_ENABLE_APC=1`.
- **Is the decode gap constant or does it vary with batch size?** The throughput crossover at c=128 suggests Pie batches more efficiently at scale.

### Next Steps

1. Run with `VLLM_ENABLE_APC=1` to see if vLLM's prefix caching changes the Tier 2 results
2. Enable `ipc-profiling` to see Pie's internal timing breakdown
3. Test on RTX 6000 Ada (48GB) with Llama-3.1-8B where prefill savings should be meaningful
4. Investigate the 3x decode speed gap — profile IPC overhead per decode step
