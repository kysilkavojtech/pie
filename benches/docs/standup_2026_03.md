# Pie vs SGLang Benchmark — Standup Notes

**Hardware:** RTX 4000 Ada (16 GB) | **Model:** Qwen3-0.6B | **5 runs per test, with warmup**

---

## What I Did

- Built a benchmark harness comparing Pie against SGLang across 5 test scenarios
- Wrote 3 custom WASM inferlets (chain-of-gen, best-of-n, constrained-retry) to exercise Pie's KV cache features
- Fixed several inferlet bugs (seed token panic in constrained-retry, unused variables in chain-of-gen)
- Iterated on the 2C (constrained retry) prompt and validator through 3 rounds to get realistic retry behavior
- Deployed and ran everything on RunPod

## Why These Benchmarks

- **Tier 1A (single request):** Baseline — how much overhead does Pie add on a simple request?
- **Tier 1B (throughput):** Does Pie's batch scheduler compensate for per-request overhead under load?
- **Tier 2A (chain of generations):** Multi-turn conversation — Pie keeps KV cache across steps, SGLang re-prefills the growing history each call
- **Tier 2B (best-of-N):** Shared prefix with parallel branches — Pie uses fork() (copy-on-write KV cache), SGLang prefills N times
- **Tier 2C (constrained retry):** Generate JSON, validate, retry on failure — Pie forks from checkpoint, SGLang re-prefills from scratch each attempt

## Key Results

| Test | SGLang p50 | Pie p50 | Pie advantage |
|------|-----------|---------|---------------|
| 1A (128 in/128 out) | 570 ms | 1,714 ms | SGLang ~3x faster |
| 1B (c=32 throughput) | 19.8 req/s | **23.6 req/s** | **Pie 19% higher throughput** |
| 2A (3-step chain) | 3,750 ms | 10,943 ms | SGLang ~3x faster |
| 2B (best-of-4) | 1,428 ms | 4,593 ms | SGLang ~3x faster |
| 2C (constrained retry, 1.2 attempts) | 1,341 ms | 6,720 ms | SGLang ~5x faster |

## Analysis

- **Pie has ~500-600ms WASM instantiation overhead per request** — this is the dominant cost gap. Every request creates a fresh Store, linker, and component instance (isolation model). SGLang has no equivalent cost.
- **Pie's prefill savings work as designed** — in 2A, Pie prefills ~2K tokens once vs SGLang's ~3.2K cumulative. But on a 0.6B model, prefilling 2K tokens costs ~10-20ms, so savings are invisible in wall time.
- **SGLang's RadixAttention partially neutralizes Pie's advantage** — SGLang automatically caches shared prefixes between sequential calls, reducing redundant prefill. This means Pie's fork() advantage over SGLang is smaller than over a naive stateless API.
- **Throughput crossover at c=32** — the most interesting result. Under concurrent load, Pie's batch scheduler keeps the GPU fed while WASM overhead overlaps with GPU work. Pie actually overtakes SGLang here.

## Open Questions / What May Be Wrong With the Benchmarks

- **Model too small** — 0.6B makes prefill nearly free (~10ms for 2K tokens). Pie's prefill savings become meaningful on larger models where prefilling 2K tokens costs ~100ms+ (8B) or ~500ms (70B)
- **System prompt may still be too short** — ~2K tokens is moderate. Real-world RAG/agentic prompts can be 8-16K tokens, which would dramatically increase SGLang's redundant prefill cost
- **2C retry count too low** — averaging 1.2 retries means the retry mechanism is barely exercised. Need to find a prompt/schema combo that forces ~3-5 retries to properly showcase the fork advantage
- **Per-request overhead may be inflated** — need to profile WASM instantiation breakdown (Store creation vs linker vs dependency instantiation) to understand if there are optimization opportunities
- **Missing TTFT metric** — only measuring wall time. Time-to-first-token would separate prefill latency from decode time and give clearer picture

## Next Steps

- **Test with larger model (8B)** — shifts cost balance toward Pie since prefill becomes expensive
- **Tune 2C difficulty** — find a prompt that produces ~3-5 retries on average
- **Test higher concurrency (64, 128)** — Pie overtook SGLang at c=32, find the ceiling
- **Profile WASM instantiation** — identify which part of the ~500ms overhead is optimizable
- **Design Tier 3 benchmarks** — tree-of-thought (8-15 branches from shared prefix) would compound fork advantage multiplicatively
- **Add TTFT metric** to separate prefill and decode costs
