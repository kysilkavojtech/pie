# Pie Benchmarking Ideas

Comprehensive benchmark suite for Pie's distributed LLM inference engine. Organized into
two categories: **performance benchmarks** (how fast/efficient) and **correctness
benchmarks** (does Pie's architecture preserve correct behavior).

---
---

# Part 1: Performance Benchmarks

## 1.1 Inferlet Overhead

### 1.1.1 WASM Cold Start Latency

**What it tests:** Time from `launch_instance` to the first forward pass completing,
when the WASM component has never been compiled before.

**Why it matters:** Cold start determines the user experience when deploying a new
inferlet for the first time, or after a cache eviction. If compilation takes seconds, it
dominates short-lived inferlet executions and makes interactive workflows sluggish.

**Method:**
1. Clear the compiled component cache (`{cache_dir}/programs/`)
2. Upload a standard inferlet (e.g., `text-completion`)
3. Measure wall-clock time from `launch_instance` call to the first `Stdout` event
   containing a generated token
4. Repeat with inferlets of varying complexity (small hello-world vs large
   tree-of-thought with multiple dependencies)

**Desired results:**
- Cold start should be bounded and predictable (not 10x variance between runs)
- Dependency resolution overhead should scale linearly with dependency count, not
  exponentially
- Report: median, p95, p99 latency in milliseconds

### 1.1.2 WASM Warm Start Latency

**What it tests:** Same as above, but with the WASM component already compiled and
cached via `wasmtime::Component::compile()`.

**Why it matters:** This is the common case in production — the inferlet has been deployed
before. The warm start cost is pure overhead on every invocation. If it's high, even
cached inferlets feel slow.

**Method:**
1. Launch the inferlet once (to populate cache), discard results
2. Measure time from `launch_instance` to first generated token
3. Repeat 100 times, report distribution

**Desired results:**
- Warm start should be significantly faster than cold start (10x+ improvement expected)
- Should be stable (low variance) — no GC pauses, no lock contention on cache lookup
- Target: sub-10ms for simple inferlets, sub-50ms for complex ones

### 1.1.3 Per-Call WIT Boundary Overhead

**What it tests:** How much time the WebAssembly Interface Types (WIT) boundary adds per
host call (forward pass, tokenize, KV allocate, etc.) compared to a native call.

**Why it matters:** Inferlets call host functions thousands of times per generation.
If each call has 100us of overhead, a 1000-token generation adds 100ms of pure WASM tax.
This is the fundamental cost of Pie's programmability model — it needs to be low enough
that users don't feel punished for using inferlets instead of a monolithic server.

**Method:**
1. Write a minimal inferlet that calls `create_forward_pass` + `execute` in a tight loop
   with trivial inputs (1 token, no KV cache)
2. Measure time per forward pass call
3. Compare against the same model inference triggered directly from Rust (bypassing WASM)
4. Isolate: forward pass, tokenize, allocate_resources, deallocate_resources

**Desired results:**
- WIT boundary overhead should be < 1% of a typical forward pass duration
- Tokenize/detokenize overhead should be negligible (< 10us per call)
- Resource allocation overhead should not dominate for small allocations

---

## 1.2 KV Cache Performance

### 1.2.1 Fork Cost

**What it tests:** Time to fork a context (sharing KV cache pages via copy-on-write or
pointer sharing) as a function of the number of shared KV pages.

**Why it matters:** Forking is the mechanism that enables parallel generation,
tree-of-thought, beam search, and speculative decoding. If forking a 4K-token context
takes 50ms, tree-of-thought with 10 branches at each level is impractical. The fork
should be near-instantaneous regardless of context length.

**Method:**
1. Build a context with N KV pages (by generating tokens)
2. Measure time to fork (call `export_resources` + `import_resources`, or the SDK's
   `ctx.fork()`)
3. Vary N: 1, 10, 100, 1000, 4000 pages
4. Measure both the fork operation itself and the first forward pass on the forked context

**Desired results:**
- Fork time should be O(1) or O(N) with a very small constant (metadata copy, not data
  copy)
- First forward pass on forked context should have same latency as on the original
- No memory spike during fork (should share, not copy)

### 1.2.2 Prefix Cache Hit Rate Under Load

**What it tests:** When N concurrent inferlet instances share a common system prompt, how
effectively does Pie's prefix caching avoid redundant KV computation?

**Why it matters:** In production, many instances share the same system prompt (e.g.,
"You are a helpful assistant"). Recomputing the KV cache for this prefix on every instance
wastes GPU compute. Prefix caching is the optimization — but it can fail under
concurrency (race conditions, cache eviction, hash collisions).

**Method:**
1. Define a long system prompt (1000+ tokens)
2. Launch N instances concurrently (N = 1, 2, 4, 8, 16, 32, 64) with the same system
   prompt but different user messages
3. Measure: total forward pass calls, total tokens computed, GPU time
4. Compare against baseline (no prefix caching, or N=1 sequential)

**Desired results:**
- With prefix caching: system prompt KV should be computed once, reused N-1 times
- Total GPU time should scale with `system_prompt_tokens + N * user_message_tokens`,
  not `N * (system_prompt_tokens + user_message_tokens)`
- Report: cache hit rate (%), compute savings (%), memory savings (%)

### 1.2.3 KV Page Allocation Churn

**What it tests:** Performance of the resource manager under rapid allocation/deallocation
cycles. Measures overhead and fragmentation.

**Why it matters:** Short-lived inferlets (e.g., single-shot completions) allocate KV
pages, use them briefly, and deallocate. Under high request rates, the allocator handles
thousands of alloc/dealloc cycles per second. Fragmentation or allocator lock contention
would degrade throughput.

**Method:**
1. In a tight loop: allocate K pages, immediately deallocate, repeat 10,000 times
2. Measure: time per cycle, memory usage over time (should be flat)
3. Vary K: 1, 10, 100 pages per cycle
4. Run with concurrent instances doing the same

**Desired results:**
- Allocation time should be constant (no degradation over 10K cycles)
- Memory usage should be flat (no fragmentation-induced growth)
- No lock contention visible under concurrent allocation

---

## 1.3 Scheduling & Batching

### 1.3.1 Batch Formation Latency

**What it tests:** Time from a request arriving at the scheduler to it being dispatched
as part of a batch to the Python backend via `fire_batch`.

**Why it matters:** The scheduler in `model.rs` accumulates requests and dispatches them
as batches. There's a trade-off: waiting longer forms bigger batches (higher throughput)
but adds latency. This benchmark measures the actual wait time under various loads.

**Method:**
1. Send requests at controlled rates: 1/s, 10/s, 100/s, 1000/s
2. Timestamp each request at submission and at batch dispatch
3. Measure: batch wait time, batch size, dispatch frequency

**Desired results:**
- At low load: batch wait time should be minimal (single request dispatched immediately)
- At high load: batches should form quickly and reach the configured max batch size
- Report: batch wait time distribution, average batch size vs request rate

### 1.3.2 Throughput vs Concurrency Curve

**What it tests:** Tokens per second as concurrent inferlet instances scale from 1 to N.

**Why it matters:** This is the primary scaling benchmark. Pie's value proposition is
efficient multi-tenant inference. If throughput doesn't scale with concurrency (or worse,
decreases), the scheduling and batching are not working.

**Method:**
1. Launch N concurrent instances, each generating 256 tokens from the same prompt
2. Measure total tokens/sec across all instances
3. Measure per-instance latency (time to first token, inter-token latency)
4. Vary N: 1, 2, 4, 8, 16, 32, 64, 128
5. Run on single GPU and multi-GPU (TP=2, TP=4)

**Desired results:**
- Throughput should increase with N up to GPU saturation
- After saturation: throughput plateaus, per-instance latency increases linearly
- No throughput cliff (sudden drop at some N)
- Report: throughput curve, latency curve, saturation point

### 1.3.3 Priority Scheduling

**What it tests:** Whether the `queue.set_priority()` API actually delivers lower latency
for high-priority requests under contention.

**Why it matters:** Priority scheduling is advertised as a feature. If it doesn't work
(e.g., priorities are ignored by the batch scheduler, or high-priority requests are
starved by a large low-priority batch already in flight), users can't rely on it for
latency-sensitive workloads.

**Method:**
1. Saturate the system with 64 low-priority instances generating long outputs
2. Inject a high-priority instance requesting a short completion
3. Measure: time-to-first-token for the high-priority instance vs a low-priority instance
   injected at the same time
4. Vary the load level and the ratio of high:low priority requests

**Desired results:**
- High-priority TTFT should be significantly lower than low-priority under contention
- High-priority should not starve low-priority completely (fairness)
- Report: TTFT ratio (high vs low priority), fairness metrics

---

## 1.4 IPC Boundary (Rust <-> Python)

### 1.4.1 fire_batch Round-Trip vs Batch Size

**What it tests:** How `fire_batch` IPC round-trip time scales with batch size and tensor
dimensions.

**Why it matters:** Every forward pass goes through the IPC boundary: Rust serializes the
batch request, sends it to Python, Python runs the model, serializes the response, sends
it back. If serialization cost scales poorly with batch size, it becomes the bottleneck
before GPU compute does.

**Method:**
1. Send `fire_batch` with batch sizes: 1, 2, 4, 8, 16, 32, 64
2. Measure: total round-trip time, serialization time, deserialization time, model
   compute time (by instrumenting the Python side)
3. Vary sequence length: 128, 512, 2048 tokens

**Desired results:**
- Serialization overhead should be < 5% of total round-trip at typical batch sizes
- Round-trip should be dominated by model compute, not IPC overhead
- Linear scaling with batch size (no superlinear blowup from serialization)

### 1.4.2 Multi-Group IPC Scaling

**What it tests:** IPC overhead per group as the number of DP groups increases (with
TP > 1).

**Why it matters:** Each DP group has its own IPC channel and Python worker. If the Rust
scheduler has per-group overhead that scales with group count (e.g., sequential dispatch,
lock contention), multi-GPU setups won't scale.

**Method:**
1. Configure TP=1 with varying DP groups: 1, 2, 4, 8
2. Send equal load to each group
3. Measure: per-group dispatch latency, total throughput, scheduler overhead

**Desired results:**
- Per-group latency should be independent of group count
- Total throughput should scale linearly with group count
- No scheduler bottleneck visible

---

## 1.5 Advanced Inference Patterns

### 1.5.1 Speculative Decoding

**What it tests:** Acceptance rate, wall-clock speedup, and overhead of speculative
decoding (as implemented in the `cacheback-decoding` example).

**Why it matters:** Speculative decoding trades compute for latency — a draft model
proposes tokens that the main model verifies in parallel. The speedup depends on the
acceptance rate. If the draft model is poor or the verification overhead is high, it can
be slower than vanilla autoregressive.

**Method:**
1. Run the same prompt through vanilla autoregressive and speculative decoding
2. Measure: tokens/sec, acceptance rate, average accepted length per step
3. Vary: draft model size, speculation length (k), prompt domain (code vs prose)

**Desired results:**
- Speedup of 1.5-3x over vanilla autoregressive for well-matched draft models
- Acceptance rate > 70% for in-domain prompts
- No quality degradation (output distribution should be identical to vanilla)
- Report: speedup, acceptance rate, overhead breakdown

### 1.5.2 Tree-of-Thought Efficiency

**What it tests:** Total tokens generated vs useful tokens, and wall-clock time vs
equivalent sequential generation.

**Why it matters:** Tree-of-thought explores multiple reasoning paths in parallel. Pie's
KV cache forking should make this efficient (shared prefix computation). But if the
forking overhead or scheduling of many concurrent branches is high, the wall-clock benefit
disappears.

**Method:**
1. Run `tree-of-thought` inferlet with 3 levels, branching factor 3
2. Measure: total tokens generated across all branches, tokens in the selected
   (best) path, wall-clock time
3. Compare against: sequential exploration (one path at a time, same total tokens)
4. Vary: branching factor (2, 3, 5), depth (2, 3, 4)

**Desired results:**
- Wall-clock time should be significantly less than sequential (ideally close to
  single-path time, since branches run in parallel)
- KV cache memory should grow with unique suffixes, not total branches
  (prefix sharing)
- Report: wall-clock speedup, compute efficiency (useful tokens / total tokens),
  memory efficiency

### 1.5.3 Parallel Generation Scaling

**What it tests:** Throughput when forking a single context into N parallel branches.

**Why it matters:** This is the core use case for KV cache sharing — one system prompt,
N different completions. It should scale well because the shared prefix is computed once.

**Method:**
1. Create a context with a 500-token system prompt
2. Fork into N branches (N = 1, 2, 4, 8, 16, 32)
3. Each branch generates 128 tokens
4. Measure: total tokens/sec, per-branch latency, KV memory usage

**Desired results:**
- Throughput should scale close to linearly with N (since prefix is shared)
- KV memory should be: `prefix_pages + N * suffix_pages`, not `N * (prefix + suffix)`
- Per-branch latency should remain stable (no starvation)

### 1.5.4 Beam Search vs Sequential

**What it tests:** The inferlet-based beam search against a naive sequential
implementation.

**Why it matters:** Beam search requires maintaining K beams, each with their own KV
cache. Pie's resource sharing (forking, prefix caching) should make this significantly
more memory-efficient than K independent generations.

**Method:**
1. Run beam search with beam width K = 1, 2, 4, 8, 16
2. Measure: wall-clock time, peak KV memory, tokens/sec
3. Compare against K independent sequential generations
4. Report memory savings from shared prefixes

**Desired results:**
- Peak memory should be sublinear in K (shared prefixes)
- Wall-clock time should be sublinear in K (batched forward passes)
- Beam quality (final score) should be monotonically non-decreasing with K

---

## 1.6 Multi-Model

### 1.6.1 Model Switching Latency

**What it tests:** Time for an inferlet to switch between two registered models via
`get_model(name)`.

**Why it matters:** Some workflows require multiple models (e.g., a small model for
classification, a large model for generation). If switching has high latency (e.g., model
needs to be loaded/unloaded), pipeline inferlets become impractical.

**Method:**
1. Register two models
2. In an inferlet: call `get_model("model_a")`, do a forward pass, call
   `get_model("model_b")`, do a forward pass. Repeat 100 times.
3. Measure: time for each `get_model` call and each forward pass

**Desired results:**
- `get_model` should be near-instantaneous (it's a lookup, not a load)
- Forward pass latency should be independent of which model was used previously
- No memory spike from switching

### 1.6.2 Cross-Model Pipeline

**What it tests:** End-to-end latency when an inferlet chains two models sequentially
(e.g., embedding model then LLM).

**Why it matters:** Multi-model pipelines are a key use case. The overhead of the pipeline
(IPC round-trips, queue switching, resource management) should be small compared to actual
model compute.

**Method:**
1. Register an embedding model and a generation model
2. Run an inferlet that: embeds input with model A, passes embeddings to model B via
   resource export/import, generates output
3. Measure: total pipeline time, per-stage time, IPC overhead between stages
4. Compare against: running each stage separately as independent inferlets

**Desired results:**
- Pipeline overhead (glue between stages) should be < 5% of total time
- Resource export/import should not require data copying (zero-copy or minimal)

---

## 1.7 Stress & Limits

### 1.7.1 Max Concurrent Instances

**What it tests:** The maximum number of concurrent inferlet instances before throughput
degrades significantly or the system becomes unstable.

**Why it matters:** In production, many users connect simultaneously. The system should
degrade gracefully (higher latency, same throughput) rather than catastrophically
(crashes, timeouts, OOM).

**Method:**
1. Launch instances incrementally: 1, 2, 4, 8, 16, 32, 64, 128, 256
2. Each instance generates 64 tokens
3. Measure: aggregate throughput, per-instance TTFT, error rate, memory usage
4. Continue until errors appear or throughput drops below 50% of peak

**Desired results:**
- Throughput should plateau, not cliff
- No crashes or panics at any concurrency level
- Error responses should be clean (`OutOfResources`) not WASM panics
- Report: saturation point, max stable concurrency, degradation curve

### 1.7.2 Memory Pressure & Backpressure

**What it tests:** System behavior when KV cache memory is exhausted.

**Why it matters:** Running out of KV pages is inevitable under high load. The question
is whether Pie handles it gracefully (backpressure, queuing, clean error) or
catastrophically (crash, corruption, silent failure).

**Method:**
1. Determine total KV page capacity
2. Launch instances that each allocate many KV pages (long contexts)
3. Continue launching until allocation fails
4. Measure: what error the inferlet receives, whether existing instances are affected,
   whether freed pages are immediately reusable

**Desired results:**
- New allocations should fail with `OutOfResources`, not crash
- Existing instances should be unaffected (no corruption of their KV caches)
- After an instance exits and frees pages, new allocations should succeed immediately
- No leaked pages after instance cleanup

### 1.7.3 Long Context Per-Token Latency

**What it tests:** How per-token generation latency changes as context length grows.

**Why it matters:** Attention is O(n^2) in context length (or O(n) with flash attention,
but with a growing constant). Users need to know the practical limits — at what context
length does generation become unacceptably slow?

**Method:**
1. Generate tokens with progressively longer prefixes: 128, 256, 512, 1024, 2048,
   4096, 8192, 16384 tokens of context
2. For each prefix length, measure per-token latency for the next 32 tokens
3. Run on different models (small vs large, different attention implementations)

**Desired results:**
- Per-token latency should scale sub-quadratically if flash attention is in use
- Report: latency curve, the context length at which latency exceeds 100ms/token
- Memory usage curve (should be linear in context length with paged KV)

---
---

# Part 2: Correctness Benchmarks

## 2.1 Sampling Correctness

### 2.1.1 Cross-Run Determinism

**What it tests:** Given a fixed seed and temperature, does the same prompt produce
identical output across multiple runs?

**Why it matters:** Non-determinism makes debugging impossible and benchmarking unreliable.
If two runs of the same inferlet with the same seed produce different tokens, there's
uncontrolled randomness somewhere — floating-point non-determinism in the forward pass,
race conditions in the scheduler, or seeding bugs.

**Method:**
1. Fix seed, temperature, and prompt
2. Run the same inferlet 10 times, collect generated token sequences
3. Compare: all 10 should be identical
4. Repeat with different TP degrees (TP=1, TP=2, TP=4) — compare across TP configurations
5. Repeat with different DP groups — compare across groups

**Desired results:**
- Same TP degree: all runs produce identical token sequences
- Different TP degrees: logits may differ within floating-point tolerance, but sampled
  tokens should be identical given the same seed (if the sampling is deterministic on
  the logits)
- If cross-TP determinism is not achievable, document the expected divergence

**Why this can fail:**
- TP all-reduce uses non-deterministic floating-point operations
- Batch padding affects computation for other requests, which can affect numerical results
- CUDA non-determinism in certain operations (atomics, reductions)

### 2.1.2 Distribution Fidelity

**What it tests:** Whether batching, TP sharding, or KV cache sharing shift the output
token distribution.

**Why it matters:** Even if individual runs aren't deterministic, the statistical
distribution should be correct. If TP=2 consistently produces lower-entropy output than
TP=1, the sharding is introducing a systematic bias. This is subtle and dangerous —
the model "works" but its quality is silently degraded.

**Method:**
1. Choose 10 prompts where the next token has high entropy (multiple plausible
   continuations)
2. For each prompt, sample the next token 1000 times at temperature=1.0
3. Collect the empirical distribution over the vocabulary
4. Run under different configurations: single-GPU baseline, TP=2, TP=4, batch_size=1
   vs batch_size=32
5. Compare distributions using KL divergence or chi-squared test

**Desired results:**
- KL divergence between configurations should be < 0.01 nats
- No statistically significant difference (p > 0.05 on chi-squared)
- If systematic bias exists, quantify it and document which configurations are affected

### 2.1.3 Sampler Boundary Cases

**What it tests:** Correctness of top-k, top-p, min-p, and temperature sampling at edge
cases.

**Why it matters:** Sampling bugs are subtle. A top-k implementation that includes k+1
tokens, or a top-p that uses `>=` instead of `>`, can go unnoticed for months because the
output "looks reasonable." These bugs matter for constrained-decoding and
output-validation inferlets that depend on exact probability semantics.

**Method:**
1. Craft logit distributions where the boundary matters:
   - Exactly k tokens with nonzero probability → top-k=k should include all, top-k=k-1
     should exclude exactly one
   - Cumulative probability hits exactly p at token i → top-p should include tokens
     0..i, not 0..i+1
   - Temperature=0 → should always select argmax (greedy)
   - Temperature=infinity → should approach uniform over nonzero-probability tokens
2. Feed these crafted logits through each sampler
3. Verify the token set and probabilities

**Desired results:**
- All boundary cases produce the mathematically correct token set
- Temperature=0 is perfectly greedy (no randomness)
- top-k=1 is equivalent to temperature=0
- top-p=1.0 includes all nonzero-probability tokens
- min-p=0.0 includes all nonzero-probability tokens

---

## 2.2 KV Cache Correctness

### 2.2.1 Fork-Then-Diverge Integrity

**What it tests:** After forking a context, does the shared prefix remain immutable? Does
divergent generation on one fork corrupt the other?

**Why it matters:** Forking is implemented as KV page sharing (not copying). If the
implementation has a copy-on-write bug — or worse, no copy-on-write at all — one fork's
new tokens could overwrite the shared prefix, corrupting all other forks.

**Method:**
1. Create a context, generate 100 tokens (call this the "prefix")
2. Detokenize and save the prefix text
3. Fork into 4 branches
4. On each branch, generate 50 different tokens (using different prompts or seeds)
5. On each branch, detokenize the first 100 tokens (the prefix region)
6. Compare all 4 prefix texts — they must be identical to the saved prefix

**Desired results:**
- All 4 forks produce identical prefix text
- Each fork's suffix is different (confirming they actually diverged)
- No memory corruption or WASM panic during the process

**Why this can fail:**
- KV pages are shared by pointer without copy-on-write
- Copy-on-write triggers incorrectly (copies too late, or not at all)
- Page metadata (sequence position) is shared when it should be per-fork

### 2.2.2 Export/Import Round-Trip

**What it tests:** KV cache pages exported by one instance and imported by another produce
identical generation to a single-instance baseline.

**Why it matters:** Export/import is used for cross-instance cache sharing (e.g., a warm
cache service that pre-computes system prompts). If the serialization is lossy (truncated
floats, wrong byte order, missing metadata), the importing instance will generate
different (wrong) output.

**Method:**
1. Instance A: generate 100 tokens from a prompt, export KV pages as "cache_x"
2. Instance A: generate 50 more tokens, record them as the baseline
3. Instance B: import "cache_x", generate 50 tokens from the same state
4. Compare Instance A's and Instance B's 50-token continuations

**Desired results:**
- Token sequences from A and B should be identical (given same seed/temperature)
- If not identical, logits should be within floating-point tolerance
- Import should not take significantly longer than a fresh forward pass of the same
  prefix

### 2.2.3 Prefix Cache Collision

**What it tests:** Two different prompts that share a long common prefix diverge correctly
at the first differing token.

**Why it matters:** Prefix caching identifies shared prefixes by their token content. A
hash collision or off-by-one in prefix matching would cause two different prompts to share
KV cache past their actual divergence point, producing identical (wrong) output for
different inputs.

**Method:**
1. Construct two prompts that share the first 500 tokens but differ at token 501
2. Run both through the system with prefix caching enabled
3. Verify that outputs diverge at or after the first differing token
4. Repeat with many prompt pairs to stress the hash function

**Desired results:**
- Outputs diverge exactly where the prompts diverge
- No false sharing (identical output past the divergence point)
- Prefix cache correctly identifies the shared prefix length

### 2.2.4 Eviction Under Pressure

**What it tests:** When KV cache is full and pages are evicted, evicted resources are
truly gone — not stale, not accessible, not corrupting new allocations.

**Why it matters:** If evicted pages are still accessible (stale pointer), an inferlet
could read garbage data as if it were valid KV cache. If the freed memory is reused but
the old reference isn't invalidated, two inferlets could share memory they don't intend
to.

**Method:**
1. Fill KV cache to capacity with instance A
2. Instance A exports pages as "my_cache"
3. Instance A exits (pages freed)
4. Launch instance B, allocate new pages (which reuse the freed physical memory)
5. Attempt to import "my_cache" from instance B

**Desired results:**
- Import should fail cleanly (resource not found or expired)
- Instance B's newly allocated pages should contain no data from instance A
- No crash, no silent data corruption

---

## 2.3 Forward Pass Determinism

### 2.3.1 Batch Position Independence

**What it tests:** Whether the same request produces identical logits regardless of its
position in a batch or the batch size.

**Why it matters:** Batched inference uses padding and attention masks to process multiple
requests simultaneously. If the padding scheme or mask computation has a bug, a request's
logits will differ depending on what other requests are in the batch. This is a silent
quality degradation — the model "works" but gives different answers depending on server
load.

**Method:**
1. Send the same request as:
   - The only request in a batch of 1
   - Position 0 in a batch of 8
   - Position 7 in a batch of 8
   - Position 3 in a batch of 8 where all other requests have different lengths
2. Collect raw logits (not sampled tokens) for each case
3. Compare logits element-wise

**Desired results:**
- Logits should be identical across all configurations (within floating-point tolerance,
  e.g., relative error < 1e-5)
- If tolerance is exceeded, document which operations cause the divergence
- Sampled tokens (given same seed) should always be identical

### 2.3.2 TP Consistency

**What it tests:** Whether the same request produces equivalent logits under TP=1 vs
TP=2 vs TP=4.

**Why it matters:** Tensor parallelism splits model weights across GPUs and uses
all-reduce to combine results. Floating-point addition is not associative — different
reduction orders produce different results. The question is whether this divergence is
within acceptable bounds or large enough to affect generation quality.

**Method:**
1. Run the same prompt under TP=1 (baseline), TP=2, TP=4
2. Collect full logit distributions for the first 10 tokens
3. Compare: max absolute error, mean absolute error, KL divergence of softmax
   distributions
4. Sample 100 tokens under each configuration (fixed seed), compare token sequences

**Desired results:**
- Max absolute logit error < 1e-3 (fp16) or < 1e-5 (fp32)
- KL divergence of softmax distributions < 0.001
- Token sequences should be identical for at least the first ~50 tokens (divergence
  may accumulate)
- Document the expected error bounds for each TP configuration

---

## 2.4 Tokenizer Correctness

### 2.4.1 Round-Trip Fidelity

**What it tests:** `detokenize(tokenize(s)) == s` for a comprehensive test corpus.

**Why it matters:** Tokenizer bugs are insidious. A single wrong token ID can shift the
entire generation. Round-trip failures mean the model sees different text than what the
user wrote. These bugs are especially common with Unicode, whitespace handling, and
special characters.

**Method:**
1. Test corpus:
   - ASCII: English text, code, JSON, XML, markdown
   - Unicode: CJK, Arabic, Devanagari, emoji, mixed scripts
   - Whitespace: tabs, multiple spaces, leading/trailing whitespace, empty string
   - Edge cases: single character, single token, maximum-length input
   - Special: null bytes, control characters, BOM
2. For each string: `assert detokenize(tokenize(s)) == normalize(s)`
3. Document which normalizations are expected (e.g., NFC normalization)

**Desired results:**
- All ASCII round-trips should be exact
- Unicode round-trips should be exact after documented normalization
- Empty string should round-trip to empty string
- No crashes on any input (including adversarial inputs)

### 2.4.2 Cross-Instance Consistency

**What it tests:** Two different inferlet instances tokenize the same string identically.

**Why it matters:** If the tokenizer has per-instance state (e.g., a cache that affects
behavior, or a lazy initialization that's non-deterministic), different instances could
tokenize the same input differently. This would cause KV cache prefix sharing to silently
produce wrong results (the prefix tokens don't match even though the prefix text does).

**Method:**
1. Launch 10 inferlet instances
2. Each instance tokenizes the same 100 strings
3. Collect token ID sequences from all instances
4. Compare all pairs

**Desired results:**
- All 10 instances produce identical token ID sequences for every string
- No instance-dependent state affects tokenization

---

## 2.5 Inferlet Isolation

### 2.5.1 State Leakage Between Instances

**What it tests:** A new inferlet instance sees no artifacts from a previously-run
instance.

**Why it matters:** WASM provides memory isolation, but Pie's host-side resources (KVS,
KV cache, exported resources, model state) are shared. If cleanup is incomplete, a new
instance could see stale data — a security issue (data leakage between tenants) and a
correctness issue (wrong state).

**Method:**
1. Instance A: write to KVS (`store_set("key", "secret")`), allocate KV pages, export
   as "shared_name"
2. Instance A exits
3. Instance B (same inferlet, fresh launch): try `store_get("key")`, try
   `import_resources("shared_name")`
4. Both should return None/fail

**Desired results:**
- KVS: `store_get("key")` returns `None`
- Resources: `import_resources("shared_name")` fails (not found)
- No KV pages from A are accessible to B
- WASM linear memory is zeroed (no data from A visible in B's memory)

### 2.5.2 Concurrent KVS Mutation

**What it tests:** Two instances writing to the same KVS key simultaneously produce
consistent results (no corruption, one writer wins).

**Why it matters:** The KVS is a shared persistent store. Without proper synchronization,
concurrent writes can corrupt values (partial writes, torn reads). Even if the KVS is
per-instance, the API contract needs to be clear and tested.

**Method:**
1. Launch 10 instances simultaneously
2. Each writes `store_set("counter", str(instance_id))` in a loop, 100 times
3. After all finish, read "counter"
4. Also: each instance reads "counter" between writes and checks it's a valid value
   (not garbled)

**Desired results:**
- Final value is one of the 10 valid instance IDs (not corrupted)
- No read ever returns a partial/garbled value
- If KVS is per-instance (not shared), document that — writes from one instance should
  not be visible to another

### 2.5.3 Resource Leak Detection

**What it tests:** KV pages allocated by an inferlet that exits without deallocating are
reclaimed by the runtime.

**Why it matters:** If resources leak, the system slowly runs out of KV pages. After
enough inferlet launches, no new instances can allocate resources. This is especially
dangerous for short-lived inferlets that are launched thousands of times.

**Method:**
1. Record initial free KV page count
2. Launch an inferlet that allocates 100 KV pages and exits without deallocating
3. Wait for instance cleanup
4. Record free KV page count — should be back to initial
5. Repeat 100 times in a loop
6. Final free count should equal initial free count

**Desired results:**
- Free page count returns to initial after each instance exits
- No degradation over 100 iterations
- Report: pages leaked per iteration (should be 0)

---

## 2.6 Output Quality & Behavioral

### 2.6.1 Chat Template Fidelity

**What it tests:** The SDK's chat formatter produces the exact prompt format expected by
each supported model.

**Why it matters:** Each model family (Llama, Qwen, Mistral, etc.) has a specific chat
template with special tokens, role markers, and formatting rules. A wrong template silently
degrades quality — the model "works" but produces worse output because it sees a prompt
format it wasn't trained on. This is one of the most common sources of quality loss.

**Method:**
1. For each supported model:
   a. Get the reference chat template (from HuggingFace tokenizer config or model card)
   b. Construct a test conversation: system message, user message, assistant message,
      user follow-up
   c. Format it using the SDK: `ctx.fill_system()`, `ctx.fill_user()`,
      `ctx.fill_assistant()`, `ctx.flush()`
   d. Format it using the reference template
   e. Compare byte-for-byte

**Desired results:**
- Exact match for all supported models
- Any divergence is a bug (wrong special tokens, missing BOS/EOS, wrong role markers)
- Report: per-model pass/fail, specific diff on failures

### 2.6.2 Stop Token Handling

**What it tests:** Generation stops correctly when a stop token is produced, and the stop
token is handled according to the API contract (included or excluded from output).

**Why it matters:** Stop token bugs cause two problems: (1) generation doesn't stop,
wasting compute and producing garbage; (2) the stop token leaks into the output,
corrupting downstream processing (e.g., JSON parsing fails because of a trailing `</s>`).

**Method:**
1. Use a prompt that reliably produces a stop token (e.g., a short completion task)
2. Verify generation stops at the stop token
3. Check output: stop token should not be included in the returned text
4. Edge case: what if the model produces the stop token as the first token?
5. Edge case: what if multiple stop tokens are defined?

**Desired results:**
- Generation stops immediately when any stop token is produced
- Stop token is not included in the output text
- First-token stop produces empty output (not an error)
- Works correctly with multi-token stop sequences (if supported)

### 2.6.3 Max Token Limit

**What it tests:** Requesting exactly N tokens produces exactly N tokens.

**Why it matters:** Off-by-one errors in token counting are common and cause subtle
downstream issues. An API that returns 127 tokens when you ask for 128 breaks
applications that depend on exact counts (e.g., filling a fixed-size buffer).

**Method:**
1. Request generation of exactly N tokens for N = 0, 1, 2, 10, 100, 1000
2. Count the returned tokens
3. Verify: returned count == requested count (unless stopped by a stop token first)

**Desired results:**
- N=0: returns empty output, no error
- N=1: returns exactly 1 token
- N=1000: returns exactly 1000 tokens (or fewer if stop token hit, with stop token
  reported)
- No off-by-one in any case

---

## 2.7 End-to-End Behavioral Equivalence

### 2.7.1 Reference Output Comparison

**What it tests:** Given a set of reference prompts with known-good outputs (from a
trusted baseline), do various inferlets produce equivalent quality?

**Why it matters:** This is the ultimate integration test. Individual components may pass
unit tests but the full pipeline (tokenize → KV allocate → forward pass → sample →
detokenize) may introduce subtle quality loss. A reference comparison catches regressions
that no individual benchmark would.

**Method:**
1. Generate a reference corpus: 100 prompts with outputs from a trusted baseline
   (single-GPU, no batching, vanilla autoregressive, known-good tokenizer)
2. Run each prompt through various inferlets: text-completion, beam-search,
   parallel-generation
3. Compare outputs:
   - For greedy (temperature=0): exact token match expected
   - For stochastic: statistical comparison (ROUGE, BLEU, or exact match for factual
     queries like "What is 2+2?")

**Desired results:**
- Greedy: 100% exact match with baseline
- Stochastic: no statistically significant quality difference
- Any divergence is a regression to investigate

### 2.7.2 Strategy A/B Comparison

**What it tests:** More sophisticated inference strategies (beam search, tree-of-thought)
produce equal or better quality than greedy decoding.

**Why it matters:** If beam search produces worse output than greedy, its implementation
is buggy. These strategies are computationally expensive — they must justify their cost
with measurably better results. A broken implementation wastes GPU time while giving
worse answers.

**Method:**
1. Select 50 prompts with objectively evaluable outputs (math problems, code completion,
   factual questions)
2. Run each with: greedy, beam search (k=4), tree-of-thought (depth=3, branch=3)
3. Score outputs by correctness (exact match for math, test pass rate for code)
4. Compare scores across strategies

**Desired results:**
- Beam search score >= greedy score (on average, across all prompts)
- Tree-of-thought score >= greedy score
- If a strategy scores lower, investigate: it's likely a bug, not an inherent limitation
- Report: per-strategy accuracy, with confidence intervals

---

## 2.8 Error Injection & Resilience

### 2.8.1 Malformed Input Tokens

**What it tests:** Sending out-of-vocabulary token IDs through the forward pass produces
a clean error, not a crash or garbage output.

**Why it matters:** In a multi-tenant system, one misbehaving inferlet should not crash
the runtime or affect other instances. If OOV tokens cause a WASM panic or a Python
segfault, the entire server goes down.

**Method:**
1. Write an inferlet that calls `input_tokens` with token IDs beyond the vocabulary size
   (e.g., vocab size is 32000, send token ID 99999)
2. Run it

**Desired results:**
- Inferlet receives a clean error (Result::Err, not a panic)
- Other running instances are unaffected
- Server continues operating normally
- Error message identifies the problem (e.g., "token ID 99999 out of vocabulary range
  0-31999")

### 2.8.2 Client Disconnect Mid-Generation

**What it tests:** When a client disconnects while its inferlet is generating, all
resources are cleaned up correctly.

**Why it matters:** Client disconnects happen constantly in production (network issues,
user closes browser, timeout). If the runtime doesn't clean up KV pages, queue entries,
and instance state, resources leak until the server must be restarted.

**Method:**
1. Launch an inferlet that generates 1000 tokens (takes a while)
2. Record initial resource state (free KV pages, active instances)
3. Disconnect the client WebSocket after 100 tokens
4. Wait for cleanup
5. Record final resource state

**Desired results:**
- Instance is terminated (not left running in the background)
- All KV pages are freed (free count returns to initial)
- No queue entries remain for the terminated instance
- No WASM panic or error in the server log
- Other instances are unaffected

### 2.8.3 OOM Behavior

**What it tests:** What happens when an inferlet requests more KV pages than available.

**Why it matters:** OOM is the most common runtime failure under load. The response must
be clean and predictable — not a crash, not a hang, not corruption of other instances'
memory.

**Method:**
1. Determine total KV page capacity
2. Launch instance A that allocates 90% of pages
3. Launch instance B that tries to allocate 20% of pages (exceeds remaining)
4. Observe instance B's behavior

**Desired results:**
- Instance B receives `OutOfResources` (or equivalent) from `allocate_resources`
- Instance B can handle this error and either retry, reduce allocation, or exit cleanly
- Instance A is completely unaffected
- Freed pages from either instance are immediately available
- No partial allocation (if requesting 100 pages and only 50 are free, the entire
  allocation fails atomically — no partial state)
