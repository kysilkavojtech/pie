# Pie Benchmark Suite

Benchmarks for correctness, performance, and resilience of the Pie inference engine.
All benchmarks connect to a running `pie serve` instance via WebSocket and use the
`text-completion` inferlet.

## Quick Start

```bash
# Build the text-completion inferlet first
cd std/text-completion && cargo build --target wasm32-wasip2 --release && cd ../..

# Start the server
pie serve &

# Run all benchmarks
./benches/run_benchmarks.sh

# Run quick subset (smaller parameters)
./benches/run_benchmarks.sh --quick

# Run a single benchmark
./benches/run_benchmarks.sh --only determinism

# Custom server
./benches/run_benchmarks.sh --server ws://192.168.1.100:8080
```

Results are saved as JSON files in `benches/results/<timestamp>/`.

## Benchmarks

### Correctness

#### `bench_max_tokens.py` — Max Token Limit

Tests that requesting N tokens produces approximately N tokens.

- **What it tests:** Token count accuracy for N = 1, 2, 10, 50, 100, 256
- **Why it matters:** Off-by-one errors in token counting break applications that
  depend on exact output sizes
- **Method:** Sends requests with `--max-tokens N` (greedy, temperature=0), estimates
  output tokens from character count
- **Pass criteria:** Estimated output tokens within 0.3x–3x of requested (wide tolerance
  because we estimate from chars, not actual token IDs)

```bash
python benches/bench_max_tokens.py
```

#### `bench_determinism.py` — Cross-Run Determinism

Tests that greedy decoding produces identical output across multiple runs.

- **What it tests:** Same prompt with temperature=0 run K times → all K outputs identical
- **Why it matters:** Non-determinism makes debugging impossible and benchmarking
  unreliable. Indicates floating-point instability, scheduler race conditions, or seeding
  bugs
- **Method:** Runs 3 different prompts 5 times each (configurable with `--runs`),
  compares all outputs for exact string match
- **Pass criteria:** All runs for each prompt produce identical text
- **What a failure means:** If outputs diverge, the report shows the character position
  and surrounding text where divergence occurs

```bash
python benches/bench_determinism.py --runs 10
```

#### `bench_batch_position.py` — Batch Position Independence

Tests that the same prompt produces identical output regardless of batch size.

- **What it tests:** Launch N identical requests concurrently (so they land in the same
  batch), verify all N outputs are identical
- **Why it matters:** Batched inference uses padding and attention masks. If the
  padding/masking has a bug, a request's output depends on what other requests are in the
  batch — a silent quality degradation
- **Method:** Tests batch sizes 1, 2, 4, 8, 16 with the same prompt (greedy). Compares
  all outputs within each batch for exact match
- **Pass criteria:** All outputs within each batch are identical

```bash
python benches/bench_batch_position.py
```

### Performance

#### `bench_cold_warm_start.py` — WASM Start Latency

Measures launch-to-first-output latency.

- **What it tests:** First launch latency (compilation overhead) vs subsequent launches
  (cached component)
- **Why it matters:** Cold start determines UX for new inferlet deployments. Warm start
  is the overhead on every invocation
- **Method:** Measures 1 first-launch and N warm launches (configurable with
  `--warm-runs`), reports latency distributions
- **Output:** p50/p90/p99 latencies, warm vs cold speedup ratio

```bash
python benches/bench_cold_warm_start.py --warm-runs 20
```

#### `bench_concurrent_scaling.py` — Throughput vs Concurrency

Measures how throughput scales with concurrent instances.

- **What it tests:** Tokens/sec and per-instance latency at concurrency levels 1, 2, 4,
  8, ..., N
- **Why it matters:** Identifies the saturation point and checks for throughput cliffs
  (sudden drops at some concurrency level)
- **Method:** At each level, runs `max(level, requests_per_level)` requests, measures
  aggregate throughput and latency distribution
- **Output:** Scaling table (concurrency vs tok/s, req/s, p50, p99)

```bash
python benches/bench_concurrent_scaling.py --max-concurrency 128 --requests-per-level 32
```

#### `bench_long_context.py` — Long Context Latency

Measures per-token latency as context length grows.

- **What it tests:** Generation latency after prefixes of 128, 256, 512, ..., N tokens
- **Why it matters:** Attention is O(n) or O(n^2) in context length. Users need to know
  at what context length generation becomes unacceptably slow
- **Method:** Builds prompts of increasing length, generates 32 tokens after each,
  measures total and per-token latency
- **Output:** Context length vs per-token latency table. Stops if any request takes > 60s

```bash
python benches/bench_long_context.py --max-context 8192
```

### Resilience

#### `bench_client_disconnect.py` — Client Disconnect Cleanup

Tests that the server survives client disconnects.

- **What it tests:** Three phases:
  1. Launch a long generation, disconnect mid-stream
  2. Connect fresh, verify server still works
  3. Rapid connect/disconnect cycles (5x), verify server survives
- **Why it matters:** Client disconnects happen constantly in production. If the server
  doesn't clean up resources, they leak until restart
- **Pass criteria:** Server responds normally after all disconnect scenarios

```bash
python benches/bench_client_disconnect.py
```

#### `bench_stress.py` — Max Concurrent Instances

Scales concurrent instances until failures appear.

- **What it tests:** Success rate at concurrency levels 1, 2, 4, ..., N
- **Why it matters:** Identifies the maximum stable concurrency. Verifies graceful
  degradation (clean errors, not crashes)
- **Method:** At each level, launches all instances simultaneously with 60s timeout.
  Stops if majority fail. Reports first failure point
- **Pass criteria:** At least concurrency=1 succeeds. Higher levels may fail but should
  fail with clean errors (OutOfResources, Timeout), not crashes

```bash
python benches/bench_stress.py --max-instances 256
```

### Existing

#### `tput.py` — Throughput

The original throughput benchmark. Measures requests/sec and estimated tokens/sec.

```bash
python benches/tput.py --num-requests 64 --concurrency 64
```

## Shared Utilities

`bench_utils.py` provides:
- `connect_and_install()` — connect to server, install text-completion inferlet
- `run_completion()` — run a single completion and return (output, latency, event)
- `BenchmarkResult` — structured result with pass/fail, details, errors
- `print_results()` / `save_results()` — formatted output and JSON export
- `percentile()` / `latency_stats()` — statistics helpers
- `add_common_args()` — standard CLI args (--server, --wasm-path, --output-json)

## Adding New Benchmarks

1. Create `benches/bench_<name>.py`
2. Import from `bench_utils`
3. Use `connect_and_install()` for setup
4. Return `BenchmarkResult` objects
5. Add to `run_benchmarks.sh`
6. Document in this README
