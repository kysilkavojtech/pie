# Pie Follow-up Fixes — Discovered Through Benchmarking

Bugs, missing instrumentation, and rough edges hit while running the
`benchmarks/vs-sglang` suite. None of these are fixed upstream yet. Each entry
records the symptom, root cause, the workaround I used on the pod, and the
proper fix.

**Status legend:** 🔴 bug · 🟡 missing feature · 🟠 packaging/build issue

---

## 🔴 1. `qwen3` init path never calls `warmup_cuda_graphs()`

**Symptom:** Setting `use_cuda_graphs = true` in the Pie config has no effect
for Qwen3 models. Benchmark wall time is unchanged, and the server log never
prints the "Capturing CUDA graphs" line.

**Root cause:** In `pie_worker/runtime.py`, the model init path for qwen3
creates the engine but does not call
`self.engine.warmup_cuda_graphs(self.kv_cache_at_layer)`. Only the `llama3`
init branch does. Without warmup, `self.cuda_graph_img` is empty and
`_run_layers_graphed()` in `pie_worker/model/qwen3.py` silently falls back
to the per-layer Python loop on every decode step.

**Workaround applied on Ada-6000 pod:** lazy warmup on first `fire_batch()`.
Roughly:

```python
# In fire_batch(), before processing the first batch:
if not hasattr(self, '_cuda_graphs_warmed_up'):
    self._cuda_graphs_warmed_up = True
    if hasattr(self.engine, 'warmup_cuda_graphs') and self.config.use_cuda_graphs:
        self.engine.warmup_cuda_graphs(self.kv_cache_at_layer)
```

This works around both this bug and #2 below.

**Proper fix:** add the `warmup_cuda_graphs()` call into the qwen3 init path
alongside the llama3 one, **and** make the Rust IPC handshake tolerate the
warmup latency (#2). Even better: put warmup behind the same config flag so
it's a single code path shared by all model backends.

**Impact:** ~130 ms saved on 8B Tier 1A 128/128 (ratio 1.27× → 1.22× vs
vLLM). Predicted ~4–5 ms/token saved on 0.6B (not yet verified because the
pod was killed before 0.6B could be downloaded).

---

## 🔴 2. Rust IPC accept times out during CUDA graph warmup

**Symptom:** Adding `warmup_cuda_graphs()` into the qwen3 `Runtime.__init__`
block causes Pie startup to fail with IPC connection timeouts. The Rust
server dies before the Python worker finishes initializing.

**Root cause:** `warmup_cuda_graphs()` takes ~0.75 s per 13-bin warmup. The
Rust side's `IpcOneShotServer::accept()` has a timeout (and the handshake RPC
itself has one) that fires before the Python worker is ready. Python-side
`ipc_queue.connect()` finishes, but the immediately-following handshake RPC
hits the accept timeout.

**Workaround applied:** lazy warmup on first `fire_batch()` (see #1). Pushes
warmup past the handshake window.

**Proper fix:** extend (or make configurable) the IPC accept/handshake
timeout in `runtime/src/model/ffi_ipc.rs` and its Rust-side callers; or
explicitly split "Python worker ready for handshake" from "Python worker
ready to serve". Ideally warmup should happen after the handshake completes
and the Rust side should be told about it so it can defer traffic.

---

## 🟡 3. Python worker profiling helpers are uncommitted

**Symptom:** The per-batch `[PROFILING]` lines used for all of the
"Python-side forward pass is ~11 ms on 0.6B, ~19 ms on 8B" numbers exist only
as ad-hoc print statements patched onto the running pods. They are not in
`main`, `benchmarks/vs-sglang`, or anywhere else in git.

**What the patch did:** added `time.perf_counter()` calls around the pieces
of `fire_batch()` in `pie_worker/runtime.py`:

```python
# Pseudocode — the real patch was applied with sed on both pods
t0 = time.perf_counter()
batch = Batch(**kwargs)
t1 = time.perf_counter()
model_inputs = batch.get_model_inputs(device)
t2 = time.perf_counter()
# ... _run_step() ...
t3 = time.perf_counter()
responses = batch.create_responses(...)
t4 = time.perf_counter()

# Running averages, flushed every 10s:
[PROFILING] Local avg: {avg}ms ({count}) | Last step: build_batch={t1-t0}ms
  get_inputs={t2-t1}ms inference={t3-t2}ms create_resp={t4-t3}ms total={t4-t0}ms
```

**Proper fix:** add this as a real, feature-flagged helper in
`pie_worker/runtime.py`, gated on something like an env var
(`PIE_WORKER_PROFILING=1`) or a config flag. Match the naming of the Rust
`ipc-profiling` feature so that "profiling on" means the same thing at both
layers. Even better: emit structured lines (JSON or `key=value`) so
`grep '[PROFILING]'` can feed a parser.

**Why this matters:** without it, the 3-layer profiling methodology
(Rust WASM + Python worker + client wall time) cannot be reproduced by
anyone else, and the next person debugging a decode-speed regression will
have to re-derive the patch.

---

## 🟠 4. `maturin develop -F ipc-profiling` doesn't always copy the feature-enabled binary

**Symptom:** Running
`maturin develop --release -F ipc-profiling --manifest-path ../runtime/Cargo.toml`
reports success. `strings target/release/lib_pie.so | grep LAUNCH-PROFILE`
shows the string is present in cargo's output. But the installed
`_pie.cpython-*.so` in `pie/src/pie/` and `.venv/lib/.../site-packages/_pie/`
does NOT contain it, so runtime profiling never fires.

**Root cause:** maturin's editable install path appears to copy a stale
artifact in some cache states. The cargo build is correct; the copy-into-venv
step picks the wrong file.

**Workaround applied:** `setup_bench_env.sh --with-profiling` now manually
copies `runtime/target/release/lib_pie.so` over both installed locations
after the maturin build. This works.

**Proper fix:** either file an upstream maturin bug with a repro, or skip
`maturin develop` entirely for profiling builds and do the copy-into-venv
ourselves. Alternatively, make `ipc-profiling` a runtime env var instead of
a compile-time feature so this whole class of problem goes away.

---

## 🟡 5. Benchmark script needs sudo-ish GPU cleanup on sequential-mode transitions

**Symptom:** When `run_bench_vs_vllm.sh` runs in sequential mode (8B
workflow), killing the Pie server sometimes leaves a zombie GPU process
holding ~40 GB of VRAM. The process isn't visible in `ps` but shows up in
`nvidia-smi`. vLLM then fails to start with OOM.

**Root cause:** Pie spawns Python GPU worker subprocesses via `mp.spawn`. If
the main Rust process is killed before the workers receive a shutdown
signal, they can linger. On the Ada-6000 pod I tracked one down by grepping
`/proc/*/maps` for `nvidia` — it was a `VLLM::EngineCore` leftover from a
previous run, not even from this session.

**Workaround applied in script:** `run_bench_vs_vllm.sh` now runs
`nvidia-smi --query-compute-apps=pid | xargs kill -9` as a pre-flight clean,
and does `pkill -f pie_worker` between sequential phases. Good enough in
practice but it's a big hammer.

**Proper fix:** ensure Pie's manager code in `pie/src/pie/manager.py`
properly signals and waits for all workers on shutdown (Ctrl-C, TERM, and
parent death). The memory module note says error reporting is already
tracked via `error_queue` — similar plumbing for shutdown would finish the
job. Alternative: use `prctl(PR_SET_PDEATHSIG)` on Linux so worker processes
die automatically when the manager dies.

---

## 🟡 6. `bench-noop` panics in one-token mode unless you flush first

**Symptom:** Running `bench-noop` with the one-token variant panics the
inferlet. Workaround pattern that works:

```rust
ctx.fill_user("x");
ctx.flush();            // prefill the one token
ctx.fill_user("y");     // add the next token
ctx.generate(...);      // now this works
```

If you call `flush()` immediately before `generate()` without re-filling,
the pending-token buffer is empty and the inferlet panics.

**Proper fix:** `ctx.generate()` (or the SDK `decode_step()` it calls) should
either accept an empty pending buffer or produce a clear error instead of a
panic. Alternatively, document the expected call pattern in the SDK
inferlet docs.

---

## 🟡 7. vLLM install wipes Pie's Rust native module

**Symptom:** Installing vLLM into Pie's venv overwrites or deletes the
maturin-built `_pie.cpython-*.so`. First Pie run after vLLM install fails
with `ModuleNotFoundError: No module named '_pie'`.

**Root cause:** pip/uv's install path somehow clobbers extension modules
under certain conditions — likely because both projects publish to the same
namespace package layout.

**Workaround:** `setup_bench_env.sh` rebuilds the native module after every
vLLM install. Has been reliable.

**Proper fix:** likely not fixable in Pie itself without a release-engineering
change (move `_pie` out of the shared package path, or pin its location in
`pyproject.toml`). Acceptable to keep as a documented setup step.

---

## Priority

| # | Title | Priority | Effort |
|---|-------|----------|--------|
| 1 | qwen3 missing warmup_cuda_graphs | **high** — blocks the "enable CUDA graphs" experiment on every Qwen3 run | low — ~5 lines |
| 2 | IPC accept timeout vs warmup | high (needed to fix #1 cleanly) | medium |
| 3 | Commit the Python worker profiling helper | **high** — otherwise the 3-layer profiling methodology is not reproducible | low |
| 4 | maturin `-F` flag copy bug | medium — setup script works around it | medium (upstream bug) |
| 5 | Worker process cleanup on shutdown | medium — script works around it | medium |
| 6 | `bench-noop` one-token panic | low — known pattern | low |
| 7 | vLLM install wipes `_pie.so` | low — documented | hard (packaging) |

Items 1, 2, and 3 are the ones most worth doing before the next benchmarking
session, because they unblock the CUDA-graphs-on-0.6B experiment which is
still the biggest outstanding question.
