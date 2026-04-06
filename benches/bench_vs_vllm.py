"""Benchmark: Pie vs vLLM Head-to-Head

Compares Pie and vLLM across Tier 1 (baseline serving) and Tier 2
(multi-step workflows) benchmarks. Measures wall time, throughput,
TTFT, and total prefill tokens.

vLLM exposes an OpenAI-compatible API, so the request format is
identical to SGLang. The key difference is that vLLM uses Automatic
Prefix Caching (APC) instead of RadixAttention for shared-prefix
optimization.

Prerequisites:
    - Pie server running (default: ws://127.0.0.1:8080)
    - vLLM server running (default: http://localhost:8000)
        vllm serve <model> --port 8000
    - Benchmark inferlets built:
        cd benches/inferlets && cargo build --target wasm32-wasip2 --release

Usage:
    python benches/bench_vs_vllm.py
    python benches/bench_vs_vllm.py --tiers 1a,1b     # Tier 1 only
    python benches/bench_vs_vllm.py --tiers 2a,2b      # Specific tests
    python benches/bench_vs_vllm.py --pie-only          # Skip vLLM
    python benches/bench_vs_vllm.py --vllm-only         # Skip Pie
"""

import argparse
import asyncio
import json
import sys
import time
import tomllib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bench_utils import (
    BenchmarkResult,
    latency_stats,
    print_results,
    save_results,
    REPO_ROOT,
)
from pie_client import PieClient, Event

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent.resolve()
INFERLETS_DIR = BENCH_DIR / "inferlets"

TEXT_COMPLETION_WASM = (
    REPO_ROOT / "std" / "text-completion" / "target"
    / "wasm32-wasip2" / "release" / "text_completion.wasm"
)
TEXT_COMPLETION_MANIFEST = REPO_ROOT / "std" / "text-completion" / "Pie.toml"

INFERLET_WASMS = {
    "bench-chain-of-gen": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_chain_of_gen.wasm",
    "bench-best-of-n": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_best_of_n.wasm",
    "bench-constrained-retry": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_constrained_retry.wasm",
    "bench-noop": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_noop.wasm",
}

INFERLET_MANIFESTS = {
    name: INFERLETS_DIR / name / "Pie.toml" for name in INFERLET_WASMS
}

# ---------------------------------------------------------------------------
# Shared prompts (same as SGLang benchmarks for reproducibility)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

BASELINE_PROMPT = "Write a short paragraph about distributed systems."

CHAIN_PROMPT = "Explain how garbage collection works in modern programming languages."

BEST_OF_N_PROMPT = "Write a concise summary of the benefits of renewable energy."

CONSTRAINED_PROMPT = (
    "List 3 famous scientists and their key contributions. "
    "Respond as a JSON array of objects with keys: name, field, contribution."
)

LONG_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "You always provide detailed, well-structured answers. "
    "When asked about technical topics, you explain concepts clearly "
    "with examples. You use proper formatting and organize your "
    "responses logically. You are precise and factual. " * 10  # ~500 tokens
)

# vLLM model name — auto-detected at runtime via /v1/models, or override with --vllm-model
VLLM_MODEL_NAME = "default"


# ---------------------------------------------------------------------------
# Pie helpers (shared with SGLang benchmark — TODO: extract to common module)
# ---------------------------------------------------------------------------

async def pie_connect_and_install_inferlet(
    client: PieClient,
    wasm_path: Path,
    manifest_path: Path,
) -> str:
    """Install an inferlet and return its name@version string."""
    if not wasm_path.exists():
        print(f"  Error: WASM not found at {wasm_path}")
        print("  Run: cd benches/inferlets && cargo build --target wasm32-wasip2 --release")
        sys.exit(1)

    pkg = tomllib.loads(manifest_path.read_text()).get("package", {})
    name = pkg.get("name", "unknown")
    version = pkg.get("version", "0.1.0")
    inferlet_name = f"{name}@{version}"

    if not await client.program_exists(inferlet_name, wasm_path, manifest_path):
        print(f"  Installing {inferlet_name}...")
        await client.install_program(wasm_path, manifest_path)
    return inferlet_name


async def pie_run_inferlet(
    client: PieClient,
    inferlet_name: str,
    arguments: list[str],
    timeout: float = 120.0,
) -> tuple[str, float, dict[str, str], Event]:
    """Run an inferlet and collect output + structured metrics from stdout.

    Returns (output, wall_ms, metrics_dict, final_event).
    """
    start = time.perf_counter()
    instance = await client.launch_instance(inferlet_name, arguments=arguments)

    output = ""
    metrics = {}
    final_event = Event.Completed

    while True:
        event, msg = await asyncio.wait_for(instance.recv(), timeout=timeout)
        if event == Event.Stdout:
            if ":" in msg and msg.split(":")[0].replace("_", "").isalpha():
                key, _, val = msg.partition(":")
                metrics[key] = val.strip()
            else:
                output += msg
        elif event == Event.Completed:
            output = msg if msg else output
            final_event = Event.Completed
            break
        elif event in (Event.Exception, Event.Aborted, Event.ServerError, Event.OutOfResources):
            final_event = event
            output = msg
            break

    wall_ms = (time.perf_counter() - start) * 1000.0
    return output, wall_ms, metrics, final_event


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

async def vllm_completion(
    base_url: str,
    prompt: str,
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float, int]:
    """Send a chat completion to vLLM's OpenAI-compatible API.

    Returns (output, wall_ms, prompt_tokens).
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        start = time.perf_counter()
        resp = await client.post("/v1/chat/completions", json={
            "model": VLLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        wall_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()

    output = data["choices"][0]["message"]["content"] or ""
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

    return output, wall_ms, prompt_tokens


async def vllm_chat_multi_turn(
    base_url: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float, int]:
    """Multi-turn chat completion for chain-of-gen.

    Returns (output, wall_ms, prompt_tokens).
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        start = time.perf_counter()
        resp = await client.post("/v1/chat/completions", json={
            "model": VLLM_MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        wall_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()

    output = data["choices"][0]["message"]["content"] or ""
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

    return output, wall_ms, prompt_tokens


async def vllm_completion_streaming(
    base_url: str,
    prompt: str,
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float, float, int]:
    """Send a streaming chat completion to vLLM. Measures TTFT.

    Returns (output, wall_ms, ttft_ms, prompt_tokens).
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        start = time.perf_counter()
        ttft_ms = 0.0
        output_chunks = []
        prompt_tokens = 0

        async with client.stream("POST", "/v1/chat/completions", json={
            "model": VLLM_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }) as resp:
            resp.raise_for_status()
            first_token_seen = False
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content and not first_token_seen:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                    first_token_seen = True
                if content:
                    output_chunks.append(content)
                # Try to get usage from the final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)

        wall_ms = (time.perf_counter() - start) * 1000.0

    return "".join(output_chunks), wall_ms, ttft_ms, prompt_tokens


async def vllm_scrape_metrics(base_url: str) -> dict[str, float]:
    """Scrape vLLM's Prometheus /metrics endpoint.

    Returns a dict of metric_name -> value for key metrics.
    Requires vLLM started with metrics enabled (default).
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            resp = await client.get("/metrics")
            resp.raise_for_status()
            text = resp.text
    except Exception:
        return {}

    metrics = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        # Parse Prometheus text format: metric_name{labels} value
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].split("{")[0]
            try:
                metrics[name] = float(parts[-1])
            except ValueError:
                pass
    return metrics


# ===========================================================================
# TIER 0: Overhead Isolation
# ===========================================================================

async def tier0_overhead(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    vllm_url: str | None,
    runs: int = 10,
) -> list[BenchmarkResult]:
    """Tier 0: Pure framework overhead — no meaningful generation."""
    results = []

    # --- Pie: noop inferlet in three modes ---
    if pie_client and pie_inferlets:
        noop_inferlet = pie_inferlets.get("bench-noop")
        if noop_inferlet:
            for mode in ["noop", "flush-only", "one-token"]:
                latencies = []
                all_metrics = []
                for _ in range(runs):
                    _, wall_ms, metrics, ev = await pie_run_inferlet(
                        pie_client, noop_inferlet,
                        ["--mode", mode],
                    )
                    if ev == Event.Completed:
                        latencies.append(wall_ms)
                        all_metrics.append(metrics)
                    else:
                        print(f"    [WARN] {mode} run failed: event={ev}, msg={_[:2000]}")

                stats = latency_stats(latencies)
                results.append(BenchmarkResult(
                    name=f"0_overhead_{mode}_pie",
                    passed=len(latencies) == runs,
                    duration_sec=sum(latencies) / 1000.0,
                    details={
                        "engine": "pie",
                        "mode": mode,
                        "runs": runs,
                        "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                        "last_inferlet_metrics": all_metrics[-1] if all_metrics else {},
                    },
                ))
                print(f"  0 overhead/{mode} [pie]: p50={stats['p50']:.0f}ms")

    # --- vLLM: max_tokens=1 (minimal generation) ---
    if vllm_url:
        latencies = []
        for _ in range(runs):
            _, wall_ms, _ = await vllm_completion(
                vllm_url, "x", max_tokens=1, temperature=0.0,
            )
            latencies.append(wall_ms)

        stats = latency_stats(latencies)
        results.append(BenchmarkResult(
            name="0_overhead_one_token_vllm",
            passed=True,
            duration_sec=sum(latencies) / 1000.0,
            details={
                "engine": "vllm",
                "mode": "one-token",
                "runs": runs,
                "latency_ms": {k: round(v, 2) for k, v in stats.items()},
            },
        ))
        print(f"  0 overhead/one-token [vllm]: p50={stats['p50']:.0f}ms")

    return results


# ===========================================================================
# TIER 1: Baseline Serving
# ===========================================================================

async def tier1a_single_request(
    pie_client: PieClient | None,
    pie_inferlet: str | None,
    vllm_url: str | None,
    runs: int = 5,
) -> list[BenchmarkResult]:
    """Tier 1A: Single-request latency comparison."""
    results = []
    configs = [
        (128, 128),
        (512, 128),
        (2048, 256),
    ]

    for input_len, output_len in configs:
        base = "Explain distributed systems in detail. "
        prompt = (base * ((input_len * 4) // len(base) + 1))[:input_len * 4]
        test_name = f"1a_latency_{input_len}in_{output_len}out"

        for engine, label in [("pie", "pie"), ("vllm", "vllm")]:
            if engine == "pie" and pie_client is None:
                continue
            if engine == "vllm" and vllm_url is None:
                continue

            latencies = []
            for _ in range(runs):
                if engine == "pie":
                    _, wall_ms, _, ev = await pie_run_inferlet(
                        pie_client, pie_inferlet,
                        ["--prompt", prompt, "--max-tokens", str(output_len),
                         "--temperature", "0"],
                    )
                    if ev == Event.Completed:
                        latencies.append(wall_ms)
                else:
                    _, wall_ms, _ = await vllm_completion(
                        vllm_url, prompt, max_tokens=output_len, temperature=0.0,
                    )
                    latencies.append(wall_ms)

            stats = latency_stats(latencies)
            results.append(BenchmarkResult(
                name=f"{test_name}_{label}",
                passed=len(latencies) == runs,
                duration_sec=sum(latencies) / 1000.0,
                details={
                    "engine": label,
                    "input_tokens_approx": input_len,
                    "output_tokens": output_len,
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                },
            ))
            print(f"  {test_name} [{label}]: p50={stats['p50']:.0f}ms, p99={stats['p99']:.0f}ms")

    return results


async def tier1b_throughput_scaling(
    pie_client: PieClient | None,
    pie_inferlet: str | None,
    vllm_url: str | None,
    max_concurrency: int = 32,
    requests_per_level: int = 16,
) -> list[BenchmarkResult]:
    """Tier 1B: Throughput scaling with concurrency."""
    results = []
    prompt = BASELINE_PROMPT
    max_tokens = 50

    levels = []
    c = 1
    while c <= max_concurrency:
        levels.append(c)
        c *= 2

    for engine, label in [("pie", "pie"), ("vllm", "vllm")]:
        if engine == "pie" and pie_client is None:
            continue
        if engine == "vllm" and vllm_url is None:
            continue

        for level in levels:
            reqs = max(level, requests_per_level)

            async def run_one_pie():
                _, wall, _, ev = await pie_run_inferlet(
                    pie_client, pie_inferlet,
                    ["--prompt", prompt, "--max-tokens", str(max_tokens),
                     "--temperature", "0.6"],
                )
                return wall, ev == Event.Completed

            async def run_one_vllm():
                _, wall, _ = await vllm_completion(
                    vllm_url, prompt, max_tokens=max_tokens, temperature=0.6,
                )
                return wall, True

            run_fn = run_one_pie if engine == "pie" else run_one_vllm

            queue = asyncio.Queue()
            for i in range(reqs):
                queue.put_nowait(i)

            latencies = []
            failures = 0

            async def worker():
                nonlocal failures
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    wall, ok = await run_fn()
                    if ok:
                        latencies.append(wall)
                    else:
                        failures += 1

            start = time.perf_counter()
            workers = [asyncio.create_task(worker()) for _ in range(level)]
            await asyncio.gather(*workers)
            duration = time.perf_counter() - start

            throughput_rps = len(latencies) / duration if duration > 0 else 0
            stats = latency_stats(latencies)

            results.append(BenchmarkResult(
                name=f"1b_throughput_c{level}_{label}",
                passed=failures == 0,
                duration_sec=duration,
                details={
                    "engine": label,
                    "concurrency": level,
                    "requests": reqs,
                    "completed": len(latencies),
                    "failures": failures,
                    "throughput_rps": round(throughput_rps, 2),
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                },
            ))
            print(f"  1b c={level} [{label}]: {throughput_rps:.1f} req/s, p50={stats['p50']:.0f}ms")

    return results


async def tier1c_ttft(
    pie_client: PieClient | None,
    pie_inferlet: str | None,
    pie_inferlets: dict[str, str] | None,
    vllm_url: str | None,
    runs: int = 5,
) -> list[BenchmarkResult]:
    """Tier 1C: Time-to-first-token comparison.

    For Pie: uses the noop inferlet in 'one-token' mode — TTFT includes WASM
    instantiation + prefill + first decode step.
    For vLLM: uses streaming mode to measure time to first SSE chunk.
    """
    results = []
    configs = [
        ("short", "Hello", 128),
        ("medium", "Explain distributed systems in detail. " * 10, 128),
        ("long", "Explain distributed systems in detail. " * 100, 256),
    ]

    for label, prompt, max_tokens in configs:
        test_name = f"1c_ttft_{label}"

        # --- Pie: use text-completion inferlet, measure wall time for one token ---
        # (TTFT �� wall time when max_tokens=1 since there's only one decode step)
        if pie_client and pie_inferlet:
            latencies = []
            for _ in range(runs):
                _, wall_ms, _, ev = await pie_run_inferlet(
                    pie_client, pie_inferlet,
                    ["--prompt", prompt, "--max-tokens", "1", "--temperature", "0"],
                )
                if ev == Event.Completed:
                    latencies.append(wall_ms)

            stats = latency_stats(latencies)
            results.append(BenchmarkResult(
                name=f"{test_name}_pie",
                passed=len(latencies) == runs,
                duration_sec=sum(latencies) / 1000.0,
                details={
                    "engine": "pie",
                    "prompt_label": label,
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "note": "TTFT approximated as wall time with max_tokens=1",
                },
            ))
            print(f"  {test_name} [pie]: p50={stats['p50']:.0f}ms")

        # --- vLLM: streaming mode, measure time to first chunk ---
        if vllm_url:
            ttfts = []
            for _ in range(runs):
                _, _, ttft_ms, _ = await vllm_completion_streaming(
                    vllm_url, prompt, max_tokens=max_tokens, temperature=0.0,
                )
                ttfts.append(ttft_ms)

            stats = latency_stats(ttfts)
            results.append(BenchmarkResult(
                name=f"{test_name}_vllm",
                passed=True,
                duration_sec=sum(ttfts) / 1000.0,
                details={
                    "engine": "vllm",
                    "prompt_label": label,
                    "runs": runs,
                    "ttft_ms": {k: round(v, 2) for k, v in stats.items()},
                },
            ))
            print(f"  {test_name} [vllm]: p50={stats['p50']:.0f}ms")

    return results


# ===========================================================================
# TIER 2: Multi-Step Workflows
# ===========================================================================

async def tier2a_chain_of_gen(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    vllm_url: str | None,
    runs: int = 3,
    max_tokens_per_step: int = 256,
) -> list[BenchmarkResult]:
    """Tier 2A: Chain-of-generations (draft -> critique -> revise)."""
    results = []

    # --- Pie ---
    if pie_client and pie_inferlets:
        inferlet = pie_inferlets.get("bench-chain-of-gen")
        if inferlet:
            pie_latencies = []
            pie_prefill_tokens = []
            for _ in range(runs):
                _, wall_ms, metrics, ev = await pie_run_inferlet(
                    pie_client, inferlet,
                    ["--prompt", CHAIN_PROMPT,
                     "--max-tokens-per-step", str(max_tokens_per_step),
                     "--system", LONG_SYSTEM_PROMPT,
                     "--temperature", "0"],
                )
                if ev == Event.Completed:
                    pie_latencies.append(wall_ms)
                    total_tok = int(metrics.get("TOTAL_TOKENS", "0"))
                    pie_prefill_tokens.append(total_tok)

            stats = latency_stats(pie_latencies)
            results.append(BenchmarkResult(
                name="2a_chain_of_gen_pie",
                passed=len(pie_latencies) == runs,
                duration_sec=sum(pie_latencies) / 1000.0,
                details={
                    "engine": "pie",
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "step_ms": {
                        k: v for k, v in (metrics or {}).items()
                        if k.startswith("STEP")
                    },
                    "prefill_count": 1,
                },
            ))
            print(f"  2a chain-of-gen [pie]: p50={stats['p50']:.0f}ms")

    # --- vLLM: 3 sequential API calls ---
    if vllm_url:
        vllm_latencies = []
        vllm_total_prefill = []

        critique_prompt = (
            "Now critically evaluate the response above. Identify any "
            "inaccuracies, missing information, or areas that could be clearer."
        )
        revise_prompt = (
            "Based on the critique above, write an improved and corrected "
            "version of the original response."
        )

        for _ in range(runs):
            total_prefill = 0
            messages = [
                {"role": "system", "content": LONG_SYSTEM_PROMPT},
                {"role": "user", "content": CHAIN_PROMPT},
            ]

            run_start = time.perf_counter()

            # Step 1: Draft
            draft, _, prefill1 = await vllm_chat_multi_turn(
                vllm_url, messages,
                max_tokens=max_tokens_per_step, temperature=0.0,
            )
            total_prefill += prefill1

            # Step 2: Critique
            messages.append({"role": "assistant", "content": draft})
            messages.append({"role": "user", "content": critique_prompt})

            critique, _, prefill2 = await vllm_chat_multi_turn(
                vllm_url, messages,
                max_tokens=max_tokens_per_step, temperature=0.0,
            )
            total_prefill += prefill2

            # Step 3: Revise
            messages.append({"role": "assistant", "content": critique})
            messages.append({"role": "user", "content": revise_prompt})

            _, _, prefill3 = await vllm_chat_multi_turn(
                vllm_url, messages,
                max_tokens=max_tokens_per_step, temperature=0.0,
            )
            total_prefill += prefill3

            wall_ms = (time.perf_counter() - run_start) * 1000.0
            vllm_latencies.append(wall_ms)
            vllm_total_prefill.append(total_prefill)

        if vllm_latencies:
            stats = latency_stats(vllm_latencies)
            avg_prefill = sum(vllm_total_prefill) / len(vllm_total_prefill)
            results.append(BenchmarkResult(
                name="2a_chain_of_gen_vllm",
                passed=True,
                duration_sec=sum(vllm_latencies) / 1000.0,
                details={
                    "engine": "vllm",
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "avg_total_prefill_tokens": round(avg_prefill),
                    "prefill_count": 3,
                    "note": "vLLM APC may partially cache shared prefix between calls",
                },
            ))
            print(f"  2a chain-of-gen [vllm]: p50={stats['p50']:.0f}ms, "
                  f"avg_prefill={avg_prefill:.0f} tokens")

    return results


async def tier2b_best_of_n(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    vllm_url: str | None,
    runs: int = 3,
    num_candidates: int = 4,
    max_tokens: int = 256,
) -> list[BenchmarkResult]:
    """Tier 2B: Best-of-N with shared prefix."""
    results = []

    # --- Pie ---
    if pie_client and pie_inferlets:
        inferlet = pie_inferlets.get("bench-best-of-n")
        if inferlet:
            pie_latencies = []
            for _ in range(runs):
                _, wall_ms, metrics, ev = await pie_run_inferlet(
                    pie_client, inferlet,
                    ["--prompt", BEST_OF_N_PROMPT,
                     "--max-tokens", str(max_tokens),
                     "--num-candidates", str(num_candidates),
                     "--system", LONG_SYSTEM_PROMPT,
                     "--temperature", "0.6"],
                )
                if ev == Event.Completed:
                    pie_latencies.append(wall_ms)

            if pie_latencies:
                stats = latency_stats(pie_latencies)
                last_metrics = metrics or {}
                results.append(BenchmarkResult(
                    name="2b_best_of_n_pie",
                    passed=len(pie_latencies) == runs,
                    duration_sec=sum(pie_latencies) / 1000.0,
                    details={
                        "engine": "pie",
                        "runs": runs,
                        "num_candidates": num_candidates,
                        "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                        "prefill_count": 1,
                    },
                ))
                print(f"  2b best-of-{num_candidates} [pie]: p50={stats['p50']:.0f}ms, "
                      f"prefix_prefilled=1x")

    # --- vLLM: N independent requests ---
    if vllm_url:
        vllm_latencies = []
        vllm_total_prefill = []

        for _ in range(runs):
            start = time.perf_counter()

            tasks = []
            for _ in range(num_candidates):
                tasks.append(vllm_completion(
                    vllm_url, BEST_OF_N_PROMPT,
                    system=LONG_SYSTEM_PROMPT,
                    max_tokens=max_tokens,
                    temperature=0.6,
                ))
            responses = await asyncio.gather(*tasks)

            wall_ms = (time.perf_counter() - start) * 1000.0
            vllm_latencies.append(wall_ms)
            total_prefill = sum(r[2] for r in responses)
            vllm_total_prefill.append(total_prefill)

        if vllm_latencies:
            stats = latency_stats(vllm_latencies)
            avg_prefill = sum(vllm_total_prefill) / len(vllm_total_prefill)
            results.append(BenchmarkResult(
                name="2b_best_of_n_vllm",
                passed=True,
                duration_sec=sum(vllm_latencies) / 1000.0,
                details={
                    "engine": "vllm",
                    "runs": runs,
                    "num_candidates": num_candidates,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "avg_total_prefill_tokens": round(avg_prefill),
                    "prefill_count": num_candidates,
                    "note": "vLLM APC may cache prefix after first request",
                },
            ))
            print(f"  2b best-of-{num_candidates} [vllm]: p50={stats['p50']:.0f}ms, "
                  f"avg_prefill={avg_prefill:.0f} tokens ({num_candidates}x)")

    return results


async def tier2c_constrained_retry(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    vllm_url: str | None,
    runs: int = 3,
    max_tokens: int = 512,
    max_retries: int = 5,
) -> list[BenchmarkResult]:
    """Tier 2C: Constrained JSON generation with retry."""
    results = []

    json_system = (
        "You are a JSON assistant. You MUST respond with valid JSON only. "
        "No markdown, no explanation, no text outside the JSON object. " * 5
    )

    # --- Pie ---
    if pie_client and pie_inferlets:
        inferlet = pie_inferlets.get("bench-constrained-retry")
        if inferlet:
            pie_latencies = []
            pie_attempts = []
            for _ in range(runs):
                _, wall_ms, metrics, ev = await pie_run_inferlet(
                    pie_client, inferlet,
                    ["--prompt", CONSTRAINED_PROMPT,
                     "--max-tokens", str(max_tokens),
                     "--max-retries", str(max_retries),
                     "--system", json_system,
                     "--temperature", "0.6"],
                )
                if ev == Event.Completed:
                    pie_latencies.append(wall_ms)
                    attempts = int(metrics.get("TOTAL_ATTEMPTS", "1"))
                    pie_attempts.append(attempts)

            if pie_latencies:
                stats = latency_stats(pie_latencies)
                avg_attempts = sum(pie_attempts) / len(pie_attempts)
                results.append(BenchmarkResult(
                    name="2c_constrained_retry_pie",
                    passed=len(pie_latencies) == runs,
                    duration_sec=sum(pie_latencies) / 1000.0,
                    details={
                        "engine": "pie",
                        "runs": runs,
                        "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                        "avg_attempts": round(avg_attempts, 1),
                        "prefill_count": 1,
                    },
                ))
                print(f"  2c constrained-retry [pie]: p50={stats['p50']:.0f}ms, "
                      f"avg_attempts={avg_attempts:.1f}, prefill=1x")

    # --- vLLM: retry from scratch each time ---
    if vllm_url:
        vllm_latencies = []
        vllm_attempts = []
        vllm_total_prefill = []

        for _ in range(runs):
            total_prefill = 0
            attempts = 0
            start = time.perf_counter()

            for attempt in range(max_retries + 1):
                attempts += 1
                output, _, prompt_tokens = await vllm_completion(
                    vllm_url, CONSTRAINED_PROMPT,
                    system=json_system,
                    max_tokens=max_tokens,
                    temperature=0.6,
                )
                total_prefill += prompt_tokens

                trimmed = output.strip()
                try:
                    for start_char in ['{', '[']:
                        idx = trimmed.find(start_char)
                        if idx >= 0:
                            json.loads(trimmed[idx:])
                            break
                    else:
                        json.loads(trimmed)
                    break  # Valid JSON
                except (json.JSONDecodeError, ValueError):
                    if attempt == max_retries:
                        break  # Give up

            wall_ms = (time.perf_counter() - start) * 1000.0
            vllm_latencies.append(wall_ms)
            vllm_attempts.append(attempts)
            vllm_total_prefill.append(total_prefill)

        if vllm_latencies:
            stats = latency_stats(vllm_latencies)
            avg_attempts = sum(vllm_attempts) / len(vllm_attempts)
            avg_prefill = sum(vllm_total_prefill) / len(vllm_total_prefill)
            results.append(BenchmarkResult(
                name="2c_constrained_retry_vllm",
                passed=True,
                duration_sec=sum(vllm_latencies) / 1000.0,
                details={
                    "engine": "vllm",
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "avg_attempts": round(avg_attempts, 1),
                    "avg_total_prefill_tokens": round(avg_prefill),
                    "prefill_count_per_attempt": 1,
                    "note": "Each retry re-prefills the full prompt (APC may partially help)",
                },
            ))
            print(f"  2c constrained-retry [vllm]: p50={stats['p50']:.0f}ms, "
                  f"avg_attempts={avg_attempts:.1f}, avg_prefill={avg_prefill:.0f} tokens")

    return results


# ===========================================================================
# Main
# ===========================================================================

async def run_benchmarks(args):
    all_results = []
    tiers = set(args.tiers.split(",")) if args.tiers else {"0", "1a", "1b", "1c", "2a", "2b", "2c"}

    # --- Connect to Pie ---
    pie_client = None
    pie_text_completion = None
    pie_inferlets = {}

    if not args.vllm_only:
        print("Connecting to Pie server...")
        pie_client = PieClient(args.pie_server)
        await pie_client.connect()
        await pie_client.authenticate("benchmark-user")

        if any(t.startswith("1") or t == "0" for t in tiers):
            pie_text_completion = await pie_connect_and_install_inferlet(
                pie_client, TEXT_COMPLETION_WASM, TEXT_COMPLETION_MANIFEST,
            )

        # Install all benchmark inferlets (needed for Tier 0 noop + Tier 2)
        if any(t.startswith("2") or t == "0" for t in tiers):
            for name in INFERLET_WASMS:
                wasm = INFERLET_WASMS[name]
                manifest = INFERLET_MANIFESTS[name]
                if wasm.exists():
                    iname = await pie_connect_and_install_inferlet(
                        pie_client, wasm, manifest,
                    )
                    pie_inferlets[name] = iname
                else:
                    print(f"  Warning: {name} WASM not found, skipping")

    vllm_url = args.vllm_server if not args.pie_only else None

    # --- Auto-detect vLLM model name ---
    global VLLM_MODEL_NAME
    if args.vllm_model:
        VLLM_MODEL_NAME = args.vllm_model
        print(f"vLLM model (override): {VLLM_MODEL_NAME}")
    elif vllm_url:
        try:
            async with httpx.AsyncClient(base_url=vllm_url, timeout=10.0) as client:
                resp = await client.get("/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    VLLM_MODEL_NAME = models[0]["id"]
                    print(f"vLLM model: {VLLM_MODEL_NAME}")
        except Exception as e:
            print(f"Warning: could not detect vLLM model name ({e}), using '{VLLM_MODEL_NAME}'")

    # --- Scrape initial vLLM metrics ---
    vllm_metrics_before = {}
    if vllm_url and args.vllm_metrics:
        print("Scraping initial vLLM metrics...")
        vllm_metrics_before = await vllm_scrape_metrics(vllm_url)

    # --- Run tests ---
    try:
        if "0" in tiers:
            print("\n--- Tier 0: Overhead Isolation ---")
            r = await tier0_overhead(
                pie_client, pie_inferlets, vllm_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "1a" in tiers:
            print("\n--- Tier 1A: Single-Request Latency ---")
            r = await tier1a_single_request(
                pie_client, pie_text_completion, vllm_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "1b" in tiers:
            print("\n--- Tier 1B: Throughput Scaling ---")
            r = await tier1b_throughput_scaling(
                pie_client, pie_text_completion, vllm_url,
                max_concurrency=args.max_concurrency,
            )
            all_results.extend(r)

        if "1c" in tiers:
            print("\n--- Tier 1C: Time-to-First-Token ---")
            r = await tier1c_ttft(
                pie_client, pie_text_completion, pie_inferlets, vllm_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "2a" in tiers:
            print("\n--- Tier 2A: Chain-of-Generations ---")
            r = await tier2a_chain_of_gen(
                pie_client, pie_inferlets, vllm_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "2b" in tiers:
            print("\n--- Tier 2B: Best-of-N ---")
            r = await tier2b_best_of_n(
                pie_client, pie_inferlets, vllm_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "2c" in tiers:
            print("\n--- Tier 2C: Constrained Retry ---")
            r = await tier2c_constrained_retry(
                pie_client, pie_inferlets, vllm_url,
                runs=args.runs,
            )
            all_results.extend(r)

    finally:
        if pie_client:
            await pie_client.close()

    # --- Scrape final vLLM metrics and compute deltas ---
    if vllm_url and args.vllm_metrics and vllm_metrics_before:
        print("\n--- vLLM Metrics Delta ---")
        vllm_metrics_after = await vllm_scrape_metrics(vllm_url)
        interesting = [
            "vllm:time_to_first_token_seconds_sum",
            "vllm:time_to_first_token_seconds_count",
            "vllm:time_per_output_token_seconds_sum",
            "vllm:time_per_output_token_seconds_count",
            "vllm:e2e_request_latency_seconds_sum",
            "vllm:e2e_request_latency_seconds_count",
            "vllm:request_queue_time_seconds_sum",
            "vllm:request_queue_time_seconds_count",
        ]
        for key in interesting:
            before = vllm_metrics_before.get(key, 0)
            after = vllm_metrics_after.get(key, 0)
            delta = after - before
            if delta != 0:
                print(f"  {key}: {delta:.4f}")

    # --- Print comparison tables ---
    print_comparison_summary(all_results)
    print_results(all_results)

    if args.output_json:
        save_results(all_results, Path(args.output_json))

    return all(r.passed for r in all_results)


def print_comparison_summary(results: list[BenchmarkResult]):
    """Print side-by-side comparison tables."""
    tests = {}
    for r in results:
        name = r.name
        engine = r.details.get("engine", "")
        base = name.rsplit(f"_{engine}", 1)[0] if engine else name
        tests.setdefault(base, {})[engine] = r

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for test_name, engines in sorted(tests.items()):
        print(f"\n  {test_name}:")
        for engine, r in sorted(engines.items()):
            lat = r.details.get("latency_ms", r.details.get("ttft_ms", {}))
            p50 = lat.get("p50", 0)

            extra = ""
            if "prefill_count" in r.details:
                extra += f", prefill={r.details['prefill_count']}x"
            if "avg_total_prefill_tokens" in r.details:
                extra += f", prefill_tokens={r.details['avg_total_prefill_tokens']}"
            if "avg_attempts" in r.details:
                extra += f", attempts={r.details['avg_attempts']}"
            if "throughput_rps" in r.details:
                extra += f", {r.details['throughput_rps']} req/s"

            print(f"    {engine:>8}: p50={p50:>8.1f}ms{extra}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Pie vs vLLM Head-to-Head"
    )
    parser.add_argument(
        "--pie-server", default="ws://127.0.0.1:8080",
        help="Pie WebSocket server URI",
    )
    parser.add_argument(
        "--vllm-server", default="http://localhost:8000",
        help="vLLM HTTP server URL (default port 8000)",
    )
    parser.add_argument(
        "--tiers", default=None,
        help="Comma-separated tier list (e.g., 0,1a,1b,1c,2a,2b,2c). Default: all",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per test (default: 3)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=128,
        help="Max concurrency for Tier 1B (default: 128)",
    )
    parser.add_argument(
        "--pie-only", action="store_true",
        help="Only run Pie benchmarks (skip vLLM)",
    )
    parser.add_argument(
        "--vllm-only", action="store_true",
        help="Only run vLLM benchmarks (skip Pie)",
    )
    parser.add_argument(
        "--vllm-model", default=None,
        help="Override vLLM model name (auto-detected from /v1/models by default)",
    )
    parser.add_argument(
        "--vllm-metrics", action="store_true",
        help="Scrape vLLM /metrics endpoint for detailed timing breakdown",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Write results to JSON file",
    )
    args = parser.parse_args()
    success = asyncio.run(run_benchmarks(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
