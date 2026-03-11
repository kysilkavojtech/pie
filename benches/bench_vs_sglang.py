"""Benchmark: Pie vs SGLang Head-to-Head

Compares Pie and SGLang across Tier 1 (baseline serving) and Tier 2
(multi-step workflows) benchmarks. Measures wall time, throughput,
TTFT, and total prefill tokens.

Prerequisites:
    - Pie server running (default: ws://127.0.0.1:8080)
    - SGLang server running (default: http://localhost:30000)
    - Benchmark inferlets built:
        cd benches/inferlets && cargo build --target wasm32-wasip2 --release

Usage:
    python benches/bench_vs_sglang.py
    python benches/bench_vs_sglang.py --tiers 1        # Tier 1 only
    python benches/bench_vs_sglang.py --tiers 2a,2b    # Specific tests
    python benches/bench_vs_sglang.py --pie-only        # Skip SGLang
    python benches/bench_vs_sglang.py --sglang-only     # Skip Pie
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

# Standard text-completion inferlet (for Tier 1)
TEXT_COMPLETION_WASM = (
    REPO_ROOT / "std" / "text-completion" / "target"
    / "wasm32-wasip2" / "release" / "text_completion.wasm"
)
TEXT_COMPLETION_MANIFEST = REPO_ROOT / "std" / "text-completion" / "Pie.toml"

# Benchmark inferlets (for Tier 2)
INFERLET_WASMS = {
    "bench-chain-of-gen": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_chain_of_gen.wasm",
    "bench-best-of-n": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_best_of_n.wasm",
    "bench-constrained-retry": INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / "bench_constrained_retry.wasm",
}

INFERLET_MANIFESTS = {
    name: INFERLETS_DIR / name / "Pie.toml" for name in INFERLET_WASMS
}

# ---------------------------------------------------------------------------
# Shared prompts for reproducibility
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

# Tier 1: baseline
BASELINE_PROMPT = "Write a short paragraph about distributed systems."

# Tier 2A: chain-of-gen
CHAIN_PROMPT = "Explain how garbage collection works in modern programming languages."

# Tier 2B: best-of-N
BEST_OF_N_PROMPT = "Write a concise summary of the benefits of renewable energy."

# Tier 2C: constrained retry
CONSTRAINED_PROMPT = (
    "List 3 famous scientists and their key contributions. "
    "Respond as a JSON array of objects with keys: name, field, contribution."
)

# Context prefix for Tier 2 tests — long enough to make prefill cost visible
LONG_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "You always provide detailed, well-structured answers. "
    "When asked about technical topics, you explain concepts clearly "
    "with examples. You use proper formatting and organize your "
    "responses logically. You are precise and factual. " * 10  # ~500 tokens
)


# ---------------------------------------------------------------------------
# Pie helpers
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
    Metrics are extracted from stdout lines matching PATTERN:VALUE.
    """
    start = time.perf_counter()
    instance = await client.launch_instance(inferlet_name, arguments=arguments)

    output = ""
    metrics = {}
    final_event = Event.Completed

    while True:
        event, msg = await asyncio.wait_for(instance.recv(), timeout=timeout)
        if event == Event.Stdout:
            # Extract structured metrics from stdout
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
# SGLang helpers
# ---------------------------------------------------------------------------

async def sglang_completion(
    base_url: str,
    prompt: str,
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float, int]:
    """Send a chat completion to SGLang's OpenAI-compatible API.

    Uses httpx directly — no extra dependencies needed.
    Returns (output, wall_ms, prompt_tokens).
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        start = time.perf_counter()
        resp = await client.post("/v1/chat/completions", json={
            "model": "default",
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


async def sglang_chat_multi_turn(
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
            "model": "default",
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


# ===========================================================================
# TIER 1: Baseline Serving
# ===========================================================================

async def tier1a_single_request(
    pie_client: PieClient | None,
    pie_inferlet: str | None,
    sglang_url: str | None,
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
        # Build a prompt of approximately input_len tokens (~4 chars/token)
        base = "Explain distributed systems in detail. "
        prompt = (base * ((input_len * 4) // len(base) + 1))[:input_len * 4]
        test_name = f"1a_latency_{input_len}in_{output_len}out"

        for engine, label in [("pie", "pie"), ("sglang", "sglang")]:
            if engine == "pie" and pie_client is None:
                continue
            if engine == "sglang" and sglang_url is None:
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
                    _, wall_ms, _ = await sglang_completion(
                        sglang_url, prompt, max_tokens=output_len, temperature=0.0,
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
    sglang_url: str | None,
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

    for engine, label in [("pie", "pie"), ("sglang", "sglang")]:
        if engine == "pie" and pie_client is None:
            continue
        if engine == "sglang" and sglang_url is None:
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

            async def run_one_sglang():
                _, wall, _ = await sglang_completion(
                    sglang_url, prompt, max_tokens=max_tokens, temperature=0.6,
                )
                return wall, True

            run_fn = run_one_pie if engine == "pie" else run_one_sglang

            # Worker pool
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


# ===========================================================================
# TIER 2: Multi-Step Workflows
# ===========================================================================

async def tier2a_chain_of_gen(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    sglang_url: str | None,
    runs: int = 3,
    max_tokens_per_step: int = 256,
) -> list[BenchmarkResult]:
    """Tier 2A: Chain-of-generations (draft → critique → revise)."""
    results = []

    # --- Pie: single inferlet, 3 steps, KV cache persists ---
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
                    # Pie prefills the prompt once; total_tokens includes gen tokens
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

    # --- SGLang: 3 sequential API calls, re-sending context each time ---
    if sglang_url:
        sglang_latencies = []
        sglang_total_prefill = []

        critique_prompt_template = (
            "Now critically evaluate the response above. Identify any "
            "inaccuracies, missing information, or areas that could be clearer."
        )
        revise_prompt_template = (
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
            draft, _, prefill1 = await sglang_chat_multi_turn(
                sglang_url, messages,
                max_tokens=max_tokens_per_step, temperature=0.0,
            )
            total_prefill += prefill1

            # Step 2: Critique (must re-send full history)
            messages.append({"role": "assistant", "content": draft})
            messages.append({"role": "user", "content": critique_prompt_template})

            critique, _, prefill2 = await sglang_chat_multi_turn(
                sglang_url, messages,
                max_tokens=max_tokens_per_step, temperature=0.0,
            )
            total_prefill += prefill2

            # Step 3: Revise (must re-send even more history)
            messages.append({"role": "assistant", "content": critique})
            messages.append({"role": "user", "content": revise_prompt_template})

            _, _, prefill3 = await sglang_chat_multi_turn(
                sglang_url, messages,
                max_tokens=max_tokens_per_step, temperature=0.0,
            )
            total_prefill += prefill3

            wall_ms = (time.perf_counter() - run_start) * 1000.0
            sglang_latencies.append(wall_ms)
            sglang_total_prefill.append(total_prefill)

        if sglang_latencies:
            stats = latency_stats(sglang_latencies)
            avg_prefill = sum(sglang_total_prefill) / len(sglang_total_prefill)
            results.append(BenchmarkResult(
                name="2a_chain_of_gen_sglang",
                passed=True,
                duration_sec=sum(sglang_latencies) / 1000.0,
                details={
                    "engine": "sglang",
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "avg_total_prefill_tokens": round(avg_prefill),
                    "prefill_count": 3,
                },
            ))
            print(f"  2a chain-of-gen [sglang]: p50={stats['p50']:.0f}ms, "
                  f"avg_prefill={avg_prefill:.0f} tokens")

    return results


async def tier2b_best_of_n(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    sglang_url: str | None,
    runs: int = 3,
    num_candidates: int = 4,
    max_tokens: int = 256,
) -> list[BenchmarkResult]:
    """Tier 2B: Best-of-N with shared prefix."""
    results = []

    # --- Pie: single inferlet forks N contexts from shared prefix ---
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
                        "prefill_ms": last_metrics.get("PREFILL_MS"),
                        "generation_ms": last_metrics.get("GENERATION_MS"),
                        "prefix_tokens": last_metrics.get("PREFIX_TOKENS"),
                        "total_prefill_tokens": last_metrics.get("TOTAL_PREFILL_TOKENS"),
                        "prefill_count": 1,
                    },
                ))
                print(f"  2b best-of-{num_candidates} [pie]: p50={stats['p50']:.0f}ms, "
                      f"prefix_prefilled=1x")

    # --- SGLang: N independent requests (prefix may or may not be cached) ---
    if sglang_url:
        sglang_latencies = []
        sglang_total_prefill = []

        for _ in range(runs):
            total_prefill = 0
            start = time.perf_counter()

            tasks = []
            for _ in range(num_candidates):
                tasks.append(sglang_completion(
                    sglang_url, BEST_OF_N_PROMPT,
                    system=LONG_SYSTEM_PROMPT,
                    max_tokens=max_tokens,
                    temperature=0.6,
                ))
            responses = await asyncio.gather(*tasks)

            wall_ms = (time.perf_counter() - start) * 1000.0
            sglang_latencies.append(wall_ms)
            total_prefill = sum(r[2] for r in responses)
            sglang_total_prefill.append(total_prefill)

        if sglang_latencies:
            stats = latency_stats(sglang_latencies)
            avg_prefill = sum(sglang_total_prefill) / len(sglang_total_prefill)
            results.append(BenchmarkResult(
                name="2b_best_of_n_sglang",
                passed=True,
                duration_sec=sum(sglang_latencies) / 1000.0,
                details={
                    "engine": "sglang",
                    "runs": runs,
                    "num_candidates": num_candidates,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "avg_total_prefill_tokens": round(avg_prefill),
                    "prefill_count": num_candidates,
                },
            ))
            print(f"  2b best-of-{num_candidates} [sglang]: p50={stats['p50']:.0f}ms, "
                  f"avg_prefill={avg_prefill:.0f} tokens ({num_candidates}x)")

    return results


async def tier2c_constrained_retry(
    pie_client: PieClient | None,
    pie_inferlets: dict[str, str] | None,
    sglang_url: str | None,
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

    # --- Pie: retry from KV cache checkpoint ---
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
                last_metrics = metrics or {}
                results.append(BenchmarkResult(
                    name="2c_constrained_retry_pie",
                    passed=len(pie_latencies) == runs,
                    duration_sec=sum(pie_latencies) / 1000.0,
                    details={
                        "engine": "pie",
                        "runs": runs,
                        "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                        "avg_attempts": round(avg_attempts, 1),
                        "prefill_tokens": last_metrics.get("TOTAL_PREFILL_TOKENS"),
                        "prefill_count": 1,
                    },
                ))
                print(f"  2c constrained-retry [pie]: p50={stats['p50']:.0f}ms, "
                      f"avg_attempts={avg_attempts:.1f}, prefill=1x")

    # --- SGLang: retry from scratch each time ---
    if sglang_url:
        sglang_latencies = []
        sglang_attempts = []
        sglang_total_prefill = []

        for _ in range(runs):
            total_prefill = 0
            attempts = 0
            start = time.perf_counter()

            for attempt in range(max_retries + 1):
                attempts += 1
                output, _, prompt_tokens = await sglang_completion(
                    sglang_url, CONSTRAINED_PROMPT,
                    system=json_system,
                    max_tokens=max_tokens,
                    temperature=0.6,
                )
                total_prefill += prompt_tokens

                # Check if valid JSON
                trimmed = output.strip()
                try:
                    for start_char in ['{', '[']:
                        idx = trimmed.find(start_char)
                        if idx >= 0:
                            json.loads(trimmed[idx:])
                            break
                    else:
                        json.loads(trimmed)
                    break  # Valid JSON, done
                except (json.JSONDecodeError, ValueError):
                    if attempt == max_retries:
                        break  # Give up

            wall_ms = (time.perf_counter() - start) * 1000.0
            sglang_latencies.append(wall_ms)
            sglang_attempts.append(attempts)
            sglang_total_prefill.append(total_prefill)

        if sglang_latencies:
            stats = latency_stats(sglang_latencies)
            avg_attempts = sum(sglang_attempts) / len(sglang_attempts)
            avg_prefill = sum(sglang_total_prefill) / len(sglang_total_prefill)
            results.append(BenchmarkResult(
                name="2c_constrained_retry_sglang",
                passed=True,
                duration_sec=sum(sglang_latencies) / 1000.0,
                details={
                    "engine": "sglang",
                    "runs": runs,
                    "latency_ms": {k: round(v, 2) for k, v in stats.items()},
                    "avg_attempts": round(avg_attempts, 1),
                    "avg_total_prefill_tokens": round(avg_prefill),
                    "prefill_count_per_attempt": 1,
                    "note": "Each retry re-prefills the full prompt",
                },
            ))
            print(f"  2c constrained-retry [sglang]: p50={stats['p50']:.0f}ms, "
                  f"avg_attempts={avg_attempts:.1f}, avg_prefill={avg_prefill:.0f} tokens")

    return results


# ===========================================================================
# Main
# ===========================================================================

async def run_benchmarks(args):
    all_results = []
    tiers = set(args.tiers.split(",")) if args.tiers else {"1a", "1b", "2a", "2b", "2c"}

    # --- Connect to Pie ---
    pie_client = None
    pie_text_completion = None
    pie_inferlets = {}

    if not args.sglang_only:
        print("Connecting to Pie server...")
        pie_client = PieClient(args.pie_server)
        await pie_client.connect()
        await pie_client.authenticate("benchmark-user")

        # Install text-completion for Tier 1
        if any(t.startswith("1") for t in tiers):
            pie_text_completion = await pie_connect_and_install_inferlet(
                pie_client, TEXT_COMPLETION_WASM, TEXT_COMPLETION_MANIFEST,
            )

        # Install benchmark inferlets for Tier 2
        if any(t.startswith("2") for t in tiers):
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

    sglang_url = args.sglang_server if not args.pie_only else None

    # --- Run tests ---
    try:
        if "1a" in tiers:
            print("\n--- Tier 1A: Single-Request Latency ---")
            r = await tier1a_single_request(
                pie_client, pie_text_completion, sglang_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "1b" in tiers:
            print("\n--- Tier 1B: Throughput Scaling ---")
            r = await tier1b_throughput_scaling(
                pie_client, pie_text_completion, sglang_url,
                max_concurrency=args.max_concurrency,
            )
            all_results.extend(r)

        if "2a" in tiers:
            print("\n--- Tier 2A: Chain-of-Generations ---")
            r = await tier2a_chain_of_gen(
                pie_client, pie_inferlets, sglang_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "2b" in tiers:
            print("\n--- Tier 2B: Best-of-N ---")
            r = await tier2b_best_of_n(
                pie_client, pie_inferlets, sglang_url,
                runs=args.runs,
            )
            all_results.extend(r)

        if "2c" in tiers:
            print("\n--- Tier 2C: Constrained Retry ---")
            r = await tier2c_constrained_retry(
                pie_client, pie_inferlets, sglang_url,
                runs=args.runs,
            )
            all_results.extend(r)

    finally:
        if pie_client:
            await pie_client.close()

    # --- Print comparison tables ---
    print_comparison_summary(all_results)
    print_results(all_results)

    if args.output_json:
        save_results(all_results, Path(args.output_json))

    return all(r.passed for r in all_results)


def print_comparison_summary(results: list[BenchmarkResult]):
    """Print side-by-side comparison tables."""
    # Group by test (strip engine suffix)
    tests = {}
    for r in results:
        # Extract test name without engine
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
            lat = r.details.get("latency_ms", {})
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
        description="Benchmark: Pie vs SGLang Head-to-Head"
    )
    parser.add_argument(
        "--pie-server", default="ws://127.0.0.1:8080",
        help="Pie WebSocket server URI",
    )
    parser.add_argument(
        "--sglang-server", default="http://localhost:30000",
        help="SGLang HTTP server URL",
    )
    parser.add_argument(
        "--tiers", default=None,
        help="Comma-separated tier list (e.g., 1a,1b,2a,2b,2c). Default: all",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per test (default: 3)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=32,
        help="Max concurrency for Tier 1B (default: 32)",
    )
    parser.add_argument(
        "--pie-only", action="store_true",
        help="Only run Pie benchmarks (skip SGLang)",
    )
    parser.add_argument(
        "--sglang-only", action="store_true",
        help="Only run SGLang benchmarks (skip Pie)",
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
