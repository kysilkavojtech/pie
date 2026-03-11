"""Benchmark: Long Context Per-Token Latency

Measures how per-token generation latency changes as context length grows.
Helps identify practical context length limits and attention scaling behavior.

Usage:
    python benches/bench_long_context.py
    python benches/bench_long_context.py --max-context 8192 --server ws://host:port
"""

import argparse
import asyncio
import time
from pathlib import Path

from bench_utils import (
    BenchmarkResult,
    add_common_args,
    connect_and_install,
    print_results,
    run_completion,
    save_results,
)
from pie_client import Event


def build_long_prompt(target_tokens: int) -> str:
    """Build a prompt that's approximately target_tokens long.

    Uses repetitive but coherent text to fill the context window.
    ~4 chars per token as a rough estimate.
    """
    base = (
        "The history of computing is a long and fascinating journey. "
        "From the earliest mechanical calculators to modern quantum computers, "
        "each era has brought revolutionary advances. "
    )
    # Repeat to reach target length (in chars ~ 4x tokens)
    target_chars = target_tokens * 4
    repetitions = max(1, target_chars // len(base))
    prefix = (base * repetitions)[:target_chars]
    return prefix + "\n\nNow, briefly summarize the above text in one sentence."


async def test_context_length(
    client, inferlet_name, context_tokens: int, gen_tokens: int = 32,
) -> BenchmarkResult:
    """Measure per-token latency for generation after a prefix of context_tokens."""
    name = f"context_{context_tokens}"
    prompt = build_long_prompt(context_tokens)
    start = time.perf_counter()

    output, latency_ms, event = await run_completion(
        client, inferlet_name, prompt,
        max_tokens=gen_tokens, temperature=0.0,
    )
    duration = time.perf_counter() - start

    if event != Event.Completed:
        return BenchmarkResult(
            name=name, passed=False, duration_sec=duration,
            details={"context_tokens": context_tokens},
            errors=[f"Did not complete: {event}"],
        )

    # Estimate tokens generated
    est_output_tokens = len(output) / 4.0
    per_token_ms = latency_ms / max(est_output_tokens, 1)

    return BenchmarkResult(
        name=name, passed=True, duration_sec=duration,
        details={
            "context_tokens_target": context_tokens,
            "prompt_chars": len(prompt),
            "output_chars": len(output),
            "est_output_tokens": round(est_output_tokens, 1),
            "total_latency_ms": round(latency_ms, 2),
            "per_token_ms": round(per_token_ms, 2),
        },
    )


async def run_benchmark(args):
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )

    # Exponential scaling of context lengths
    lengths = []
    c = 128
    while c <= args.max_context:
        lengths.append(c)
        c *= 2

    results = []
    try:
        print(f"{'Context':>10} {'Total ms':>10} {'Per-tok ms':>12} {'Output chars':>14}")
        print("-" * 50)

        for ctx_len in lengths:
            r = await test_context_length(client, inferlet_name, ctx_len)
            d = r.details
            print(
                f"{d.get('context_tokens_target', 0):>10} "
                f"{d.get('total_latency_ms', 0):>10.1f} "
                f"{d.get('per_token_ms', 0):>12.1f} "
                f"{d.get('output_chars', 0):>14}"
            )
            results.append(r)

            # Stop if it's taking too long (> 60s per request)
            if d.get("total_latency_ms", 0) > 60000:
                print(f"  Stopping: latency exceeded 60s at context={ctx_len}")
                break

    finally:
        await client.close()

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Long Context Latency")
    add_common_args(parser)
    parser.add_argument(
        "--max-context", type=int, default=4096,
        help="Maximum context length in tokens to test",
    )
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
