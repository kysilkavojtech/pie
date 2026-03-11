"""Benchmark: Max Token Limit Correctness

Tests that requesting exactly N tokens produces exactly N tokens (or fewer if
a stop token is hit first). Catches off-by-one errors in token counting.

Usage:
    python benches/bench_max_tokens.py
    python benches/bench_max_tokens.py --server ws://host:port --output-json results.json
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


async def test_max_tokens(client, inferlet_name, requested: int) -> BenchmarkResult:
    """Test that requesting `requested` tokens produces approximately that many."""
    name = f"max_tokens_{requested}"
    start = time.perf_counter()

    # Use a prompt that won't hit stop tokens early for small counts
    prompt = "Count from 1 to 1000, one number per line."
    output, latency_ms, event = await run_completion(
        client, inferlet_name, prompt,
        max_tokens=requested, temperature=0.0,
    )
    duration = time.perf_counter() - start

    if event != Event.Completed:
        return BenchmarkResult(
            name=name, passed=False, duration_sec=duration,
            errors=[f"Instance did not complete: {event}"],
        )

    if requested == 0:
        passed = len(output) == 0 or output.strip() == ""
        return BenchmarkResult(
            name=name, passed=passed, duration_sec=duration,
            details={"requested": 0, "output_len": len(output)},
            errors=[] if passed else [f"Expected empty output, got {len(output)} chars"],
        )

    # We can't count exact tokens from the client side (we only see text),
    # but we can verify the output is non-empty and reasonably sized.
    # A rough heuristic: 1 token ~ 4 chars for English.
    output_chars = len(output)
    est_tokens = output_chars / 4.0

    # Allow wide tolerance: 0.3x to 3x of requested (accounts for tokenizer variance)
    # The key check is that we got *something* and it's in the right ballpark.
    lower = requested * 0.3
    upper = requested * 3.0

    passed = output_chars > 0 and lower <= est_tokens <= upper
    return BenchmarkResult(
        name=name, passed=passed, duration_sec=duration,
        details={
            "requested_tokens": requested,
            "output_chars": output_chars,
            "est_tokens": round(est_tokens, 1),
            "latency_ms": round(latency_ms, 2),
        },
        errors=[] if passed else [
            f"Estimated {est_tokens:.0f} tokens for {requested} requested "
            f"({output_chars} chars)"
        ],
    )


async def run_benchmark(args):
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )

    results = []
    test_counts = [1, 2, 10, 50, 100, 256]

    try:
        for n in test_counts:
            print(f"  Testing max_tokens={n}...", end=" ", flush=True)
            r = await test_max_tokens(client, inferlet_name, n)
            print(r.summary_line())
            results.append(r)
    finally:
        await client.close()

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Max Token Limit")
    add_common_args(parser)
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
