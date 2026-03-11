"""Benchmark: Cross-Run Determinism

Tests that greedy decoding (temperature=0) produces identical output across
multiple runs. Non-determinism indicates floating-point instability, race
conditions in the scheduler, or seeding bugs.

Usage:
    python benches/bench_determinism.py
    python benches/bench_determinism.py --runs 10 --server ws://host:port
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


TEST_PROMPTS = [
    "What is 2 + 2?",
    "List the first 5 prime numbers.",
    "Write a haiku about the ocean.",
]


async def test_determinism(
    client, inferlet_name, prompt: str, num_runs: int,
) -> BenchmarkResult:
    """Run the same prompt num_runs times with temperature=0, check outputs match."""
    name = f"determinism_{prompt[:30].replace(' ', '_')}"
    start = time.perf_counter()

    outputs = []
    latencies = []
    for i in range(num_runs):
        output, latency_ms, event = await run_completion(
            client, inferlet_name, prompt,
            max_tokens=50, temperature=0.0,
        )
        if event != Event.Completed:
            return BenchmarkResult(
                name=name, passed=False,
                duration_sec=time.perf_counter() - start,
                errors=[f"Run {i} did not complete: {event}"],
            )
        outputs.append(output)
        latencies.append(latency_ms)

    duration = time.perf_counter() - start

    # All outputs should be identical
    unique_outputs = set(outputs)
    passed = len(unique_outputs) == 1

    details = {
        "prompt": prompt,
        "num_runs": num_runs,
        "unique_outputs": len(unique_outputs),
        "output_length": len(outputs[0]) if outputs else 0,
        "latencies_ms": [round(l, 2) for l in latencies],
    }

    errors = []
    if not passed:
        # Show the differences
        for i, o in enumerate(outputs):
            if o != outputs[0]:
                # Find first divergence point
                for j, (a, b) in enumerate(zip(outputs[0], o)):
                    if a != b:
                        errors.append(
                            f"Run {i} diverges at char {j}: "
                            f"'{outputs[0][max(0,j-10):j+10]}' vs '{o[max(0,j-10):j+10]}'"
                        )
                        break
                else:
                    errors.append(
                        f"Run {i} differs in length: {len(outputs[0])} vs {len(o)}"
                    )

    return BenchmarkResult(
        name=name, passed=passed, duration_sec=duration,
        details=details, errors=errors,
    )


async def run_benchmark(args):
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )

    results = []
    try:
        for prompt in TEST_PROMPTS:
            print(f"  Testing determinism: '{prompt[:40]}'...", end=" ", flush=True)
            r = await test_determinism(client, inferlet_name, prompt, args.runs)
            print(r.summary_line())
            results.append(r)
    finally:
        await client.close()

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Cross-Run Determinism")
    add_common_args(parser)
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per prompt")
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
