"""Benchmark: WASM Cold Start vs Warm Start Latency

Measures time from launch_instance to first output for:
- Cold start: first launch (WASM compilation required)
- Warm start: subsequent launches (compiled component cached)

Usage:
    python benches/bench_cold_warm_start.py
    python benches/bench_cold_warm_start.py --warm-runs 20 --server ws://host:port
"""

import argparse
import asyncio
import time
from pathlib import Path

from bench_utils import (
    BenchmarkResult,
    add_common_args,
    connect_and_install,
    latency_stats,
    print_results,
    run_completion,
    save_results,
)
from pie_client import Event


async def measure_launch_latency(
    client, inferlet_name, label: str, num_runs: int,
) -> BenchmarkResult:
    """Measure launch-to-first-output latency over num_runs."""
    name = f"start_latency_{label}"
    prompt = "Say hello."
    latencies = []
    errors = []

    for i in range(num_runs):
        output, latency_ms, event = await run_completion(
            client, inferlet_name, prompt,
            max_tokens=5, temperature=0.0,
        )
        if event == Event.Completed:
            latencies.append(latency_ms)
        else:
            errors.append(f"Run {i}: {event}")

    duration = sum(latencies) / 1000.0 if latencies else 0
    stats = latency_stats(latencies)

    return BenchmarkResult(
        name=name, passed=len(errors) == 0 and len(latencies) > 0,
        duration_sec=duration,
        details={
            "label": label,
            "num_runs": num_runs,
            "completed": len(latencies),
            "latency_ms": {k: round(v, 2) for k, v in stats.items()},
        },
        errors=errors,
    )


async def run_benchmark(args):
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )

    results = []
    try:
        # The install step in connect_and_install already does the first compile.
        # The first launch is still "warmish" (compiled but not instantiated).
        # We measure it separately.
        print("  Measuring first launch (cold-ish)...", end=" ", flush=True)
        r_cold = await measure_launch_latency(client, inferlet_name, "first_launch", 1)
        cold_lat = r_cold.details.get("latency_ms", {}).get("p50", 0)
        print(f"{cold_lat:.0f}ms")
        results.append(r_cold)

        # Warm starts
        print(f"  Measuring warm starts ({args.warm_runs} runs)...", end=" ", flush=True)
        r_warm = await measure_launch_latency(
            client, inferlet_name, "warm", args.warm_runs,
        )
        warm_lat = r_warm.details.get("latency_ms", {}).get("p50", 0)
        print(f"p50={warm_lat:.0f}ms")
        results.append(r_warm)

        # Compare
        if cold_lat > 0 and warm_lat > 0:
            speedup = cold_lat / warm_lat
            print(f"  Warm vs first launch speedup: {speedup:.1f}x")
    finally:
        await client.close()

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Cold/Warm Start Latency")
    add_common_args(parser)
    parser.add_argument("--warm-runs", type=int, default=10, help="Number of warm start runs")
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
