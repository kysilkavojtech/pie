"""Benchmark: Throughput vs Concurrency Scaling

Measures tokens/sec and per-instance latency as concurrent inferlet instances
scale from 1 to N. Identifies the saturation point and checks for throughput
cliffs.

Usage:
    python benches/bench_concurrent_scaling.py
    python benches/bench_concurrent_scaling.py --max-concurrency 128 --server ws://host:port
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


async def test_concurrency_level(
    client, inferlet_name, concurrency: int, requests_per_level: int,
) -> BenchmarkResult:
    """Run requests_per_level requests at a given concurrency, measure throughput."""
    name = f"concurrency_{concurrency}"
    prompt = "Write a short paragraph about distributed systems."
    max_tokens = 50

    latencies = []
    failures = 0
    total_chars = 0

    queue = asyncio.Queue()
    for i in range(requests_per_level):
        queue.put_nowait(i)

    async def worker():
        nonlocal failures, total_chars
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            output, latency_ms, event = await run_completion(
                client, inferlet_name, prompt,
                max_tokens=max_tokens, temperature=0.6,
            )
            if event == Event.Completed:
                latencies.append(latency_ms)
                total_chars += len(output)
            else:
                failures += 1
            queue.task_done()

    start = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    duration = time.perf_counter() - start

    completed = len(latencies)
    est_tokens = total_chars / 4.0
    throughput_tps = est_tokens / duration if duration > 0 else 0
    throughput_rps = completed / duration if duration > 0 else 0

    stats = latency_stats(latencies)

    return BenchmarkResult(
        name=name, passed=failures == 0, duration_sec=duration,
        details={
            "concurrency": concurrency,
            "requests": requests_per_level,
            "completed": completed,
            "failures": failures,
            "throughput_rps": round(throughput_rps, 2),
            "throughput_est_tps": round(throughput_tps, 2),
            "total_chars": total_chars,
            "latency_ms": {k: round(v, 2) for k, v in stats.items()},
        },
    )


async def run_benchmark(args):
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )

    # Scale concurrency: 1, 2, 4, 8, ... up to max
    levels = []
    c = 1
    while c <= args.max_concurrency:
        levels.append(c)
        c *= 2

    results = []
    try:
        for level in levels:
            reqs = max(level, args.requests_per_level)
            print(f"  Concurrency={level} ({reqs} requests)...", end=" ", flush=True)
            r = await test_concurrency_level(
                client, inferlet_name, level, reqs,
            )
            tps = r.details.get("throughput_est_tps", 0)
            p50 = r.details.get("latency_ms", {}).get("p50", 0)
            print(f"{tps:.0f} tok/s, p50={p50:.0f}ms")
            results.append(r)
    finally:
        await client.close()

    # Print scaling summary
    print("\n--- Scaling Summary ---")
    print(f"{'Concurrency':>12} {'Tok/s':>10} {'Req/s':>10} {'p50 ms':>10} {'p99 ms':>10}")
    for r in results:
        d = r.details
        lat = d.get("latency_ms", {})
        print(
            f"{d['concurrency']:>12} "
            f"{d['throughput_est_tps']:>10.1f} "
            f"{d['throughput_rps']:>10.2f} "
            f"{lat.get('p50', 0):>10.1f} "
            f"{lat.get('p99', 0):>10.1f}"
        )

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Throughput vs Concurrency")
    add_common_args(parser)
    parser.add_argument(
        "--max-concurrency", type=int, default=64,
        help="Maximum concurrency level to test",
    )
    parser.add_argument(
        "--requests-per-level", type=int, default=16,
        help="Minimum requests per concurrency level",
    )
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
