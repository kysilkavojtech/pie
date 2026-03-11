"""Benchmark: Stress Test — Max Concurrent Instances

Scales concurrent instances until failures appear. Measures the saturation
point and verifies graceful degradation (no crashes, clean errors).

Usage:
    python benches/bench_stress.py
    python benches/bench_stress.py --max-instances 256 --server ws://host:port
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
    save_results,
)
from pie_client import Event


async def stress_at_level(client, inferlet_name, num_instances: int) -> dict:
    """Launch num_instances concurrently, measure success/failure/latency."""
    prompt = "Say hello in one sentence."
    max_tokens = 20

    latencies = []
    failures_by_type = {}
    lock = asyncio.Lock()

    async def single_request():
        args = [
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", "0.6",
        ]
        start = time.perf_counter()
        try:
            instance = await asyncio.wait_for(
                client.launch_instance(inferlet_name, arguments=args),
                timeout=60.0,
            )
            while True:
                event, msg = await asyncio.wait_for(instance.recv(), timeout=60.0)
                if event == Event.Completed:
                    lat = (time.perf_counter() - start) * 1000.0
                    async with lock:
                        latencies.append(lat)
                    return
                elif event in (Event.Exception, Event.Aborted, Event.ServerError,
                               Event.OutOfResources):
                    key = event.name
                    async with lock:
                        failures_by_type[key] = failures_by_type.get(key, 0) + 1
                    return
        except asyncio.TimeoutError:
            async with lock:
                failures_by_type["Timeout"] = failures_by_type.get("Timeout", 0) + 1
        except Exception as e:
            async with lock:
                key = type(e).__name__
                failures_by_type[key] = failures_by_type.get(key, 0) + 1

    start = time.perf_counter()
    tasks = [asyncio.create_task(single_request()) for _ in range(num_instances)]
    await asyncio.gather(*tasks)
    duration = time.perf_counter() - start

    completed = len(latencies)
    failed = sum(failures_by_type.values())
    stats = latency_stats(latencies)

    return {
        "num_instances": num_instances,
        "completed": completed,
        "failed": failed,
        "failures_by_type": failures_by_type,
        "duration_sec": round(duration, 2),
        "throughput_rps": round(completed / duration, 2) if duration > 0 else 0,
        "latency_ms": {k: round(v, 2) for k, v in stats.items()},
    }


async def run_benchmark(args):
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )

    levels = []
    c = 1
    while c <= args.max_instances:
        levels.append(c)
        c *= 2

    results = []
    saturation_point = None

    try:
        print(f"{'Instances':>10} {'OK':>6} {'Fail':>6} {'Req/s':>8} {'p50 ms':>10} {'p99 ms':>10}")
        print("-" * 55)

        for level in levels:
            data = await stress_at_level(client, inferlet_name, level)
            lat = data["latency_ms"]
            print(
                f"{data['num_instances']:>10} "
                f"{data['completed']:>6} "
                f"{data['failed']:>6} "
                f"{data['throughput_rps']:>8.1f} "
                f"{lat.get('p50', 0):>10.1f} "
                f"{lat.get('p99', 0):>10.1f}"
            )

            passed = data["failed"] == 0
            results.append(BenchmarkResult(
                name=f"stress_{level}", passed=passed,
                duration_sec=data["duration_sec"],
                details=data,
                errors=[f"Failures: {data['failures_by_type']}"] if not passed else [],
            ))

            if not passed and saturation_point is None:
                saturation_point = level

            # If more than half failed, stop escalating
            if data["failed"] > data["completed"]:
                print(f"  Stopping: majority failures at {level} instances")
                break

    finally:
        await client.close()

    if saturation_point:
        print(f"\nFirst failures at {saturation_point} concurrent instances")
    else:
        print(f"\nNo failures up to {levels[-1]} concurrent instances")

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    # Pass if at least the first level (1 instance) succeeded
    return results[0].passed if results else False


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Stress Test")
    add_common_args(parser)
    parser.add_argument(
        "--max-instances", type=int, default=128,
        help="Maximum concurrent instances to test",
    )
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
