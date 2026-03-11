"""Benchmark: Client Disconnect Cleanup

Tests that when a client disconnects mid-generation, the server cleans up
properly and continues operating normally. Verifies no resource leaks or
crashes.

Usage:
    python benches/bench_client_disconnect.py
    python benches/bench_client_disconnect.py --server ws://host:port
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
from pie_client import PieClient, Event


async def test_disconnect_and_recover(
    server: str, inferlet_name: str, wasm_path, manifest_path,
) -> list[BenchmarkResult]:
    """Disconnect mid-generation, then verify the server still works."""
    results = []

    # Phase 1: Launch a long generation and disconnect mid-stream
    print("  Phase 1: Launching long generation and disconnecting...", end=" ", flush=True)
    start = time.perf_counter()

    try:
        client1 = PieClient(server)
        await client1.connect()
        await client1.authenticate("benchmark-user")

        instance = await client1.launch_instance(
            inferlet_name,
            arguments=[
                "--prompt", "Write a very long essay about the history of computing, "
                "covering every decade from the 1940s to the 2020s in detail.",
                "--max-tokens", "500",
                "--temperature", "0.6",
            ],
        )

        # Read a few events then disconnect abruptly
        events_received = 0
        for _ in range(5):
            try:
                event, msg = await asyncio.wait_for(instance.recv(), timeout=30.0)
                events_received += 1
                if event in (Event.Completed, Event.Exception, Event.Aborted):
                    break
            except asyncio.TimeoutError:
                break

        # Abrupt disconnect — close without terminating instance
        await client1.close()
        duration1 = time.perf_counter() - start

        results.append(BenchmarkResult(
            name="disconnect_midstream", passed=True, duration_sec=duration1,
            details={"events_before_disconnect": events_received},
        ))
        print(f"OK ({events_received} events received before disconnect)")

    except Exception as e:
        results.append(BenchmarkResult(
            name="disconnect_midstream", passed=False,
            duration_sec=time.perf_counter() - start,
            errors=[str(e)],
        ))
        print(f"FAIL: {e}")

    # Brief pause for server cleanup
    await asyncio.sleep(2.0)

    # Phase 2: Connect fresh and verify server still works
    print("  Phase 2: Verifying server still works...", end=" ", flush=True)
    start = time.perf_counter()

    try:
        client2 = PieClient(server)
        await client2.connect()
        await client2.authenticate("benchmark-user")

        output, latency_ms, event = await run_completion(
            client2, inferlet_name,
            prompt="What is 1 + 1?",
            max_tokens=10, temperature=0.0,
        )
        await client2.close()
        duration2 = time.perf_counter() - start

        passed = event == Event.Completed and len(output) > 0
        results.append(BenchmarkResult(
            name="post_disconnect_recovery", passed=passed, duration_sec=duration2,
            details={
                "output": output[:100],
                "latency_ms": round(latency_ms, 2),
                "event": event.name,
            },
            errors=[] if passed else [f"Expected Completed, got {event.name}"],
        ))
        print(f"{'OK' if passed else 'FAIL'} ({latency_ms:.0f}ms)")

    except Exception as e:
        results.append(BenchmarkResult(
            name="post_disconnect_recovery", passed=False,
            duration_sec=time.perf_counter() - start,
            errors=[f"Server unreachable after disconnect: {e}"],
        ))
        print(f"FAIL: {e}")

    # Phase 3: Multiple rapid disconnects
    print("  Phase 3: Rapid connect/disconnect cycles...", end=" ", flush=True)
    start = time.perf_counter()
    rapid_errors = []

    for i in range(5):
        try:
            c = PieClient(server)
            await c.connect()
            await c.authenticate("benchmark-user")
            inst = await c.launch_instance(
                inferlet_name,
                arguments=["--prompt", "Hello", "--max-tokens", "50", "--temperature", "0.6"],
            )
            # Disconnect immediately without reading anything
            await c.close()
        except Exception as e:
            rapid_errors.append(f"Cycle {i}: {e}")

    await asyncio.sleep(2.0)

    # Verify server is still alive
    try:
        c_final = PieClient(server)
        await c_final.connect()
        await c_final.authenticate("benchmark-user")
        output, _, event = await run_completion(
            c_final, inferlet_name,
            prompt="Are you still working?",
            max_tokens=10, temperature=0.0,
        )
        await c_final.close()
        alive = event == Event.Completed
    except Exception as e:
        alive = False
        rapid_errors.append(f"Final check: {e}")

    duration3 = time.perf_counter() - start
    results.append(BenchmarkResult(
        name="rapid_disconnect_cycles", passed=alive and len(rapid_errors) == 0,
        duration_sec=duration3,
        details={"cycles": 5, "server_alive_after": alive},
        errors=rapid_errors,
    ))
    print(f"{'OK' if alive else 'FAIL'} (server alive: {alive})")

    return results


async def run_benchmark(args):
    # First, ensure the inferlet is installed
    client, inferlet_name = await connect_and_install(
        args.server,
        Path(args.wasm_path) if args.wasm_path else None,
        Path(args.manifest_path) if args.manifest_path else None,
    )
    await client.close()

    results = await test_disconnect_and_recover(
        args.server, inferlet_name,
        args.wasm_path, args.manifest_path,
    )

    print_results(results)
    if args.output_json:
        save_results(results, Path(args.output_json))

    return all(r.passed for r in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Client Disconnect Cleanup")
    add_common_args(parser)
    args = parser.parse_args()
    success = asyncio.run(run_benchmark(args))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
