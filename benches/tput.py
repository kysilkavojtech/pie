import asyncio
import argparse
import json
import math
import time
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from pie_client import PieClient, Event


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


async def run_benchmark(args):
    # 1. Setup paths
    script_dir = Path(__file__).parent.resolve()
    # Default paths assuming standard repository layout.
    default_wasm_path = (
        script_dir.parent
        / "std"
        / "text-completion"
        / "target"
        / "wasm32-wasip2"
        / "release"
        / "text_completion.wasm"
    )
    default_manifest_path = script_dir.parent / "std" / "text-completion" / "Pie.toml"

    wasm_path = (
        Path(args.wasm_path).expanduser().resolve()
        if args.wasm_path
        else default_wasm_path
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else default_manifest_path
    )

    if not wasm_path.exists():
        print(f"Error: WASM binary not found at {wasm_path}")
        print(
            "Please run `cargo build --target wasm32-wasip2 --release` in `text-completion` first."
        )
        sys.exit(1)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    print(f"Using WASM: {wasm_path}")
    print(f"Using Manifest: {manifest_path}")
    manifest = tomllib.loads(manifest_path.read_text())
    package = manifest.get("package")
    if not package:
        print("Error: Manifest missing [package] section")
        sys.exit(1)
    package_name = package.get("name")
    version = package.get("version")
    if not package_name or not version:
        print("Error: Manifest missing package.name or package.version")
        sys.exit(1)

    # Current naming convention across the repo is "name@version".
    # Keep package_name as-is to support both plain and namespaced names.
    inferlet_name = f"{package_name}@{version}"
    print(f"Inferlet: {inferlet_name}")

    # 2. Connect to server
    print(f"Connecting to {args.server}...")
    async with PieClient(args.server) as client:
        await client.authenticate("benchmark-user")

        # 3. Install program (check both name and hashes match)
        if not await client.program_exists(inferlet_name, wasm_path, manifest_path):
            print("Installing program...")
            await client.install_program(wasm_path, manifest_path)
        else:
            print("Program already exists on server.")

        # 4. Prepare workload
        print(
            f"Starting benchmark: {args.num_requests} requests, concurrency {args.concurrency}"
        )
        print(f"Prompt: {args.prompt}")
        print(f"Max Tokens: {args.max_tokens}")

        inferlet_args = [
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--system",
            "You are a helpful benchmarking assistant.",
        ]

        # 5. Execution Loop
        start_time = time.time()
        completed = 0
        total_chars = 0
        total_tokens_est = 0
        latencies_ms = []
        failures_by_reason = {}

        queue = asyncio.Queue()
        for i in range(args.num_requests):
            queue.put_nowait(i)

        async def worker(worker_id):
            nonlocal completed, total_chars, total_tokens_est
            while not queue.empty():
                try:
                    req_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                # Launch instance
                try:
                    request_start = time.perf_counter()
                    instance = await client.launch_instance(
                        inferlet_name, arguments=inferlet_args
                    )
                    while True:
                        event, msg = await instance.recv()
                        if event == Event.Completed:
                            text = msg
                            chars = len(text)
                            tokens = chars / 4.0
                            latency_ms = (time.perf_counter() - request_start) * 1000.0

                            total_chars += chars
                            total_tokens_est += tokens
                            latencies_ms.append(latency_ms)
                            completed += 1
                            print(".", end="", flush=True)
                            break
                        elif event == Event.Exception:
                            print(f"[{worker_id}] Req {req_id} failed: {msg}")
                            failures_by_reason["Event.Exception"] = (
                                failures_by_reason.get("Event.Exception", 0) + 1
                            )
                            break
                        # Handle other potential closing events
                        elif event in (
                            Event.Aborted,
                            Event.ServerError,
                            Event.OutOfResources,
                        ):
                            print(
                                f"[{worker_id}] Req {req_id} aborted/failed: {event} {msg}"
                            )
                            key = f"Event.{event.name}"
                            failures_by_reason[key] = failures_by_reason.get(key, 0) + 1
                            break
                except Exception as e:
                    print(f"[{worker_id}] Error: {e}")
                    failures_by_reason["Worker.Exception"] = (
                        failures_by_reason.get("Worker.Exception", 0) + 1
                    )
                finally:
                    queue.task_done()

        # Creates workers
        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        await asyncio.wait(workers)

        duration = time.time() - start_time
        failed = args.num_requests - completed
        success_rate = completed / args.num_requests if args.num_requests else 0.0
        sorted_latencies = sorted(latencies_ms)

        p50_ms = _percentile(sorted_latencies, 50)
        p90_ms = _percentile(sorted_latencies, 90)
        p99_ms = _percentile(sorted_latencies, 99)

        print("\n--- Benchmark Results ---")
        print(f"Total Time: {duration:.2f} s")
        print(f"Total Requests: {completed}/{args.num_requests}")
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Failures: {failed}")
        print(f"Total Chars: {total_chars}")
        print(f"Est. Total Tokens: {total_tokens_est:.0f}")
        req_throughput = completed / duration if duration > 0 else 0.0
        token_throughput = total_tokens_est / duration if duration > 0 else 0.0
        print(f"Throughput (Requests/sec): {req_throughput:.2f}")
        print(f"Throughput (Est. Tokens/sec): {token_throughput:.2f}")
        print(f"Latency p50/p90/p99 (ms): {p50_ms:.2f} / {p90_ms:.2f} / {p99_ms:.2f}")
        if failures_by_reason:
            print("Failure Breakdown:")
            for reason, count in sorted(failures_by_reason.items()):
                print(f"  - {reason}: {count}")

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "server": args.server,
                "run_id": args.run_id,
                "model": args.model,
                "gpu": args.gpu,
                "num_requests": args.num_requests,
                "concurrency": args.concurrency,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "duration_sec": duration,
                "completed_requests": completed,
                "failed_requests": failed,
                "success_rate": success_rate,
                "total_chars": total_chars,
                "total_tokens_est": total_tokens_est,
                "throughput_rps": req_throughput,
                "throughput_tokens_per_sec_est": token_throughput,
                "latency_ms": {
                    "p50": p50_ms,
                    "p90": p90_ms,
                    "p99": p99_ms,
                    "min": min(sorted_latencies) if sorted_latencies else 0.0,
                    "max": max(sorted_latencies) if sorted_latencies else 0.0,
                    "mean": (
                        sum(sorted_latencies) / len(sorted_latencies)
                        if sorted_latencies
                        else 0.0
                    ),
                },
                "failure_breakdown": failures_by_reason,
            }
            output_path.write_text(json.dumps(summary, indent=2))
            print(f"Saved benchmark JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pie Throughput Benchmark")
    parser.add_argument("--server", default="ws://127.0.0.1:8080", help="Server URI")
    parser.add_argument(
        "--num-requests", type=int, default=64, help="Total number of requests"
    )
    parser.add_argument(
        "--concurrency", type=int, default=64, help="Concurrent requests"
    )
    parser.add_argument(
        "--prompt", default="Write a short story about a robot.", help="Prompt to use"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate per request"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--wasm-path", default=None, help="Path to inferlet WASM")
    parser.add_argument("--manifest-path", default=None, help="Path to Pie.toml manifest")
    parser.add_argument("--output-json", default=None, help="Write benchmark summary JSON")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument("--model", default=None, help="Model name for metadata only")
    parser.add_argument("--gpu", default=None, help="GPU label for metadata only")

    args = parser.parse_args()

    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")


if __name__ == "__main__":
    main()
