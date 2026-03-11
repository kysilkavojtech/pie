"""Shared utilities for Pie benchmarks.

Provides common helpers for connecting to the server, installing inferlets,
measuring latencies, and formatting results.
"""

import argparse
import asyncio
import json
import math
import sys
import time
import tomllib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from pie_client import PieClient, Event


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BENCH_DIR.parent

DEFAULT_TEXT_COMPLETION_WASM = (
    REPO_ROOT / "std" / "text-completion" / "target"
    / "wasm32-wasip2" / "release" / "text_completion.wasm"
)
DEFAULT_TEXT_COMPLETION_MANIFEST = REPO_ROOT / "std" / "text-completion" / "Pie.toml"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def latency_stats(latencies_ms: list[float]) -> dict:
    """Compute standard latency statistics from a list of latencies in ms."""
    if not latencies_ms:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p90": 0, "p99": 0}
    s = sorted(latencies_ms)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / len(s),
        "p50": percentile(s, 50),
        "p90": percentile(s, 90),
        "p99": percentile(s, 99),
    }


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    passed: bool
    duration_sec: float = 0.0
    details: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = f"[{status}] {self.name} ({self.duration_sec:.2f}s)"
        if self.errors:
            line += f" - {self.errors[0]}"
        return line


def print_results(results: list[BenchmarkResult]) -> None:
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for r in results:
        print(r.summary_line())
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    print("=" * 60)


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmarks": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

async def connect_and_install(
    server: str,
    wasm_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[PieClient, str]:
    """Connect to server, install the text-completion inferlet, return (client, inferlet_name).

    The caller is responsible for closing the client (use as async context manager
    or call client.close()).
    """
    wasm = wasm_path or DEFAULT_TEXT_COMPLETION_WASM
    manifest = manifest_path or DEFAULT_TEXT_COMPLETION_MANIFEST

    if not wasm.exists():
        print(f"Error: WASM not found at {wasm}")
        print("Run `cargo build --target wasm32-wasip2 --release` in std/text-completion/")
        sys.exit(1)
    if not manifest.exists():
        print(f"Error: Manifest not found at {manifest}")
        sys.exit(1)

    pkg = tomllib.loads(manifest.read_text()).get("package", {})
    name = pkg.get("name", "text-completion")
    version = pkg.get("version", "0.1.0")
    inferlet_name = f"{name}@{version}"

    client = PieClient(server)
    await client.connect()
    await client.authenticate("benchmark-user")

    if not await client.program_exists(inferlet_name, wasm, manifest):
        print(f"Installing {inferlet_name}...")
        await client.install_program(wasm, manifest)
    else:
        print(f"{inferlet_name} already installed.")

    return client, inferlet_name


async def run_completion(
    client: PieClient,
    inferlet_name: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    system: str = "You are a helpful assistant.",
) -> tuple[str, float, Event]:
    """Run a single text completion and return (output_text, latency_ms, final_event).

    Uses temperature=0.0 (greedy) by default for deterministic benchmarks.
    """
    args = [
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--system", system,
    ]
    start = time.perf_counter()
    instance = await client.launch_instance(inferlet_name, arguments=args)

    output = ""
    final_event = Event.Completed
    while True:
        event, msg = await instance.recv()
        if event == Event.Stdout:
            output += msg
        elif event == Event.Completed:
            # Completed message contains the return value
            output = msg if msg else output
            final_event = Event.Completed
            break
        elif event in (Event.Exception, Event.Aborted, Event.ServerError, Event.OutOfResources):
            final_event = event
            output = msg
            break

    latency_ms = (time.perf_counter() - start) * 1000.0
    return output, latency_ms, final_event


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add standard arguments shared by all benchmarks."""
    parser.add_argument("--server", default="ws://127.0.0.1:8080", help="Server URI")
    parser.add_argument("--wasm-path", default=None, help="Path to inferlet WASM")
    parser.add_argument("--manifest-path", default=None, help="Path to Pie.toml manifest")
    parser.add_argument("--output-json", default=None, help="Write results to JSON file")
