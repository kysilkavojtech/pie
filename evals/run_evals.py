#!/usr/bin/env python3
"""CLI entry point for the Pie accuracy evaluation framework.

Usage:
    python evals/run_evals.py --config evals/eval_config.toml
    python evals/run_evals.py --config evals/eval_config.toml --datasets arc_easy --engines vllm --limit 5
    python evals/run_evals.py --config evals/eval_config.toml --output-json evals/results/run1.json
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .runner import run_eval



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pie Accuracy Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset filter (e.g. arc_easy,math500)",
    )
    parser.add_argument(
        "--engines", default=None,
        help="Comma-separated engine filter (e.g. pie,vllm)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions per dataset (overrides config)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Max concurrent requests per engine (overrides config)",
    )
    parser.add_argument(
        "--output-json", default=None, type=Path,
        help="Output JSON path (default: <output_dir>/<timestamp>.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-question results as they complete",
    )
    parser.add_argument(
        "--no-think", action="store_true",
        help="Append /no_think to prompts (disables thinking mode on Qwen3, etc.)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.concurrency is not None:
        config.concurrency = args.concurrency

    dataset_filter = args.datasets.split(",") if args.datasets else None
    engine_filter = args.engines.split(",") if args.engines else None

    results = asyncio.run(
        run_eval(
            config,
            engine_filter=engine_filter,
            dataset_filter=dataset_filter,
            limit_override=args.limit,
            verbose=args.verbose,
            no_think=args.no_think,
        )
    )

    results.print_table()

    output_path = args.output_json
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path(config.output_dir) / f"eval_{ts}.json"
    results.save_json(output_path)


if __name__ == "__main__":
    main()
