"""Results collection, JSON output, and stdout comparison table."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class QuestionResult:
    """Result for a single question."""
    id: str
    correct: bool
    expected: str
    extracted: str
    raw_output: str
    latency_ms: float
    error: str | None = None


@dataclass
class DatasetResult:
    """Aggregated results for one (engine, dataset) pair."""
    engine: str
    dataset: str
    accuracy: float = 0.0
    total: int = 0
    correct_count: int = 0
    failed: int = 0  # questions that errored (timeout, server error)
    questions: list[QuestionResult] = field(default_factory=list)

    def compute_accuracy(self) -> None:
        answerable = self.total - self.failed
        if answerable > 0:
            self.correct_count = sum(1 for q in self.questions if q.correct)
            self.accuracy = self.correct_count / answerable
        else:
            self.accuracy = 0.0


@dataclass
class EvalResults:
    """Top-level results container."""
    config_summary: dict = field(default_factory=dict)
    dataset_results: list[DatasetResult] = field(default_factory=list)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build comparison table
        comparison: dict[str, dict[str, float]] = {}
        for dr in self.dataset_results:
            comparison.setdefault(dr.dataset, {})[dr.engine] = dr.accuracy

        data = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config_summary,
            "results": [asdict(dr) for dr in self.dataset_results],
            "comparison": comparison,
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Results saved to {path}")

    def print_table(self) -> None:
        """Print a comparison table to stdout."""
        if not self.dataset_results:
            print("No results.")
            return

        # Collect unique engines and datasets (preserve order)
        engines: list[str] = []
        datasets: list[str] = []
        grid: dict[tuple[str, str], float] = {}
        for dr in self.dataset_results:
            if dr.engine not in engines:
                engines.append(dr.engine)
            if dr.dataset not in datasets:
                datasets.append(dr.dataset)
            grid[(dr.engine, dr.dataset)] = dr.accuracy

        # Column widths
        engine_w = max(len("Engine"), max(len(e) for e in engines))
        col_ws = {d: max(len(d), 7) for d in datasets}  # 7 = "100.0%"

        # Header
        header = f"{'Engine':<{engine_w}}"
        for d in datasets:
            header += f" | {d:^{col_ws[d]}}"
        sep = "-" * engine_w
        for d in datasets:
            sep += "-+-" + "-" * col_ws[d]

        print()
        print(header)
        print(sep)

        # Rows
        for eng in engines:
            row = f"{eng:<{engine_w}}"
            for d in datasets:
                acc = grid.get((eng, d))
                cell = f"{acc * 100:.1f}%" if acc is not None else "N/A"
                row += f" | {cell:^{col_ws[d]}}"
            print(row)

        print()
