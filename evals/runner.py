"""Evaluation runner — orchestrates engines, datasets, and result collection."""

import asyncio
import time

from collections.abc import Callable

from .config import EvalConfig, EngineConfig, DatasetConfig
from .engine.base import Engine, GenerationParams
from .engine.openai_engine import OpenAIEngine
from .dataset.base import Dataset, Question
from .dataset.arc import ARCEasy, ARCChallenge
from .results import EvalResults, DatasetResult, QuestionResult


# ---------------------------------------------------------------------------
# Registry — maps config names to constructors
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, type[Dataset] | Callable] = {
    "arc_easy": ARCEasy,
    "arc_challenge": ARCChallenge,
}

# MATH-500 registered lazily to avoid hard dep on math_verify at import time
_math500_registered = False


def _ensure_math500():
    global _math500_registered
    if not _math500_registered:
        try:
            from .dataset.math500 import MATH500
            DATASET_REGISTRY["math500"] = MATH500
        except ImportError:
            pass
        _math500_registered = True


def build_engine(cfg: EngineConfig) -> Engine:
    if cfg.type == "openai":
        return OpenAIEngine(
            name=cfg.name,
            base_url=cfg.base_url,
            model_name=cfg.model_name,
        )
    elif cfg.type == "pie":
        from .engine.pie_engine import PieEngine
        return PieEngine(
            name=cfg.name,
            server=cfg.server,
            wasm_path=cfg.wasm_path,
            manifest_path=cfg.manifest_path,
        )
    else:
        raise ValueError(f"Unknown engine type: {cfg.type!r}")


def build_dataset(cfg: DatasetConfig) -> Dataset:
    _ensure_math500()
    cls = DATASET_REGISTRY.get(cfg.name)
    if cls is None:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset: {cfg.name!r}. Available: {available}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def _run_question(
    engine: Engine,
    dataset: Dataset,
    question: Question,
    params: GenerationParams,
    semaphore: asyncio.Semaphore,
    verbose: bool = False,
) -> QuestionResult:
    """Run a single question through an engine and evaluate."""
    prompt = dataset.format_prompt(question)
    async with semaphore:
        start = time.perf_counter()
        try:
            raw_output = await engine.generate(prompt, params)
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            if verbose:
                print(f"  [{engine.name}] {question.id}: ERROR — {e}")
            return QuestionResult(
                id=question.id,
                correct=False,
                expected=question.correct_answer,
                extracted="",
                raw_output="",
                latency_ms=latency_ms,
                error=str(e),
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

    correct, extracted = dataset.evaluate(question, raw_output)

    if verbose:
        status = "OK" if correct else "WRONG"
        # Truncate raw output for display, collapse whitespace
        raw_preview = " ".join(raw_output.split())[:150]
        print(
            f"  [{engine.name}] {question.id}: {status} "
            f"(expected={question.correct_answer}, extracted={extracted})\n"
            f"    raw: {raw_preview}..."
        )

    return QuestionResult(
        id=question.id,
        correct=correct,
        expected=question.correct_answer,
        extracted=extracted,
        raw_output=raw_output,
        latency_ms=latency_ms,
    )


async def run_eval(
    config: EvalConfig,
    engine_filter: list[str] | None = None,
    dataset_filter: list[str] | None = None,
    limit_override: int | None = None,
    verbose: bool = False,
) -> EvalResults:
    """Run the full evaluation and return results."""
    limit = limit_override if limit_override is not None else config.limit
    if limit == 0:
        limit = None  # load all

    # Build engines and datasets
    engine_cfgs = config.engines
    if engine_filter:
        engine_cfgs = [e for e in engine_cfgs if e.name in engine_filter]
    dataset_cfgs = config.datasets
    if dataset_filter:
        dataset_cfgs = [d for d in dataset_cfgs if d.name in dataset_filter]

    if not engine_cfgs:
        raise ValueError("No engines selected. Check --engines filter.")
    if not dataset_cfgs:
        raise ValueError("No datasets selected. Check --datasets filter.")

    engines = [build_engine(cfg) for cfg in engine_cfgs]
    datasets = [build_dataset(cfg) for cfg in dataset_cfgs]

    # Setup engines
    for eng in engines:
        print(f"Setting up engine: {eng.name}")
        await eng.setup()

    # Load datasets
    loaded: dict[str, list[Question]] = {}
    for ds in datasets:
        print(f"Loading dataset: {ds.name} (limit={limit})")
        loaded[ds.name] = ds.load(limit=limit)
        print(f"  → {len(loaded[ds.name])} questions")

    # Run evaluations
    results = EvalResults(
        config_summary={
            "generation": {
                "max_tokens": config.generation.max_tokens,
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
            },
            "concurrency": config.concurrency,
            "limit": limit,
        }
    )

    for eng in engines:
        for ds in datasets:
            questions = loaded[ds.name]
            print(f"\nRunning {ds.name} on {eng.name} ({len(questions)} questions)...")

            sem = asyncio.Semaphore(config.concurrency)
            tasks = [
                _run_question(eng, ds, q, config.generation, sem, verbose)
                for q in questions
            ]
            question_results = await asyncio.gather(*tasks)

            dr = DatasetResult(
                engine=eng.name,
                dataset=ds.name,
                total=len(questions),
                failed=sum(1 for qr in question_results if qr.error is not None),
                questions=list(question_results),
            )
            dr.compute_accuracy()
            results.dataset_results.append(dr)

            print(
                f"  → {ds.name}/{eng.name}: "
                f"{dr.accuracy * 100:.1f}% "
                f"({dr.correct_count}/{dr.total - dr.failed} correct, "
                f"{dr.failed} errors)"
            )

    # Teardown engines
    for eng in engines:
        await eng.teardown()

    return results
