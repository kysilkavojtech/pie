"""Harness runner — runs evals via lm-evaluation-harness (EleutherAI).

This module bridges our CLI/config with ``lm_eval.simple_evaluate()``,
mapping our engine configs to lm-eval model types and converting lm-eval
results back to our ``EvalResults`` format for unified output.

Engine/task compatibility:
  - OpenAI engines (vLLM, SGLang) use ``local-completions`` which supports
    both loglikelihood (MCQ) and generate_until tasks.
  - Pie engine uses our custom PieLM which only supports generate_until.
    Loglikelihood-based tasks (ARC MCQ) will error with Pie — use
    ``--backend custom`` for those.
"""

import httpx
import lm_eval

from .config import EvalConfig, EngineConfig
from .results import EvalResults, DatasetResult, QuestionResult


# ---------------------------------------------------------------------------
# Task name mapping: our names → lm-eval task names
# ---------------------------------------------------------------------------
# Verify available tasks with: lm_eval --tasks list
TASK_NAME_MAP: dict[str, str] = {
    "arc_easy": "arc_easy",          # MCQ (loglikelihood) — OpenAI engines only
    "arc_challenge": "arc_challenge",  # MCQ (loglikelihood) — OpenAI engines only
    "math500": "minerva_math500",    # generate_until — all engines
}

# Tasks that require loglikelihood (not supported by Pie)
_LOGLIKELIHOOD_TASKS: set[str] = {"arc_easy", "arc_challenge"}

# Reverse map for result conversion
_TASK_NAME_REVERSE: dict[str, str] = {v: k for k, v in TASK_NAME_MAP.items()}


def _map_task_names(our_names: list[str]) -> list[str]:
    """Convert our dataset names to lm-eval task names."""
    lm_tasks = []
    for name in our_names:
        mapped = TASK_NAME_MAP.get(name)
        if mapped is None:
            # Pass through — the user may be using a native lm-eval task name
            mapped = name
        lm_tasks.append(mapped)
    return lm_tasks


def _our_name(lm_task: str) -> str:
    """Convert an lm-eval task name back to ours."""
    return _TASK_NAME_REVERSE.get(lm_task, lm_task)


def _detect_model_name(base_url: str) -> str:
    """Query /v1/models to discover the served model name."""
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    try:
        resp = httpx.get(f"{api_base}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        print(f"  Warning: could not auto-detect model name from {api_base}/models: {e}")
    return "default"


def _build_model_spec(engine_cfg: EngineConfig) -> tuple[str, str]:
    """Map an engine config to (model_type, model_args) for lm-eval.

    Returns:
        (model_type, model_args_string) suitable for ``lm_eval.simple_evaluate()``.
    """
    if engine_cfg.type == "openai":
        # Use local-completions (not local-chat-completions) because it
        # supports both loglikelihood (MCQ tasks like ARC) and generate_until
        # (math tasks). vLLM and SGLang both expose /v1/completions.
        base_url = engine_cfg.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        completions_url = f"{base_url}/completions"

        # Model name is required (used for API calls + tokenizer loading)
        model_name = engine_cfg.model_name
        if not model_name:
            model_name = _detect_model_name(engine_cfg.base_url)
            print(f"  Auto-detected model: {model_name}")

        parts = [
            f"model={model_name}",
            f"base_url={completions_url}",
            "num_concurrent=4",
            "tokenized_requests=False",
        ]
        return "local-completions", ",".join(parts)

    elif engine_cfg.type == "pie":
        # Use our custom PieLM (registered via @register_model("pie"))
        # Import triggers the registration side-effect
        from .engine import lm_eval_pie as _  # noqa: F401

        parts = [f"server={engine_cfg.server}"]
        if engine_cfg.wasm_path:
            parts.append(f"wasm_path={engine_cfg.wasm_path}")
        if engine_cfg.manifest_path:
            parts.append(f"manifest_path={engine_cfg.manifest_path}")
        return "pie", ",".join(parts)

    else:
        raise ValueError(f"Unknown engine type for harness backend: {engine_cfg.type!r}")


def _extract_accuracy(task_results: dict) -> float:
    """Extract the primary accuracy metric from lm-eval task results.

    lm-eval reports metrics like ``acc,none``, ``acc_norm,none``, ``exact_match,none``.
    We prefer ``acc_norm`` → ``acc`` → ``exact_match`` in that order.
    """
    for key in ("acc_norm,none", "acc,none", "exact_match,none"):
        if key in task_results:
            return task_results[key]
    # Fallback: take the first numeric value that looks like accuracy
    for key, val in task_results.items():
        if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
            return val
    return 0.0


def _get_filtered_resp(sample: dict) -> str:
    """Extract the filtered (post-extraction) response from an lm-eval sample.

    ``filtered_resps`` can be:
      - a list of strings: ``["9"]``
      - a dict keyed by filter name: ``{"flexible-extract": ["9"]}``
    """
    filtered = sample.get("filtered_resps")
    if filtered is None:
        return ""
    if isinstance(filtered, dict):
        # Take the first filter's result
        for val in filtered.values():
            if isinstance(val, list) and val:
                return str(val[0])
            return str(val)
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    return str(filtered)


def _convert_samples(
    lm_task: str,
    samples: list[dict],
    engine_name: str,
    verbose: bool = False,
) -> DatasetResult:
    """Convert lm-eval per-sample results to our DatasetResult format."""
    our_name = _our_name(lm_task)
    question_results = []

    # Debug: print keys of the first sample so we can see what lm-eval returns
    if samples and verbose:
        s = samples[0]
        print(f"  [debug] sample keys: {list(s.keys())}")
        print(f"  [debug] filtered_resps type={type(s.get('filtered_resps'))}, "
              f"value={repr(s.get('filtered_resps'))[:200]}")

    for sample in samples:
        doc_id = sample.get("doc_id", len(question_results))
        target = str(sample.get("target", ""))

        # lm-eval stores responses in "resps" (list of list of strings)
        resps = sample.get("resps", [[]])
        raw_output = resps[0][0] if resps and resps[0] else ""

        # 1. Get the filtered/extracted answer from lm-eval's post-processing
        extracted = _get_filtered_resp(sample)

        # 2. Determine correctness — check per-sample metric keys that
        #    lm-eval may attach (varies by task and version)
        correct = None
        for key in ("exact_match", "acc", "acc_norm",
                     "exact_match,none", "acc,none", "acc_norm,none"):
            if key in sample:
                correct = bool(sample[key])
                break

        # 3. Fallback: compare extracted answer to target
        if correct is None and extracted:
            correct = extracted.strip() == target.strip()

        if correct is None:
            correct = False

        if not extracted:
            extracted = str(raw_output)[:200]

        question_results.append(
            QuestionResult(
                id=str(doc_id),
                correct=correct,
                expected=target,
                extracted=extracted,
                raw_output=str(raw_output),
                latency_ms=0.0,  # lm-eval doesn't report per-question latency
            )
        )

    dr = DatasetResult(
        engine=engine_name,
        dataset=our_name,
        total=len(question_results),
        failed=0,
        questions=question_results,
    )
    dr.compute_accuracy()
    return dr


def run_harness_eval(
    config: EvalConfig,
    engine_filter: list[str] | None = None,
    dataset_filter: list[str] | None = None,
    limit_override: int | None = None,
    verbose: bool = False,
    no_think: bool = False,
) -> EvalResults:
    """Run evaluations using lm-evaluation-harness.

    This is a synchronous function (lm_eval.simple_evaluate is sync).
    """
    # Determine limit
    limit = limit_override if limit_override is not None else config.limit
    if limit == 0:
        limit = None  # lm-eval uses None for "all"

    # Filter engines and datasets
    engine_cfgs = config.engines
    if engine_filter:
        engine_cfgs = [e for e in engine_cfgs if e.name in engine_filter]
    dataset_names = [d.name for d in config.datasets]
    if dataset_filter:
        dataset_names = [d for d in dataset_names if d in dataset_filter]

    if not engine_cfgs:
        raise ValueError("No engines selected. Check --engines filter.")
    if not dataset_names:
        raise ValueError("No datasets selected. Check --datasets filter.")

    lm_tasks = _map_task_names(dataset_names)

    # Build gen_kwargs from our config
    gen_kwargs_parts = [
        f"temperature={config.generation.temperature}",
        f"top_p={config.generation.top_p}",
        f"max_gen_toks={config.generation.max_tokens}",
    ]
    if no_think:
        # Append /no_think instruction to system prompt
        gen_kwargs_parts.append("do_sample=False")
    gen_kwargs_str = ",".join(gen_kwargs_parts)

    # Harness config
    num_fewshot = config.harness_num_fewshot

    results = EvalResults(
        config_summary={
            "backend": "harness",
            "generation": {
                "max_tokens": config.generation.max_tokens,
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
            },
            "concurrency": config.concurrency,
            "limit": limit,
            "num_fewshot": num_fewshot,
        }
    )

    for engine_cfg in engine_cfgs:
        model_type, model_args = _build_model_spec(engine_cfg)

        # Filter out loglikelihood tasks for Pie (only supports generate_until)
        engine_tasks = lm_tasks
        if engine_cfg.type == "pie":
            skipped = [t for t in lm_tasks if _our_name(t) in _LOGLIKELIHOOD_TASKS]
            engine_tasks = [t for t in lm_tasks if _our_name(t) not in _LOGLIKELIHOOD_TASKS]
            if skipped:
                skipped_names = [_our_name(t) for t in skipped]
                print(f"\n  Skipping {skipped_names} for {engine_cfg.name} "
                      f"(loglikelihood tasks not supported — use --backend custom)")
            if not engine_tasks:
                print(f"  No compatible tasks for {engine_cfg.name}, skipping.")
                continue

        print(f"\nRunning lm-eval: engine={engine_cfg.name} ({model_type}), "
              f"tasks={engine_tasks}, limit={limit}")

        lm_results = lm_eval.simple_evaluate(
            model=model_type,
            model_args=model_args,
            tasks=engine_tasks,
            num_fewshot=num_fewshot,
            batch_size=1,  # API models process one request at a time (concurrency handled internally)
            limit=limit,
            log_samples=True,
            gen_kwargs=gen_kwargs_str,
            verbosity="DEBUG" if verbose else "INFO",
        )

        # Extract per-task results
        task_results = lm_results.get("results", {})
        task_samples = lm_results.get("samples", {})

        for lm_task in engine_tasks:
            our_name = _our_name(lm_task)

            if lm_task in task_results:
                acc = _extract_accuracy(task_results[lm_task])
                print(f"  → {our_name}/{engine_cfg.name}: {acc * 100:.1f}% (lm-eval)")
                if verbose:
                    print(f"  [debug] task-level metrics: {task_results[lm_task]}")

            # Convert per-sample results if available
            if lm_task in task_samples:
                dr = _convert_samples(
                    lm_task, task_samples[lm_task], engine_cfg.name,
                    verbose=verbose,
                )
                results.dataset_results.append(dr)
            elif lm_task in task_results:
                # No per-sample data, create a summary-only DatasetResult
                acc = _extract_accuracy(task_results[lm_task])
                total = task_results[lm_task].get("samples", 0)
                dr = DatasetResult(
                    engine=engine_cfg.name,
                    dataset=our_name,
                    accuracy=acc,
                    total=total,
                    correct_count=int(acc * total) if total else 0,
                )
                results.dataset_results.append(dr)

    return results
