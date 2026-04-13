"""TOML config loading for the eval framework."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .engine.base import GenerationParams


@dataclass
class EngineConfig:
    name: str
    type: str  # "pie" or "openai"
    # OpenAI-engine fields
    base_url: str = "http://localhost:8000"
    model_name: str | None = None
    # Pie-engine fields
    server: str = "ws://127.0.0.1:8080"
    wasm_path: str | None = None
    manifest_path: str | None = None


@dataclass
class DatasetConfig:
    name: str
    split: str = "test"


@dataclass
class EvalConfig:
    generation: GenerationParams
    engines: list[EngineConfig]
    datasets: list[DatasetConfig]
    concurrency: int = 4
    limit: int = 0  # 0 = all questions
    output_dir: str = "evals/results"
    # Harness backend settings
    harness_num_fewshot: int | None = None  # None = use task default


def load_config(path: Path) -> EvalConfig:
    """Load an eval config from a TOML file."""
    raw = tomllib.loads(path.read_text())

    run = raw.get("run", {})
    gen_raw = raw.get("generation", {})

    generation = GenerationParams(
        max_tokens=gen_raw.get("max_tokens", 2048),
        temperature=gen_raw.get("temperature", 0.0),
        top_p=gen_raw.get("top_p", 1.0),
        system=gen_raw.get("system", "You are a helpful assistant."),
    )

    engines = []
    for e in raw.get("engine", []):
        engines.append(
            EngineConfig(
                name=e["name"],
                type=e["type"],
                base_url=e.get("base_url", "http://localhost:8000"),
                model_name=e.get("model_name"),
                server=e.get("server", "ws://127.0.0.1:8080"),
                wasm_path=e.get("wasm_path"),
                manifest_path=e.get("manifest_path"),
            )
        )

    datasets = []
    for d in raw.get("dataset", []):
        datasets.append(
            DatasetConfig(
                name=d["name"],
                split=d.get("split", "test"),
            )
        )

    # Optional [harness] section
    harness_raw = raw.get("harness", {})

    return EvalConfig(
        generation=generation,
        engines=engines,
        datasets=datasets,
        concurrency=run.get("concurrency", 4),
        limit=run.get("limit", 0),
        output_dir=run.get("output_dir", "evals/results"),
        harness_num_fewshot=harness_raw.get("num_fewshot"),
    )
