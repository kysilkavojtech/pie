"""Pie as an lm-evaluation-harness model.

Wraps Pie's inferlets as an ``lm_eval.api.model.LM`` subclass so that
lm-eval-harness can drive inference through Pie.

Supports:
- ``generate_until`` — via the text-completion inferlet
- ``loglikelihood`` — via the loglikelihood inferlet (context forking +
  decode_step_dist for token-level log probabilities)
"""

import asyncio
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from pie_client import PieClient, Event


# Default paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Text-completion inferlet (generate_until)
_DEFAULT_WASM = (
    _REPO_ROOT / "std" / "text-completion" / "target"
    / "wasm32-wasip2" / "release" / "text_completion.wasm"
)
_DEFAULT_MANIFEST = _REPO_ROOT / "std" / "text-completion" / "Pie.toml"

# Loglikelihood inferlet
_DEFAULT_LL_WASM = (
    _REPO_ROOT / "std" / "loglikelihood" / "target"
    / "wasm32-wasip2" / "release" / "loglikelihood.wasm"
)
_DEFAULT_LL_MANIFEST = _REPO_ROOT / "std" / "loglikelihood" / "Pie.toml"


def _install_inferlet(loop, client, wasm, manifest, label):
    """Install an inferlet if not already present. Returns the inferlet name."""
    if not wasm.exists():
        print(f"Error: WASM not found at {wasm}")
        print(f"Run `cargo build --target wasm32-wasip2 --release` in {wasm.parent.parent.parent}")
        sys.exit(1)
    if not manifest.exists():
        print(f"Error: Manifest not found at {manifest}")
        sys.exit(1)

    pkg = tomllib.loads(manifest.read_text()).get("package", {})
    name = f"{pkg.get('name', label)}@{pkg.get('version', '0.1.0')}"

    if not loop.run_until_complete(client.program_exists(name, wasm, manifest)):
        print(f"  Installing {name}...")
        loop.run_until_complete(client.install_program(wasm, manifest))
    else:
        print(f"  {name} already installed.")
    return name


@register_model("pie")
class PieLM(LM):
    """lm-eval model backed by a running Pie server.

    Uses the text-completion inferlet for generation and the loglikelihood
    inferlet for scoring continuations. Instantiated by lm-eval via
    ``model_args`` string, e.g.::

        model_args="server=ws://127.0.0.1:8080,num_concurrent=4"
    """

    def __init__(
        self,
        server: str = "ws://127.0.0.1:8080",
        wasm_path: str | None = None,
        manifest_path: str | None = None,
        num_concurrent: int = 4,
        max_gen_toks: int = 8192,
        batch_size: int | str = 1,
        **kwargs,
    ):
        super().__init__()
        self.server = server
        self.wasm = Path(wasm_path) if wasm_path else _DEFAULT_WASM
        self.manifest = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
        self.num_concurrent = int(num_concurrent)
        self._max_gen_toks = int(max_gen_toks)
        self._batch_size = batch_size

        # Will be set during _setup()
        self._client: PieClient | None = None
        self._inferlet_name: str | None = None
        self._ll_inferlet_name: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Eagerly set up connection
        self._setup()

    def _setup(self) -> None:
        """Connect to Pie server and install both inferlets."""
        self._loop = asyncio.new_event_loop()
        self._client = PieClient(self.server)
        self._loop.run_until_complete(self._client.connect())
        self._loop.run_until_complete(self._client.authenticate("eval-user"))

        # Install text-completion (generate_until)
        self._inferlet_name = _install_inferlet(
            self._loop, self._client, self.wasm, self.manifest, "text-completion"
        )

        # Install loglikelihood
        self._ll_inferlet_name = _install_inferlet(
            self._loop, self._client, _DEFAULT_LL_WASM, _DEFAULT_LL_MANIFEST, "loglikelihood"
        )

    # ------------------------------------------------------------------
    # Shared WebSocket helper
    # ------------------------------------------------------------------

    async def _run_inferlet(self, inferlet_name: str, args: list[str]) -> str:
        """Launch an inferlet instance and collect its output."""
        assert self._client is not None
        instance = await self._client.launch_instance(inferlet_name, arguments=args)
        output = ""
        while True:
            event, msg = await instance.recv()
            if event == Event.Stdout:
                output += msg
            elif event == Event.Completed:
                output = msg if msg else output
                break
            elif event in (
                Event.Exception, Event.Aborted,
                Event.ServerError, Event.OutOfResources,
            ):
                raise RuntimeError(f"Pie error ({event.name}): {msg}")
        return output

    # ------------------------------------------------------------------
    # generate_until — text-completion inferlet
    # ------------------------------------------------------------------

    async def _generate_one(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str]
    ) -> str:
        """Generate a single completion via Pie WebSocket."""
        args = [
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--system", "You are a helpful assistant.",
        ]
        output = await self._run_inferlet(self._inferlet_name, args)

        # Apply stop sequences (truncate at first match)
        for s in stop:
            idx = output.find(s)
            if idx >= 0:
                output = output[:idx]
        return output

    async def _generate_batch(self, requests) -> list[str]:
        """Run multiple generation requests concurrently."""
        sem = asyncio.Semaphore(self.num_concurrent)

        async def _bounded(prompt, max_tokens, temperature, stop):
            async with sem:
                return await self._generate_one(
                    prompt, max_tokens, temperature, stop
                )

        tasks = []
        for req in requests:
            context, gen_kwargs = req.args
            max_tokens = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            tasks.append(_bounded(context, max_tokens, temperature, until))

        return await asyncio.gather(*tasks)

    def generate_until(self, requests) -> list[str]:
        """Generate completions for a batch of lm-eval Instance requests."""
        assert self._loop is not None
        return self._loop.run_until_complete(self._generate_batch(requests))

    # ------------------------------------------------------------------
    # loglikelihood — loglikelihood inferlet
    # ------------------------------------------------------------------

    async def _loglikelihood_batch(self, requests) -> list[tuple[float, bool]]:
        """Score (context, continuation) pairs via the loglikelihood inferlet.

        Groups requests by context so that MCQ choices sharing the same prompt
        are evaluated in a single inferlet call (shared KV cache via fork).
        """
        # Group by context for efficiency
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, req in enumerate(requests):
            context, continuation = req.args
            groups[context].append((i, continuation))

        results: list[tuple[float, bool] | None] = [None] * len(requests)
        sem = asyncio.Semaphore(self.num_concurrent)

        async def _score_group(context: str, items: list[tuple[int, str]]):
            async with sem:
                continuations = [cont for _, cont in items]
                args = [
                    "--context", context,
                    "--continuations", json.dumps(continuations),
                ]
                output = await self._run_inferlet(self._ll_inferlet_name, args)
                scores = json.loads(output)
                for (idx, _), score in zip(items, scores):
                    logprob = score["logprob"]
                    # Handle JSON infinity (serde_json writes null for inf)
                    if logprob is None or logprob == float("-inf"):
                        logprob = float("-inf")
                    results[idx] = (float(logprob), bool(score["is_greedy"]))

        tasks = [_score_group(ctx, items) for ctx, items in groups.items()]
        await asyncio.gather(*tasks)
        return results

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Score continuations for loglikelihood-based tasks (ARC MCQ, etc.)."""
        assert self._loop is not None
        return self._loop.run_until_complete(self._loglikelihood_batch(requests))

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling is not supported by Pie — "
            "use loglikelihood or generate_until tasks."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return 32768

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "pie"

    def __del__(self):
        """Clean up connection."""
        if self._client is not None and self._loop is not None:
            try:
                self._loop.run_until_complete(self._client.close())
            except Exception:
                pass
            self._loop.close()
