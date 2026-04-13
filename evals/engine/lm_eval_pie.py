"""Pie as an lm-evaluation-harness model.

Wraps PieEngine (WebSocket + text-completion inferlet) as an ``lm_eval.api.model.LM``
subclass so that lm-eval-harness can drive inference through Pie.

Only ``generate_until`` is supported — Pie doesn't expose token-level logprobs
over its WebSocket protocol, so loglikelihood-based tasks are not available.
"""

import asyncio
import sys
import tomllib
from pathlib import Path

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from pie_client import PieClient, Event


# Default paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WASM = (
    _REPO_ROOT / "std" / "text-completion" / "target"
    / "wasm32-wasip2" / "release" / "text_completion.wasm"
)
_DEFAULT_MANIFEST = _REPO_ROOT / "std" / "text-completion" / "Pie.toml"


@register_model("pie")
class PieLM(LM):
    """lm-eval model backed by a running Pie server.

    Uses the text-completion inferlet for generation, same as PieEngine.
    Instantiated by lm-eval via ``model_args`` string, e.g.::

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
        self._loop: asyncio.AbstractEventLoop | None = None

        # Eagerly set up connection
        self._setup()

    def _setup(self) -> None:
        """Connect to Pie server and install inferlet."""
        if not self.wasm.exists():
            print(f"Error: WASM not found at {self.wasm}")
            print(
                "Run `cargo build --target wasm32-wasip2 --release` "
                "in std/text-completion/"
            )
            sys.exit(1)
        if not self.manifest.exists():
            print(f"Error: Manifest not found at {self.manifest}")
            sys.exit(1)

        pkg = tomllib.loads(self.manifest.read_text()).get("package", {})
        pkg_name = pkg.get("name", "text-completion")
        version = pkg.get("version", "0.1.0")
        self._inferlet_name = f"{pkg_name}@{version}"

        # Create a new event loop for async operations
        self._loop = asyncio.new_event_loop()
        self._client = PieClient(self.server)
        self._loop.run_until_complete(self._client.connect())
        self._loop.run_until_complete(self._client.authenticate("eval-user"))

        if not self._loop.run_until_complete(
            self._client.program_exists(
                self._inferlet_name, self.wasm, self.manifest
            )
        ):
            print(f"  Installing {self._inferlet_name}...")
            self._loop.run_until_complete(
                self._client.install_program(self.wasm, self.manifest)
            )
        else:
            print(f"  {self._inferlet_name} already installed.")

    async def _generate_one(
        self, prompt: str, max_tokens: int, temperature: float, stop: list[str]
    ) -> str:
        """Generate a single completion via Pie WebSocket."""
        assert self._client is not None
        assert self._inferlet_name is not None

        args = [
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--system", "You are a helpful assistant.",
        ]
        instance = await self._client.launch_instance(
            self._inferlet_name, arguments=args
        )

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

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "Pie doesn't expose token logprobs — "
            "only generate_until tasks are supported."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "Pie doesn't expose token logprobs — "
            "only generate_until tasks are supported."
        )

    @property
    def eot_token_id(self):
        # Not applicable for API-based models
        return None

    @property
    def max_length(self):
        # Pie doesn't expose context window info over WebSocket
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
