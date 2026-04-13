"""Pie inference engine — WebSocket + text-completion inferlet."""

import sys
import tomllib
from pathlib import Path

from pie_client import PieClient, Event

from .base import Engine, GenerationParams

# Default paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WASM = (
    _REPO_ROOT / "std" / "text-completion" / "target"
    / "wasm32-wasip2" / "release" / "text_completion.wasm"
)
_DEFAULT_MANIFEST = _REPO_ROOT / "std" / "text-completion" / "Pie.toml"


class PieEngine(Engine):
    """Engine that talks to a running Pie server via WebSocket.

    Uses the text-completion inferlet for generation. Follows the same
    connection/install pattern as ``benches/bench_utils.py``.
    """

    def __init__(
        self,
        name: str = "pie",
        server: str = "ws://127.0.0.1:8080",
        wasm_path: str | None = None,
        manifest_path: str | None = None,
    ):
        super().__init__(name)
        self.server = server
        self.wasm = Path(wasm_path) if wasm_path else _DEFAULT_WASM
        self.manifest = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
        self._client: PieClient | None = None
        self._inferlet_name: str | None = None

    async def setup(self) -> None:
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

        self._client = PieClient(self.server)
        await self._client.connect()
        await self._client.authenticate("eval-user")

        if not await self._client.program_exists(
            self._inferlet_name, self.wasm, self.manifest
        ):
            print(f"  Installing {self._inferlet_name}...")
            await self._client.install_program(self.wasm, self.manifest)
        else:
            print(f"  {self._inferlet_name} already installed.")

    async def generate(self, prompt: str, params: GenerationParams) -> str:
        assert self._client is not None
        assert self._inferlet_name is not None

        args = [
            "--prompt", prompt,
            "--max-tokens", str(params.max_tokens),
            "--temperature", str(params.temperature),
            "--system", params.system,
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

        return output

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
