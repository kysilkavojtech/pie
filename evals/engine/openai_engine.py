"""OpenAI-compatible HTTP engine for vLLM and SGLang."""

import httpx

from .base import Engine, GenerationParams


class OpenAIEngine(Engine):
    """Engine that talks to any OpenAI-compatible /v1/chat/completions endpoint.

    Works with vLLM (default port 8000) and SGLang (default port 30000).
    """

    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8000",
        model_name: str | None = None,
        timeout: float = 120.0,
    ):
        super().__init__(name)
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        )
        # Auto-detect model name if not provided
        if self._model_name is None:
            self._model_name = await self._detect_model()

    async def _detect_model(self) -> str:
        """Query /v1/models to find the served model name."""
        assert self._client is not None
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError(
                f"No models found at {self.base_url}/v1/models — "
                "is the server running?"
            )
        return models[0]["id"]

    async def generate(self, prompt: str, params: GenerationParams) -> str:
        assert self._client is not None
        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": params.system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
