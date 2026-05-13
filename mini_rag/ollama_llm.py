"""HTTP client for Ollama chat completions."""

from __future__ import annotations

import httpx

from mini_rag.config import Settings

# Reuse one client per process: avoids new TCP handshakes to localhost on each ask.
_http: httpx.Client | None = None


def _http_client(timeout_s: float) -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=timeout_s)
    return _http


class OllamaError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def chat(settings: Settings, system: str, user: str, timeout_s: float = 120.0) -> str:
    url = f"{settings.ollama_base_url}/api/chat"
    options: dict[str, int | float] = {
        "num_predict": settings.ollama_num_predict,
        "num_ctx": settings.ollama_num_ctx,
        "temperature": settings.ollama_temperature,
    }
    payload: dict[str, object] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
        "keep_alive": settings.ollama_keep_alive,
    }
    try:
        client = _http_client(timeout_s)
        r = client.post(url, json=payload)
    except httpx.RequestError as e:
        raise OllamaError(f"Could not reach Ollama at {url}: {e}") from e

    if r.status_code != 200:
        raise OllamaError(
            f"Ollama returned HTTP {r.status_code}: {r.text[:500]}",
            status_code=r.status_code,
        )

    data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OllamaError(f"Unexpected Ollama response: {data!r}")
    return content.strip()
