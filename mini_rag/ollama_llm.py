"""HTTP client for Ollama chat completions."""

from __future__ import annotations

import httpx

from mini_rag.config import Settings


class OllamaError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def chat(settings: Settings, system: str, user: str, timeout_s: float = 120.0) -> str:
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
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
